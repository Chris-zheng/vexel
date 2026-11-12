"""OME-Zarr (NGFF 0.4) multiscale cache: write a SpatialVolume as a pyramid and read it back."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import zarr

from vexel.core.volume import AXES, SpatialVolume

_NGFF_VERSION = "0.4"
_UNIT_NAMES = {"um": "micrometer", "nm": "nanometer", "mm": "millimeter", "s": "second", "ms": "millisecond"}
_UNIT_SHORT = {v: k for k, v in _UNIT_NAMES.items()}


def _chunks_for(shape: tuple[int, ...]) -> tuple[int, ...]:
    y = min(shape[3], 1024)
    x = min(shape[4], 1024)
    return (1, 1, 1, y, x)


def write_ome_zarr(volume: SpatialVolume, path: str | Path, *, levels: int = 3, overwrite: bool = False) -> Path:
    """Write ``volume`` as an OME-Zarr pyramid, downsampling Y/X by 2 per level (mean).

    Data is streamed chunk by chunk; the full volume is never held in memory.
    """
    path = Path(path)
    root = zarr.open_group(str(path), mode="w" if overwrite else "w-", zarr_format=2)
    datasets = []
    level = volume.data
    for i in range(max(1, levels)):
        if i > 0:
            if min(level.shape[3], level.shape[4]) // 2 < 64:  # stop before a level gets tiny
                break
            level = da.coarsen(da.mean, level, {3: 2, 4: 2}, trim_excess=True).astype(volume.dtype)
        chunks = _chunks_for(level.shape)
        target = root.create_array(name=str(i), shape=level.shape, chunks=chunks, dtype=level.dtype)
        da.store(level.rechunk(chunks), target, lock=False)
        f = 2**i
        scale = [volume.spacing["t"], 1.0, volume.spacing["z"], volume.spacing["y"] * f, volume.spacing["x"] * f]
        datasets.append({"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": scale}]})

    def axis(name: str) -> dict:
        if name == "c":
            return {"name": "c", "type": "channel"}
        if name == "t":
            out = {"name": "t", "type": "time"}
            unit = _UNIT_NAMES.get(volume.time_unit)
        else:
            out = {"name": name, "type": "space"}
            unit = _UNIT_NAMES.get(volume.space_unit)
        if unit:
            out["unit"] = unit
        return out

    root.attrs["multiscales"] = [
        {"version": _NGFF_VERSION, "name": path.stem, "axes": [axis(a) for a in AXES.lower()], "datasets": datasets}
    ]
    root.attrs["vexel"] = {
        "space_unit": volume.space_unit,
        "time_unit": volume.time_unit,
        "channel_names": volume.channel_names,
        "source": volume.source,
    }
    return path


def read_ome_zarr(path: str | Path, *, level: int = 0) -> SpatialVolume:
    """Open one resolution level of an OME-Zarr image lazily."""
    path = Path(path)
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
    if not multiscales:
        raise ValueError(f"{path} has no OME-Zarr multiscales metadata")
    ms = multiscales[0]
    axes = "".join(a["name"] for a in ms["axes"]).upper()
    dataset = ms["datasets"][level]
    data = da.from_zarr(root[dataset["path"]])
    scale = next((t["scale"] for t in dataset.get("coordinateTransformations", []) if t["type"] == "scale"), None)
    spacing = {a.lower(): float(s) for a, s in zip(axes, scale or []) if a != "C"}
    units = {a["name"]: _UNIT_SHORT.get(a.get("unit", ""), None) for a in ms["axes"]}
    extra = attrs.get("vexel", {})
    return SpatialVolume.from_array(
        data,
        axes,
        spacing=spacing,
        space_unit=extra.get("space_unit") or units.get("x") or "px",
        time_unit=extra.get("time_unit") or units.get("t") or "frame",
        channel_names=extra.get("channel_names"),
        source=str(path),
        metadata={"reader": "ome-zarr", "level": level, "n_levels": len(ms["datasets"])},
    )
