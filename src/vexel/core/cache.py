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


class OmeZarrAppender:
    """An OME-Zarr image (single resolution, TCZYX) that grows by one time point per call,
    for acquisitions written as they arrive. Readable with :func:`read_ome_zarr` at any
    time; the time axis is resized on every append."""

    def __init__(
        self,
        path: str | Path,
        frame_shape: tuple[int, int],
        dtype: str | object,
        n_channels: int = 1,
        *,
        spacing: dict[str, float] | None = None,
        space_unit: str = "px",
        time_unit: str = "frame",
        channel_names: list[str] | None = None,
        source: str | None = None,
    ) -> None:
        self.path = Path(path)
        sp = {"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0, **(spacing or {})}
        root = zarr.open_group(str(self.path), mode="w-", zarr_format=2)
        y, x = frame_shape
        self._array = root.create_array(name="0", shape=(0, n_channels, 1, y, x), chunks=(1, 1, 1, *_chunks_for((1, 1, 1, y, x))[3:]),
                                        dtype=dtype)  # fmt: skip

        def axis(name: str) -> dict:
            if name == "c":
                return {"name": "c", "type": "channel"}
            unit = _UNIT_NAMES.get(time_unit if name == "t" else space_unit)
            out = {"name": name, "type": "time" if name == "t" else "space"}
            if unit:
                out["unit"] = unit
            return out

        scale = [sp["t"], 1.0, sp["z"], sp["y"], sp["x"]]
        root.attrs["multiscales"] = [
            {"version": _NGFF_VERSION, "name": self.path.stem, "axes": [axis(a) for a in AXES.lower()],
             "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}]}
        ]  # fmt: skip
        root.attrs["vexel"] = {"space_unit": space_unit, "time_unit": time_unit, "channel_names": channel_names, "source": source}
        self.n_frames = 0

    def append(self, channels: list[object]) -> None:
        """Add one time point; ``channels`` holds one 2-D array per channel."""
        import numpy as np

        stack = np.stack([np.asarray(c) for c in channels])[:, None]  # (C, 1, Y, X)
        t = self.n_frames
        self._array.resize((t + 1, *self._array.shape[1:]))
        self._array[t] = stack
        self.n_frames = t + 1
