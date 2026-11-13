"""TIFF / OME-TIFF / ImageJ TIFF reader (lazy, one chunk per page)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import dask.array as da
import tifffile
import zarr

from vexel.core.volume import SpatialVolume
from vexel.readers.base import Reader, normalize_unit, register_reader

# tifffile axis letters -> TCZYX. S (samples, e.g. RGB) is treated as channels;
# I/Q (unspecified sequence) as time.
_AXIS_MAP = {"T": "T", "C": "C", "Z": "Z", "Y": "Y", "X": "X", "S": "C", "I": "T", "Q": "T"}


def _ome_calibration(xml: str) -> dict:
    out: dict = {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    pixels = next((el for el in root.iter() if el.tag.endswith("}Pixels") or el.tag == "Pixels"), None)
    if pixels is None:
        return out
    a = pixels.attrib
    for ax in "XYZ":
        if f"PhysicalSize{ax}" in a:
            out[ax.lower()] = float(a[f"PhysicalSize{ax}"])
    if "x" in out or "y" in out:
        out["space_unit"] = normalize_unit(a.get("PhysicalSizeXUnit", "µm"))
    if "TimeIncrement" in a:
        out["t"] = float(a["TimeIncrement"])
        out["time_unit"] = normalize_unit(a.get("TimeIncrementUnit", "s"))
    names = [el.attrib.get("Name", "") for el in pixels if el.tag.endswith("}Channel") or el.tag == "Channel"]
    if names and any(names):
        out["channel_names"] = names
    return out


def _imagej_calibration(tif: tifffile.TiffFile) -> dict:
    out: dict = {}
    meta = tif.imagej_metadata or {}
    unit = normalize_unit(meta.get("unit"))
    page = tif.pages[0]
    for ax, tag in (("x", "XResolution"), ("y", "YResolution")):
        t = page.tags.get(tag)
        if t is not None and unit:
            num, den = t.value
            if num:
                out[ax] = den / num
    if "x" in out:
        out["space_unit"] = unit
    if "spacing" in meta:
        out["z"] = float(meta["spacing"])
    if meta.get("finterval"):
        out["t"] = float(meta["finterval"])
        out["time_unit"] = "s"
    return out


class TiffReader(Reader):
    name = "tiff"

    def can_read(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {".tif", ".tiff"}

    def open(self, path: Path) -> SpatialVolume:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            raw_axes = series.axes
            if tif.is_ome and tif.ome_metadata:
                calib = _ome_calibration(tif.ome_metadata)
            elif tif.is_imagej:
                calib = _imagej_calibration(tif)
            else:
                calib = {}
        axes = "".join(_AXIS_MAP.get(a, "?") for a in raw_axes)
        if "?" in axes or len(set(axes)) != len(axes):
            raise ValueError(f"{path}: cannot map TIFF axes {raw_axes!r} onto TCZYX")
        store = tifffile.imread(path, aszarr=True, series=0, chunkmode="page")
        data = da.from_zarr(zarr.open(store, mode="r"))
        return SpatialVolume.from_array(
            data,
            axes,
            spacing={k: calib[k] for k in "tzyx" if k in calib},
            space_unit=calib.get("space_unit") or "px",
            time_unit=calib.get("time_unit") or "frame",
            channel_names=calib.get("channel_names"),
            source=str(path),
            metadata={"reader": self.name, "tiff_axes": raw_axes},
        )


register_reader(TiffReader())
