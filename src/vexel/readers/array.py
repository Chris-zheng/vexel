"""Readers for ``.npy`` arrays (memory-mapped) and OME-Zarr."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vexel.core.cache import read_ome_zarr
from vexel.core.volume import SpatialVolume
from vexel.readers.base import Reader, register_reader

_NPY_AXES = {2: "YX", 3: "TYX", 4: "TCYX", 5: "TCZYX"}


class NpyReader(Reader):
    name = "npy"

    def can_read(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".npy"

    def open(self, path: Path) -> SpatialVolume:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim not in _NPY_AXES:
            raise ValueError(f"{path}: unsupported array with {arr.ndim} dimensions")
        return SpatialVolume.from_array(arr, _NPY_AXES[arr.ndim], source=str(path), metadata={"reader": self.name})


class OmeZarrReader(Reader):
    name = "ome-zarr"

    def can_read(self, path: Path) -> bool:
        return path.is_dir() and ((path / ".zgroup").exists() or (path / "zarr.json").exists())

    def open(self, path: Path) -> SpatialVolume:
        return read_ome_zarr(path)


register_reader(NpyReader())
register_reader(OmeZarrReader())
