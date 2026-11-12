"""Image-sequence reader: a directory of same-sized 2-D images, one file per frame."""

from __future__ import annotations

import re
from pathlib import Path

import dask
import dask.array as da
import imageio.v3 as iio
import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.readers.base import Reader, register_reader

_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(name: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def list_frames(path: Path) -> list[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _EXTS and not p.name.startswith(".")]
    return sorted(files, key=lambda p: natural_key(p.name))


def _read(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    frame = np.asarray(iio.imread(path))
    if frame.shape != shape or frame.dtype != dtype:
        raise ValueError(f"{path}: expected {shape} {dtype}, got {frame.shape} {frame.dtype}")
    return frame


class SequenceReader(Reader):
    name = "image-sequence"

    def can_read(self, path: Path) -> bool:
        return path.is_dir() and not (path / ".zgroup").exists() and not (path / "zarr.json").exists() and bool(list_frames(path))

    def open(self, path: Path) -> SpatialVolume:
        files = list_frames(path)
        first = np.asarray(iio.imread(files[0]))  # one frame decides shape and dtype
        if first.ndim not in (2, 3):
            raise ValueError(f"{files[0]}: expected a 2-D or RGB image, got shape {first.shape}")
        read = dask.delayed(_read, pure=True)
        frames = [da.from_delayed(read(f, first.shape, first.dtype), shape=first.shape, dtype=first.dtype) for f in files]
        data = da.stack(frames)
        return SpatialVolume.from_array(
            data,
            "TYX" if first.ndim == 2 else "TYXC",
            source=str(path),
            metadata={"reader": self.name, "n_files": len(files), "first_file": files[0].name},
        )


register_reader(SequenceReader())
