"""Single-image reader for standard 2-D formats (PNG, JPEG, BMP, WebP).

Complements ``tiff.py``, ``sequence.py``, ``video.py``, and ``array.py`` so both SEM and
microfluidic single-frame images open cleanly into a unified ``(1, C, 1, Y, X)``
:class:`SpatialVolume`.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import imageio.v3 as iio
import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.readers.base import Reader, register_reader

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class SingleImageReader(Reader):
    name = "image"

    def can_read(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in _IMAGE_EXTS

    def open(self, path: Path) -> SpatialVolume:
        arr = np.asarray(iio.imread(path))
        if arr.ndim == 2:
            tczyx = arr[None, None, None, :, :]
        elif arr.ndim == 3:
            # Convert HWC -> (1, C, 1, H, W); drop alpha if RGBA
            c = min(arr.shape[2], 3)
            tczyx = np.transpose(arr[:, :, :c], (2, 0, 1))[None, :, None, :, :]
        else:
            raise ValueError(f"unsupported image dimensions {arr.shape} in {path}")
        lazy = da.from_array(tczyx, chunks=(1, 1, 1, tczyx.shape[3], tczyx.shape[4]))
        return SpatialVolume(
            data=lazy,
            spacing={"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
            space_unit="px",
            time_unit="frame",
            source=str(path),
            metadata={"reader": self.name, "format": path.suffix.lower().lstrip(".")},
        )


register_reader(SingleImageReader())
