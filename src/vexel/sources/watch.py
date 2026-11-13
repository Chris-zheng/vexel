"""Watch source: an acquisition folder processed image by image as files arrive.

Acquisition software that writes one image file per frame (or per slice) into a folder
is followed live: a file is taken once its size has stopped changing, files are taken
in natural name order (``img_2`` before ``img_10``), and each frame is appended to an
OME-Zarr store if one is requested, so the acquisition is available as a single volume
the moment it ends. Watching ends when no new file has arrived for ``idle_timeout_s``,
when a file named ``STOP`` appears in the folder, or when the executor asks it to.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.sources.replay import Frame

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
STOP_FILE = "STOP"


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _read_channels(path: Path) -> list[np.ndarray]:
    """One 2-D array per channel: grayscale -> 1 channel; (Y, X, 3|4) -> RGB channels;
    (C, Y, X) with few planes -> those channels."""
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    else:
        import imageio.v3 as iio

        arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 8:
        return [arr[..., i] for i in range(3)]
    if arr.ndim == 3 and arr.shape[0] <= 8:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"{path.name}: expected a 2-D image (or a few channels), got shape {arr.shape}")


class WatchSource:
    def __init__(
        self,
        folder: str | Path,
        sources: dict[str, int],
        *,
        pixel_size: float | None = None,
        space_unit: str | None = None,
        frame_interval_s: float | None = None,
        poll_s: float = 0.2,
        idle_timeout_s: float = 10.0,
        include_existing: bool = True,
        max_frames: int | None = None,
        zarr_path: str | Path | None = None,
    ) -> None:
        """``sources`` maps value references to channel numbers (``Recipe.input.sources()``).
        Calibration not given here is read from ``<folder>/vexel.json`` if present."""
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"not a folder: {self.folder}")
        sidecar = self.folder / "vexel.json"
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.pixel_size = pixel_size if pixel_size is not None else meta.get("pixel_size")
        self.space_unit = space_unit or meta.get("space_unit") or ("um" if self.pixel_size else "px")
        self.frame_interval_s = frame_interval_s if frame_interval_s is not None else meta.get("frame_interval")
        self.sources = dict(sources)
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self.max_frames = max_frames
        self.zarr_path = Path(zarr_path) if zarr_path else None
        self.n_frames: int | None = None
        self.live = True
        self.files: list[str] = []
        self.ended_by: str | None = None
        self._stop = threading.Event()
        self._skip = set() if include_existing else {p.name for p in self._candidates()}
        self._frame: tuple[tuple[int, int], np.dtype] = ((1, 1), np.dtype(np.uint8))

    @property
    def volume(self) -> SpatialVolume:
        """Calibration and size of what has arrived so far (no pixels are held)."""
        import dask.array as da

        shape, dtype = self._frame
        spacing = {"t": self.frame_interval_s or 1.0}
        if self.pixel_size:
            spacing.update({"y": float(self.pixel_size), "x": float(self.pixel_size)})
        return SpatialVolume.from_array(
            da.zeros((max(1, len(self.files)), *shape), dtype=dtype, chunks=(1, *shape)), "TYX", spacing=spacing,
            space_unit=self.space_unit if self.pixel_size else "px", time_unit="s" if self.frame_interval_s else "frame",
            source=str(self.folder), metadata={"reader": "watch", "files": len(self.files)},
        )  # fmt: skip

    def _candidates(self) -> list[Path]:
        return [p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")]

    def request_stop(self) -> None:
        self._stop.set()

    def describe(self) -> dict:
        return {
            "kind": "watch",
            "source": str(self.folder),
            "n_frames": len(self.files),
            "frame_interval_s": self.frame_interval_s,
            "idle_timeout_s": self.idle_timeout_s,
            "ended_by": self.ended_by,
            "zarr": str(self.zarr_path) if self.zarr_path else None,
        }

    def __iter__(self) -> Iterator[Frame]:
        sizes: dict[str, int] = {}  # name -> size at the previous poll (unchanged twice = complete)
        appender = None
        last_new = time.monotonic()
        index = 0
        while True:
            if self._stop.is_set():
                self.ended_by = "stop request"
                return
            if (self.folder / STOP_FILE).exists():
                self.ended_by = "STOP file"
                return
            ready = []
            for path in sorted(self._candidates(), key=_natural_key):
                if path.name in self._skip:
                    continue
                size = path.stat().st_size
                if size > 0 and sizes.get(path.name) == size:
                    ready.append(path)
                sizes[path.name] = size
            for path in ready:
                self._skip.add(path.name)
                channels = _read_channels(path)
                if max(self.sources.values()) >= len(channels):
                    raise ValueError(f"{path.name} has {len(channels)} channel(s); the recipe reads channel {max(self.sources.values())}")
                due = time.monotonic()
                if index == 0:
                    self._frame = (channels[0].shape, channels[0].dtype)
                if self.zarr_path is not None:
                    if appender is None:
                        from vexel.core.cache import OmeZarrAppender

                        spacing = {"t": self.frame_interval_s or 1.0}
                        if self.pixel_size:
                            spacing.update({"y": float(self.pixel_size), "x": float(self.pixel_size)})
                        appender = OmeZarrAppender(
                            self.zarr_path, channels[0].shape, channels[0].dtype, len(channels), spacing=spacing,
                            space_unit=self.space_unit if self.pixel_size else "px",
                            time_unit="s" if self.frame_interval_s else "frame", source=str(self.folder),
                        )  # fmt: skip
                    appender.append(channels)
                self.files.append(path.name)
                yield Frame(index, {ref: channels[c] for ref, c in self.sources.items()}, due)
                index += 1
                last_new = time.monotonic()
                if self.max_frames is not None and index >= self.max_frames:
                    self.ended_by = "max_frames"
                    return
            if time.monotonic() - last_new > self.idle_timeout_s:
                self.ended_by = "idle timeout"
                return
            self._stop.wait(self.poll_s)
