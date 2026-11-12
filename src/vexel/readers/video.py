"""Video reader (PyAV). Frames are decoded lazily in sequential blocks."""

from __future__ import annotations

from pathlib import Path

import av
import dask
import dask.array as da
import imageio.v3 as iio
import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.readers.base import Reader, register_reader

_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
_BLOCK = 64


def _probe(path: Path) -> tuple[int, int, int, float | None]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        n = stream.frames
        if not n:  # header has no count: demux (no decoding) to count packets
            n = sum(1 for p in container.demux(stream) if p.size)
        rate = stream.average_rate
        fps = float(rate) if rate else None
        return int(n), int(stream.codec_context.height), int(stream.codec_context.width), fps


def _read_block(path: str, start: int, stop: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.empty((stop - start, *shape, 3), dtype=np.uint8)
    with iio.imopen(path, "r", plugin="pyav") as f:
        for i in range(start, stop):
            out[i - start] = f.read(index=i, format="rgb24")
    return out


class VideoReader(Reader):
    name = "video"

    def can_read(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in _EXTS

    def open(self, path: Path) -> SpatialVolume:
        n, h, w, fps = _probe(path)
        if n <= 0:
            raise ValueError(f"{path}: no video frames found")
        read = dask.delayed(_read_block, pure=True)
        blocks = [
            da.from_delayed(read(str(path), a, min(n, a + _BLOCK), (h, w)), shape=(min(n, a + _BLOCK) - a, h, w, 3), dtype=np.uint8)
            for a in range(0, n, _BLOCK)
        ]
        data = da.concatenate(blocks)
        return SpatialVolume.from_array(
            data,
            "TYXC",
            spacing={"t": 1.0 / fps} if fps else {},
            time_unit="s" if fps else "frame",
            channel_names=["R", "G", "B"],
            source=str(path),
            metadata={"reader": self.name, "fps": fps},
        )


register_reader(VideoReader())
