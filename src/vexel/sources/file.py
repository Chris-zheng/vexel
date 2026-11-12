"""File data source: a recorded acquisition, fully available up front."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.readers import open_volume


class FrameSource(Protocol):
    """What executors consume. Batch execution uses ``volume()``; stream execution
    (Phase 3) will use ``frames()``."""

    def volume(self) -> SpatialVolume: ...

    def frames(self, channel: int = 0, z: int = 0) -> Iterator[tuple[int, np.ndarray]]: ...


class FileSource:
    def __init__(self, path: str | Path, **calibration: float | str | None) -> None:
        self.path = Path(path)
        self._volume = open_volume(self.path, **calibration)

    def volume(self) -> SpatialVolume:
        return self._volume

    def frames(self, channel: int = 0, z: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        lazy = self._volume.frames(channel, z)
        for t in range(lazy.shape[0]):
            yield t, np.asarray(lazy[t])
