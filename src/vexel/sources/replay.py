"""Replay source: feeds a recorded acquisition frame by frame, as a camera would.

Frames are released at the original frame rate times ``speed`` (``speed=None``: as fast
as they can be read). The stream executor treats them exactly like live frames, which
is how online monitoring is demonstrated and how processing throughput and frame drops
are measured. The timing excludes camera readout and transfer: it is the software's
speed, not end-to-end latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from vexel.core.volume import SpatialVolume


@dataclass
class Frame:
    index: int  # position in the acquisition (gaps mean frames were dropped)
    images: dict[str, np.ndarray]  # value reference ("source", "source.fluo", ...) -> 2-D frame
    due: float  # time.monotonic() at which the frame became available


class ReplaySource:
    def __init__(
        self,
        volume: SpatialVolume,
        sources: dict[str, int],
        z: int = 0,
        *,
        speed: float | None = 1.0,
        frame_interval_s: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        """``sources`` maps value references to channel numbers (``Recipe.input.sources()``).

        The frame interval comes from the data (``volume.time_unit == 's'``) unless given.
        Paced replay (``speed`` set) needs it."""
        if speed is not None and speed <= 0:
            raise ValueError("speed must be positive, or None for as fast as possible")
        self.volume = volume
        self._lazy = {ref: volume.frames(index, z) for ref, index in sources.items()}
        n = int(next(iter(self._lazy.values())).shape[0])
        self.n_frames = min(n, max_frames) if max_frames is not None else n
        if frame_interval_s is None and volume.time_unit == "s":
            frame_interval_s = float(volume.spacing["t"])
        if speed is not None and frame_interval_s is None:
            raise ValueError("paced replay needs the frame interval: give frame_interval_s, or replay as fast as possible")
        self.frame_interval_s = frame_interval_s
        self.speed = speed

    def describe(self) -> dict:
        return {
            "kind": "replay",
            "source": self.volume.source,
            "n_frames": self.n_frames,
            "speed": self.speed,
            "frame_interval_s": self.frame_interval_s,
        }

    def __iter__(self) -> Iterator[Frame]:
        start = time.monotonic()
        for t in range(self.n_frames):
            if self.speed is not None:
                due = start + t * self.frame_interval_s / self.speed
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            else:
                due = time.monotonic()
            images = {ref: np.asarray(lazy[t]) for ref, lazy in self._lazy.items()}
            yield Frame(t, images, due)
