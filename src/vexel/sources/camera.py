"""Camera source: interface only.

Hardware integration (camera SDKs, GenICam) is deferred. When it happens, only this
module gains an implementation; the engine does not change.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from vexel.core.volume import SpatialVolume


class CameraSource:
    def volume(self) -> SpatialVolume:
        raise NotImplementedError("camera sources are deferred until hardware integration")

    def frames(self, channel: int = 0, z: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        raise NotImplementedError("camera sources are deferred until hardware integration")
