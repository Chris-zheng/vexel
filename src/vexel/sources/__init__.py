"""Data sources. Phase 1 ships the file source; ``watch`` and ``replay`` arrive with the
stream executor (Phase 3); ``camera`` is an interface only until hardware integration."""

from vexel.sources.camera import CameraSource
from vexel.sources.file import FileSource, FrameSource

__all__ = ["CameraSource", "FileSource", "FrameSource"]
