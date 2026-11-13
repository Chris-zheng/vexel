"""Data sources: the file source (batch), the replay source (stream executor; a recorded
acquisition delivered frame by frame), and ``camera``, an interface only until hardware
integration."""

from vexel.sources.camera import CameraSource
from vexel.sources.file import FileSource, FrameSource
from vexel.sources.replay import Frame, ReplaySource

__all__ = ["CameraSource", "FileSource", "Frame", "FrameSource", "ReplaySource"]
