"""Reader plugin registry.

A reader turns a path into a lazy :class:`SpatialVolume`. Opening reads metadata only;
pixels are decoded when a downstream consumer asks for them. Add a format by
subclassing :class:`Reader` and calling :func:`register_reader`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vexel.core.volume import SpatialVolume

_UNIT_ALIASES = {"µm": "um", "μm": "um", "micron": "um", "microns": "um", "micrometer": "um", "sec": "s", "second": "s"}


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    unit = unit.strip()
    return _UNIT_ALIASES.get(unit, _UNIT_ALIASES.get(unit.lower(), unit))


class Reader:
    name: str = ""

    def can_read(self, path: Path) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def open(self, path: Path) -> SpatialVolume:  # pragma: no cover - interface
        raise NotImplementedError


_READERS: list[Reader] = []


def register_reader(reader: Reader) -> Reader:
    _READERS.append(reader)
    return reader


def list_readers() -> list[str]:
    return [r.name for r in _READERS]


def _sidecar(path: Path) -> dict[str, Any]:
    """Optional metadata next to the data: ``<file>.vexel.json`` or ``<dir>/vexel.json``."""
    candidates = [path / "vexel.json"] if path.is_dir() else [path.with_name(path.name + ".vexel.json")]
    for c in candidates:
        if c.is_file():
            return json.loads(c.read_text(encoding="utf-8"))
    return {}


def open_volume(
    path: str | Path,
    *,
    pixel_size: float | None = None,
    space_unit: str | None = None,
    frame_interval: float | None = None,
) -> SpatialVolume:
    """Open ``path`` with the first reader that accepts it.

    Calibration precedence: explicit arguments > sidecar JSON > file metadata.
    ``pixel_size`` is in ``space_unit`` (default ``um``); ``frame_interval`` in seconds.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    for reader in _READERS:
        if reader.can_read(path):
            volume = reader.open(path)
            break
    else:
        raise ValueError(f"no reader for {path} (available: {', '.join(list_readers())})")

    side = _sidecar(path)
    px = pixel_size if pixel_size is not None else side.get("pixel_size")
    dt = frame_interval if frame_interval is not None else side.get("frame_interval")
    if px is not None:
        if px <= 0:
            raise ValueError("pixel_size must be positive")
        volume.spacing["y"] = volume.spacing["x"] = float(px)
        volume.space_unit = normalize_unit(space_unit or side.get("space_unit")) or "um"
    if dt is not None:
        if dt <= 0:
            raise ValueError("frame_interval must be positive")
        volume.spacing["t"] = float(dt)
        volume.time_unit = "s"
    return volume
