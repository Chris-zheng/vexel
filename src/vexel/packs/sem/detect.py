"""How likely a file is an SEM micrograph (0..1), from the instrument metadata it carries.

See ``vexel.packs.sem.metadata`` for where each vendor writes it.
"""

from __future__ import annotations

from pathlib import Path

from vexel.packs.sem.metadata import VENDOR_NAMES, vendor_of

IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VENDORS = VENDOR_NAMES


def tiff_pages(path: Path) -> int:
    import tifffile

    try:
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages)
    except Exception:  # noqa: BLE001  not a readable TIFF
        return 0


def score(path: Path) -> float:
    path = Path(path)
    if path.is_dir() or path.suffix.lower() not in IMAGES:
        return 0.0
    if vendor_of(path):
        return 1.0
    if path.suffix.lower() in (".tif", ".tiff") and tiff_pages(path) > 1:
        return 0.1  # a stack or recording
    return 0.4  # a single image: possibly a micrograph, the person decides
