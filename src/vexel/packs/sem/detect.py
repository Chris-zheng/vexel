"""How likely a file is an SEM micrograph (0..1), from vendor metadata.

Thermo Fisher / FEI write an INI-style block in private TIFF tags 34680/34682; Zeiss
SmartSEM writes its parameters in tag 34118; TESCAN writes a ``.hdr`` text file and
Hitachi and JEOL a ``.txt`` file next to the image.
"""

from __future__ import annotations

from pathlib import Path

VENDOR_TAGS = {34680: "thermo_fisher", 34682: "thermo_fisher", 34118: "zeiss"}
IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def tiff_info(path: Path) -> tuple[int, str | None]:
    """Number of pages and the SEM vendor whose private tag the first page carries."""
    import tifffile

    try:
        with tifffile.TiffFile(path) as tf:
            tags = tf.pages[0].tags
            vendor = next((v for code, v in VENDOR_TAGS.items() if code in tags), None)
            return len(tf.pages), vendor
    except Exception:  # noqa: BLE001  not a readable TIFF
        return 0, None


def sidecar_vendor(path: Path) -> str | None:
    """TESCAN .hdr, or Hitachi / JEOL .txt next to the image."""
    for suffix, keys, vendor in ((".hdr", ("PixelSizeX",), "tescan"), (".txt", ("PixelSize", "$CM_", "SemImageFile"), "hitachi_jeol")):
        side = path.with_suffix(suffix)
        if side.is_file():
            try:
                head = side.read_text(encoding="utf-8", errors="replace")[:20000]
            except OSError:
                continue
            if any(k in head for k in keys):
                return vendor
    return None


def score(path: Path) -> float:
    path = Path(path)
    if path.is_dir():
        return 0.0
    suffix = path.suffix.lower()
    if suffix not in IMAGES:
        return 0.0
    if suffix in (".tif", ".tiff"):
        n_pages, vendor = tiff_info(path)
        if vendor:
            return 1.0
        if n_pages > 1:
            return 0.1
    if sidecar_vendor(path):
        return 0.95
    return 0.4  # a single image: possibly a micrograph, the person decides
