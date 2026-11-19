"""How likely a file is a droplet recording (0..1): many frames over time."""

from __future__ import annotations

from pathlib import Path

VIDEO = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def score(path: Path) -> float:
    path = Path(path)
    if path.is_dir():
        if path.suffix.lower() == ".zarr":
            return 0.7
        images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGES]
        return 0.75 if len(images) >= 5 else 0.0
    suffix = path.suffix.lower()
    if suffix in VIDEO:
        return 0.9
    if suffix == ".npy":
        return 0.6
    if suffix in (".tif", ".tiff"):
        n_pages, microscope = _tiff_info(path)
        if microscope:
            return 0.05
        return 0.85 if n_pages > 1 else 0.2
    if suffix in IMAGES:
        return 0.15
    return 0.0


# private TIFF tags electron microscopes write (Thermo Fisher, Zeiss, TESCAN): not a droplet recording
_MICROSCOPE_TAGS = (34680, 34682, 34118, 50431)


def _tiff_info(path: Path) -> tuple[int, bool]:
    import tifffile

    try:
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages), any(code in tf.pages[0].tags for code in _MICROSCOPE_TAGS)
    except Exception:  # noqa: BLE001  not a readable TIFF
        return 0, False
