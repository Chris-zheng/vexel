"""SEM calibration: how many metres one pixel is, where that came from, and whether a person confirmed it.

Three sources, tried in order (plan §2.1):

1. **metadata** - the instrument's own record (``metadata.read_metadata``). Adopted
   automatically (confidence "high").
2. **databar** - the scale bar measured in pixels, divided into the length its label
   reads (OCR). Cross-checked against the field width when the bar shows it. Needs a
   person's confirmation before it counts.
3. **manual** - the person gives the scale bar's length (Vexel measures its pixels) or
   the pixel size itself.

When no source works the result says so (``ok`` is False with a reason): Vexel never
returns a calibration it could not establish.

Every calibration is recorded with the image's SHA-256, so a confirmation applies to
exactly that file and is found again next time (``$VEXEL_HOME/sem/calibrations``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vexel.packs.sem import databar as db
from vexel.packs.sem.metadata import SemMetadata, parse_length, read_metadata

AGREE = 0.02  # two independent estimates within 2 %: consistent


@dataclass
class Calibration:
    file: str
    sha256: str
    image_shape: tuple[int, int]
    pixel_size: float | None = None  # metres per pixel (x)
    pixel_size_y: float | None = None
    source: str = "none"  # metadata | databar | manual | none
    confidence: str = "none"  # high | medium | low | manual | none
    confirmed: bool = False
    confirmed_by: str = ""  # "instrument metadata" or "person"
    confirmed_at: str = ""
    vendor: str | None = None
    crop_rows: int | None = None  # rows [0, crop_rows) are the micrograph; None: no data bar
    crop_source: str = "none"  # metadata | detected | none
    scale_bar: dict[str, Any] | None = None
    label: str = ""  # the scale label as read, e.g. "500 µm"
    checks: list[dict[str, Any]] = field(default_factory=list)  # independent estimates and how they agree
    warnings: list[str] = field(default_factory=list)
    reason: str = ""  # why there is no calibration
    metadata: dict[str, Any] | None = None
    texts: list[dict[str, Any]] = field(default_factory=list)  # what OCR read
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def ok(self) -> bool:
        return self.pixel_size is not None

    @property
    def needs_confirmation(self) -> bool:
        return self.ok and not self.confirmed

    def confirm(self, by: str = "person") -> "Calibration":
        self.confirmed, self.confirmed_by = True, by
        self.confirmed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def describe(self) -> str:
        if not self.ok:
            return f"未标定：{self.reason}"
        src = {"metadata": "仪器元数据", "databar": "信息栏比例尺识别", "manual": "手动"}.get(self.source, self.source)
        state = "已确认" if self.confirmed else "待确认"
        return f"{format_length(self.pixel_size)}/像素（{src}，{state}）"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["image_shape"] = list(self.image_shape)
        d["ok"] = self.ok
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibration":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["image_shape"] = tuple(d.get("image_shape", (0, 0)))
        return cls(**d)


def format_length(metres: float | None, digits: int = 4) -> str:
    if metres is None:
        return "-"
    for unit, f in (("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9), ("pm", 1e-12)):
        if abs(metres) >= f:
            return f"{metres / f:.{digits}g} {unit}"
    return f"{metres:.{digits}g} m"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: str | Path) -> np.ndarray:
    """The first image of the file as a 2-D array (colour averaged to grey), unscaled."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        img = tifffile.imread(path, key=0)
    else:
        import imageio.v3 as iio

        img = iio.imread(path)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].astype(np.float32).mean(-1)
    while img.ndim > 2:
        img = img[0]
    return img


def _bar(img: np.ndarray, md: SemMetadata | None, cal: Calibration) -> db.DataBar | None:
    if md is not None and md.databar_rows:
        bar = db.databar_from_rows(img, md.scan_rows or 0)
        if bar is not None:
            cal.crop_rows, cal.crop_source = bar.top, "metadata"
            return bar
    bar = db.find_databar(img)
    if bar is not None:
        cal.crop_rows, cal.crop_source = bar.top, "detected"
    return bar


def _read_bar(img: np.ndarray, bar: db.DataBar | None, cal: Calibration, *, use_ocr: bool) -> tuple[float | None, dict[str, Any]]:
    """Pixel size from the scale bar and its label; also the bar's other fields (field width...)."""
    from vexel.packs.sem import ocr

    top = bar.top if bar is not None else int(img.shape[0] * 0.75)
    texts: list[ocr.Text] = []
    if use_ocr:
        ok, why = ocr.available()
        if ok:
            texts = ocr.read_text(img[top:], top=top)
            cal.texts = [t.to_dict() for t in texts]
        else:
            cal.warnings.append(why)
    labels = ocr.scale_labels(texts)
    near = labels[0].centre if len(labels) == 1 else None
    sb = db.find_scalebar(img, bar, near=near, text_boxes=[t.box for t in texts])
    if sb is not None:
        cal.scale_bar = sb.to_dict()
        if labels and near is None:  # several length labels: the one closest to the bar
            label = min(labels, key=lambda t: np.hypot(t.centre[0] - (sb.x0 + sb.x1) / 2, t.centre[1] - sb.y))
        else:
            label = labels[0] if labels else None
        if label is not None:
            cal.label = label.text
    found = ocr.fields(texts)
    if sb is None or not cal.label:
        return None, found
    value = ocr.lengths_in(cal.label)[0]
    return value / sb.length_px, found


def calibrate(path: str | Path, *, use_ocr: bool = True, cross_check: bool = True, use_metadata: bool = True) -> Calibration:
    """Calibrate one SEM image from what the file and its data bar say (sources 1 and 2).

    ``use_metadata=False`` ignores the instrument record (to evaluate the data-bar route
    against it); ``cross_check`` also reads the bar when metadata exists, and warns when
    the two disagree."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    img = load_image(path)
    cal = Calibration(file=path.name, sha256=file_sha256(path), image_shape=(int(img.shape[0]), int(img.shape[1])))
    md = read_metadata(path) if use_metadata else None
    if md is not None:
        cal.vendor, cal.metadata = md.vendor, md.to_dict()
        cal.warnings.extend(md.warnings)
    bar = _bar(img, md, cal)
    width = cal.image_shape[1]

    if md is not None and md.pixel_size:
        cal.pixel_size, cal.pixel_size_y = md.pixel_size, md.pixel_size_y or md.pixel_size
        cal.source, cal.confidence = "metadata", "high"
        cal.confirm("instrument metadata")
        if cross_check and bar is not None and use_ocr:
            from vexel.packs.sem import ocr

            if ocr.available()[0]:
                px, _ = _read_bar(img, bar, cal, use_ocr=True)
                if px is not None:
                    _check(cal, "scale bar", px, md.pixel_size, "信息栏比例尺")
        return cal

    px, found = _read_bar(img, bar, cal, use_ocr=use_ocr)
    if px is None:
        if bar is None and not cal.label:
            cal.scale_bar = None  # a stroke in the picture itself, not a scale bar
            cal.reason = "文件中没有仪器元数据，也没有找到信息栏；请手动标定"
        elif cal.scale_bar is not None:
            cal.reason = f"找到比例尺（{cal.scale_bar['length_px']:.1f} 像素），但没有读出它代表的长度；请输入长度或手动标定"
        else:
            cal.reason = "找到信息栏，但没有找到比例尺；请手动标定"
        return cal
    if not 1e-13 < px < 1e-2:
        cal.reason = f"比例尺识别结果不合理（{px!r} 米/像素）；请手动标定"
        return cal
    cal.pixel_size = cal.pixel_size_y = px
    cal.source, cal.confidence = "databar", "low"
    if "hfw" in found:
        if _check(cal, "field width", found["hfw"] / width, px, "视场宽度 ÷ 图片宽度"):
            cal.confidence = "medium"
    else:
        cal.warnings.append("信息栏中没有可核对的视场宽度，只有比例尺一个依据")
    if cal.scale_bar and cal.scale_bar["length_px"] < 50:
        cal.warnings.append(f"比例尺只有 {cal.scale_bar['length_px']:.0f} 像素长，1 像素的误差就是 {100 / cal.scale_bar['length_px']:.1f}%")
    return cal


def _check(cal: Calibration, name: str, other: float, ours: float, label: str) -> bool:
    diff = other / ours - 1
    agree = abs(diff) <= AGREE
    cal.checks.append({"check": name, "pixel_size": other, "difference": diff, "agree": agree})
    if not agree:
        cal.warnings.append(f"{label}得到 {format_length(other)}/像素，与采用值相差 {diff * 100:+.1f}%，请核对")
    return agree


def manual(path: str | Path, *, pixel_size: float | None = None, bar_length: float | None = None,
           bar_px: float | None = None, crop_rows: int | None = None) -> Calibration:  # fmt: skip
    """Source 3: the person gives the pixel size, or the scale bar's length (Vexel measures
    its pixels unless ``bar_px`` is given, e.g. from two clicks)."""
    path = Path(path)
    img = load_image(path)
    cal = Calibration(file=path.name, sha256=file_sha256(path), image_shape=(int(img.shape[0]), int(img.shape[1])))
    md = read_metadata(path)
    if md is not None:
        cal.vendor, cal.metadata = md.vendor, md.to_dict()
    bar = _bar(img, md, cal)
    if crop_rows is not None:
        cal.crop_rows, cal.crop_source = crop_rows, "manual"
    if pixel_size is None:
        if bar_length is None:
            raise ValueError("give the pixel size, or the scale bar's length")
        if bar_px is None:
            sb = db.find_scalebar(img, bar)
            if sb is None:
                raise ValueError("no scale bar found: give its length in pixels too (two clicks on its ends)")
            cal.scale_bar, bar_px = sb.to_dict(), sb.length_px
        else:
            cal.scale_bar = {"length_px": float(bar_px), "ends": "clicked"}
        cal.label = format_length(bar_length)
        pixel_size = bar_length / bar_px
    if not 1e-13 < pixel_size < 1e-2:
        raise ValueError(f"implausible pixel size: {pixel_size!r} m")
    cal.pixel_size = cal.pixel_size_y = float(pixel_size)
    cal.source, cal.confidence = "manual", "manual"
    if md is not None and md.pixel_size:
        _check(cal, "metadata", md.pixel_size, cal.pixel_size, "仪器元数据")
    return cal.confirm("person")


def parse_quantity(text: str) -> float:
    """``"500nm"``, ``"3.37 µm"`` -> metres (ValueError otherwise)."""
    value = parse_length(text.replace("um", "µm") if "µ" not in text else text)
    if value is None:
        raise ValueError(f"not a length with a unit (e.g. 500nm, 2.5um): {text!r}")
    return value


# -- records ---------------------------------------------------------------------------------


def store_dir() -> Path:
    return Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel")) / "sem" / "calibrations"


def save(cal: Calibration) -> Path:
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{cal.sha256}.json"
    out.write_text(json.dumps(cal.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(sha256: str) -> Calibration | None:
    p = store_dir() / f"{sha256}.json"
    if not p.is_file():
        return None
    try:
        return Calibration.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def for_file(path: str | Path, *, use_ocr: bool = True) -> Calibration:
    """The recorded calibration of this exact file when a person confirmed one, else a fresh one."""
    path = Path(path)
    saved = load(file_sha256(path))
    if saved is not None and saved.ok and saved.confirmed:
        return saved
    return calibrate(path, use_ocr=use_ocr)


def micrograph(path: str | Path, cal: Calibration) -> np.ndarray:
    """The image without its data bar (measure and analyse only this part)."""
    img = load_image(path)
    return img[: cal.crop_rows] if cal.crop_rows else img


def to_volume(path: str | Path, cal: Calibration) -> Any:
    """The micrograph as a platform ``SpatialVolume`` with spacing in µm (px when uncalibrated)."""
    from vexel.core.volume import SpatialVolume

    img = micrograph(path, cal)
    if cal.ok:
        spacing = {"y": (cal.pixel_size_y or cal.pixel_size) * 1e6, "x": cal.pixel_size * 1e6}  # type: ignore[operator]
        unit = "µm"
    else:
        spacing, unit = {}, "px"
    meta = {"sem_calibration": {k: v for k, v in cal.to_dict().items() if k not in ("texts", "metadata")}}
    return SpatialVolume.from_array(img, "YX", spacing=spacing, space_unit=unit, source=str(path), metadata=meta)
