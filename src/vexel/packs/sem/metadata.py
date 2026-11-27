"""SEM instrument metadata: pixel size and acquisition settings written by the microscope.

Calibration level 1 (the most reliable). Every vendor stores it differently:

=============  ==================================================================  ===============
Vendor         Where                                                               Pixel size
=============  ==================================================================  ===============
Thermo Fisher  TIFF tag 34682 (or 34680), INI text; ``[Scan] PixelWidth``           metres
(FEI)          ``[Image] ResolutionY`` < image height: the rest is the data bar
Zeiss          TIFF tag 34118 (SmartSEM); ``AP_IMAGE_PIXEL_SIZE``, ``AP_WIDTH``      with unit
TESCAN         TIFF tag 50431 text, or a ``.hdr`` file next to the image;           metres
               ``PixelSizeX``
Hitachi        ``.txt`` next to the image, ``[SemImageFile]``; ``PixelSize``;       nanometres
               ``DataSize=WxH`` < image size: the rest is the data bar
JEOL           ``.txt`` next to the image; ``$$SM_MICRON_MARKER`` (e.g. ``1um``)    marker / bar
               over ``$$SM_MICRON_BAR`` (pixels)
=============  ==================================================================  ===============

Thermo Fisher and Zeiss were checked against real instrument files; TESCAN, Hitachi and
JEOL field names come from published import scripts and are to be confirmed with real
files (roadmap 0.0.5, M1). Anything unexpected is reported in ``warnings`` rather than
guessed.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNITS = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "å": 1e-10, "a": 1e-10}
VENDOR_NAMES = {"thermo_fisher": "Thermo Fisher（FEI）", "zeiss": "Zeiss", "tescan": "TESCAN", "hitachi": "Hitachi", "jeol": "JEOL", "imagej": "ImageJ 标定"}
TESCAN_TAG = 50431


@dataclass
class SemMetadata:
    vendor: str
    pixel_size: float | None = None  # metres per pixel along x
    pixel_size_y: float | None = None
    image_shape: tuple[int, int] = (0, 0)  # rows, columns of the file
    scan_rows: int | None = None  # rows that are the micrograph; the rest (bottom) is the data bar
    instrument: str = ""
    hfw: float | None = None  # horizontal field width, metres
    magnification: float | None = None
    voltage: float | None = None  # accelerating voltage, volts
    working_distance: float | None = None  # metres
    tilt: float | None = None  # stage tilt, degrees
    detector: str = ""
    date: str = ""
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)  # the fields used, as written in the file

    @property
    def vendor_name(self) -> str:
        return VENDOR_NAMES.get(self.vendor, self.vendor)

    @property
    def databar_rows(self) -> int:
        if self.scan_rows is None or not self.image_shape[0]:
            return 0
        return max(0, self.image_shape[0] - self.scan_rows)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["image_shape"] = list(self.image_shape)
        return d


def parse_length(text: str) -> float | None:
    """``"500 nm"``, ``"1.00um"``, ``"2 µm"``, ``"5.965 µm"`` -> metres. None if not a length."""
    m = re.fullmatch(r"\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(pm|nm|um|µm|μm|mm|cm|m|Å|A)\s*", text or "")
    if not m:
        return None
    return float(m.group(1).replace(",", ".")) * UNITS[m.group(2).lower()]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    m = re.match(r"\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)", str(value))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v if math.isfinite(v) else None


def _mag(value: Any) -> float | None:
    """``"50.00 K X"``, ``"120 x"``, ``105`` -> 50000, 120, 105."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = re.search(r"(\d[\d.,]*)\s*([kKmM]?)\s*[xX×]?", str(value or ""))
    if not m:
        return None
    digits = m.group(1)
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", digits):  # thousands separators: "9,656"
        digits = digits.replace(",", "")
    v = _float(digits)
    if v is None:
        return None
    return v * {"k": 1e3, "m": 1e6}.get(m.group(2).lower(), 1.0)


def _plausible(pixel_size: float | None) -> bool:
    return pixel_size is not None and 1e-13 < pixel_size < 1e-2


# -- vendors ---------------------------------------------------------------------------------


def _thermo_fisher(fm: dict[str, Any], shape: tuple[int, int]) -> SemMetadata:
    scan, image, beam = fm.get("Scan", {}), fm.get("Image", {}), fm.get("Beam", {})
    ebeam = fm.get(beam.get("Beam", "EBeam"), {}) or fm.get("EBeam", {})
    stage, system = fm.get("Stage", {}), fm.get("System", {})
    md = SemMetadata(vendor="thermo_fisher", image_shape=shape)
    md.pixel_size = _float(scan.get("PixelWidth"))
    md.pixel_size_y = _float(scan.get("PixelHeight")) or md.pixel_size
    res_x, res_y = _float(image.get("ResolutionX")), _float(image.get("ResolutionY"))
    if res_y and 0 < res_y <= shape[0]:
        md.scan_rows = int(res_y)
    if res_x and shape[1] and int(res_x) != shape[1]:
        md.warnings.append(f"成像分辨率宽度 {int(res_x)} 与图片宽度 {shape[1]} 不同：图片可能在采集后被缩放过")
    md.hfw = _float(scan.get("HorFieldsize")) or _float(ebeam.get("HFW"))
    md.voltage = _float(beam.get("HV")) or _float(ebeam.get("HV"))
    md.working_distance = _float(stage.get("WorkingDistance")) or _float(ebeam.get("WD"))
    tilt = _float(stage.get("StageT"))
    md.tilt = math.degrees(tilt) if tilt is not None else None  # radians in the file
    md.instrument = str(system.get("SystemType", "") or system.get("Type", "")).strip()
    md.detector = str(fm.get("Detectors", {}).get("Name", "")).strip()
    md.date = " ".join(str(fm.get("User", {}).get(k, "")) for k in ("Date", "Time")).strip()
    canvas = _float(image.get("MagCanvasRealWidth"))
    if canvas and md.hfw:
        md.magnification = canvas / md.hfw
    md.raw = {k: str(v) for k, v in (("Scan.PixelWidth", scan.get("PixelWidth")), ("Scan.PixelHeight", scan.get("PixelHeight")),
                                      ("Image.ResolutionX", res_x), ("Image.ResolutionY", res_y), ("Scan.HorFieldsize", scan.get("HorFieldsize"))) if v is not None}  # fmt: skip
    return md


def _zeiss(sm: dict[str, Any], shape: tuple[int, int]) -> SemMetadata:
    def value(key: str) -> tuple[Any, str]:
        entry = sm.get(key)
        if not isinstance(entry, tuple) or len(entry) < 2:
            return None, ""
        return entry[1], (entry[2] if len(entry) > 2 else "")

    def length(key: str) -> float | None:
        v, unit = value(key)
        v = _float(v)
        if v is None:
            return None
        return v * UNITS.get(unit.lower().replace("μ", "µ"), float("nan")) if unit else None

    md = SemMetadata(vendor="zeiss", image_shape=shape)
    image_px, store_px, width = length("ap_image_pixel_size"), length("ap_pixel_size"), length("ap_width")
    md.pixel_size = image_px or store_px
    md.hfw = width
    if width and shape[1]:
        from_width = width / shape[1]
        if md.pixel_size is None:
            md.pixel_size = from_width
        elif abs(from_width - md.pixel_size) / md.pixel_size > 0.01:
            md.warnings.append(f"记录的像素尺寸 {md.pixel_size * 1e9:.4g} nm 与视场宽度 ÷ 图片宽度 {from_width * 1e9:.4g} nm 不一致："
                               "图片可能在采集后被缩放过，采用后者")  # fmt: skip
            md.pixel_size = from_width
    md.pixel_size_y = md.pixel_size
    md.magnification = _mag(value("ap_mag")[0])
    v, unit = value("ap_actualkv")
    if _float(v) is not None:
        md.voltage = _float(v) * (1e3 if unit.lower() == "kv" else 1.0)
    md.working_distance = length("ap_wd")
    tilt, _ = value("ap_stage_at_t")
    md.tilt = _float(tilt)
    md.detector = str(value("dp_detector_channel")[0] or "")
    md.date = " ".join(str(value(k)[0] or "") for k in ("ap_date", "ap_time")).strip()
    md.instrument = str(value("sv_serial_number")[0] or "")
    md.raw = {k: " ".join(str(x) for x in sm[k][1:]) for k in ("ap_image_pixel_size", "ap_pixel_size", "ap_width", "ap_mag", "dp_image_store") if k in sm}
    return md


def _key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("["):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif line.startswith("$"):
            parts = line.split(None, 1)
            k, v = parts[0], (parts[1] if len(parts) > 1 else "")
        else:
            continue
        out.setdefault(k.strip(), v.strip())
    return out


def _tescan(kv: dict[str, str], shape: tuple[int, int], where: str) -> SemMetadata:
    md = SemMetadata(vendor="tescan", image_shape=shape)
    md.pixel_size = _float(kv.get("PixelSizeX"))
    md.pixel_size_y = _float(kv.get("PixelSizeY")) or md.pixel_size
    md.voltage = _float(kv.get("HV"))
    md.working_distance = _float(kv.get("WD"))
    md.magnification = _mag(kv.get("Magnification"))
    md.tilt = _float(kv.get("StageTilt") or kv.get("TiltAngle"))  # degrees (to be confirmed)
    md.instrument = kv.get("Device", "") or kv.get("DeviceModel", "")
    md.date = " ".join(kv.get(k, "") for k in ("Date", "Time")).strip()
    md.detector = kv.get("Detector", "")
    if md.pixel_size and shape[1]:
        md.hfw = md.pixel_size * shape[1]
    md.raw = {k: kv[k] for k in ("PixelSizeX", "PixelSizeY", "HV", "WD", "Magnification") if k in kv} | {"source": where}
    return md


def _hitachi(kv: dict[str, str], shape: tuple[int, int]) -> SemMetadata:
    md = SemMetadata(vendor="hitachi", image_shape=shape)
    px = _float(kv.get("PixelSize"))
    md.pixel_size = px * 1e-9 if px is not None else None  # nanometres in the file
    md.pixel_size_y = md.pixel_size
    size = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", kv.get("DataSize", ""))
    if size:
        w, h = int(size.group(1)), int(size.group(2))
        if 0 < h <= shape[0]:
            md.scan_rows = h
        if shape[1] and w != shape[1]:
            md.warnings.append(f"DataSize 宽度 {w} 与图片宽度 {shape[1]} 不同：图片可能在采集后被缩放过")
    md.magnification = _mag(kv.get("Magnification"))
    volts = _float(kv.get("AcceleratingVoltage") or kv.get("Accelerating Voltage"))
    if volts is not None:
        unit = (kv.get("AcceleratingVoltage") or kv.get("Accelerating Voltage", "")).lower()
        md.voltage = volts * (1e3 if "kv" in unit else 1.0)
    wd = kv.get("WorkingDistance", "")
    if _float(wd) is not None:
        md.working_distance = _float(wd) * (1e-6 if "um" in wd.lower() else 1e-3)
    md.tilt = _float(kv.get("StagePositionT"))
    md.instrument = kv.get("InstructName", "")
    md.detector = kv.get("SignalName", "")
    md.date = " ".join(kv.get(k, "") for k in ("Date", "Time")).strip()
    if md.pixel_size and shape[1]:
        md.hfw = md.pixel_size * (int(size.group(1)) if size else shape[1])
    md.raw = {k: kv[k] for k in ("PixelSize", "DataSize", "Magnification", "InstructName") if k in kv}
    return md


def _jeol(kv: dict[str, str], shape: tuple[int, int]) -> SemMetadata:
    md = SemMetadata(vendor="jeol", image_shape=shape)
    bar_px = _float(kv.get("$$SM_MICRON_BAR") or kv.get("$SM_MICRON_BAR"))
    marker = parse_length(kv.get("$$SM_MICRON_MARKER") or kv.get("$SM_MICRON_MARKER") or "")
    if bar_px and marker:
        md.pixel_size = md.pixel_size_y = marker / bar_px
    size = (kv.get("$CM_FULL_SIZE") or "").split()
    if len(size) == 2 and all(s.isdigit() for s in size):
        w, h = int(size[0]), int(size[1])
        if 0 < h <= shape[0]:
            md.scan_rows = h
        if shape[1] and w != shape[1]:
            # the marker's pixels refer to the full-size image
            if md.pixel_size:
                md.pixel_size = md.pixel_size_y = md.pixel_size * w / shape[1]
            md.warnings.append(f"$CM_FULL_SIZE 宽度 {w} 与图片宽度 {shape[1]} 不同，已按比例换算像素尺寸")
    md.magnification = _mag(kv.get("$CM_MAG"))
    kv_volt = _float(kv.get("$CM_ACCEL_VOLT"))
    md.voltage = kv_volt * 1e3 if kv_volt is not None else None  # kV in the file
    wd = _float(kv.get("$$SM_WD") or kv.get("$SM_WD"))
    md.working_distance = wd * 1e-3 if wd is not None else None
    md.instrument = kv.get("$CM_INSTRUMENT", "")
    md.detector = kv.get("$CM_SIGNAL", "")
    md.date = kv.get("$CM_DATE", "")
    if md.pixel_size and shape[1]:
        md.hfw = md.pixel_size * shape[1]
    md.raw = {k: kv[k] for k in ("$$SM_MICRON_BAR", "$$SM_MICRON_MARKER", "$CM_FULL_SIZE", "$CM_MAG", "$CM_INSTRUMENT") if k in kv}
    return md


_IMAGEJ_UNITS = {"micron": 1e-6, "microns": 1e-6, "µm": 1e-6, "\u00b5m": 1e-6, "μm": 1e-6, "um": 1e-6, "\\u00b5m": 1e-6,
                 "nm": 1e-9, "nanometer": 1e-9, "mm": 1e-3, "m": 1.0, "pm": 1e-12, "a": 1e-10, "å": 1e-10}  # fmt: skip


def _imagej(tf: Any, shape: tuple[int, int]) -> SemMetadata | None:
    """A pixel size written by ImageJ / Fiji (resolution tags in pixels per unit, the unit in
    the ImageJ description): not the instrument's record, but a calibration someone set."""
    ij = tf.imagej_metadata or {}
    unit = str(ij.get("unit", "")).strip().lower()
    factor = _IMAGEJ_UNITS.get(unit)
    tags = tf.pages[0].tags
    if factor is None or "XResolution" not in tags:
        return None

    def per_pixel(tag: str) -> float | None:
        if tag not in tags:
            return None
        num, den = tags[tag].value
        return factor * den / num if num else None

    px = per_pixel("XResolution")
    if px is None:
        return None
    md = SemMetadata(vendor="imagej", pixel_size=px, pixel_size_y=per_pixel("YResolution") or px, image_shape=shape,
                     instrument=f"ImageJ {ij.get('ImageJ', '')}".strip())  # fmt: skip
    md.raw = {"unit": str(ij.get("unit")), "XResolution": str(tags["XResolution"].value)}
    if ij.get("spacing"):
        md.raw["spacing"] = str(ij["spacing"])
    md.warnings.append("像素尺寸来自 ImageJ 写入的标定（不是仪器原始记录）：图像若在 ImageJ 中缩放过，请核对")
    return md


def _sidecar(path: Path, suffix: str) -> str | None:
    for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper()), Path(str(path) + suffix)):
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace")[:200_000]
            except OSError:
                return None
    return None


def image_shape(path: Path) -> tuple[int, int]:
    """Rows and columns of the (first page of the) image, without reading its pixels."""
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        try:
            with tifffile.TiffFile(path) as tf:
                s = tf.pages[0].shape
                return (int(s[0]), int(s[1])) if len(s) == 2 or s[-1] in (3, 4) else (int(s[-2]), int(s[-1]))
        except Exception:  # noqa: BLE001 - fall back to a full read
            pass
    from PIL import Image

    with Image.open(path) as im:
        return im.height, im.width


def read_metadata(path: str | Path) -> SemMetadata | None:
    """The instrument's own calibration and settings, or None when the file carries none."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    shape = image_shape(path)
    md: SemMetadata | None = None
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tf:
            tags = tf.pages[0].tags
            if 34682 in tags or 34680 in tags:
                fm = tf.fei_metadata
                if fm:
                    md = _thermo_fisher(fm, shape)
            elif 34118 in tags:
                sm = tf.sem_metadata
                if sm:
                    md = _zeiss(sm, shape)
            elif TESCAN_TAG in tags:
                value = tags[TESCAN_TAG].value
                text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
                if "PixelSizeX" in text:
                    md = _tescan(_key_values(text), shape, f"TIFF tag {TESCAN_TAG}")
            elif tf.is_imagej:
                md = _imagej(tf, shape)
    if md is None:
        text = _sidecar(path, ".hdr")
        if text and "PixelSizeX" in text:
            md = _tescan(_key_values(text), shape, ".hdr")
    if md is None:
        text = _sidecar(path, ".txt")
        if text:
            kv = _key_values(text)
            if "$$SM_MICRON_BAR" in kv or "$CM_MAG" in kv or "$CM_FORMAT" in kv:
                md = _jeol(kv, shape)
            elif "PixelSize" in kv or "SemImageFile" in text:
                md = _hitachi(kv, shape)
    if md is None:
        return None
    if md.pixel_size is not None and not _plausible(md.pixel_size):
        md.warnings.append(f"元数据中的像素尺寸 {md.pixel_size!r} 不合理，未采用")
        md.pixel_size = md.pixel_size_y = None
    if md.pixel_size is None:
        md.warnings.append("元数据中没有可用的像素尺寸")
    return md


def vendor_of(path: str | Path) -> str | None:
    try:
        md = read_metadata(path)
    except Exception:  # noqa: BLE001 - not a readable image
        return None
    return md.vendor if md else None
