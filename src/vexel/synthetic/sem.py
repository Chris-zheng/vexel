"""Synthetic SEM micrographs with exact calibration truth, in each vendor's file layout.

Not pictures of any real instrument: generic data bars that reproduce the features
calibration depends on (bar colour, table borders, where the label sits, end ticks,
rounded caps or a dotted scale bar) and the metadata each vendor writes.

=========  ====================================================================  ==============
style      data bar                                                               metadata
=========  ====================================================================  ==============
thermo     dark table; header row (HFW...) over values; capped bar, label on it   TIFF tag 34682
zeiss      light; ticked bar under its label; "Mag = ..." text                    TIFF tag 34118
tescan     light; "View field: ..." and a plain bar                               TIFF tag 50431
hitachi    dark; dotted scale bar (11 marks) under its label                      ``.txt`` sidecar
jeol       dark; plain bar beside its label                                       ``.txt`` sidecar
plain      light; ticked bar, field width text; **no metadata**                   none
none       **no data bar, no metadata**                                           none
=========  ====================================================================  ==============
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

STYLES = ("thermo", "zeiss", "tescan", "hitachi", "jeol", "plain", "none")
NICE = (1, 2, 5)


@dataclass
class SemTruth:
    style: str
    pixel_size: float  # metres per pixel
    scan_rows: int  # micrograph rows; the data bar is below
    image_shape: tuple[int, int]
    bar_length_px: float  # scale bar length, in the convention the style draws (see module doc)
    bar_value: float  # metres the scale bar stands for
    label: str
    hfw: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_shape"] = list(self.image_shape)
        return d


def _fmt(metres: float) -> tuple[str, str]:
    for unit, f in (("mm", 1e-3), ("um", 1e-6), ("nm", 1e-9)):  # "um": Pillow's built-in font has no "µ"
        if metres >= f * 0.999:
            v = metres / f
            return (f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.3g}"), unit
    return f"{metres * 1e12:.0f}", "pm"


def _nice_bar(pixel_size: float, width: int) -> float:
    target = pixel_size * width * 0.2
    exp = math.floor(math.log10(target))
    best = min((n * 10**e for e in (exp - 1, exp, exp + 1) for n in NICE), key=lambda v: abs(v - target))
    return best


def micrograph(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Grainy background with bright-edged particles: an SEM-like picture (float 0..1)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.normal(size=(rows, cols)), 8) * 4 + ndi.gaussian_filter(rng.normal(size=(rows, cols)), 1.5)
    img = 0.45 + 0.08 * base / (np.abs(base).max() + 1e-9)
    yy, xx = np.mgrid[:rows, :cols]
    for _ in range(int(rows * cols / 9000)):
        cy, cx, r = rng.uniform(0, rows), rng.uniform(0, cols), rng.uniform(4, 22)
        d = np.hypot(yy - cy, xx - cx) / r
        img += np.where(d < 1, 0.15 + 0.35 * d**4, 0)  # edge brightening
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1)


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def _bar(style: str, cols: int, height: int, pixel_size: float, value: float, hv: float, mag: float, hfw: float, wd: float):
    """Draw the data bar; return it (uint8) and the scale bar length in px as drawn."""
    from PIL import Image, ImageDraw

    dark = style in ("thermo", "hitachi", "jeol")
    bg, fg = (0, 255) if dark else (255, 0)
    im = Image.new("L", (cols, height), bg)
    dr = ImageDraw.Draw(im)
    n = round(value / pixel_size)  # pixels the vendor draws for the scale value
    num, unit = _fmt(value)
    label = f"{num} {unit}"
    hfw_n, hfw_u = _fmt(hfw)
    f_small, f_big = _font(max(11, height // 5)), _font(max(13, height // 4))
    if style == "thermo":
        dr.rectangle([0, 0, cols - 1, height - 1], outline=fg)
        cells = [("HV", f"{hv / 1e3:.2f} kV"), ("mag", f"{mag:,.0f} x"), ("HFW", f"{hfw_n} {hfw_u}"), ("WD", f"{wd * 1e3:.1f} mm"), ("det", "ETD")]
        cw = int(cols * 0.55) // len(cells)
        for i, (head, val) in enumerate(cells):
            x = i * cw
            dr.line([x, 0, x, height - 1], fill=fg)
            dr.text((x + cw // 2, height * 0.28), head, fill=fg, font=f_small, anchor="mm")
            dr.text((x + cw // 2, height * 0.72), val, fill=fg, font=f_small, anchor="mm")
        x_cell = len(cells) * cw
        dr.line([x_cell, 0, x_cell, height - 1], fill=fg)
        mid = height // 2
        dr.line([x_cell, mid, cols - 1, mid], fill=fg)  # a table line inside the scale cell
        x0 = x_cell + (cols - x_cell - n) // 2
        y = int(height * 0.25)
        dr.rectangle([x0 - 1, y - 1, x0 + n + 1, y + 1], fill=fg)  # the stroke
        for xc in (x0, x0 + n):
            dr.point([(xc, y - 2), (xc, y + 2)], fill=fg)  # rounded caps, centred on the ends
        tw = dr.textlength(label, font=f_small) + 8
        dr.rectangle([x0 + n / 2 - tw / 2, y - 6, x0 + n / 2 + tw / 2, y + 6], fill=bg)
        dr.text((x0 + n / 2, y), label, fill=fg, font=f_small, anchor="mm")
        length = float(n)
    elif style in ("zeiss", "plain"):
        x0, y = 20, int(height * 0.68)
        dr.text((x0 + n / 2, height * 0.32), label, fill=fg, font=f_big, anchor="mm")
        dr.rectangle([x0 - 1, y - 1, x0 + n, y], fill=fg)
        for xc in (x0, x0 + n):
            dr.rectangle([xc - 1, y - 8, xc, y + 8], fill=fg)  # 2-px ticks; centre at xc - 0.5
        length = float(n)
        dr.text((x0 + n + 60, height * 0.3), f"EHT = {hv / 1e3:.2f} kV", fill=fg, font=f_small, anchor="lm")
        dr.text((x0 + n + 60, height * 0.7), f"Mag = {mag:,.0f} X", fill=fg, font=f_small, anchor="lm")
        if style == "plain":
            dr.text((cols * 0.65, height * 0.5), f"Width = {hfw_n} {hfw_u}", fill=fg, font=f_small, anchor="lm")
    elif style == "tescan":
        dr.text((10, height * 0.3), f"SEM HV: {hv / 1e3:.1f} kV", fill=fg, font=f_small, anchor="lm")
        dr.text((10, height * 0.7), f"View field: {hfw_n} {hfw_u}", fill=fg, font=f_small, anchor="lm")
        x0, y = int(cols * 0.6), int(height * 0.7)
        dr.rectangle([x0, y - 1, x0 + n - 1, y + 1], fill=fg)  # plain: n lit pixels
        dr.text((x0 + n / 2, height * 0.3), label, fill=fg, font=f_small, anchor="mm")
        length = float(n)
    elif style == "hitachi":
        dr.text((10, height * 0.5), f"{hv / 1e3:.1f}kV  x{mag / 1e3:.1f}k  SE(U)", fill=fg, font=f_small, anchor="lm")
        x0, y = int(cols * 0.65), int(height * 0.7)
        step = n / 10
        for i in range(11):
            xc = round(x0 + i * step)
            dr.rectangle([xc - 1, y - 1, xc + 1, y + 1], fill=fg)
        length = float(round(x0 + 10 * step) - x0)
        dr.text((x0 + n / 2, height * 0.3), label, fill=fg, font=f_small, anchor="mm")
    elif style == "jeol":
        dr.text((10, height * 0.5), f"{hv / 1e3:.0f}kV   X{mag:,.0f}", fill=fg, font=f_small, anchor="lm")
        x0, y = int(cols * 0.45), int(height * 0.5)
        dr.text((x0 - 10, y), label, fill=fg, font=f_small, anchor="rm")
        dr.rectangle([x0, y - 1, x0 + n - 1, y + 1], fill=fg)
        length = float(n)
        dr.text((cols - 10, y), "SEI", fill=fg, font=f_small, anchor="rm")
    else:
        raise ValueError(style)
    return np.asarray(im, dtype=np.uint8), length, label


def write_sem(path: str | Path, style: str = "thermo", *, pixel_size: float = 5e-9, rows: int = 600, cols: int = 800,
              bar_height: int = 60, seed: int = 0, tilt: float = 0.0) -> SemTruth:  # fmt: skip
    """Write one synthetic SEM image (and its sidecar) at ``path``; return the truth."""
    import tifffile

    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (micrograph(rows, cols, seed) * 255).astype(np.uint8)
    hfw = pixel_size * cols
    hv, wd, mag = 5000.0, 4.2e-3, 0.127 / hfw
    value = _nice_bar(pixel_size, cols)
    length, label = 0.0, ""
    if style != "none":
        bar, length, label = _bar(style, cols, bar_height, pixel_size, value, hv, mag, hfw, wd)
        img = np.vstack([img, bar])
    truth = SemTruth(style=style, pixel_size=pixel_size, scan_rows=rows, image_shape=img.shape, bar_length_px=length,
                     bar_value=value, label=label, hfw=hfw)  # fmt: skip
    extratags = []
    if style == "thermo":
        ini = (f"[User]\r\nDate=11/19/2026\r\nTime=10:00:00 AM\r\n[System]\r\nSystemType=Synthetic SEM\r\n[Beam]\r\nHV={hv:g}\r\nBeam=EBeam\r\n"
               f"[EBeam]\r\nHV={hv:g}\r\nHFW={hfw:g}\r\nWD={wd:g}\r\n[Scan]\r\nPixelWidth={pixel_size:g}\r\nPixelHeight={pixel_size:g}\r\n"
               f"HorFieldsize={hfw:g}\r\n[Stage]\r\nStageT={math.radians(tilt):g}\r\nWorkingDistance={wd:g}\r\n"
               f"[Image]\r\nResolutionX={cols}\r\nResolutionY={rows}\r\nMagCanvasRealWidth=0.127\r\n[Detectors]\r\nName=ETD\r\n")  # fmt: skip
        extratags.append((34682, "s", 0, ini, True))
    elif style == "zeiss":
        unit_v, unit = _fmt(pixel_size)
        w_v, w_u = _fmt(hfw)
        text = ("0\r\n0\r\nAP_IMAGE_PIXEL_SIZE\r\nImage Pixel Size = " + f"{float(unit_v):.4g} {unit}\r\n"
                f"AP_WIDTH\r\nWidth = {float(w_v):.4g} {w_u}\r\nAP_MAG\r\nMag = {mag:,.0f} X\r\nAP_ACTUALKV\r\nEHT = {hv / 1e3:.2f} kV\r\n"
                f"AP_WD\r\nWD = {wd * 1e3:.1f} mm\r\nAP_STAGE_AT_T\r\nStage at T = {tilt:.1f} °\r\nDP_DETECTOR_CHANNEL\r\nSignal A = InLens\r\n")  # fmt: skip
        raw = text.encode("latin-1")  # SmartSEM writes "µ" and "°" as Latin-1 bytes
        extratags.append((34118, "B", len(raw), raw, True))
    elif style == "tescan":
        text = (f"AccFrames=1\r\nDate=2026-11-19\r\nDevice=Synthetic\r\n[SEM]\r\nHV={hv:g}\r\nPixelSizeX={pixel_size:g}\r\n"
                f"PixelSizeY={pixel_size:g}\r\nWD={wd:g}\r\nStageTilt={tilt:g}\r\n")  # fmt: skip
        extratags.append((50431, "s", 0, text, True))
    tifffile.imwrite(path, img, extratags=extratags)
    if style == "hitachi":
        txt = (f"[SemImageFile]\r\nInstructName=Synthetic\r\nDate=2026/11/19\r\nImageName={path.name}\r\nDataSize={cols}x{rows}\r\n"
               f"PixelSize={pixel_size * 1e9:.6g}\r\nSignalName=SE(U)\r\nAcceleratingVoltage={hv:g} Volt\r\nMagnification={mag:.0f}\r\n"
               f"WorkingDistance={wd * 1e6:.0f} um\r\nStagePositionT={tilt:g}\r\n")  # fmt: skip
        path.with_suffix(".txt").write_text(txt, encoding="utf-8")
    elif style == "jeol":
        bar_px = round(value / pixel_size)
        num, unit = _fmt(value)
        txt = (f"$CM_FORMAT JEOL/SEM\n$CM_INSTRUMENT Synthetic\n$CM_MAG {mag:.0f}\n$CM_FULL_SIZE {cols} {rows}\n"
               f"$CM_ACCEL_VOLT {hv / 1e3:.2f}\n$$SM_MICRON_BAR {bar_px}\n$$SM_MICRON_MARKER {num}{unit}\n")  # fmt: skip
        path.with_suffix(".txt").write_text(txt, encoding="utf-8")
    path.with_suffix(".truth.json").write_text(json.dumps(truth.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def write_set(folder: str | Path, *, seed: int = 0) -> list[tuple[Path, SemTruth]]:
    """One image per style, with varied pixel sizes and sizes."""
    folder = Path(folder)
    out = []
    sizes = {"thermo": (2.5e-9, 600, 800, 56), "zeiss": (13.7e-9, 640, 960, 80), "tescan": (0.82e-6, 512, 768, 48),
             "hitachi": (4.96e-9, 480, 640, 64), "jeol": (31.3e-9, 480, 640, 50), "plain": (1.9e-9, 600, 900, 70), "none": (5e-9, 400, 500, 0)}  # fmt: skip
    for i, style in enumerate(STYLES):
        px, rows, cols, bh = sizes[style]
        p = folder / f"{style}.tif"
        out.append((p, write_sem(p, style, pixel_size=px, rows=rows, cols=cols, bar_height=bh or 60, seed=seed + i)))
    return out


def line_space(rows: int = 200, cols: int = 800, *, width: float = 13.3, pitch: float = 31.7, offset: float = 5.2,
               edge_peaks: bool = False, blur: float = 1.5, noise: float = 0.03, seed: int = 0) -> np.ndarray:  # fmt: skip
    """Vertical bright lines of exact (sub-pixel) ``width`` and ``pitch``, pixels: a top-down
    line/space pattern. ``edge_peaks`` adds the bright edges SEM shows at steps (the line
    body is then only slightly brighter than the space). Float image, 0..1."""
    from scipy import ndimage as ndi

    x = np.arange(cols, dtype=float)

    def cover(a: float, b: float) -> np.ndarray:  # fraction of each pixel [i, i+1) inside [a, b)
        return np.clip(np.minimum(x + 1, b) - np.maximum(x, a), 0, 1)

    row = np.zeros(cols)
    edges = np.zeros(cols)
    k = 0
    while offset + k * pitch < cols:
        s = offset + k * pitch
        row += cover(s, s + width)
        if edge_peaks:
            edges += cover(s - 1, s + 1) + cover(s + width - 1, s + width + 1)
        k += 1
    img = 0.2 + (0.15 if edge_peaks else 0.6) * row + 0.6 * edges
    img = np.tile(img, (rows, 1))
    if blur:
        img = ndi.gaussian_filter(img, blur)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
