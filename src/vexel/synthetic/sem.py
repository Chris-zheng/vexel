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


def micrograph(rows: int, cols: int, seed: int = 0, noise: float = 0.03) -> np.ndarray:
    """Grainy background with bright-edged particles: an SEM-like picture (float 0..1).
    ``noise``: standard deviation of the added pixel noise (0: a clean picture)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.normal(size=(rows, cols)), 8) * 4 + ndi.gaussian_filter(rng.normal(size=(rows, cols)), 1.5)
    img = 0.45 + 0.08 * base / (np.abs(base).max() + 1e-9)
    yy, xx = np.mgrid[:rows, :cols]
    for _ in range(int(rows * cols / 9000)):
        cy, cx, r = rng.uniform(0, rows), rng.uniform(0, cols), rng.uniform(4, 22)
        d = np.hypot(yy - cy, xx - cx) / r
        img += np.where(d < 1, 0.15 + 0.35 * d**4, 0)  # edge brightening
    if noise:
        img += rng.normal(0, noise, img.shape)
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


# -- artefacts for testing image processing (plan §2.3) -----------------------------------------


def add_stripes(img: np.ndarray, amplitude: float = 0.06, *, direction: str = "horizontal", vary: float = 0.0,
                seed: int = 0) -> np.ndarray:  # fmt: skip
    """Charging bands along the scan direction: each line (horizontal) gets a random offset,
    plus slower bands a few lines wide. ``vary`` > 0: the bands fade in and out along the
    line (by that fraction), as charging builds up and leaks away during the scan."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    n = img.shape[0] if direction == "horizontal" else img.shape[1]
    off = rng.normal(0, 1, n) + ndi.gaussian_filter1d(rng.normal(0, 3, n), 3)
    off = amplitude * off / (off.std() + 1e-12)
    m = img.shape[1] if direction == "horizontal" else img.shape[0]
    along = 1 + vary * np.sin(2 * np.pi * np.arange(m) / m * 1.5 + 0.7)
    return img + (off[:, None] * along[None, :] if direction == "horizontal" else off[None, :] * along[:, None])


def add_gradient(img: np.ndarray, amplitude: float = 0.25) -> np.ndarray:
    """Uneven brightness: brighter towards one corner, plus a bright centre."""
    r, c = img.shape
    yy, xx = np.mgrid[:r, :c]
    ramp = (xx / max(c - 1, 1) * 0.7 + yy / max(r - 1, 1) * 0.3) - 0.5
    centre = np.exp(-(((yy - r / 2) / (0.6 * r)) ** 2 + ((xx - c / 2) / (0.6 * c)) ** 2))
    return img + amplitude * (ramp + 0.5 * (centre - centre.mean()))


def add_noise(img: np.ndarray, sigma: float = 0.05, *, seed: int = 0) -> np.ndarray:
    return img + np.random.default_rng(seed).normal(0, sigma, img.shape)


def add_row_jitter(img: np.ndarray, *, fraction: float = 0.15, max_shift: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Shift a random ``fraction`` of the scan lines sideways by up to ``max_shift`` pixels
    (sub-pixel). Returns the image and each line's shift (positive: moved right)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    r, c = img.shape
    shifts = np.where(rng.random(r) < fraction, rng.uniform(-max_shift, max_shift, r), 0.0)
    rr, cc = np.meshgrid(np.arange(r, dtype=float), np.arange(c, dtype=float), indexing="ij")
    out = ndi.map_coordinates(img, [rr, cc - shifts[:, None]], order=3, mode="nearest")
    return out, shifts


def add_saturation(img: np.ndarray, *, spots: int = 3, radius: float = 18.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Charged spots clipped at white (1.0). Returns the image and the clipped pixels."""
    rng = np.random.default_rng(seed)
    r, c = img.shape
    yy, xx = np.mgrid[:r, :c]
    out = img.copy()
    for _ in range(spots):
        cy, cx, rad = rng.uniform(0.15, 0.85) * r, rng.uniform(0.15, 0.85) * c, rng.uniform(0.5, 1.0) * radius
        out += 1.2 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad**2)))
    out = np.clip(out, 0, 1)
    return out, out >= 1.0


# -- semiconductor patterns with known geometry (M4) ---------------------------------------------


def rough_edge(n: int, sigma: float, corr: float, rng: np.random.Generator) -> np.ndarray:
    """Edge displacement series (n lines) with exponential autocorrelation: correlation
    length ``corr`` lines, standard deviation ``sigma`` pixels (AR(1) process)."""
    phi = math.exp(-1.0 / corr) if corr > 0 else 0.0
    e = rng.normal(0, sigma * math.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _cover(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of pixel [i, i+1) inside [a, b), per line (a, b: one value per line)."""
    return np.clip(np.minimum(x[None, :] + 1, b[:, None]) - np.maximum(x[None, :], a[:, None]), 0, 1)


def rough_lines(rows: int = 512, cols: int = 640, *, width: float = 14.3, pitch: float = 36.7, offset: float = 9.4, sigma: float = 1.0,
                corr: float = 8.0, blur: float = 1.0, noise: float = 0.03, edge_peaks: bool = False, slant: float = 0.0,
                seed: int = 0, noise_seed: int | None = None) -> tuple[np.ndarray, dict]:  # fmt: skip
    """Vertical bright lines whose two edges wander independently (``sigma`` px, correlation
    length ``corr`` lines); ``slant`` px of drift per line. ``noise_seed``: a different noise
    on the same pattern (repeated images of one place). Returns the image (0..1) and the
    truth: per line its left and right edge series (pixel boundaries)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    x = np.arange(cols, dtype=float)
    y = np.arange(rows, dtype=float)
    img = np.full((rows, cols), 0.2)
    truth = {"lines": [], "width": width, "pitch": pitch, "sigma": sigma, "corr": corr}
    k = 0
    while offset + k * pitch + width < cols - 1:
        s = offset + k * pitch + slant * (y - rows / 2)
        left = s + rough_edge(rows, sigma, corr, rng)
        right = s + width + rough_edge(rows, sigma, corr, rng)
        if s.min() > 1:
            img += (0.15 if edge_peaks else 0.55) * _cover(x, left, right)
            if edge_peaks:
                img += 0.5 * (_cover(x, left - 1, left + 1) + _cover(x, right - 1, right + 1))
            truth["lines"].append({"left": left, "right": right})
        k += 1
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img += (rng if noise_seed is None else np.random.default_rng(noise_seed)).normal(0, noise, img.shape)
    return img, truth


def layer_stack(rows: int = 300, cols: int = 500, *, thicknesses: tuple[float, ...] = (40.3, 23.6, 61.2), top: float = 50.5,
                levels: tuple[float, ...] = (0.15, 0.7, 0.35, 0.85, 0.5), roughness: float = 0.6, blur: float = 1.2, noise: float = 0.03,
                seed: int = 0) -> tuple[np.ndarray, dict]:  # fmt: skip
    """A cross-section: horizontal layers of the given ``thicknesses`` (px) starting at row
    ``top``, grey ``levels`` from the top (vacuum/protective layer first), interfaces
    rough by ``roughness`` px. Returns the image and the interface series."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    yy = np.arange(rows, dtype=float)[:, None]
    bounds = [top + sum(thicknesses[:i]) for i in range(len(thicknesses) + 1)]
    series = [b + rough_edge(cols, roughness, 15, rng) for b in bounds]
    img = np.full((rows, cols), levels[0])
    for i, s in enumerate(series):
        frac = np.clip(yy + 1 - s[None, :], 0, 1)  # fraction of pixel row below the interface
        img = img + (levels[i + 1] - levels[i]) * frac
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img += rng.normal(0, noise, img.shape)
    return img, {"interfaces": series, "thicknesses": thicknesses}


def xsection_feature(rows: int = 260, cols: int = 300, *, top: float = 60.3, height: float = 140.0, width_top: float = 60.0,
                     angle: float = 84.0, feature: str = "bright", blur: float = 1.2, noise: float = 0.03,
                     seed: int = 0) -> tuple[np.ndarray, dict]:  # fmt: skip
    """A trapezoid line (``bright``) or trench (``dark``) in cross-section: ``height`` rows,
    ``width_top`` px at its top, sidewalls at ``angle`` degrees to the base inside the
    feature (< 90: wider at the bottom)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    x = np.arange(cols, dtype=float)
    y = np.arange(rows, dtype=float) + 0.5
    run = 1 / math.tan(math.radians(angle))  # widening per line on each side, going down
    cx = cols / 2
    depth = np.clip(y - top, 0, None)
    half = width_top / 2 + run * depth
    inside_rows = (y >= top) & (y < top + height)
    left = np.where(inside_rows, cx - half, cx)
    right = np.where(inside_rows, cx + half, cx)
    cov = _cover(x, left, right)
    # partial first and last rows
    cov[np.floor(top).astype(int)] *= 1 - (top - math.floor(top))
    base, fg = (0.25, 0.75) if feature == "bright" else (0.75, 0.25)
    img = base + (fg - base) * cov
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img += rng.normal(0, noise, img.shape)
    mid = top + height / 2
    return img, {"height": height, "width_top": width_top, "width_bottom": width_top + 2 * run * height,
                 "width_middle": width_top + 2 * run * (mid - top), "angle": angle, "top": top}  # fmt: skip


def _disks(rows: int, cols: int, discs: list[tuple[float, float, float]], ss: int = 4) -> np.ndarray:
    """Area coverage of discs (cx, cy, r) per pixel, ``ss``×``ss`` supersampled."""
    off = (np.arange(ss) + 0.5) / ss
    yy = (np.arange(rows)[:, None] + off[None, :]).reshape(-1)
    xx = (np.arange(cols)[:, None] + off[None, :]).reshape(-1)
    cov = np.zeros((rows * ss, cols * ss), bool)
    for cx, cy, r in discs:
        y0, y1 = max(int((cy - r - 1) * ss), 0), min(int((cy + r + 1) * ss) + 1, rows * ss)
        x0, x1 = max(int((cx - r - 1) * ss), 0), min(int((cx + r + 1) * ss) + 1, cols * ss)
        if y0 >= y1 or x0 >= x1:
            continue
        cov[y0:y1, x0:x1] |= (yy[y0:y1, None] - cy) ** 2 + (xx[None, x0:x1] - cx) ** 2 <= r * r
    return cov.reshape(rows, ss, cols, ss).mean(axis=(1, 3))


def hole_array(rows: int = 400, cols: int = 500, *, pitch: tuple[float, float] = (48.3, 48.3), diameter: float = 24.6, sd: float = 0.0,
               missing: int = 3, blur: float = 1.2, noise: float = 0.03, rim: bool = True,
               seed: int = 0, noise_seed: int | None = None) -> tuple[np.ndarray, dict]:  # fmt: skip
    """A contact-hole array: dark holes (``diameter`` ± ``sd`` px) on a bright surface with
    bright rims, ``missing`` sites left empty. Truth: every hole (cx, cy, d) and the missing sites."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    sites = [(x, y) for y in np.arange(pitch[1] * 0.75, rows - pitch[1] * 0.5, pitch[1])
             for x in np.arange(pitch[0] * 0.75, cols - pitch[0] * 0.5, pitch[0])]  # fmt: skip
    gone = set(rng.choice(len(sites), size=min(missing, len(sites)), replace=False).tolist()) if missing else set()
    holes, discs = [], []
    for i, (x, y) in enumerate(sites):
        if i in gone:
            continue
        d = diameter + rng.normal(0, sd) if sd else diameter
        holes.append((x, y, d))
        discs.append((x, y, d / 2))
    cov = _disks(rows, cols, discs)
    img = 0.7 - 0.5 * cov
    if rim:
        ring = _disks(rows, cols, [(x, y, r + 1.5) for x, y, r in discs]) - cov
        img += 0.25 * ring
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img += (rng if noise_seed is None else np.random.default_rng(noise_seed)).normal(0, noise, img.shape)
    return img, {"holes": holes, "missing": [sites[i] for i in sorted(gone)], "pitch": pitch, "diameter": diameter}


def particle_field(rows: int = 500, cols: int = 600, *, n: int = 60, d_mean: float = 24.0, d_sd: float = 6.0, touching: float = 0.2,
                   blur: float = 1.0, noise: float = 0.03, seed: int = 0) -> tuple[np.ndarray, dict]:  # fmt: skip
    """Bright round particles (diameters ~ normal) on a dark background; a fraction
    ``touching`` placed against another one. Truth: every particle (cx, cy, d)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    parts: list[tuple[float, float, float]] = []
    tries = 0
    while len(parts) < n and tries < n * 200:
        tries += 1
        d = max(rng.normal(d_mean, d_sd), 6.0)
        r = d / 2
        if parts and rng.random() < touching:
            px, py, pd = parts[rng.integers(len(parts))]
            ang = rng.uniform(0, 2 * np.pi)
            cx, cy = px + (pd / 2 + r - 0.15 * min(r, pd / 2)) * np.cos(ang), py + (pd / 2 + r - 0.15 * min(r, pd / 2)) * np.sin(ang)
            if any(np.hypot(cx - x, cy - y) < (r + q / 2) * 0.9 for x, y, q in parts if (x, y, q) != (px, py, pd)):
                continue
        else:
            cx, cy = rng.uniform(r + 2, cols - r - 2), rng.uniform(r + 2, rows - r - 2)
            if any(np.hypot(cx - x, cy - y) < r + q / 2 + 3 for x, y, q in parts):
                continue
        if not (r + 1 < cx < cols - r - 1 and r + 1 < cy < rows - r - 1):
            continue
        parts.append((cx, cy, d))
    img = 0.2 + 0.6 * np.clip(_disks(rows, cols, [(x, y, d / 2) for x, y, d in parts]), 0, 1)
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img += rng.normal(0, noise, img.shape)
    return img, {"particles": parts}


def tip(rows: int = 400, cols: int = 300, *, radius: float = 20.0, half_angle: float = 12.0, apex: tuple[float, float] = (150.3, 80.6),
        blur: float = 1.0, noise: float = 0.03, seed: int = 0) -> tuple[np.ndarray, dict]:  # fmt: skip
    """A bright needle pointing up: a cone of ``half_angle`` degrees capped by a sphere of
    ``radius`` px tangent to it, apex at ``apex`` (x, y). Truth: radius, cone angle."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    ss = 4
    off = (np.arange(ss) + 0.5) / ss
    yy = (np.arange(rows)[:, None] + off[None, :]).reshape(-1)[:, None]
    xx = (np.arange(cols)[:, None] + off[None, :]).reshape(-1)[None, :]
    a = math.radians(half_angle)
    ax, ay = apex
    cy = ay + radius  # sphere centre
    ty = cy - radius * math.sin(a)  # where the cap meets the cone (tangent)
    half_at_t = radius * math.cos(a)
    inside_cap = (xx - ax) ** 2 + (yy - cy) ** 2 <= radius**2
    half = half_at_t + (yy - ty) * math.tan(a)
    inside_cone = (yy >= ty) & (np.abs(xx - ax) <= half)
    cov = (inside_cap | inside_cone).reshape(rows, ss, cols, ss).mean(axis=(1, 3))
    img = 0.2 + 0.6 * cov
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img += rng.normal(0, noise, img.shape)
    return img, {"radius": radius, "cone_angle": 2 * half_angle, "apex": apex}


# -- tilt series with a known height map (M6) -----------------------------------------------------


def height_map(kind: str = "pyramid", rows: int = 320, cols: int = 384, *, height: float = 30.0, seed: int = 0) -> np.ndarray:
    """A surface height map, in pixels (the same unit as x and y):

    - ``pyramid``: a truncated pyramid of ``height`` on a flat base (side walls ~35°);
    - ``steps``: three terraces ``height`` / 3 apart (step heights);
    - ``spheres``: spherical caps of up to ``height`` on a flat base;
    - ``sine``: a sinusoidal corrugation of amplitude ``height`` / 2;
    - ``rough``: a random rough surface, RMS height ``height`` / 6.
    """
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:rows, :cols].astype(float)
    if kind == "pyramid":
        cy, cx = rows / 2, cols / 2
        d = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
        top, base = min(rows, cols) * 0.12, min(rows, cols) * 0.12 + height / math.tan(math.radians(35))
        z = np.clip((base - d) / (base - top), 0, 1) * height
    elif kind == "steps":
        z = np.floor(np.clip((xx - cols * 0.2) / (cols * 0.25), 0, 2.999)) * height / 3
        z = ndi.gaussian_filter(z, 1.5)  # steps with a short slope
    elif kind == "spheres":
        z = np.zeros((rows, cols))
        for _ in range(6):
            r = rng.uniform(0.08, 0.16) * min(rows, cols)
            cy, cx = rng.uniform(r, rows - r), rng.uniform(r, cols - r)
            h = min(height, r * 0.6)
            R = (r * r + h * h) / (2 * h)  # cap of height h and base radius r
            dd = (yy - cy) ** 2 + (xx - cx) ** 2
            cap = np.where(dd < r * r, np.sqrt(np.maximum(R * R - dd, 0)) - (R - h), 0)
            z = np.maximum(z, cap)
    elif kind == "sine":
        z = height / 2 * np.sin(2 * np.pi * yy / (rows / 3)) * np.sin(2 * np.pi * xx / (cols / 2.5))
    elif kind == "rough":
        z = ndi.gaussian_filter(rng.normal(0, 1, (rows, cols)), 6)
        z = z / z.std() * height / 6
    else:
        raise ValueError(kind)
    return z


def surface_texture(rows: int, cols: int, *, seed: int = 0) -> np.ndarray:
    """Albedo of a textured surface (grains and fine contrast): what makes views matchable."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    t = 0.6 * ndi.gaussian_filter(rng.normal(0, 1, (rows, cols)), 1.2) + 0.4 * ndi.gaussian_filter(rng.normal(0, 1, (rows, cols)), 4)
    t = (t - t.min()) / np.ptp(t)
    return 0.3 + 0.6 * t


def tilt_view(z: np.ndarray, albedo: np.ndarray, tilt: float, *, noise: float = 0.02, blur: float = 0.8, flat: np.ndarray | None = None,
              seed: int = 0) -> np.ndarray:  # fmt: skip
    """The surface seen by the beam after a eucentric tilt of ``tilt`` degrees about the image's
    horizontal axis through its centre row (parallel projection, as SEM at usual magnifications):
    a point (x, y, z) appears at row (y - yc)·cos(tilt) + z·sin(tilt) + yc. Brightness: the albedo
    times a secondary-electron-like slope term (brighter where the surface faces away from the
    beam). ``flat``: where the albedo is replaced by a uniform grey (a textureless area)."""
    from scipy import ndimage as ndi

    rows, cols = z.shape
    phi = math.radians(tilt)
    yc = (rows - 1) / 2
    gy, gx = np.gradient(z)
    nrm = np.sqrt(gx**2 + gy**2 + 1)
    cos_t = (-gy * math.sin(phi) + math.cos(phi)) / nrm  # angle between the surface normal and the beam
    shade = np.clip(cos_t, 0.2, 1) ** -0.4
    a = albedo if flat is None else np.where(flat, 0.6, albedo)
    bright = a * shade
    out = np.empty((rows, cols))
    y = np.arange(rows, dtype=float)
    for c in range(cols):
        yp = (y - yc) * math.cos(phi) + z[:, c] * math.sin(phi) + yc  # where each surface row lands
        order = np.argsort(yp)  # a monotonic mapping unless the surface hides itself
        out[:, c] = np.interp(y, yp[order], bright[order, c], left=bright[0, c], right=bright[-1, c])
    if blur:
        out = ndi.gaussian_filter(out, blur)
    return out + np.random.default_rng(seed).normal(0, noise, out.shape)
