"""SEM image processing (plan §2.3): corrections for charging and scanning artefacts,
noise and contrast, the saturated-area mask, and multi-frame averaging.

A processing chain is a list of steps (``Step``: an operation and its settings), kept in
the image's measurement record, so the processed image can always be rebuilt from the
original file and the record. Steps come in two roles:

- **correct** (line levelling, Fourier destriping, background levelling, scan-line
  alignment, denoising): their result is the image that measurements and analysis use.
- **display** (contrast, sharpening): only change what is shown and exported as a
  picture. Grey-level measurements (magic wand, profiles) never see them.

Display steps always run after the corrections, whatever their place in the list.

Only the micrograph is processed; the data bar under it is left as it was. Pixels keep
their places (scan-line alignment moves grey levels within a row by a fraction of a
pixel), so measurements drawn before processing stay where they are.

**Saturated areas** (severe charging, clipped detector) are marked, never "repaired":
the information there is lost. The mask comes from the original image, and the magic
wand and line profiles leave masked pixels out (``measure.compute``).

Charging is best avoided at acquisition (lower voltage, shorter dwell, interlaced scans,
coating); this module only reduces what is left.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# -- the chain ---------------------------------------------------------------------------------


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class Saturation(BaseModel):
    """How saturated pixels are found. ``high``/``low``: the grey level at or beyond which a
    pixel is saturated; None finds it (the dtype's limit, or a plateau at the image's max)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    high: float | None = None
    low: float | None = None
    dark: bool = False  # also mark pixels clipped at the bottom
    dilate: int = 1  # widen the mask by this many pixels: the edge of a clipped area is unreliable too


class Processing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[Step] = Field(default_factory=list)
    saturation: Saturation = Field(default_factory=Saturation)

    def active(self) -> list[Step]:
        return [s for s in self.steps if s.enabled]


@dataclass
class Param:
    name: str
    label: str
    kind: str  # choice | float | int | bool
    default: Any
    choices: tuple[tuple[str, str], ...] = ()  # (value, label)
    lo: float = 0.0
    hi: float = 1e9
    step: float = 1.0
    help: str = ""


@dataclass
class Op:
    name: str
    label: str
    role: str  # correct | display
    func: Callable[..., np.ndarray]
    params: list[Param]
    help: str = ""
    caution: str = ""  # shown when the step is added

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        out = self.defaults()
        for p in self.params:
            if p.name not in params:
                continue
            v = params[p.name]
            if p.kind == "choice":
                if v not in [c[0] for c in p.choices]:
                    raise ValueError(f"{self.name}: {p.name} must be one of {[c[0] for c in p.choices]}")
            elif p.kind == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif p.kind == "int":
                v = int(v)
            else:
                v = float(v)
            if p.kind in ("int", "float") and not (p.lo <= v <= p.hi):
                raise ValueError(f"{self.name}: {p.name} must be between {p.lo:g} and {p.hi:g}")
            out[p.name] = v
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise ValueError(f"{self.name}: unknown setting {sorted(unknown)}")
        return out


# -- helpers -------------------------------------------------------------------------------------


def _range(a: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(a, (0.5, 99.5))
    return float(lo), float(max(hi, lo + 1e-12))


def _reduce(a: np.ndarray, f: int) -> np.ndarray:
    """Block mean by ``f`` (edges padded by reflection)."""
    if f <= 1:
        return a
    r, c = a.shape
    pr, pc = (-r) % f, (-c) % f
    p = np.pad(a, ((0, pr), (0, pc)), mode="reflect") if pr or pc else a
    return p.reshape(p.shape[0] // f, f, p.shape[1] // f, f).mean(axis=(1, 3))


def _expand(small: np.ndarray, shape: tuple[int, int], f: int) -> np.ndarray:
    """Back to full size: block centres interpolated linearly (inverse of ``_reduce``)."""
    from scipy import ndimage as ndi

    if f <= 1:
        return small
    rows = (np.arange(shape[0]) + 0.5) / f - 0.5
    cols = (np.arange(shape[1]) + 0.5) / f - 0.5
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return ndi.map_coordinates(small, [rr, cc], order=1, mode="nearest")


def _blur_large(a: np.ndarray, sigma: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Gaussian blur with a large sigma, fast (on a reduced copy) and, with ``weights``,
    normalised so that weight-0 pixels (saturated) do not pull the result."""
    from scipy import ndimage as ndi

    f = max(1, int(sigma // 6))
    w = np.ones_like(a) if weights is None else weights.astype(np.float64)
    num = ndi.gaussian_filter(_reduce(a * w, f), sigma / f, mode="reflect")
    den = ndi.gaussian_filter(_reduce(w, f), sigma / f, mode="reflect")
    small = num / np.maximum(den, 1e-9)
    return _expand(small, a.shape, f)


# -- corrections ---------------------------------------------------------------------------------


def line_level(a: np.ndarray, *, direction: str = "rows", statistic: str = "median", keep: float = 0.0,
               mask: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Make each scan line's level (median or mean) agree with its neighbours: removes
    bands along the scan direction. ``keep`` > 0 keeps level changes that span more than
    about ``keep`` lines (a slow gradient, wide real features)."""
    from scipy import ndimage as ndi

    t = direction == "columns"
    x = a.T if t else a
    m = None if mask is None else (mask.T if t else mask)
    if m is not None and m.any():
        xm = np.where(m, np.nan, x)
        stat = np.nanmedian(xm, 1) if statistic == "median" else np.nanmean(xm, 1)
        stat = np.where(np.isnan(stat), np.nanmedian(stat) if np.isfinite(stat).any() else 0.0, stat)
    else:
        stat = np.median(x, 1) if statistic == "median" else x.mean(1)
    base = ndi.gaussian_filter1d(stat, keep, mode="nearest") if keep > 0 else np.full_like(stat, np.median(stat))
    out = x - (stat - base)[:, None]
    return out.T if t else out


def destripe(a: np.ndarray, *, direction: str = "horizontal", width: float = 3.0, keep: float = 0.0, strength: float = 1.0,
             mask: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Fourier destriping: stripes along one direction put their energy on one axis of the
    spectrum (horizontal stripes: the vertical axis, near zero horizontal frequency).
    That axis is damped by a Gaussian ``width`` frequency bins wide (wider: stripes whose
    strength changes more along the line). With ``keep`` > 0, bands wider than about
    ``keep`` pixels (slow brightness changes) are kept.

    Removes real structure that runs exactly along the stripes as well (horizontal lines
    of a line/space pattern under horizontal destriping)."""
    t = direction == "vertical"
    x = a.T if t else a
    r, c = x.shape
    p = np.pad(x, ((r // 2, r // 2), (c // 2, c // 2)), mode="symmetric")  # mirror: no seam at the borders
    pr, pc = p.shape
    fy = np.fft.fftfreq(pr)[:, None]
    kx = np.fft.rfftfreq(pc)[None, :] * pc  # in frequency bins
    across = 1 - np.exp(-((fy * keep) ** 2) / 2) if keep > 0 else (fy != 0).astype(float)
    damp = np.exp(-(kx**2) / (2 * max(width, 1e-3) ** 2)) * across
    spec = np.fft.rfft2(p) * (1 - strength * damp)
    out = np.fft.irfft2(spec, s=p.shape)[r // 2 : r // 2 + r, c // 2 : c // 2 + c]
    return out.T if t else out


def _poly_background(a: np.ndarray, order: int, weights: np.ndarray | None) -> np.ndarray:
    f = max(1, int(math.ceil(max(a.shape) / 160)))
    s = _reduce(a, f)
    w = np.ones_like(s, dtype=bool) if weights is None else _reduce(weights.astype(float), f) > 0.99
    yy, xx = np.mgrid[: s.shape[0], : s.shape[1]]
    yn, xn = yy / max(s.shape[0] - 1, 1) * 2 - 1, xx / max(s.shape[1] - 1, 1) * 2 - 1
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([xn.ravel() ** i * yn.ravel() ** j for i, j in terms], 1)
    keep = w.ravel().copy()
    for _ in range(3):  # refit without outliers (particles, defects): the background is the bulk
        coef, *_ = np.linalg.lstsq(A[keep], s.ravel()[keep], rcond=None)
        res = s.ravel() - A @ coef
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = w.ravel() & (np.abs(res) <= 2.5 * max(sd, 1e-12))
    rows = np.linspace(-1, 1, a.shape[0])[:, None]
    cols = np.linspace(-1, 1, a.shape[1])[None, :]
    return sum(cf * cols**i * rows**j for cf, (i, j) in zip(coef, terms))


def _rolling_ball(a: np.ndarray, radius: float) -> np.ndarray:
    from skimage.restoration import rolling_ball

    f = max(1, int(radius // 12))
    s = _reduce(a, f)
    rad = radius / f
    lo, hi = _range(s)
    scale = rad / (hi - lo)  # grey levels in units of the ball's radius: the ball rolls under the features
    bg = rolling_ball((s - lo) * scale, radius=rad) / scale + lo
    return _expand(bg, a.shape, f)


def background(a: np.ndarray, *, method: str = "polynomial", size: float = 0.0, order: int = 2, mode: str = "subtract",
               mask: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Estimate the large-scale brightness (one side bright, a bright centre) and remove it.

    - ``polynomial``: a smooth surface of the given ``order``, fitted to the bulk of the
      image (outliers left out).
    - ``gaussian``: a blur ``size`` pixels wide (default: 1/8 of the image).
    - ``rolling_ball``: the surface a ball of radius ``size`` reaches from below; keeps
      bright features smaller than the ball.

    ``subtract`` keeps the mean level; ``divide`` suits brightness that scales the signal
    (uneven detector collection)."""
    weights = None if mask is None or not mask.any() else ~mask
    size = size or max(a.shape) / 8
    if method == "polynomial":
        bg = _poly_background(a, order, weights)
    elif method == "gaussian":
        bg = _blur_large(a, size, weights)
    else:
        bg = _rolling_ball(a, size)
    mean = float(bg.mean())
    if mode == "divide":
        floor = max(abs(mean) * 1e-3, 1e-12)
        return a / np.maximum(bg, floor) * mean
    return a - bg + mean


def align_rows(a: np.ndarray, *, max_shift: float = 5.0, neighbours: int = 2, mask: np.ndarray | None = None) -> np.ndarray:
    """Undo horizontal jumps of single scan lines (charging, vibration): each line is
    cross-correlated with the mean of its ``neighbours`` above and below and moved by the
    sub-pixel offset found. Features that slant across many lines are kept: their offset
    against the neighbours' mean is zero."""
    shifts = row_shifts(a, max_shift=max_shift, neighbours=neighbours)
    return shift_rows(a, shifts)


def row_shifts(a: np.ndarray, *, max_shift: float = 5.0, neighbours: int = 2, iterations: int = 12, min_corr: float = 0.6) -> np.ndarray:
    """Per-line horizontal offsets (pixels; positive: the line sits to the right)."""
    from scipy import ndimage as ndi

    rows, cols = a.shape
    k = max(1, int(neighbours))
    m = int(math.ceil(max_shift)) + 1
    n = 1 << int(math.ceil(math.log2(cols + 2 * m)))
    total = np.zeros(rows)
    base = ndi.gaussian_filter1d(np.asarray(a, np.float64), 0.8, axis=1)
    for _ in range(iterations):
        cur = shift_rows(base, total, order=1) if total.any() else base
        g = np.diff(cur, axis=1)
        g -= g.mean(1, keepdims=True)
        # the mean of the kk lines above and kk below (kk shrinks at the top and bottom, so the
        # window stays symmetric: a slanted feature then has no offset against it)
        cs = np.cumsum(np.vstack([np.zeros((1, g.shape[1])), g]), 0)
        r = np.arange(rows)
        kk = np.minimum(k, np.minimum(r, rows - 1 - r))
        ref = (cs[r + kk + 1] - cs[r - kk] - g) / np.maximum(2 * kk, 1)[:, None]
        ref[kk == 0] = 0.0  # first and last line: no estimate
        cc = np.fft.irfft(np.fft.rfft(g, n) * np.conj(np.fft.rfft(ref, n)), n)
        lags = np.concatenate([cc[:, n - m :], cc[:, : m + 1]], 1)  # lags -m..m
        norm = np.sqrt((g**2).sum(1) * (ref**2).sum(1)) + 1e-12
        i = np.clip(np.argmax(lags[:, 1:-1], 1) + 1, 1, 2 * m - 1)
        y0, y1, y2 = (lags[np.arange(rows), i + d] for d in (-1, 0, 1))
        den = y0 - 2 * y1 + y2
        sub = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
        d = (i - m) + np.clip(sub, -0.5, 0.5)
        d = np.where((y1 / norm > min_corr) & (np.abs(d) <= max_shift), d, 0.0)  # weak correlation (flat lines): leave alone
        step = 0.5 * d  # half a step: alternate-line (interlace) offsets converge instead of swapping
        total += step
        if np.abs(step).max() < 0.01:
            break
    total -= np.median(total)
    return np.clip(total, -max_shift, max_shift)


def shift_rows(a: np.ndarray, shifts: np.ndarray, order: int = 3) -> np.ndarray:
    """Move each line left by its shift (undo an offset of ``shifts`` pixels)."""
    from scipy import ndimage as ndi

    rows, cols = a.shape
    rr, cc = np.meshgrid(np.arange(rows, dtype=float), np.arange(cols, dtype=float), indexing="ij")
    return ndi.map_coordinates(np.asarray(a, np.float64), [rr, cc + shifts[:, None]], order=order, mode="nearest")


def noise_sigma(a: np.ndarray) -> float:
    """Standard deviation of the pixel noise (wavelet-free MAD estimate on pixel differences)."""
    d = np.diff(np.asarray(a, np.float64), axis=1)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2))


def denoise(a: np.ndarray, *, method: str = "nlm", strength: float = 1.0, mask: np.ndarray | None = None) -> np.ndarray:
    """Reduce pixel noise.

    - ``nlm`` (non-local means): averages similar patches; keeps edges best. Filtering
      strength ``strength`` × the estimated noise level.
    - ``median``: a (2·strength+1)-pixel median; removes isolated spikes.
    - ``bilateral``: smooths within similar grey levels.
    - ``gaussian``: a blur of ``strength`` pixels; also blurs edges (widths grow)."""
    from scipy import ndimage as ndi

    if method == "median":
        return ndi.median_filter(a, size=2 * max(1, int(round(strength))) + 1, mode="reflect")
    if method == "gaussian":
        return ndi.gaussian_filter(a, strength, mode="reflect")
    lo, hi = _range(a)
    span = hi - lo
    x = (a - lo) / span
    sigma = max(noise_sigma(x), 1e-6)
    if method == "bilateral":
        from skimage.restoration import denoise_bilateral

        y = denoise_bilateral(x, sigma_color=2.5 * sigma * strength, sigma_spatial=1.0 + strength, win_size=5, mode="reflect")
    else:
        from skimage.restoration import denoise_nl_means

        y = denoise_nl_means(x, h=0.8 * sigma * strength, sigma=sigma, patch_size=5, patch_distance=6, fast_mode=True)
    return y * span + lo


# -- display -------------------------------------------------------------------------------------


def contrast(a: np.ndarray, *, low: float = 0.5, high: float = 99.5, gamma: float = 1.0, clahe: bool = False, clip: float = 0.01,
             mask: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Grey levels to 0..1: the ``low``..``high`` percentiles stretched to black..white, then
    gamma, then (optionally) local histogram equalisation (CLAHE)."""
    sel = a if mask is None or not mask.any() or mask.all() else a[~mask]
    lo, hi = np.percentile(sel, (low, high))
    y = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    if gamma != 1.0:
        y = y**gamma
    if clahe:
        from skimage.exposure import equalize_adapthist

        y = equalize_adapthist(y, clip_limit=clip)
    return y


def sharpen(a: np.ndarray, *, radius: float = 2.0, amount: float = 0.8, mask: np.ndarray | None = None) -> np.ndarray:
    """Unsharp mask (for viewing only: it creates overshoot at edges)."""
    from scipy import ndimage as ndi

    return a + amount * (a - ndi.gaussian_filter(a, radius, mode="reflect"))


OPS: dict[str, Op] = {
    o.name: o
    for o in [
        Op("line_level", "逐行校平", "correct", line_level, [
            Param("direction", "方向", "choice", "rows", (("rows", "逐行（水平条纹）"), ("columns", "逐列（垂直条纹）"))),
            Param("statistic", "依据", "choice", "median", (("median", "中位数"), ("mean", "平均值"))),
            Param("keep", "保留宽于（行）", "float", 0.0, lo=0, hi=10000, step=5,
                  help="0：去掉所有行间亮度差；大于 0：保留跨越约这么多行的缓慢变化"),
        ], help="让每条扫描线的亮度与相邻扫描线一致，去掉沿扫描方向的明暗条纹",
           caution="会把与扫描方向平行的真实结构（例如水平线条）一起压平；“保留宽于”为 0 时也压平上下方向的缓慢亮度变化"),
        Op("destripe", "傅里叶去条纹", "correct", destripe, [
            Param("direction", "条纹方向", "choice", "horizontal", (("horizontal", "水平"), ("vertical", "垂直"))),
            Param("width", "频带宽度", "float", 3.0, lo=0.5, hi=50, step=0.5, help="在频谱上压低的带宽（频率格数）：条纹沿行的强弱变化越大，需要越宽；太宽会损伤真实结构"),
            Param("keep", "保留宽于（像素）", "float", 0.0, lo=0, hi=100000, step=10, help="0：各种宽度的条纹都去掉；大于 0：比这更宽的明暗带保留"),
            Param("strength", "强度", "float", 1.0, lo=0, hi=1, step=0.1),
        ], help="条纹集中在一个方向时，在频谱中压低该方向的成分",
           caution="与条纹同方向的真实结构也会被削弱"),
        Op("background", "背景校平", "correct", background, [
            Param("method", "方法", "choice", "polynomial", (("polynomial", "多项式曲面"), ("gaussian", "高斯"), ("rolling_ball", "滚球"))),
            Param("size", "尺度（像素）", "float", 0.0, lo=0, hi=100000, step=10, help="高斯宽度或滚球半径；0：图宽的 1/8"),
            Param("order", "多项式阶数", "int", 2, lo=1, hi=6),
            Param("mode", "方式", "choice", "subtract", (("subtract", "扣除"), ("divide", "相除"))),
        ], help="估计大范围的亮度不均（一侧亮、中央亮）并扣除"),
        Op("align_rows", "行错位校正", "correct", align_rows, [
            Param("max_shift", "最大错位（像素）", "float", 5.0, lo=0.5, hi=100, step=0.5),
            Param("neighbours", "参考行数（每侧）", "int", 2, lo=1, hi=20),
        ], help="把水平错开的扫描线对齐（与上下相邻行互相关，亚像素）"),
        Op("denoise", "去噪", "correct", denoise, [
            Param("method", "方法", "choice", "nlm", (("nlm", "非局部均值"), ("median", "中值"), ("bilateral", "双边"), ("gaussian", "高斯"))),
            Param("strength", "强度", "float", 1.0, lo=0.1, hi=10, step=0.1),
        ], help="降低像素噪声；高斯会把边缘抹宽，测线宽前慎用"),
        Op("contrast", "对比度", "display", contrast, [
            Param("low", "黑点（百分位）", "float", 0.5, lo=0, hi=50, step=0.5),
            Param("high", "白点（百分位）", "float", 99.5, lo=50, hi=100, step=0.5),
            Param("gamma", "伽马", "float", 1.0, lo=0.1, hi=5, step=0.1),
            Param("clahe", "局部直方图均衡（CLAHE）", "bool", False),
            Param("clip", "CLAHE 限幅", "float", 0.01, lo=0.001, hi=1, step=0.005),
        ], help="只改变显示和导出的图，不影响测量"),
        Op("sharpen", "锐化", "display", sharpen, [
            Param("radius", "半径（像素）", "float", 2.0, lo=0.3, hi=50, step=0.5),
            Param("amount", "量", "float", 0.8, lo=0, hi=5, step=0.1),
        ], help="反锐化掩模，只用于显示；会在边缘产生过冲"),
    ]
}  # fmt: skip


def step(op: str, **params: Any) -> Step:
    """A validated step (unknown operations and settings are refused)."""
    if op not in OPS:
        raise ValueError(f"unknown processing step {op!r}; one of {', '.join(OPS)}")
    OPS[op].resolve(params)
    return Step(op=op, params=params)


def parse_step(text: str) -> Step:
    """``op`` or ``op:key=value,key=value`` (the command-line form)."""
    op, _, rest = text.partition(":")
    params: dict[str, Any] = {}
    for part in filter(None, (s.strip() for s in rest.split(","))):
        key, eq, value = part.partition("=")
        if not eq:
            raise ValueError(f"expected key=value, got {part!r}")
        params[key.strip()] = value.strip()
    s = step(op.strip(), **params)
    s.params = {k: v for k, v in OPS[s.op].resolve(params).items() if k in params}  # typed values
    return s


# -- saturation ----------------------------------------------------------------------------------


def saturation_mask(image: np.ndarray, sat: Saturation | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Pixels at the detector's limit (and ``dilate`` pixels around them)."""
    from scipy import ndimage as ndi

    sat = sat or Saturation()
    a = np.asarray(image)
    n = a.size
    plateau = max(16, int(1e-4 * n))

    def level(top: bool, given: float | None) -> float | None:
        if given is not None:
            return given
        ext = float(a.max() if top else a.min())
        if np.issubdtype(a.dtype, np.integer):
            limit = np.iinfo(a.dtype).max if top else np.iinfo(a.dtype).min
            if ext == limit:
                return ext
        return ext if int(np.count_nonzero(a == ext)) >= plateau else None  # a plateau at the extreme: clipped

    hi = level(True, sat.high)
    lo = level(False, sat.low) if sat.dark else None
    mask = np.zeros(a.shape, bool)
    if hi is not None:
        mask |= a >= hi
    if lo is not None:
        mask |= a <= lo
    raw = int(mask.sum())
    if sat.dilate and raw:
        mask = ndi.binary_dilation(mask, iterations=int(sat.dilate))
    _, regions = ndi.label(mask)
    return mask, {"high": hi, "low": lo, "pixels": raw, "fraction": raw / n, "marked": int(mask.sum()), "regions": int(regions)}


# -- running a chain -----------------------------------------------------------------------------


@dataclass
class Result:
    analysis: np.ndarray  # float64, full image (data bar untouched): what measurements use
    display: np.ndarray  # uint8, full image: what is shown and exported as a picture
    original: np.ndarray  # uint8, full image, no processing: for the before/after view
    mask: np.ndarray | None  # saturated pixels (full image), or None
    saturation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _to_u8(a: np.ndarray, rng: tuple[float, float] | None = None) -> np.ndarray:
    lo, hi = rng or _range(a)
    return np.clip((a - lo) / max(hi - lo, 1e-12) * 255 + 0.5, 0, 255).astype(np.uint8)


def _key(steps: list[Step]) -> str:
    return json.dumps([s.model_dump() for s in steps], sort_keys=True)


class Processor:
    """Runs a chain on one image, caching each prefix of the chain: changing the last step
    (often the display contrast) does not rerun the slow ones before it. Safe to call from
    two threads (one waits for the other)."""

    def __init__(self, image: np.ndarray, crop_rows: int | None = None) -> None:
        self.image = np.asarray(image)
        self.crop_rows = crop_rows if crop_rows and 0 < crop_rows < self.image.shape[0] else None
        self._cache: dict[str, np.ndarray] = {}
        self._mask: tuple[str, np.ndarray, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    @property
    def micro(self) -> np.ndarray:
        return self.image[: self.crop_rows] if self.crop_rows else self.image

    def mask(self, sat: Saturation) -> tuple[np.ndarray | None, dict[str, Any]]:
        if not sat.enabled:
            return None, {}
        key = sat.model_dump_json()
        if self._mask is None or self._mask[0] != key:
            m, info = saturation_mask(self.micro, sat)
            self._mask = (key, m, info)
        return self._mask[1], self._mask[2]

    def _run(self, steps: list[Step], mask: np.ndarray | None) -> np.ndarray:
        if not steps:
            return np.asarray(self.micro, np.float64)
        key = _key(steps) + (f"|m{int(mask.sum())}" if mask is not None else "")
        if key not in self._cache:
            prev = self._run(steps[:-1], mask)
            s = steps[-1]
            op = OPS[s.op]
            self._cache[key] = np.asarray(op.func(prev, mask=mask, **op.resolve(s.params)), np.float64)
            if len(self._cache) > 12:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def run(self, proc: Processing | None = None) -> Result:
        with self._lock:
            return self._run_chain(proc or Processing())

    def _run_chain(self, proc: Processing) -> Result:
        active = proc.active()
        for s in active:
            if s.op not in OPS:
                raise ValueError(f"unknown processing step {s.op!r}")
        corrections = [s for s in active if OPS[s.op].role == "correct"]
        shows = [s for s in active if OPS[s.op].role == "display"]
        mask, sat = self.mask(proc.saturation)
        corrected = self._run(corrections, mask)
        shown = self._run(corrections + shows, mask) if shows else corrected
        stretched = any(s.op == "contrast" for s in shows)
        notes = [f"{OPS[s.op].label}：{OPS[s.op].caution}" for s in corrections if OPS[s.op].caution]
        bar = self.image[self.crop_rows :] if self.crop_rows else None
        full = np.asarray(self.image, np.float64)
        analysis = full.copy()
        analysis[: corrected.shape[0]] = corrected
        orig_rng = _range(self.micro)
        disp = np.clip(shown, 0, 1) * 255 + 0.5 if stretched else None
        d_micro = disp.astype(np.uint8) if disp is not None else _to_u8(shown, _range(shown) if corrections or shows else orig_rng)
        o_micro = _to_u8(self.micro, orig_rng)
        if bar is not None:
            b = _to_u8(bar, (float(bar.min()), float(max(bar.max(), bar.min() + 1e-12))))
            display, original = np.vstack([d_micro, b]), np.vstack([o_micro, b])
        else:
            display, original = d_micro, o_micro
        full_mask = None
        if mask is not None:
            full_mask = np.zeros(self.image.shape, bool)
            full_mask[: mask.shape[0]] = mask
            if sat.get("pixels"):
                notes.insert(0, f"饱和区：{sat['pixels']} 个像素（{sat['fraction']:.2%}），已标记；魔棒和线剖面不使用这些像素")
        return Result(analysis=analysis, display=display, original=original, mask=full_mask, saturation=sat, notes=notes)


def apply(image: np.ndarray, proc: Processing | None = None, crop_rows: int | None = None) -> Result:
    return Processor(image, crop_rows).run(proc)


def describe(proc: Processing) -> list[str]:
    """One line per active step, as written in reports."""
    out = []
    for s in proc.active():
        op = OPS.get(s.op)
        if op is None:
            out.append(s.op)
            continue
        vals = op.resolve(s.params)
        parts = []
        for p in op.params:
            v = vals[p.name]
            if p.kind == "choice":
                v = dict(p.choices).get(v, v)
            elif p.kind == "bool":
                v = "是" if v else "否"
            parts.append(f"{p.label} {v:g}" if isinstance(v, float) else f"{p.label} {v}")
        out.append(f"{op.label}（{'仅显示' if op.role == 'display' else '校正'}）：" + "，".join(parts))
    if proc.saturation.enabled:
        out.append("饱和区标记：开" + (f"，阈值 {proc.saturation.high:g}" if proc.saturation.high is not None else "（自动）"))
    return out


# -- multi-frame averaging -----------------------------------------------------------------------


def average_frames(frames: list[np.ndarray], *, align: bool = True, upsample: int = 100) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mean of several frames of the same place, each moved onto the first (sub-pixel
    phase correlation, then again against the first pass's mean). Returns the mean and
    each frame's (dy, dx) correction."""
    from scipy import ndimage as ndi
    from skimage.registration import phase_cross_correlation

    if len(frames) < 2:
        raise ValueError("averaging needs at least two frames")
    arrs = [np.asarray(f, np.float64) for f in frames]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("the frames differ in size")
    if not align:
        return np.mean(arrs, 0), [(0.0, 0.0)] * len(arrs)
    ref = arrs[0]
    for _ in range(2):
        shifts = [tuple(float(v) for v in phase_cross_correlation(ref, a, upsample_factor=upsample, normalization=None)[0]) for a in arrs]
        moved = [ndi.shift(a, s, order=3, mode="nearest") for a, s in zip(arrs, shifts)]
        ref = np.mean(moved, 0)
    return ref, shifts  # type: ignore[return-value]


def write_derived(path: str | Path, image: np.ndarray, cal: Any, provenance: dict[str, Any]) -> Path:
    """Write a processed or averaged image (float32 TIFF) and record its calibration (taken
    over from the source image) under the new file's SHA-256, so Vexel opens it calibrated."""
    import tifffile

    from vexel.packs.sem import calibrate as C

    path = Path(path)
    desc = {"vexel": {**provenance, "pixel_size": getattr(cal, "pixel_size", None)}}
    tifffile.imwrite(path, np.asarray(image, np.float32), description=json.dumps(desc), metadata=None)
    if cal is not None and cal.ok:
        new = C.Calibration.from_dict(cal.to_dict())
        new.file, new.sha256, new.image_shape = path.name, C.file_sha256(path), tuple(image.shape)
        new.texts = []
        new.warnings = [*cal.warnings, f"标定沿用 {provenance.get('source', '源图')}（{provenance.get('operation', '处理')}后的图）"]
        if not new.crop_rows or new.crop_rows >= image.shape[0]:
            new.crop_rows, new.crop_source = None, "none"
        C.save(new)
    return path


def average_files(paths: list[str | Path], out: str | Path, *, align: bool = True, use_ocr: bool = True) -> dict[str, Any]:
    """Average image files of the same place: the micrographs are aligned and averaged, the
    first file's data bar is kept under the result, and the first file's calibration is
    recorded for the new file."""
    from vexel.packs.sem import calibrate as C

    paths = [Path(p) for p in paths]
    cal = C.for_file(paths[0], use_ocr=use_ocr)
    imgs = [C.load_image(p) for p in paths]
    if any(i.shape != imgs[0].shape for i in imgs):
        raise ValueError("the frames differ in size")
    crop = cal.crop_rows if cal.crop_rows and cal.crop_rows < imgs[0].shape[0] else None
    micro = [i[:crop] if crop else i for i in imgs]
    mean, shifts = average_frames(micro, align=align)
    full = np.vstack([mean, np.asarray(imgs[0][crop:], np.float64)]) if crop else mean
    sigma_one = noise_sigma(np.asarray(micro[0], np.float64))
    sigma_mean = noise_sigma(mean)
    prov = {"operation": "多帧平均", "source": paths[0].name, "frames": [p.name for p in paths],
            "sha256": [C.file_sha256(p) for p in paths], "shifts": [list(s) for s in shifts]}  # fmt: skip
    write_derived(out, full, cal, prov)
    return {"output": str(out), "frames": len(paths), "shifts": shifts, "noise_single": sigma_one, "noise_mean": sigma_mean,
            "calibration": cal.describe() if cal.ok else None}  # fmt: skip


__all__ = [
    "OPS", "Op", "Param", "Processing", "Processor", "Result", "Saturation", "Step", "align_rows", "apply", "average_files",
    "average_frames", "background", "contrast", "denoise", "describe", "destripe", "line_level", "noise_sigma", "parse_step",
    "row_shifts", "saturation_mask", "sharpen", "shift_rows", "step", "write_derived",
]  # fmt: skip
