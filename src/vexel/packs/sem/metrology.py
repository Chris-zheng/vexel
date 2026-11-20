"""Semiconductor dimensional metrology on SEM images (plan §7 A, M4): edges tracked along
lines, then line width (CD), space, pitch, line-edge and line-width roughness (LER, LWR),
layer thickness in cross-sections, and feature profiles (height, top/middle/bottom
width, sidewall angle).

All of it rests on one operation, ``track_edges``: edges found on the region's mean
profile are followed line by line across the region (sub-pixel, with a chosen edge
definition), giving each edge as a position series ``x(y)``.

Roughness (Villarrubia & Bunday 2005; SEMI P47 practice): an edge's roughness is the
standard deviation of its positions about a best-fit straight line, reported as 3σ.
Image noise moves the detected edge too and inflates σ ("noise bias"). The noise adds a
flat floor to the edge's power spectrum (white, uncorrelated between lines). The power
spectrum is fitted with a roughness model plus that floor, P(f) = A / (1 + (2πfξ)²)^(H+½)
+ c, and the floor is removed: σ²(unbiased) = σ² − σ²(noise). (When the fit fails, the
mean power of the highest frequencies is taken as the floor.) The beam's blur lowers the
roughness an image can show at all; that is the instrument's resolution and is not
corrected.
Both values are reported, with the sampling (number of lines, spacing) and the
correlation length. There is no single official algorithm; results are only comparable
with the same edge definition, sampling and noise treatment.

Positions are in pixels of the analysed region; ``Analysis`` (``analysis.py``) turns them
into lengths with the calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# -- edge tracking -------------------------------------------------------------------------------


@dataclass
class Edge:
    position: np.ndarray  # x (sub-pixel) for every line y of the region; NaN where not found
    polarity: int  # +1: dark -> bright going to +x; -1: bright -> dark
    start: float  # position on the mean profile

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.position)

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.position))


def _skew(a: np.ndarray, bands: int = 8) -> float:
    """Drift of the pattern along x per line (px/line), from band means cross-correlated."""
    rows = a.shape[0]
    if rows < 2 * bands:
        return 0.0
    edges = np.linspace(0, rows, bands + 1).astype(int)
    prof = [a[edges[i] : edges[i + 1]].mean(0) for i in range(bands)]
    from vexel.packs.sem.measure import period

    ref = prof[bands // 2] - prof[bands // 2].mean()
    ys, xs = [], []
    n = len(ref)
    per = period(ref)
    m = max(1, int(0.4 * per) if per else n // 4)  # a periodic pattern repeats every pitch: search less than half of it
    for i, p in enumerate(prof):
        p = p - p.mean()
        cc = np.correlate(np.gradient(p), np.gradient(ref), mode="full")
        lag = int(np.argmax(cc[n - 1 - m : n - 1 + m + 1])) + n - 1 - m
        if 0 < lag < len(cc) - 1:
            y0, y1, y2 = cc[lag - 1], cc[lag], cc[lag + 1]
            den = y0 - 2 * y1 + y2
            lag = lag + (0.5 * (y0 - y2) / den if den else 0.0)
        xs.append(lag - (n - 1))
        ys.append((edges[i] + edges[i + 1]) / 2)
    return float(np.polyfit(ys, xs, 1)[0])


def _weak_edges(m: np.ndarray, smooth: float, min_distance: float, floor: float = 0.05) -> list[tuple[float, int]]:
    """Edges on a (low-noise) mean profile down to a few times its own noise, and at least
    ``floor`` of the strongest: a weak interface next to a strong one is still found."""
    from scipy import ndimage as ndi
    from scipy.signal import find_peaks

    v = ndi.gaussian_filter1d(np.asarray(m, float), max(smooth, 0.7))
    g = np.gradient(v)
    mag = np.abs(g)
    noise = 1.4826 * float(np.median(np.abs(np.diff(g) - np.median(np.diff(g))))) + 1e-12
    thr = max(6 * noise, floor * float(mag.max()))
    idx, _ = find_peaks(mag, height=thr, prominence=thr, distance=max(2, int(min_distance)))
    out = []
    for i in idx:
        if 0 < i < len(mag) - 1:
            y0, y1, y2 = mag[i - 1], mag[i], mag[i + 1]
            den = y0 - 2 * y1 + y2
            out.append((float(i + (0.5 * (y0 - y2) / den if den else 0.0)), 1 if g[i] > 0 else -1))
    return out


def _find_on_mean(m: np.ndarray, method: str, level: float, smooth: float, min_distance: float, weak: bool = False) -> list[tuple[float, int]]:
    from vexel.packs.sem.measure import profile_edges

    if weak:
        return [(p, s) for p, s in _weak_edges(m, smooth, min_distance) if 2 <= p <= len(m) - 3]
    if method == "peak":  # each bright edge peak is an edge
        edges = profile_edges(m, "peak", level, max(smooth, 0.7), max(2, int(min_distance)))
        return [(p, s) for p, s in edges if 2 <= p <= len(m) - 3]
    # found by their slope (works for any number of grey levels, e.g. a layer stack); the edge
    # definition (threshold or gradient) is applied when each line is searched
    edges = profile_edges(m, "gradient", level, max(smooth, 0.7), max(2, int(min_distance)))
    # drop edges too close to the region border to be followed
    return [(p, s) for p, s in edges if 2 <= p <= len(m) - 3]


def track_edges(a: np.ndarray, *, method: str = "threshold", level: float = 0.5, smooth: float = 1.0, average: int = 1,
                mask: np.ndarray | None = None, weak: bool = False) -> tuple[list[Edge], dict[str, Any]]:  # fmt: skip
    """Follow every edge of a pattern of (roughly) vertical edges down the region ``a``.

    The mean profile (after removing any slant) gives the edges and their polarity; each
    line is then searched within ±40 % of the distance to the neighbouring edges.
    ``threshold``: where the line crosses ``level`` between the local low and high grey
    (the plateaus beside the edge on the mean profile, re-levelled per line); ``gradient``:
    the steepest point (parabolic peak of the slope); ``peak``: the maximum of a bright edge
    peak (SEM edge brightening). ``average`` lines are averaged
    (moving mean) before searching: less noise, but high-frequency roughness is smoothed.
    ``mask`` (saturated) pixels make the line's edge missing."""
    from scipy import ndimage as ndi

    a = np.asarray(a, np.float64)
    rows, cols = a.shape
    if average > 1:
        a = ndi.uniform_filter1d(a, int(average), axis=0, mode="nearest")
    sm = ndi.gaussian_filter1d(a, smooth, axis=1, mode="nearest") if smooth else a
    slope = _skew(sm)
    yy = np.arange(rows)
    # mean profile with the slant taken out
    shift = slope * (yy - rows / 2)
    rr, cc = np.meshgrid(yy.astype(float), np.arange(cols, dtype=float), indexing="ij")
    straight = ndi.map_coordinates(sm, [rr, cc + shift[:, None]], order=1, mode="nearest")
    mean = straight.mean(0)
    med = np.median(straight, axis=1)
    row_offset = med - np.median(med)  # each line's brightness against the others (charging stripes, drift along the lines)
    found = _find_on_mean(mean, method, level, 0.0, 3, weak=weak)
    info: dict[str, Any] = {"slant_px_per_line": slope, "mean_profile": mean, "lines": rows}
    if not found:
        return [], info
    pos = np.array([p for p, _ in found])
    gaps = np.diff(np.concatenate([[-np.inf], pos, [np.inf]]))
    half = np.maximum(np.minimum(gaps[:-1], gaps[1:]) * 0.4, 2.0)
    half = np.minimum(half, cols / 2)
    # plateau levels beside each edge: the mean profile in the middle of the gaps to the neighbouring
    # edges (well clear of every transition, also when rough edges smear the mean profile)
    bounds = np.concatenate([[0.0], pos, [cols - 1.0]])
    gaps_mid = []
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        i0, i1 = int(round(a_ + 0.3 * (b_ - a_))), int(round(a_ + 0.7 * (b_ - a_)))
        gaps_mid.append(float(np.median(mean[i0 : max(i1, i0) + 1])))
    plateaus = [(min(gaps_mid[k], gaps_mid[k + 1]), max(gaps_mid[k], gaps_mid[k + 1])) for k in range(len(pos))]
    out: list[Edge] = []
    with np.errstate(divide="ignore", invalid="ignore"):  # all-NaN windows (masked, off the image) are expected
        out = _follow(found, half, plateaus, sm, shift, row_offset, rows, cols, level, method, mask, yy)
    return out, info


def _follow(found: list[tuple[float, int]], half: np.ndarray, plateaus: list[tuple[float, float]], sm: np.ndarray, shift: np.ndarray,
            row_offset: np.ndarray, rows: int, cols: int, level: float, method: str, mask: np.ndarray | None, yy: np.ndarray) -> list[Edge]:  # fmt: skip
    out: list[Edge] = []
    for k_edge, ((p0, pol), h) in enumerate(zip(found, half)):
        hw = int(math.ceil(h))
        centre = p0 + shift  # per line
        base = np.floor(centre).astype(int)
        idx = base[:, None] + np.arange(-hw, hw + 2)[None, :]
        ok_idx = (idx >= 0) & (idx < cols)
        idxc = np.clip(idx, 0, cols - 1)
        win = np.take_along_axis(sm, idxc, axis=1)
        win = np.where(ok_idx, win, np.nan)
        rel_centre = centre - base + hw  # centre in window samples
        if method == "peak":
            v = np.where(np.isfinite(win), win, -np.inf)
            j = np.argmax(v, axis=1)
            jj = np.clip(j, 1, v.shape[1] - 2)
            y0, y1, y2 = (v[np.arange(rows), jj + d] for d in (-1, 0, 1))
            den = y0 - 2 * y1 + y2
            sub = np.where(np.isfinite(den) & (np.abs(den) > 1e-12), 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
            x = np.where(j == jj, jj + np.clip(sub, -0.5, 0.5), np.nan)
        elif method == "gradient":
            g = np.diff(win, axis=1) * pol
            g = np.where(np.isfinite(g), g, -np.inf)
            j = np.argmax(g, axis=1)
            jj = np.clip(j, 1, g.shape[1] - 2)
            y0, y1, y2 = (g[np.arange(rows), jj + d] for d in (-1, 0, 1))
            den = y0 - 2 * y1 + y2
            sub = np.where(np.isfinite(den) & (np.abs(den) > 1e-12), 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
            x = jj + np.clip(sub, -0.5, 0.5) + 0.5  # slope sample j lies between samples j and j+1
            x = np.where(j == jj, x, np.nan)
            x = np.where(y1 > 0, x, np.nan)
        else:
            # plateaus beside the edge from the mean profile (stable), shifted by each line's own
            # brightness offset. (Plateaus taken from each line's window shrink the edge's excursions:
            # when the edge strays to one side, that side's plateau runs short and takes in edge samples.)
            lo0, hi0 = plateaus[k_edge]
            lo, hi = lo0 + row_offset, hi0 + row_offset
            lev = lo + level * (hi - lo)
            s = (win - lev[:, None]) * pol
            cross = (s[:, :-1] < 0) & (s[:, 1:] >= 0)
            jpos = np.arange(s.shape[1] - 1)[None, :] + s[:, :-1] / np.where(cross, s[:, :-1] - s[:, 1:], 1)
            # noise can make several crossings: take the one nearest the edge's equal-area position (the
            # plateau-normalised sum over the window, which is linear in the noise). Choosing the one
            # nearest the window centre or the noisy steepest point shrinks real excursions.
            frac = (win - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None]  # not clipped: linear in the noise
            below = np.nansum(1 - frac, axis=1) if pol > 0 else np.nansum(frac, axis=1)
            coarse = below - 0.5  # samples before the edge, minus half a sample: the 50 % position
            dist = np.where(cross, np.abs(jpos - coarse[:, None]), np.inf)
            k = np.argmin(dist, axis=1)
            x = np.where(np.isfinite(dist[np.arange(rows), k]), jpos[np.arange(rows), k], np.nan)
            x = np.where(hi - lo > 1e-12, x, np.nan)
        xabs = base - hw + x
        if mask is not None and mask.any():
            xi = np.clip(np.round(np.nan_to_num(xabs, nan=0)).astype(int), 0, cols - 1)
            xabs = np.where(mask[yy, xi], np.nan, xabs)
        # a jump far off the line's course is a mis-detection
        resid = xabs - (p0 + shift)
        bad = np.abs(resid - np.nanmedian(resid)) > h
        out.append(Edge(position=np.where(bad, np.nan, xabs), polarity=int(pol), start=float(p0)))
    return out


# -- roughness -----------------------------------------------------------------------------------


def _fill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.all():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def noise_floor(psd: np.ndarray, noise_band: float = 0.25) -> tuple[float, str]:
    """White-noise variance in a one-sided periodogram (bins 1..): from a fit of roughness
    model + floor on log-binned power, else the mean of the top ``noise_band`` of bins."""
    from scipy.optimize import least_squares

    P = np.asarray(psd[1:], np.float64)
    n = len(P)
    band = float(P[int(n * (1 - noise_band)) :].mean()) * n if n else 0.0
    if n < 16 or not np.all(np.isfinite(P)) or P.max() <= 0:
        return band, "band"
    f = np.arange(1, n + 1) / (2 * n)
    edges = np.unique(np.round(np.geomspace(1, n + 1, 40)).astype(int))
    fb = np.array([f[a - 1 : b - 1].mean() for a, b in zip(edges[:-1], edges[1:])])
    pb = np.array([P[a - 1 : b - 1].mean() for a, b in zip(edges[:-1], edges[1:])])
    wb = np.diff(edges).astype(float)
    if np.any(pb <= 0):
        return band, "band"

    def resid(p: np.ndarray) -> np.ndarray:
        A, xi, H, c = math.exp(p[0]), math.exp(p[1]), p[2], math.exp(p[3])
        return (np.log(A / (1 + (2 * np.pi * fb * xi) ** 2) ** (H + 0.5) + c) - np.log(pb)) * np.sqrt(wb)

    p0 = [math.log(pb[0]), math.log(5.0), 0.5, math.log(max(pb[-3:].mean() * 0.5, 1e-30))]
    try:
        r = least_squares(resid, p0, bounds=([-80, math.log(0.3), 0.1, -80], [20, math.log(max(n, 2)), 1.5, 20]))
    except (ValueError, FloatingPointError):
        return band, "band"
    if not r.success:
        return band, "band"
    return math.exp(r.x[3]) * n, "fit"


def roughness(series: np.ndarray, *, detrend: bool = True, noise_band: float = 0.25) -> dict[str, Any]:
    """σ of an edge (or width) series in pixels, biased and noise-corrected, its PSD and
    correlation length (lines). ``noise_band``: the top fraction of spatial frequencies
    whose mean power is taken as the white-noise floor."""
    x = np.asarray(series, np.float64)
    ok = np.isfinite(x)
    n_ok = int(ok.sum())
    out: dict[str, Any] = {"n": n_ok, "missing": float(1 - n_ok / max(len(x), 1))}
    if n_ok < 16:
        return out | {"sigma": None, "sigma_unbiased": None, "noise_sigma": None, "corr_length": None}
    y = np.arange(len(x))
    r = x.copy()
    if detrend:
        c = np.polyfit(y[ok], x[ok], 1)
        r = x - np.polyval(c, y)
    else:
        r = x - np.nanmean(x)
    r = _fill(r)
    r -= r.mean()
    n = len(r)
    var = float(r.var())
    f = np.fft.rfft(r)
    psd = (np.abs(f) ** 2) / n**2
    psd[1:-1 if n % 2 == 0 else None] *= 2  # one-sided: sum over k >= 1 equals the variance
    k = np.arange(len(psd))
    floor, how = noise_floor(psd, noise_band)
    noise_var = min(floor, var)
    unb = max(var - noise_var, 0.0)
    # correlation length: unbiased autocovariance (white noise sits only at lag 0) falls to 1/e
    ac = np.fft.irfft(np.abs(np.fft.rfft(r, 2 * n)) ** 2)[:n] / n
    corr = None
    if unb > 0:
        rho = ac[1:] / unb
        below = np.flatnonzero(rho < 1 / math.e)
        if len(below):
            j = int(below[0])
            if j == 0:
                corr = float(1 / (1 - math.log(max(rho[0], 1e-9))) if rho[0] > 0 else 0.5)
            else:
                r0, r1 = rho[j - 1], rho[j]
                corr = float(j + (r0 - 1 / math.e) / (r0 - r1))
    return out | {"sigma": math.sqrt(var), "sigma_unbiased": math.sqrt(unb), "noise_sigma": math.sqrt(noise_var), "noise_method": how,
                  "corr_length": corr, "psd": psd.tolist(), "freq": (k / n).tolist()}  # fmt: skip


# -- line / space ------------------------------------------------------------------------------


@dataclass
class LineResult:
    edges: list[Edge]
    lines: list[dict[str, Any]] = field(default_factory=list)  # per line: left, right edge index, widths...
    spaces: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def lines_spaces(a: np.ndarray, *, feature: str = "bright", method: str = "threshold", level: float = 0.5, smooth: float = 1.0,
                 average: int = 1, mask: np.ndarray | None = None) -> LineResult:  # fmt: skip
    """CD, space, pitch, LER and LWR of a pattern of vertical lines filling ``a``.

    ``feature``: whether the lines are the ``bright`` or the ``dark`` stripes. Values are
    in pixels of ``a`` (x) and lines (y)."""
    from vexel.packs.sem.measure import period

    edges, info = track_edges(a, method=method, level=level, smooth=smooth, average=average, mask=mask)
    res = LineResult(edges=edges, info=info)
    rows = a.shape[0]
    if len(edges) < 2:
        res.warnings.append("找不到成对的边缘：区域内要有完整的线条（两侧边缘都在区域内）")
        return res
    # slanted lines: horizontal distances are longer than the distances across the lines
    across = math.cos(math.atan(info["slant_px_per_line"]))
    if across < 0.9999:
        for e in edges:
            e.position = e.position * across  # positions scaled across the lines (consistent widths, pitch, roughness)
    rise_first = 1 if feature == "bright" else -1  # a bright line starts with a rising edge
    if method == "peak":
        # between two edge peaks lies a line or a space: the line's body is the brighter (or darker) one
        mp = info["mean_profile"]
        starts = [e.start for e in edges]
        inner = []
        for a_, b_ in zip(starts, starts[1:]):
            lo_, hi_ = int(math.ceil(a_ + 2)), int(math.floor(b_ - 2))
            inner.append(float(np.mean(mp[lo_ : hi_ + 1])) if hi_ >= lo_ else float(mp[int(round((a_ + b_) / 2))]))
        cut = (max(inner) + min(inner)) / 2 if inner else 0.0
        for k, v in enumerate(inner):
            is_line = v > cut if feature == "bright" else v < cut
            edges[k].polarity, edges[k + 1].polarity = (rise_first, -rise_first) if is_line else (edges[k].polarity, edges[k + 1].polarity)
    i = 0
    while i < len(edges) - 1:
        e1, e2 = edges[i], edges[i + 1]
        if e1.polarity == rise_first and e2.polarity == -rise_first:
            w = e2.position - e1.position
            res.lines.append({"left": i, "right": i + 1, "width": w, "centre": (e1.position + e2.position) / 2})
            i += 2
        else:
            i += 1
    for a_, b_ in zip(res.lines, res.lines[1:]):
        res.spaces.append({"left": a_["right"], "right": b_["left"], "width": edges[b_["left"]].position - edges[a_["right"]].position})
    if not res.lines:
        res.warnings.append("没有找到完整的线条：检查“线条是亮还是暗”的设置")
        return res
    # per edge and per line roughness
    edge_rough = [roughness(e.position) for e in edges]
    for ln in res.lines:
        ln["cd"] = float(np.nanmean(ln["width"]))
        ln["cd_sd"] = float(np.nanstd(ln["width"]))
        lw = roughness(ln["width"], detrend=False)
        nl, nr = edge_rough[ln["left"]].get("noise_sigma"), edge_rough[ln["right"]].get("noise_sigma")
        if lw["sigma"] is not None and nl is not None and nr is not None:
            unb = max(lw["sigma"] ** 2 - nl**2 - nr**2, 0.0)  # edge noise is independent on the two sides
            lw["sigma_unbiased"], lw["noise_sigma"] = math.sqrt(unb), math.sqrt(nl**2 + nr**2)
        ln["lwr"] = lw
        ln["ler_left"], ln["ler_right"] = edge_rough[ln["left"]], edge_rough[ln["right"]]
        ln["x"] = float(np.nanmean(ln["centre"]))
        ln["missing"] = float(np.mean(~np.isfinite(ln["width"])))
    for sp in res.spaces:
        sp["cd"] = float(np.nanmean(sp["width"]))
    cds = np.array([ln["cd"] for ln in res.lines])
    centres = np.array([ln["x"] for ln in res.lines])
    ler = [r for ln in res.lines for r in (ln["ler_left"], ln["ler_right"]) if r.get("sigma") is not None]
    lwr = [ln["lwr"] for ln in res.lines if ln["lwr"].get("sigma") is not None]

    def mean_of(items: list[dict[str, Any]], key: str) -> float | None:
        v = [it[key] for it in items if it.get(key) is not None]
        return float(np.sqrt(np.mean(np.square(v)))) if v else None  # σ combine as RMS

    per = period(info["mean_profile"])
    res.summary = {
        "lines": len(res.lines), "cd_mean": float(cds.mean()), "cd_sd": float(cds.std(ddof=1)) if len(cds) > 1 else 0.0,
        "space_mean": float(np.mean([s["cd"] for s in res.spaces])) if res.spaces else None,
        "pitch": float(np.mean(np.diff(centres))) if len(centres) > 1 else None,
        "pitch_autocorrelation": per,
        "ler_3sigma": 3 * mean_of(ler, "sigma") if ler else None, "ler_3sigma_unbiased": 3 * mean_of(ler, "sigma_unbiased") if ler else None,
        "lwr_3sigma": 3 * mean_of(lwr, "sigma") if lwr else None, "lwr_3sigma_unbiased": 3 * mean_of(lwr, "sigma_unbiased") if lwr else None,
        "ler_noise_3sigma": 3 * mean_of(ler, "noise_sigma") if ler else None,
        "corr_length_lines": float(np.nanmedian([r["corr_length"] for r in ler if r.get("corr_length") is not None])) if any(r.get("corr_length") for r in ler) else None,
        "sampled_lines": rows, "missing": float(np.mean([ln["missing"] for ln in res.lines])),
        "slant_deg": math.degrees(math.atan(info["slant_px_per_line"])),
    }  # fmt: skip
    if rows < 64:
        res.warnings.append(f"区域只有 {rows} 行：粗糙度至少需要几百行（覆盖若干个相关长度）才可靠")
    if res.summary["missing"] > 0.05:
        res.warnings.append(f"{res.summary['missing']:.0%} 的行上没有找到边缘（饱和、污染或边缘太弱），已跳过")
    cl = res.summary["corr_length_lines"]
    if cl is not None and cl < 2:
        res.warnings.append("相关长度不到 2 行：真实粗糙度的高频部分与噪声难以区分，无偏值可能偏低")
    return res


# -- cross-sections ----------------------------------------------------------------------------


def layers(a: np.ndarray, *, method: str = "gradient", level: float = 0.5, smooth: float = 1.5, average: int = 1,
           mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """Interfaces of a horizontal layer stack (cross-section), followed across every column;
    thickness of each layer between consecutive interfaces. Positions in pixels (y)."""
    edges, info = track_edges(np.asarray(a).T, method=method, level=level, smooth=smooth, average=average,
                              mask=None if mask is None else mask.T, weak=True)  # fmt: skip
    interfaces = []
    for i, e in enumerate(edges):
        r = roughness(e.position)
        interfaces.append({"index": i + 1, "y": e.mean, "polarity": e.polarity, "position": e.position,
                           "roughness_sigma": r.get("sigma"), "roughness_sigma_unbiased": r.get("sigma_unbiased"),
                           "missing": r["missing"]})  # fmt: skip
    thick = []
    for i, (e1, e2) in enumerate(zip(edges, edges[1:])):
        t = e2.position - e1.position
        ok = np.isfinite(t)
        thick.append({"layer": i + 1, "top": e1.mean, "bottom": e2.mean, "thickness": float(np.nanmean(t)) if ok.any() else None,
                      "sd": float(np.nanstd(t)) if ok.any() else None, "min": float(np.nanmin(t)) if ok.any() else None,
                      "max": float(np.nanmax(t)) if ok.any() else None, "series": t})  # fmt: skip
    warnings = []
    if not edges:
        warnings.append("没有找到界面：区域内应包含水平的层（必要时先做倾斜校正和背景校平）")
    if abs(math.degrees(math.atan(info["slant_px_per_line"]))) > 2:
        warnings.append(f"层面倾斜 {math.degrees(math.atan(info['slant_px_per_line'])):.1f}°：厚度按垂直方向测量，倾斜大时偏大")
    return {"interfaces": interfaces, "layers": thick, "warnings": warnings, "slant_deg": math.degrees(math.atan(info["slant_px_per_line"]))}


def feature_profile(a: np.ndarray, *, feature: str = "bright", level: float = 0.5, smooth: float = 1.0, fit_range: tuple[float, float] = (0.1, 0.9),
                    mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """One raised line or trench seen in cross-section, filling ``a``: its height (rows
    where both sidewalls are found), width at the top, middle and bottom, and the angle of
    each sidewall (straight line fitted over ``fit_range`` of the height; 90° = vertical,
    < 90° = the feature narrows upwards).

    ``feature`` ``bright``: a bright feature on a darker background (a line); ``dark``: a
    trench (or a dark line). Rows are image lines from top to bottom."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    a = np.asarray(a, np.float64)
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    if mask is not None:
        sm = np.where(mask, np.nan, sm)
    finite = sm[np.isfinite(sm)]
    t0 = threshold_otsu(finite)
    lo, hi = float(np.mean(finite[finite < t0])), float(np.mean(finite[finite >= t0]))
    lev = lo + level * (hi - lo)
    inside = (sm > lev) if feature == "bright" else (sm < lev)
    rows, cols = a.shape
    left = np.full(rows, np.nan)
    right = np.full(rows, np.nan)
    sgn = 1 if feature == "bright" else -1
    for y in range(rows):
        row = inside[y]
        if not row.any():
            continue
        lab, n = ndi.label(row)
        if n == 0:
            continue
        sizes = ndi.sum(row, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        idx = np.flatnonzero(lab == k)
        x0, x1 = idx[0], idx[-1]
        if x0 == 0 or x1 == cols - 1:
            continue  # the feature runs into the region border on this line: not a full width
        v = (sm[y] - lev) * sgn
        left[y] = x0 - 1 + v[x0 - 1] / (v[x0 - 1] - v[x0]) if v[x0 - 1] != v[x0] else x0  # sub-pixel crossing
        right[y] = x1 + v[x1] / (v[x1] - v[x1 + 1]) if v[x1] != v[x1 + 1] else x1
    ok = np.flatnonzero(np.isfinite(left) & np.isfinite(right))
    out: dict[str, Any] = {"left": left, "right": right, "warnings": []}
    if len(ok) < 5:
        out["warnings"].append("区域内没有找到完整的特征：框选一个线条或沟槽，两侧要留出背景")
        return out
    # the feature's rows: the longest run of consecutive rows with both walls
    runs = np.split(ok, np.flatnonzero(np.diff(ok) > 1) + 1)
    run = max(runs, key=len)
    top, bottom = int(run[0]), int(run[-1])
    height = bottom - top + 1
    width = right - left
    f0, f1 = fit_range
    sel = run[int(round(f0 * (len(run) - 1))) : int(round(f1 * (len(run) - 1))) + 1]
    if len(sel) < 3:
        sel = run

    def wall(xs: np.ndarray, side: str) -> tuple[float, float, float]:
        c = np.polyfit(sel.astype(float), xs[sel], 1)  # x = c0 * y + c1
        # angle between the wall and the horizontal, measured inside the feature at its base
        dxdy = c[0]
        ang = math.degrees(math.atan2(1.0, abs(dxdy)))
        widening_down = (dxdy < 0) if side == "left" else (dxdy > 0)  # base wider than top
        angle = ang if widening_down else 180 - ang
        resid = xs[sel] - np.polyval(c, sel)
        return angle, float(c[0]), float(resid.std())

    la, lslope, lres = wall(left, "left")
    ra, rslope, rres = wall(right, "right")

    def at(frac: float) -> float:  # width at a fraction of the height (mean of 3 lines)
        i = int(round(frac * (len(run) - 1)))
        return float(np.nanmean(width[run[max(0, i - 1) : i + 2]]))

    out.update(top=top, bottom=bottom, height=float(height), width_top=at(0.1), width_middle=at(0.5), width_bottom=at(0.9),
               angle_left=la, angle_right=ra, wall_residual_left=lres, wall_residual_right=rres, fit_range=list(fit_range),
               rows=run.tolist())  # fmt: skip
    if top == 0 or bottom == rows - 1:
        out["warnings"].append("特征碰到区域的上边或下边：高度可能被截断，框选时上下也留出背景")
    return out


__all__ = ["Edge", "LineResult", "feature_profile", "layers", "lines_spaces", "roughness", "track_edges"]
