"""SEM probe (M5): what an image shows and what state it is in - deterministic rules, no
training, offline.

Kinds:

==========  ===================================================================  ============
kind        evidence                                                              recipe
==========  ===================================================================  ============
lines       one direction dominates (edge coherence) and the spectrum has peaks   lines
            on one line through its centre (a 1-D period)
holes       many round objects of one size on a 2-D lattice (spectrum peaks in    holes
            two directions)
particles   separate compact objects, not on a lattice                           particles
layers      horizontal bands across the whole width, a few interfaces, no period  layers
profile     one compact, box-like feature with near-vertical walls, clear of the  profile
            image's sides (a line or trench in cross-section)
tip         one elongated object entering from a border and narrowing to a point  tip
other       none of these (grains, textures, devices...)                          -
==========  ===================================================================  ============

Each kind gets a score in 0..1 with the evidence for it. The probe names a kind only when
its score is high enough and clearly above the next one; otherwise the kind is
``unknown`` and the candidates are listed: it never guesses.

It also measures the image's state (noise, stripes along the scan lines, uneven
brightness, saturated pixels) and suggests processing steps, each with its reason; it
leaves out steps that would harm the measurement (no line levelling across horizontal
structures, no denoising before roughness).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

KINDS = ("lines", "holes", "particles", "layers", "profile", "tip", "other")
KIND_NAMES = {"lines": "线条与间隔", "holes": "接触孔阵列", "particles": "颗粒/缺陷", "layers": "截面层叠", "profile": "截面轮廓（单个线条或沟槽）",
              "tip": "尖端", "other": "其他", "unknown": "无法确定"}  # fmt: skip
RECIPE_OF = {"lines": "lines", "holes": "holes", "particles": "particles", "layers": "layers", "profile": "profile", "tip": "tip"}
MIN_SCORE = 0.55  # a kind needs at least this score...
MIN_MARGIN = 0.15  # ...and this much over the next one
MAX_SIDE = 768  # the probe works on a copy no larger than this


@dataclass
class SemFingerprint:
    kind: str  # one of KINDS, or "unknown"
    confidence: float
    scores: dict[str, float]
    evidence: dict[str, list[str]]  # per kind: why it scored as it did
    shape: tuple[int, int]  # analysed (micrograph) size, pixels
    calibrated: bool = False
    pixel_size: float | None = None
    vendor: str | None = None
    # pattern
    coherence: float = 0.0  # 0 isotropic .. 1 one direction
    orientation: str | None = None  # direction of the structures: "vertical" / "horizontal" / "oblique"
    period_px: float | None = None  # 1-D period (lines), image pixels
    lattice: dict[str, Any] | None = None  # 2-D lattice (holes): pitches, angle
    polarity: str | None = None  # the objects / lines are "bright" or "dark"
    objects: dict[str, Any] = field(default_factory=dict)  # count, size, roundness, box of the main object...
    edge_peaks: bool = False  # lines show bright edge peaks (edge-peak definition suits)
    # state
    noise: float = 0.0  # pixel noise / contrast
    snr: float = 0.0
    stripes: float = 0.0  # scan-line level jumps / noise
    gradient: float = 0.0  # large-scale brightness range / contrast
    saturated: float = 0.0  # fraction of clipped pixels
    suggested_steps: list[dict[str, Any]] = field(default_factory=list)  # {"op", "params", "reason"}
    notes: list[str] = field(default_factory=list)

    @property
    def recipe(self) -> str | None:
        return RECIPE_OF.get(self.kind)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    # recipe memory (``vexel.agent.memory``)
    SCALES = {"log_snr": 0.5, "coherence": 0.15, "log_period": 0.15, "noise": 0.05, "gradient": 0.15, "log_size": 0.3, "log_pixel": 0.3}

    def features(self) -> dict[str, float]:
        f = {"log_snr": math.log10(max(self.snr, 1e-3)), "coherence": self.coherence, "noise": self.noise, "gradient": self.gradient}
        if self.period_px:
            f["log_period"] = math.log10(self.period_px)
        if self.objects.get("diameter"):
            f["log_size"] = math.log10(self.objects["diameter"])
        if self.pixel_size:
            f["log_pixel"] = math.log10(self.pixel_size * 1e9)
        return f

    def meta(self) -> dict[str, Any]:
        return {"module": "sem", "kind": self.kind, "vendor": self.vendor, "calibrated": self.calibrated}


# -- measurements ----------------------------------------------------------------------------------


def _small(a: np.ndarray) -> tuple[np.ndarray, float]:
    f = max(1, int(math.ceil(max(a.shape) / MAX_SIDE)))
    if f == 1:
        return a, 1.0
    from vexel.packs.sem.process import _reduce

    return _reduce(a, f), float(f)


def structure(a: np.ndarray, sigma: float = 3.0) -> tuple[float, float]:
    """Global coherence of the gradient directions (0..1) and the dominant gradient angle
    (degrees, 0 = gradients along x, i.e. vertical structures)."""
    from scipy import ndimage as ndi

    s = ndi.gaussian_filter(a, sigma)
    gy, gx = np.gradient(s)
    jxx, jyy, jxy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    tr = jxx + jyy
    if tr <= 0:
        return 0.0, 0.0
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / tr
    ang = 0.5 * math.degrees(math.atan2(2 * jxy, jxx - jyy))
    return coh, ang


def spectrum_peaks(a: np.ndarray, *, min_excess: float = 12.0, max_peaks: int = 12) -> list[tuple[float, float, float]]:
    """Strong peaks of the power spectrum (half-plane, DC left out): (fx, fy, excess), where
    fx, fy are cycles per pixel and excess is the power over the median power at that radius."""
    from scipy import ndimage as ndi

    r, c = a.shape
    win = np.outer(np.hanning(r), np.hanning(c))
    P = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))) ** 2
    yy, xx = np.indices(P.shape)
    cy, cx = r // 2, c // 2
    fy, fx = (yy - cy) / r, (xx - cx) / c
    rad = np.hypot(fx * max(r, c), fy * max(r, c))  # in bins of the larger side
    rb = np.round(rad).astype(int)
    med = ndi.median(P, rb, index=np.arange(rb.max() + 1))
    ex = P / np.maximum(np.asarray(med)[rb], 1e-30)
    ex[rad < 3] = 0  # DC and the slowest changes
    ex[np.hypot(fx, fy) > 0.25] = 0  # periods under 4 pixels: noise, not structure
    ex[(fy < 0) | ((fy == 0) & (fx < 0))] = 0  # half-plane: a real image's spectrum is symmetric
    _BIN[0] = 1 / min(r, c)
    loc = (ex == ndi.maximum_filter(ex, size=5)) & (ex > min_excess)
    idx = np.argwhere(loc)
    peaks = sorted(((float(fx[i, j]), float(fy[i, j]), float(ex[i, j])) for i, j in idx), key=lambda p: -p[2])
    return peaks[:max_peaks]


_BIN = [1 / 512]  # one frequency bin (cycles per pixel) of the spectrum last taken


def _near(p: tuple[float, float, float], direction: float) -> bool:
    """A peak lies on the line through the centre at ``direction``: within 1.5 frequency bins
    of it (an angle test would fail near the centre, where one bin is many degrees)."""
    th = math.radians(direction)
    return abs(-p[0] * math.sin(th) + p[1] * math.cos(th)) <= 1.5 * _BIN[0] or abs(math.degrees(math.atan2(p[1], p[0])) % 180 - direction) <= 3


def _directions(peaks: list[tuple[float, float, float]]) -> list[float]:
    """Distinct directions (degrees, 0..180) of spectrum peaks, strongest first."""
    dirs: list[float] = []
    for p in peaks:
        ang = math.degrees(math.atan2(p[1], p[0])) % 180
        if not any(_near(p, d) for d in dirs):
            dirs.append(ang)
    return dirs


def _period(peaks: list[tuple[float, float, float]], direction: float) -> float | None:
    """The fundamental period along a spectrum direction: the shortest frequency there."""
    on = [math.hypot(p[0], p[1]) for p in peaks if _near(p, direction)]
    return 1 / min(on) if on else None


def objects(a: np.ndarray, polarity: str | None = None) -> dict[str, Any]:
    """Objects against the background, at the grey level halfway between the two classes:
    the minority class is the objects (their polarity), unless ``polarity`` is given.
    Counts, sizes, shapes, borders; ``valley``: the share of pixels near that grey level
    (small for objects with sharp boundaries, large for textures)."""
    from scipy import ndimage as ndi
    from skimage.measure import regionprops

    from vexel.packs.sem.objects import _mid_threshold

    sm = ndi.gaussian_filter(a, 1.0)
    t, lo, hi = _mid_threshold(sm.ravel())
    bright = sm > t
    polarity = polarity or ("bright" if bright.mean() <= 0.5 else "dark")
    valley = float(np.mean(np.abs(ndi.gaussian_filter(a, 1.5) - t) < 0.2 * (hi - lo)))
    fg = bright if polarity == "bright" else ~bright
    fg = ndi.binary_fill_holes(ndi.binary_opening(fg, iterations=1))
    lab, n = ndi.label(fg)
    rows, cols = a.shape
    props = [p for p in regionprops(lab) if p.area >= 12]
    out: dict[str, Any] = {"polarity": polarity, "fraction": float(fg.mean()), "contrast": float(hi - lo), "threshold": float(t), "count": len(props),
                           "valley": valley}
    if not props:
        return out
    areas = np.array([p.area for p in props], float)
    inner = [p for p in props if p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < rows and p.bbox[3] < cols]
    if len(inner) >= 4:  # regular spacing: nearest-neighbour distances between centres
        from scipy.spatial import cKDTree

        cen = np.array([p.centroid for p in inner])
        nn = cKDTree(cen).query(cen, k=2)[0][:, 1]
        out["nn_distance"] = float(np.median(nn))
        out["nn_cv"] = float(nn.std() / nn.mean())
        out["nn_regular"] = float(np.mean(np.abs(nn / np.median(nn) - 1) <= 0.1))  # share at the typical spacing
    circ = [min(4 * math.pi * p.area / max(p.perimeter, 1) ** 2, 1.0) for p in inner if p.perimeter > 0]
    d = np.array([p.equivalent_diameter_area for p in inner]) if inner else np.zeros(0)
    main = props[int(np.argmax(areas))]
    r0, c0, r1, c1 = main.bbox
    mm = lab == main.label
    contact = {"top": int(mm[0].sum()), "bottom": int(mm[-1].sum()), "left": int(mm[:, 0].sum()), "right": int(mm[:, -1].sum())}
    out["main_base"] = max(contact, key=contact.__getitem__) if any(contact.values()) else None
    sides = {"top": r0 == 0, "bottom": r1 == rows, "left": c0 == 0, "right": c1 == cols}
    out.update(inner=len(inner), diameter=float(np.median(d)) if len(d) else None, size_cv=float(d.std() / d.mean()) if len(d) > 1 else None,
               circularity=float(np.median(circ)) if circ else None, main_share=float(main.area / areas.sum()),
               main_box=[int(c0), int(r0), int(c1), int(r1)], main_sides=[k for k, v in sides.items() if v],
               main_circularity=float(min(4 * math.pi * main.area / max(main.perimeter, 1) ** 2, 1.0)),
               main_extent=float(main.area / max((r1 - r0) * (c1 - c0), 1)), main_solidity=float(main.solidity),
               main_axes=[float(main.axis_major_length), float(max(main.axis_minor_length, 1e-6))])  # fmt: skip
    # a tapering object: widths across its long axis, near each end
    yy, xx = np.nonzero(lab == main.label)
    pts = np.stack([xx, yy], 1).astype(float)
    ctr = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - ctr).T)) if len(pts) > 2 else (np.ones(2), np.eye(2))
    ax = v[:, int(np.argmax(w))]
    t_ = (pts - ctr) @ ax
    lat = (pts - ctr) @ np.array([-ax[1], ax[0]])
    span = float(np.ptp(t_)) if len(t_) else 0.0
    if span > 10:
        def width_at(frac: float) -> float:
            sel = np.abs(t_ - (t_.min() + frac * span)) < max(1.0, 0.03 * span)
            return float(np.ptp(lat[sel])) if sel.sum() > 1 else 0.0

        out["widths_along"] = [width_at(f) for f in (0.08, 0.5, 0.92)]
    return out


def state(a: np.ndarray, raw: np.ndarray | None, orientation: str | None, contrast: float) -> dict[str, float]:
    """Noise, stripes along the scan lines, large-scale unevenness, saturation."""
    from scipy import ndimage as ndi

    from vexel.packs.sem.process import noise_sigma, saturation_mask

    contrast = max(contrast, 1e-12)
    sig = min(noise_sigma(a), noise_sigma(a.T))  # along the direction with fewer edges
    med = np.median(a, axis=1)
    jumps = np.diff(med)
    expected = sig * math.sqrt(2 * math.pi / (2 * a.shape[1])) * 1.25 + 1e-12  # spread of row medians from noise alone
    stripes = float(1.4826 * np.median(np.abs(jumps - np.median(jumps))) / expected)
    big = ndi.gaussian_filter(a, max(a.shape) / 8)
    gradient = float(np.ptp(big[big.shape[0] // 10 : -big.shape[0] // 10 or None, big.shape[1] // 10 : -big.shape[1] // 10 or None]) / contrast)
    sat = 0.0
    if raw is not None:
        m, info = saturation_mask(raw)
        sat = float(info.get("fraction", 0.0))
    return {"noise": sig / contrast, "snr": contrast / max(sig, 1e-12), "stripes": stripes, "gradient": gradient, "saturated": sat}


# -- the probe ------------------------------------------------------------------------------------


def probe(image: np.ndarray, *, raw: np.ndarray | None = None, cal: Any = None) -> SemFingerprint:
    """Recognise ``image`` (the micrograph, without the data bar). ``raw``: the same region of
    the original image (for saturation); ``cal``: its calibration (for scale and vendor)."""
    a0 = np.asarray(image, np.float64)
    a, f = _small(a0)
    lo, hi = np.percentile(a, (1, 99))
    a = (a - lo) / max(hi - lo, 1e-12)
    coh, ang = structure(a)
    peaks = spectrum_peaks(a)
    dirs = _directions(peaks)
    ob = objects(a)
    st = state(a, None if raw is None else np.asarray(raw), None, ob.get("contrast", 1.0))
    # gradient angle 0: gradients along x -> the structures run vertically
    g = ang % 180
    orientation = "vertical" if min(g, 180 - g) < 15 else "horizontal" if abs(g - 90) < 15 else "oblique"
    scores = {k: 0.0 for k in KINDS}
    ev: dict[str, list[str]] = {k: [] for k in KINDS}

    # lines: strong coherence, peaks on one line through the centre, the period across the lines
    period = None
    if dirs:
        period = _period(peaks, dirs[0])
    collinear = len(dirs) == 1 or (len(dirs) >= 2 and sum(p[2] for p in peaks if _on(p, dirs[0])) > 0.9 * sum(p[2] for p in peaks))
    regular, n_edges = _regular_edges(a, orientation)
    if coh > 0.45 and peaks and collinear and period and not regular:
        ev["lines"].append(f"有一个方向的周期成分，但剖面重复得不规则（自相关 {n_edges:.2f}）：不像线条阵列")
    if coh > 0.45 and peaks and collinear and period and regular:
        n_periods = (a.shape[1] if orientation == "vertical" else a.shape[0]) / period
        scores["lines"] = min(1.0, 0.4 + 0.6 * min(coh / 0.7, 1.0)) * (1.0 if n_periods >= 5 else 0.5)  # few periods: could be a layer stack
        ev["lines"].append(f"边缘方向一致性 {coh:.2f}，频谱峰在一条直线上，周期 {period * f:.1f} 像素（约 {n_periods:.0f} 个周期）")
    elif coh > 0.45:
        ev["lines"].append(f"边缘方向一致性 {coh:.2f}，但没有一维周期")

    # objects must stand clearly above the noise
    obj_snr = st["snr"]
    if ob.get("valley", 1.0) >= 0.1:
        for k in ("holes", "particles", "profile", "tip"):
            ev[k].append(f"{ob.get('valley', 1):.0%} 的像素灰度在两类之间：没有清晰边界的目标（更像纹理）")
        ob["inner"], ob["main_share"] = 0, 0.0
    elif obj_snr < 3:
        for k in ("holes", "particles", "profile", "tip"):
            ev[k].append(f"目标与背景的灰度差只有噪声的 {obj_snr:.1f} 倍，分不出目标")
        ob["inner"], ob["main_share"] = 0, 0.0
    # holes: many round objects of one size, evenly spaced (a lattice)
    lattice = None
    if ob.get("inner", 0) >= 4 and ob.get("nn_cv") is not None:
        circ = ob.get("circularity") or 0
        cv = ob.get("size_cv") if ob.get("size_cv") is not None else 1.0
        if ob.get("nn_regular", 0) >= 0.8:
            lattice = {"pitch": ob["nn_distance"] * f, "regular": ob["nn_regular"]}
            if circ > 0.7 and cv < 0.2:
                scores["holes"] = min(1.0, 0.5 + 0.5 * min(ob["inner"] / 12, 1.0)) * (1.0 if ob["fraction"] < 0.5 else 0.6)
                ev["holes"].append(f"{ob['inner']} 个圆形目标（圆度 {circ:.2f}，大小离散 {cv:.0%}）等距排列，间距约 {lattice['pitch']:.1f} 像素")
            else:
                ev["holes"].append(f"目标等距排列，但圆度 {circ:.2f}、大小离散 {cv:.0%}，不像孔阵列")
        else:
            ev["holes"].append(f"只有 {ob.get('nn_regular', 0):.0%} 的目标与最近邻相距典型间距：不是阵列")

    # particles: separate compact objects without a lattice
    n_in = ob.get("inner", 0)
    if n_in >= 3 and (ob.get("circularity") or 0) > 0.45 and ob.get("main_share", 1) < 0.5:
        s = min(1.0, 0.45 + 0.4 * min(n_in / 15, 1.0) + 0.15 * min((ob["circularity"] - 0.45) / 0.4, 1.0))
        if lattice and scores["holes"] > 0:
            s *= 0.6
        if scores["lines"] > 0.5:
            s *= 0.4
        scores["particles"] = s
        ev["particles"].append(f"{n_in} 个独立目标，中位直径 {(ob.get('diameter') or 0) * f:.1f} 像素，圆度 {ob.get('circularity') or 0:.2f}")

    # layers: horizontal bands across the full width, a few interfaces, no period
    if orientation == "horizontal" and coh > 0.35:
        prof = a.mean(axis=1)
        from vexel.packs.sem.measure import profile_edges

        n_if = len(profile_edges(prof, "gradient", smooth=1.5, min_distance=3))
        from scipy import ndimage as ndi

        sm2 = ndi.gaussian_filter(a, 2.0)
        explained = float(np.var(sm2.mean(axis=1)) / max(np.var(sm2), 1e-12))  # bands across the whole width: rows explain the picture
        across = 1 - explained
        if 2 <= n_if <= 12 and explained > 0.8 and scores["lines"] <= 0.5:
            scores["layers"] = min(1.0, 0.55 + 0.45 * min((explained - 0.8) / 0.15, 1.0))
            if regular:  # bands repeating at one period: a periodic multilayer or a few horizontal lines - not decidable
                scores["layers"] = min(scores["layers"], 0.5)
                ev["layers"].append(f"水平条带横贯全图，{n_if} 个界面，但等间距重复：也可能是水平线条")
            else:
                ev["layers"].append(f"水平条带横贯全图，{n_if} 个界面，间距不规则")
        elif 2 <= n_if <= 12:
            ev["layers"].append(f"有水平结构，但{'不横贯全图' if explained <= 0.8 else '是周期性的线条'}")
        if scores["lines"] > 0 and scores["layers"] > 0:
            ev["lines"].append("线条为水平方向：也可能是截面层叠，按周期性区分")

    # one main object: profile (box-like, clear of the sides) or tip (from a border, narrowing);
    # both polarities are looked at (a wide tip can cover more than half the picture)
    single = [ob] if ob.get("main_share", 0) else []
    if ob.get("valley", 1.0) < 0.1 and st["snr"] >= 3:
        other_pol = objects(a, "dark" if ob.get("polarity") == "bright" else "bright")
        single.append(other_pol)
    for o in single:
        share = o.get("main_share", 0)
        if share <= 0.6 or o.get("count", 0) > 4:
            continue
        sides = o.get("main_sides", [])
        widths = o.get("widths_along") or [0, 0, 0]
        major, minor = o.get("main_axes", [1, 1])
        taper = (min(widths[0], widths[2]) / max(widths[0], widths[2], 1e-9)) if max(widths) > 0 else 1.0
        base = o.get("main_base")
        opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(base or "", "")
        if sides and base and opposite not in sides and major / minor > 1.1 and taper < 0.5:
            sc = min(1.0, 0.6 + 0.4 * (0.5 - taper) / 0.5) * (1.0 if len(sides) == 1 else 0.9)
            if sc > scores["tip"]:
                scores["tip"] = sc
                tip_obj = o
                ev["tip"] = [f"一个目标从图像{_side_cn(base)}进入，向另一端收窄（两端宽度比 {taper:.2f}），尖端不碰图像边缘"]
        boxy = (o.get("main_extent", 0) > 0.6 and o.get("main_solidity", 0) > 0.9 and o.get("main_circularity", 1) < 0.86
                and (major / minor >= 1.3 or o.get("main_extent", 0) > 0.75))  # box-like, not a disc
        if set(sides) <= {"bottom"} and boxy:  # a feature may stand on the substrate (the image's bottom)
            sc = min(1.0, 0.6 + 0.4 * (o["main_extent"] - 0.6) / 0.4) * (1.0 if not sides else 0.85)
            if sc > scores["profile"]:
                scores["profile"] = sc
                profile_obj = o
                ev["profile"] = [f"一个近似矩形/梯形的目标（填充 {o['main_extent']:.2f}），" + ("四周留有背景" if not sides else "立在图像底边上")]
    if max(scores.values()) < 0.3:
        scores["other"] = 0.6
        ev["other"].append("不符合任何一类的特征")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (k1, s1), (_, s2) = ranked[0], ranked[1]
    kind = k1 if s1 >= MIN_SCORE and s1 - s2 >= MIN_MARGIN else "unknown"
    fp = SemFingerprint(kind=kind, confidence=float(s1), scores={k: round(v, 3) for k, v in scores.items()}, evidence={k: v for k, v in ev.items() if v},
                        shape=tuple(a0.shape), coherence=float(coh), orientation=orientation, period_px=period * f if period else None,
                        lattice=lattice, polarity=ob.get("polarity"),
                        objects={k: (v * f if k == "diameter" and v else v) for k, v in ob.items() if k not in ("main_box", "widths_along")},
                        noise=st["noise"], snr=st["snr"], stripes=st["stripes"], gradient=st["gradient"], saturated=st["saturated"])  # fmt: skip
    chosen = {"tip": locals().get("tip_obj"), "profile": locals().get("profile_obj")}.get(kind)
    if chosen is not None:  # the object the kind was recognised on (and its polarity)
        fp.polarity = chosen.get("polarity")
        fp.objects = {k: (v * f if k == "diameter" and v else v) for k, v in chosen.items() if k not in ("main_box", "widths_along")}
        ob = chosen
    if ob.get("main_box"):
        fp.objects["main_box"] = [v * f for v in ob["main_box"]]
    if cal is not None:
        fp.calibrated = bool(getattr(cal, "ok", False))
        fp.pixel_size = getattr(cal, "pixel_size", None)
        fp.vendor = getattr(cal, "vendor", None)
    if kind == "lines":
        fp.edge_peaks = _has_edge_peaks(a, orientation, period)
        fp.polarity = _line_polarity(a, orientation)
    if kind == "unknown":
        fp.notes.append("无法确定图像类型：" + "、".join(f"{KIND_NAMES[k]} {v:.2f}" for k, v in ranked[:3] if v > 0) + "。请选择测量方法")
    fp.suggested_steps = suggest_steps(fp)
    return fp


def _regular_edges(a: np.ndarray, orientation: str | None) -> tuple[bool, float]:
    """Whether the profile across the structures repeats regularly (a line array): the
    autocorrelation at its first period. Also returns that value."""
    from vexel.packs.sem.measure import period

    prof = a.mean(1) if orientation == "horizontal" else a.mean(0)
    v = prof - prof.mean()
    per = period(prof)
    if not per or not np.any(v):
        return False, 0.0
    n = len(v)
    ac = np.correlate(v, v, mode="full")[n - 1 :] / (np.dot(v, v) + 1e-30)
    ac_unbiased = ac * n / np.maximum(n - np.arange(n), 1)  # the overlap shrinks with the lag
    i = int(round(per))
    value = float(ac_unbiased[min(i, n - 1)])
    return value >= 0.6 and n / per >= 3, value


def _on(p: tuple[float, float, float], direction: float) -> bool:
    return _near(p, direction)


def _side_cn(side: str) -> str:
    return {"top": "上方", "bottom": "下方", "left": "左侧", "right": "右侧"}[side]


def _has_edge_peaks(a: np.ndarray, orientation: str | None, period: float | None) -> bool:
    """Bright edge peaks: about two maxima per period (one at each edge of a line) instead of
    one (the line's plateau). Counted in bands of rows, so that slanted lines do not smear the
    two peaks of a line into one."""
    from scipy.signal import find_peaks

    if not period:
        return False
    b = a if orientation != "horizontal" else a.T
    band = max(16, min(64, b.shape[0] // 4))
    counts = []
    for y0 in range(0, b.shape[0] - band + 1, band):
        prof = b[y0 : y0 + band].mean(0)
        lo, hi = np.percentile(prof, (5, 95))
        idx, _ = find_peaks(prof, prominence=0.15 * (hi - lo), distance=2)
        counts.append(len(idx) / (len(prof) / period))
    return bool(counts) and float(np.median(counts)) >= 1.6


def _line_polarity(a: np.ndarray, orientation: str | None) -> str:
    """Which stripes are the lines: the narrower ones (lines are usually printed narrower than
    the spaces when the duty is not 1:1; at 1:1, bright)."""
    prof = a.mean(0) if orientation != "horizontal" else a.mean(1)
    t = (np.percentile(prof, 5) + np.percentile(prof, 95)) / 2
    return "bright" if (prof > t).mean() <= 0.55 else "dark"


def suggest_steps(fp: SemFingerprint) -> list[dict[str, Any]]:
    """Processing steps the image's state calls for - left out where they would harm the measurement."""
    out = []
    horizontal_structure = fp.orientation == "horizontal" and fp.kind in ("lines", "layers", "unknown")
    if fp.stripes > 3 and not horizontal_structure:
        out.append({"op": "line_level", "params": {}, "reason": f"扫描行之间的亮度跳变是噪声的 {fp.stripes:.1f} 倍（荷电条纹）"})
    elif fp.stripes > 3:
        fp.notes.append("有扫描行条纹，但图中结构是水平的：逐行校平会压平真实结构，未建议")
    if fp.gradient > 0.25 and fp.kind in ("particles", "holes", "other", "unknown"):
        out.append({"op": "background", "params": {}, "reason": f"大范围亮度不均，达对比度的 {fp.gradient:.0%}"})
    if fp.snr < 6 and fp.kind not in ("lines", "layers"):
        out.append({"op": "denoise", "params": {}, "reason": f"信噪比只有 {fp.snr:.1f}"})
    elif fp.snr < 6:
        fp.notes.append(f"信噪比只有 {fp.snr:.1f}，但去噪会改变边缘粗糙度：未建议，粗糙度的噪声部分会被单独扣除")
    if fp.saturated > 0.001:
        fp.notes.append(f"有 {fp.saturated:.2%} 的像素饱和：已标记，测量不使用它们")
    return out


__all__ = ["KINDS", "KIND_NAMES", "SemFingerprint", "objects", "probe", "spectrum_peaks", "structure", "suggest_steps"]
