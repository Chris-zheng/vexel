"""Objects on SEM images (M4): contact holes and vias in arrays, and particles (defects,
powders).

**Holes**: separated from the surface by a grey-level threshold; then, as in CD-SEM, the
edge is found on profiles cast outward from the hole's centre in every direction:
``threshold`` - where the profile crosses ``level`` between its minimum (the hole) and
its maximum (the bright rim, or the surface when there is none); ``gradient`` - the
steepest point. The edge points are fitted with a circle and an ellipse. In an array,
the lattice is found from the hole centres, and lattice sites without a hole are counted
as missing; merged holes (bridges) are flagged.

**Particles**: threshold halfway between particle and background grey (iterated from
Otsu), touching particles separated by a distance-transform watershed. Sizes: equivalent
circle diameter, Feret diameters, circularity, aspect ratio; distribution by number and
by volume (d10, d50, d90). Following ISO 13322-1, particles cut by the region's border
are left out of the statistics and counted separately.

Positions and sizes are in pixels of the analysed region.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _mid_threshold(v: np.ndarray) -> tuple[float, float, float]:
    """Threshold halfway between the two grey classes (Otsu, then iterated means)."""
    from skimage.filters import threshold_otsu

    v = np.asarray(v, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or not np.ptp(v) > 0:
        raise ValueError("the region has no contrast: nothing to measure")
    t = float(threshold_otsu(v))
    for _ in range(20):
        lo, hi = float(v[v < t].mean()), float(v[v >= t].mean())
        nt = (lo + hi) / 2
        if abs(nt - t) < 1e-6 * max(abs(hi - lo), 1e-12):
            break
        t = nt
    return t, lo, hi


def _polygon(points: np.ndarray) -> tuple[float, float]:
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.hypot(*np.diff(np.vstack([points, points[:1]]), axis=0).T).sum())
    return float(area), perim


def feret(points: np.ndarray) -> tuple[float, float, float]:
    """Maximum and minimum Feret diameters of an outline ([x, y] points), and the angle
    (degrees, counter-clockwise on screen) of the maximum."""
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return 0.0, 0.0, 0.0
    try:
        hull = points[ConvexHull(points).vertices]
    except Exception:  # noqa: BLE001 - degenerate (collinear) outline
        hull = points
    ang = np.radians(np.arange(0, 180, 0.5))
    d = np.stack([np.cos(ang), -np.sin(ang)], 1)  # screen y points down
    proj = hull @ d.T
    widths = proj.max(0) - proj.min(0)
    i = int(np.argmax(widths))
    return float(widths.max()), float(widths.min()), float(np.degrees(ang[i]))


def _outline(sm: np.ndarray, region: np.ndarray, level: float, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Sub-pixel outline of ``region`` at grey ``level`` (``[x, y]`` in ``sm`` coordinates)."""
    from skimage.measure import find_contours

    r0, c0, r1, c1 = bbox
    pad = 3
    R0, C0 = max(r0 - pad, 0), max(c0 - pad, 0)
    R1, C1 = min(r1 + pad, sm.shape[0]), min(c1 + pad, sm.shape[1])
    crop = sm[R0:R1, C0:C1]
    reg = region[R0:R1, C0:C1]
    inside_val = float(np.median(crop[reg]))
    sign = 1 if inside_val > level else -1
    contours = find_contours(sign * (crop - level), 0.0)
    if not contours:
        return None
    from skimage.measure import points_in_poly

    cy, cx = np.argwhere(reg).mean(0)
    best = None
    for c in contours:
        if len(c) < 6 or not np.allclose(c[0], c[-1]):
            continue
        if points_in_poly([[cy, cx]], c)[0]:
            if best is None or len(c) > len(best):
                best = c
    if best is None:
        return None
    return np.stack([best[:, 1] + C0, best[:, 0] + R0], 1)


def radial_edges(sm: np.ndarray, centre: tuple[float, float], r0: float, *, feature: str = "dark", method: str = "threshold",
                 level: float = 0.5, rays: int | None = None) -> np.ndarray:  # fmt: skip
    """Edge points ([x, y]) of a round object around ``centre`` (radius about ``r0``) on
    rays cast outward; rays without an edge are left out."""
    from scipy import ndimage as ndi

    n = rays or max(64, int(2 * math.pi * r0))
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    step = 0.25
    t = np.arange(0.0, 1.6 * r0 + 4, step)
    xs = centre[0] + np.cos(th)[:, None] * t[None, :]
    ys = centre[1] + np.sin(th)[:, None] * t[None, :]
    prof = ndi.map_coordinates(sm, [ys, xs], order=1, mode="nearest")
    if feature == "bright":
        prof = -prof
    inner = t <= 0.8 * r0
    outer = t >= 0.6 * r0
    jmin = np.argmin(np.where(inner[None, :], prof, np.inf), axis=1)
    lo = prof[np.arange(n), jmin]
    hi = np.max(np.where(outer[None, :], prof, -np.inf), axis=1)
    jmax = np.argmax(np.where(outer[None, :], prof, -np.inf), axis=1)
    pts = []
    for k in range(n):
        seg = prof[k, jmin[k] : jmax[k] + 1]
        if len(seg) < 3 or hi[k] - lo[k] <= 1e-12:
            continue
        if method == "gradient":
            g = np.diff(seg)
            j = int(np.argmax(g))
            if 0 < j < len(g) - 1:
                y0, y1, y2 = g[j - 1], g[j], g[j + 1]
                den = y0 - 2 * y1 + y2
                j = j + (0.5 * (y0 - y2) / den if den else 0.0)
            r = t[jmin[k]] + (j + 0.5) * step
        else:
            lev = lo[k] + level * (hi[k] - lo[k])
            above = np.flatnonzero(seg >= lev)
            if not len(above) or above[0] == 0:
                continue
            j = above[0]
            r = t[jmin[k]] + (j - 1 + (lev - seg[j - 1]) / (seg[j] - seg[j - 1])) * step
        pts.append((centre[0] + r * math.cos(th[k]), centre[1] + r * math.sin(th[k])))
    return np.array(pts) if pts else np.zeros((0, 2))


def holes(a: np.ndarray, *, feature: str = "dark", method: str = "threshold", level: float = 0.5, smooth: float = 1.0,
          min_diameter: float = 4.0, mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """Contact holes (``dark`` on a brighter surface, or ``bright``): one row per hole
    (centre, circle-fit diameter and residual, ellipse axes and orientation, area-equivalent
    diameter, roundness), the array's lattice and missing sites."""
    from scipy import ndimage as ndi
    from skimage.measure import regionprops

    from vexel.packs.sem.measure import fit_circle, fit_ellipse

    a = np.asarray(a, np.float64)
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    t, lo, hi = _mid_threshold(sm[~mask] if mask is not None and mask.any() else sm.ravel())
    inside = sm < t if feature == "dark" else sm > t
    inside = ndi.binary_opening(inside, iterations=1)
    lab, n = ndi.label(ndi.binary_fill_holes(inside))
    min_area = math.pi * (min_diameter / 2) ** 2
    rows_out, edge_count, warnings = [], 0, []
    props = [p for p in regionprops(lab, intensity_image=a) if p.area >= min_area]
    areas = np.array([p.area for p in props]) if props else np.zeros(0)
    typical = float(np.median(areas)) if len(areas) else 0.0
    for p in props:
        r0, c0, r1, c1 = p.bbox
        if r0 == 0 or c0 == 0 or r1 == a.shape[0] or c1 == a.shape[1]:
            edge_count += 1
            continue
        region = lab == p.label
        ring = ndi.binary_dilation(region, iterations=max(2, int(p.equivalent_diameter_area / 4))) & ~ndi.binary_dilation(region, iterations=2)
        ring &= lab == 0
        if mask is not None:
            if (mask & ndi.binary_dilation(region, iterations=2)).any():
                warnings.append(f"孔 ({p.centroid[1]:.0f}, {p.centroid[0]:.0f}) 接触饱和区，已跳过")
                continue
        in_v = float(np.median(sm[ndi.binary_erosion(region, iterations=1)] if ndi.binary_erosion(region, iterations=1).any() else sm[region]))
        out_v = float(np.median(sm[ring])) if ring.any() else (hi if feature == "dark" else lo)
        lev = in_v + 0.5 * (out_v - in_v)
        rough = _outline(sm, region, lev, p.bbox)  # first estimate of centre and radius
        if rough is None or len(rough) < 8:
            continue
        c0x, c0y, r0, _ = fit_circle(rough)
        outline = radial_edges(sm, (c0x, c0y), r0, feature=feature, method=method, level=level)
        if len(outline) < 8:
            continue
        cx, cy, r, rms = fit_circle(outline)
        el = fit_ellipse(outline)
        area, perim = _polygon(outline)
        fmax, fmin, _ = feret(outline)
        row = {"x": cx, "y": cy, "diameter": 2 * r, "fit_residual": rms, "eq_diameter": 2 * math.sqrt(area / math.pi), "area": area,
               "perimeter": perim, "feret_max": fmax, "feret_min": fmin, "merged": bool(typical and p.area > 1.8 * typical),
               "outline": outline}  # fmt: skip
        if el is not None:
            row.update(major=el[2], minor=el[3], orientation=el[4], roundness=el[3] / el[2] if el[2] else None)
        rows_out.append(row)
    good = [h for h in rows_out if not h["merged"]]
    d = np.array([h["diameter"] for h in good])
    summary: dict[str, Any] = {"holes": len(rows_out), "merged": sum(h["merged"] for h in rows_out), "at_border": edge_count,
                               "diameter_mean": float(d.mean()) if len(d) else None,
                               "diameter_3sigma": float(3 * d.std(ddof=1)) if len(d) > 1 else None,
                               "roundness_mean": float(np.mean([h["roundness"] for h in good if h.get("roundness")])) if good else None}  # fmt: skip
    lat = lattice(np.array([[h["x"], h["y"]] for h in good]), a.shape) if len(good) >= 4 else None
    if lat is not None:
        summary.update(pitch_a=lat["pitch_a"], pitch_b=lat["pitch_b"], lattice_angle=lat["angle"], missing=len(lat["missing"]))
    if any(h["merged"] for h in rows_out):
        warnings.append(f"{summary['merged']} 个孔面积超过典型值的 1.8 倍（可能相连），不计入孔径统计")
    return {"holes": rows_out, "summary": summary, "lattice": lat, "warnings": warnings, "threshold": t}


def lattice(centres: np.ndarray, shape: tuple[int, int], tolerance: float = 0.3) -> dict[str, Any] | None:
    """The 2-D lattice through hole centres: basis vectors a, b (pitch, angle), every
    lattice site inside the region (half a pitch from the border), and sites without a
    hole within ``tolerance`` × pitch (missing)."""
    from scipy.spatial import cKDTree

    if len(centres) < 4:
        return None
    tree = cKDTree(centres)
    d, idx = tree.query(centres, k=min(7, len(centres)))
    vecs = np.concatenate([centres[idx[:, j]] - centres for j in range(1, idx.shape[1])])
    lens = np.hypot(*vecs.T)
    shortest = np.median(d[:, 1])
    vecs = vecs[lens < 1.6 * shortest]
    if len(vecs) < 2:
        return None
    vecs = np.where((vecs[:, :1] < 0) | ((vecs[:, :1] == 0) & (vecs[:, 1:] < 0)), -vecs, vecs)  # fold to a half-plane
    ang = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    hist, edges = np.histogram(ang, bins=36, range=(-90, 90))
    k = int(np.argmax(hist))
    sel = (ang >= edges[k] - 5) & (ang <= edges[k + 1] + 5)
    a = np.median(vecs[sel], axis=0)
    u = vecs / np.hypot(*vecs.T)[:, None]
    ua = a / np.hypot(*a)
    rest = vecs[np.abs(u[:, 0] * ua[1] - u[:, 1] * ua[0]) > 0.5]  # not parallel to a
    if not len(rest):
        return None
    ang2 = np.degrees(np.arctan2(rest[:, 1], rest[:, 0]))
    h2, e2 = np.histogram(ang2, bins=36, range=(-90, 90))
    k2 = int(np.argmax(h2))
    b = np.median(rest[(ang2 >= e2[k2] - 5) & (ang2 <= e2[k2 + 1] + 5)], axis=0)
    # refine: integer indices of every hole, least squares for origin, a, b
    B = np.stack([a, b], 1)
    o = centres[np.argmin(np.hypot(*(centres - centres.mean(0)).T))]
    for _ in range(3):
        ij = np.round(np.linalg.solve(B, (centres - o).T).T)
        A = np.column_stack([np.ones(len(ij)), ij])
        sol, *_ = np.linalg.lstsq(A, centres, rcond=None)
        o, a, b = sol[0], sol[1], sol[2]
        B = np.stack([a, b], 1)
    pa, pb = float(np.hypot(*a)), float(np.hypot(*b))
    rows, cols = shape
    corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], float)
    ijc = np.linalg.solve(B, (corners - o).T).T
    lo, hi = np.floor(ijc.min(0)) - 1, np.ceil(ijc.max(0)) + 1
    gi, gj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    sites = o + gi.reshape(-1, 1) * a + gj.reshape(-1, 1) * b
    margin = 0.5 * min(pa, pb)
    inside = (sites[:, 0] >= margin) & (sites[:, 0] <= cols - margin) & (sites[:, 1] >= margin) & (sites[:, 1] <= rows - margin)
    sites = sites[inside]
    dist, _ = tree.query(sites)
    missing = sites[dist > tolerance * min(pa, pb)]
    angle = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (pa * pb), -1, 1))))
    return {"pitch_a": pa, "pitch_b": pb, "angle": angle, "a": a.tolist(), "b": b.tolist(), "origin": o.tolist(),
            "sites": len(sites), "missing": missing.tolist()}  # fmt: skip


def particles(a: np.ndarray, *, feature: str = "bright", threshold: str = "mid", split: bool = True, min_diameter: float = 3.0,
              smooth: float = 0.5, mask: np.ndarray | None = None, fill: bool = True, solidity_min: float = 0.85,
              outlier_k: float = 4.0) -> dict[str, Any]:  # fmt: skip
    """Particles: one row per particle (equivalent diameter, Feret max/min, circularity,
    aspect ratio, solidity, convexity, mean grey level, centre, whether it touches others,
    a class) and the size distribution.

    Classes are rules that can be checked, not a trained model:
    ``agglomerate`` touches other particles (one connected blob split by the watershed);
    ``irregular`` has a solidity (area / convex-hull area) below ``solidity_min``: broken,
    angular or fused particles; ``outlier`` differs from the population by more than
    ``outlier_k`` robust standard deviations in log size or in grey level: possibly foreign
    material; ``normal`` otherwise. ``fill`` fills holes inside objects (off for pores)."""
    from scipy import ndimage as ndi
    from skimage.measure import find_contours, regionprops
    from skimage.segmentation import watershed

    a = np.asarray(a, np.float64)
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    vals = sm[~mask] if mask is not None and mask.any() else sm.ravel()
    if threshold == "local":
        from skimage.filters import threshold_local

        t_img = threshold_local(sm, block_size=2 * (max(a.shape) // 16) + 1, offset=0)
        fg = sm > t_img if feature == "bright" else sm < t_img
        t = float(np.median(t_img))
    else:
        t, _, _ = _mid_threshold(vals)
        fg = sm > t if feature == "bright" else sm < t
        # halfway between the plateaus: the particles' cores and the background away from them
        # (class means are pulled towards each other by the blurred edge pixels)
        core = ndi.binary_erosion(fg, iterations=2)
        far = ~ndi.binary_dilation(fg, iterations=3)
        if mask is not None:
            core &= ~mask
            far &= ~mask
        if core.sum() > 20 and far.sum() > 20:
            t = float(np.median(sm[core]) + np.median(sm[far])) / 2
            fg = sm > t if feature == "bright" else sm < t
    if mask is not None:
        fg &= ~mask
    fg = ndi.binary_opening(fg, iterations=1)
    if fill:
        fg = ndi.binary_fill_holes(fg)
    comp, ncomp = ndi.label(fg)
    if split and ncomp:
        from skimage.morphology import h_maxima

        dist = ndi.distance_transform_edt(fg)
        peaks = h_maxima(ndi.gaussian_filter(dist, 1.0), max(1.0, min_diameter / 4))
        markers, _ = ndi.label(peaks)
        lab = watershed(-dist, markers, mask=fg)
        lab[(lab == 0) & fg] = 0
        lab = _merge_oversplit(lab)
    else:
        lab = comp
    min_area = math.pi * (min_diameter / 2) ** 2
    per_comp: dict[int, int] = {}
    props = [p for p in regionprops(lab, intensity_image=a) if p.area >= min_area]
    for p in props:
        c = int(comp[tuple(p.coords[0])])
        per_comp[c] = per_comp.get(c, 0) + 1
    rows_out, at_border = [], 0
    for p in props:
        r0, c0, r1, c1 = p.bbox
        border = r0 == 0 or c0 == 0 or r1 == a.shape[0] or c1 == a.shape[1]
        if border:
            at_border += 1
            continue
        region = np.pad(lab[r0:r1, c0:c1] == p.label, 1)
        cs = find_contours(region.astype(float), 0.5)
        outline = max(cs, key=len)[:, ::-1] + np.array([c0 - 1, r0 - 1]) if cs else np.zeros((0, 2))
        _, perim = _polygon(outline) if len(outline) > 2 else (0.0, 0.0)
        area = float(p.area)
        fmax, fmin, fang = feret(outline) if len(outline) > 2 else (0.0, 0.0, 0.0)
        touching = per_comp.get(int(comp[tuple(p.coords[0])]), 1) > 1
        solidity, convexity = _hull(outline, area, perim)
        rows_out.append({"x": float(p.centroid[1]), "y": float(p.centroid[0]), "area": area, "eq_diameter": 2 * math.sqrt(area / math.pi),
                         "feret_max": fmax, "feret_min": fmin, "feret_angle": fang, "perimeter": perim,
                         "circularity": float(min(4 * math.pi * area / perim**2, 1.0)) if perim else None,
                         "aspect_ratio": fmin / fmax if fmax else None, "solidity": solidity, "convexity": convexity,
                         "grey": float(p.intensity_mean), "touching": touching, "outline": outline})  # fmt: skip
    _classify(rows_out, solidity_min, outlier_k)
    d = np.array([r["eq_diameter"] for r in rows_out])
    summary: dict[str, Any] = {"particles": len(rows_out), "at_border": at_border}
    if len(d):
        summary.update(diameter_mean=float(d.mean()), diameter_sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0, **distribution(d),
                       agglomerated=float(np.mean([r["touching"] for r in rows_out])),
                       area_fraction=float(fg.sum() / fg.size))  # fmt: skip
        for c in CLASSES:
            summary[f"class_{c}"] = sum(r["class"] == c for r in rows_out)
    warnings = []
    if len(d) and np.median(d) < 10:
        warnings.append(f"颗粒中位直径只有 {np.median(d):.1f} 像素：尺寸误差相对较大，建议提高放大倍数（每个颗粒至少 10 像素宽）")
    if at_border:
        warnings.append(f"{at_border} 个颗粒被区域边界截断，按 ISO 13322-1 不计入统计")
    return {"particles": rows_out, "summary": summary, "warnings": warnings, "threshold": t, "labels": lab}


def _merge_oversplit(lab: np.ndarray, neck: float = 1.0, solidity: float = 0.93) -> np.ndarray:
    """Undo watershed splits that are not between particles. Two touching particles meet at a
    neck narrower than either of them and together are clearly not convex; a noisy outline
    can give one round particle two distance maxima, and its pieces then share a boundary
    wider than the smaller piece and together are convex. Pairs with a shared boundary over
    ``neck`` × the smaller piece's equivalent diameter and a union solidity over
    ``solidity`` are merged (on synthetic touching discs the boundary ratio stays below 0.8
    and the union solidity below 0.9)."""
    from skimage.measure import regionprops

    pairs = []
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.column_stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])]))
    pp = np.concatenate(pairs)
    if not len(pp):
        return lab
    uniq, counts = np.unique(pp, axis=0, return_counts=True)
    area = np.bincount(lab.ravel())
    boxes = {p.label: p.bbox for p in regionprops(lab)}
    parent = np.arange(area.size)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), n in zip(uniq, counts):
        if n <= neck * 2 * math.sqrt(min(area[a], area[b]) / math.pi):
            continue
        r0, c0 = min(boxes[a][0], boxes[b][0]), min(boxes[a][1], boxes[b][1])
        r1, c1 = max(boxes[a][2], boxes[b][2]), max(boxes[a][3], boxes[b][3])
        crop = lab[r0:r1, c0:c1]
        if regionprops(((crop == a) | (crop == b)).astype(np.uint8))[0].solidity > solidity:
            parent[find(b)] = find(a)
    roots = np.array([find(i) for i in range(area.size)])
    return roots[lab]


CLASSES = ("normal", "agglomerate", "irregular", "outlier")


def _hull(outline: np.ndarray, area: float, perim: float) -> tuple[float | None, float | None]:
    """Solidity (area / convex-hull area) and convexity (hull perimeter / perimeter), both 0..1."""
    if len(outline) < 4 or not perim:
        return None, None
    from scipy.spatial import ConvexHull, QhullError

    try:
        h = ConvexHull(outline)
    except QhullError:
        return None, None
    # the outline runs through pixel edges (0.5 contour): its own polygon area is the fair comparison
    poly_area, _ = _polygon(outline)
    return float(min(poly_area / h.volume, 1.0)) if h.volume else None, float(min(h.area / perim, 1.0))


def _robust_z(v: np.ndarray) -> np.ndarray:
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return (v - med) / mad if mad > 0 else np.zeros_like(v)


def _classify(rows: list[dict[str, Any]], solidity_min: float, outlier_k: float) -> None:
    """Rule-based class of each particle (see ``particles``); also the reason, for the table."""
    if not rows:
        return
    zs = _robust_z(np.log([r["eq_diameter"] for r in rows])) if len(rows) >= 8 else np.zeros(len(rows))
    zg = _robust_z(np.array([r["grey"] for r in rows])) if len(rows) >= 8 else np.zeros(len(rows))
    for r, a, b in zip(rows, zs, zg):
        if r["touching"]:
            r["class"], r["reason"] = "agglomerate", "touching"
        elif r["solidity"] is not None and r["solidity"] < solidity_min:
            r["class"], r["reason"] = "irregular", f"solidity {r['solidity']:.2f}"
        elif abs(a) > outlier_k or abs(b) > outlier_k:
            r["class"], r["reason"] = "outlier", f"size z {a:+.1f}" if abs(a) >= abs(b) else f"grey z {b:+.1f}"
        else:
            r["class"], r["reason"] = "normal", ""


def pores(a: np.ndarray, *, feature: str = "dark", threshold: str = "mid", split: bool = False, min_diameter: float = 3.0,
          smooth: float = 0.5, mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """Pores: porosity (the pore phase's share of the region, pores cut by the region's edge
    included) and one row per whole pore (equivalent diameter, Feret, circularity, aspect
    ratio); pores cut by the edge are left out of the size statistics only."""
    res = particles(a, feature=feature, threshold=threshold, split=split, min_diameter=min_diameter, smooth=smooth, mask=mask, fill=False,
                    solidity_min=0.0, outlier_k=1e9)  # fmt: skip
    from scipy import ndimage as ndi

    sm = ndi.gaussian_filter(np.asarray(a, np.float64), smooth) if smooth else np.asarray(a, np.float64)
    t = res["threshold"]
    phase = (sm < t) if feature == "dark" else (sm > t)
    if threshold == "local":
        phase = res["labels"] > 0
    valid = np.ones_like(phase) if mask is None else ~mask
    porosity = float((phase & valid).sum() / max(valid.sum(), 1))
    for r in res["particles"]:
        r.pop("class", None)
        r.pop("reason", None)
    sm_ = res["summary"]
    sm_ = {k: v for k, v in sm_.items() if not k.startswith("class_") and k not in ("agglomerated", "area_fraction")}
    sm_["porosity"] = porosity
    warnings = [w.replace("颗粒", "孔") for w in res["warnings"]]
    if porosity > 0.6:
        warnings.append(f"孔隙率 {porosity:.0%}：暗相占了大部分区域，检查“孔是”的亮暗设置，或阈值是否把阴影算成了孔")
    return {"pores": res["particles"], "summary": sm_, "warnings": warnings, "threshold": t, "phase": phase}


def _mask_outline(region: np.ndarray, step: int = 1) -> np.ndarray | None:
    """Outline of a binary region (``[x, y]``, the 0.5 contour of its mask)."""
    from skimage.measure import find_contours

    padded = np.pad(region.astype(float), 1)
    cs = find_contours(padded, 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    return np.stack([c[:, 1] - 1, c[:, 0] - 1], 1)[::step]


def nested(a: np.ndarray, *, parent: str = "membrane", child: str = "dark", membrane: float = 3.0, smooth: float = 1.0,
           min_diameter: float = 12.0, child_min: float = 0.005, child_max: float = 0.8, max_children: int = 1,
           mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """Regions inside regions - cells and their nuclei, grains and their cores: each parent
    region, the child regions inside it, their areas and the ratio.

    Parents (``parent``): ``membrane`` - regions enclosed by dark boundary lines about
    ``membrane`` px wide (tissue, where neighbouring cells share a membrane, or cells with a
    dark outline on a slide): the lines are found as dark features narrower than the
    regions (black top-hat), the enclosed areas are seeds, and each region grows to the
    middle of its boundary line (watershed on the line strength); ``bright`` or ``dark`` -
    objects brighter or darker than the background (threshold halfway between the two
    greys). Parents cut by the region's border are left out and counted.

    Children (``child`` ``dark`` or ``bright``): inside each parent, away from its boundary
    line, the pixels darker (brighter) than halfway between the parent's two greys; the
    largest ``max_children`` pieces, holes filled. A parent where the two greys differ by
    less than twice the pixel noise, or where the child would cover less than
    ``child_min`` or more than ``child_max`` of the parent, has no child.

    Areas are pixel counts (the boundary of a parent is the middle of its line; a child's at
    half its contrast). Ratios: child / parent area, and child / (parent - child) - for a
    cell, nucleus to cytoplasm (N/C)."""
    from scipy import ndimage as ndi
    from skimage.morphology import disk
    from skimage.segmentation import watershed

    a = np.asarray(a, np.float64)
    rows, cols = a.shape
    d = np.diff(a, axis=1)
    noise = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2)
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    min_area = math.pi * (min_diameter / 2) ** 2
    warnings: list[str] = []
    if parent == "membrane":
        r = max(2, int(math.ceil(1.5 * membrane)))
        line = ndi.grey_closing(sm, footprint=disk(r)) - sm  # dark lines narrower than 2r+1
        t, lo_l, _ = _mid_threshold(line if mask is None else line[~mask])
        # boundary lines are long: faint stretches are kept where they connect to strong ones
        # (hysteresis), and short dark pieces - specks and texture inside the regions - are dropped
        from skimage.filters import apply_hysteresis_threshold

        lines = apply_hysteresis_threshold(line, lo_l + 0.5 * (t - lo_l), t)
        ll, nl = ndi.label(lines, structure=np.ones((3, 3)))
        if nl:
            ext = np.zeros(nl + 1)
            for j, sl in enumerate(ndi.find_objects(ll), start=1):
                if sl is not None:
                    ext[j] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            short = ext < min_diameter / 2
            short[0] = False
            lines &= ~short[ll]
        lab, n = ndi.label(~lines)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = sizes < min_area / 4
        small[0] = True
        lab[small[lab]] = 0
        # each region grows to the middle of its boundary line: the ridge of the line's curvature (the
        # Hessian's larger eigenvalue at the line's scale), which a brighter side shifts less than
        # the top-hat's peak
        from skimage.feature import hessian_matrix, hessian_matrix_eigvals

        ridge = hessian_matrix_eigvals(hessian_matrix(a, sigma=max(1.0, membrane / 2), order="rc", use_gaussian_derivatives=True))[0]
        lab = watershed(ridge, lab)
    elif parent in ("bright", "dark"):
        t, lo, hi = _mid_threshold(sm if mask is None else sm[~mask])
        fg = sm > t if parent == "bright" else sm < t
        fg = ndi.binary_fill_holes(fg)
        lab, n = ndi.label(fg)
        # the global threshold sits between the background and the average of everything inside (a
        # dark nucleus pulls it into the cell): each object's edge is re-found halfway between the
        # background and the grey just inside its edge
        g_bg = lo if parent == "bright" else hi
        bg_pix = ~ndi.binary_dilation(fg, iterations=3)
        if bg_pix.any():
            g_bg = float(np.median(sm[bg_pix]))
        refined = np.zeros_like(lab)
        for k, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            R0, C0 = max(sl[0].start - 4, 0), max(sl[1].start - 4, 0)
            R1, C1 = min(sl[0].stop + 4, rows), min(sl[1].stop + 4, cols)
            reg = lab[R0:R1, C0:C1] == k
            ring = reg & ~ndi.binary_erosion(reg, iterations=4) & ndi.binary_erosion(reg, iterations=1)
            if ring.sum() < 8:
                refined[R0:R1, C0:C1][reg] = k
                continue
            level = (g_bg + float(np.median(sm[R0:R1, C0:C1][ring]))) / 2
            loc = sm[R0:R1, C0:C1]
            new_reg = (loc > level) if parent == "bright" else (loc < level)
            new_reg &= ndi.binary_dilation(reg, iterations=3)
            nl, _ = ndi.label(new_reg)
            ids = np.unique(nl[reg & (nl > 0)])
            new_reg = ndi.binary_fill_holes(np.isin(nl, ids)) if ids.size else reg
            tgt = refined[R0:R1, C0:C1]
            tgt[new_reg & (tgt == 0)] = k
        lab = refined
    else:
        raise ValueError("parent must be 'membrane', 'bright' or 'dark'")
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    if mask is not None and mask.any():
        border = np.union1d(border, np.unique(lab[ndi.binary_dilation(mask, iterations=2)]))
    objs = ndi.find_objects(lab)
    sizes = np.bincount(lab.ravel(), minlength=len(objs) + 1)
    at_border = 0
    out = []
    erode = int(math.ceil(membrane)) + 1 if parent == "membrane" else 2
    for k, sl in enumerate(objs, start=1):
        if sl is None or sizes[k] < min_area:
            continue
        if k in border:
            at_border += 1
            continue
        pad = 2
        R0, C0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        R1, C1 = min(sl[0].stop + pad, rows), min(sl[1].stop + pad, cols)
        reg = lab[R0:R1, C0:C1] == k
        if parent == "membrane" and sizes[k] > 0.5 * rows * cols:
            continue  # the background around separate cells
        area = float(reg.sum())
        core = ndi.binary_erosion(reg, iterations=erode)
        loc = sm[R0:R1, C0:C1]
        row: dict[str, Any] = {"area": area, "x": float(np.argwhere(reg)[:, 1].mean() + C0), "y": float(np.argwhere(reg)[:, 0].mean() + R0),
                               "eq_diameter": 2 * math.sqrt(area / math.pi), "child_area": None, "ratio": None, "nc": None,
                               "child_x": None, "child_y": None, "child_count": 0, "grey": float(loc[core].mean()) if core.any() else None,
                               "child_grey": None, "outline": None, "child_outline": None, "reason": ""}  # fmt: skip
        o = _mask_outline(reg)
        if o is not None:
            row["outline"] = (o + [C0, R0]).tolist()
        vals = loc[core]
        if vals.size < 16:
            row["reason"] = "区域太小，内部无法判断"
            out.append(row)
            continue
        try:
            tc, lo, hi = _mid_threshold(vals)
        except ValueError:
            tc, lo, hi = 0.0, 0.0, 0.0
        if hi - lo < 2 * noise:
            row["reason"] = "内部灰度均匀：没有可分辨的子区域"
            out.append(row)
            continue
        sel = core & ((loc < tc) if child == "dark" else (loc > tc))
        cl, cn = ndi.label(sel)
        if cn == 0:
            row["reason"] = "没有子区域"
            out.append(row)
            continue
        cs = np.bincount(cl.ravel(), minlength=cn + 1)
        cs[0] = 0
        keep = np.argsort(cs)[::-1][: max(1, max_children)]
        keep = [j for j in keep if cs[j] > 0]
        ch = ndi.binary_fill_holes(np.isin(cl, keep)) & reg
        frac = float(ch.sum()) / area
        if not (child_min <= frac <= child_max):
            row["reason"] = f"子区域占比 {frac:.1%} 超出 {child_min:.1%}–{child_max:.0%}：不计"
            out.append(row)
            continue
        ca = float(ch.sum())
        cyx = np.argwhere(ch).mean(0)
        row.update(child_area=ca, ratio=ca / area, nc=ca / (area - ca) if area > ca else None, child_x=float(cyx[1] + C0), child_y=float(cyx[0] + R0),
                   child_count=len(keep), child_grey=float(loc[ch].mean()))  # fmt: skip
        co = _mask_outline(ch)
        if co is not None:
            row["child_outline"] = (co + [C0, R0]).tolist()
        out.append(row)
    if len(out) >= 3 and np.median([r["eq_diameter"] for r in out]) < 1.5 * min_diameter:
        # regions piling up at the smallest size allowed are fragments of texture, not cells
        raise ValueError(f"找到的区域大多只有最小直径（{min_diameter:g} 像素）左右：多半是纹理碎片而不是细胞。"
                         "请把“区域最小直径”设为细胞的大致直径；细胞内部纹理很强（如电镜切片）时，膜线可能无法与纹理区分")  # fmt: skip
    ratios = np.array([r["ratio"] for r in out if r["ratio"] is not None])
    ncs = np.array([r["nc"] for r in out if r["nc"] is not None])
    areas = np.array([r["area"] for r in out])
    summary: dict[str, Any] = {"parents": len(out), "with_child": int(len(ratios)), "at_border": at_border,
                               "area_mean": float(areas.mean()) if len(areas) else None,
                               "child_area_mean": float(np.mean([r["child_area"] for r in out if r["child_area"]])) if len(ratios) else None,
                               "ratio_mean": float(ratios.mean()) if len(ratios) else None, "ratio_median": float(np.median(ratios)) if len(ratios) else None,
                               "ratio_sd": float(ratios.std(ddof=1)) if len(ratios) > 1 else None,
                               "nc_mean": float(ncs.mean()) if len(ncs) else None, "nc_median": float(np.median(ncs)) if len(ncs) else None}  # fmt: skip
    if not out:
        warnings.append("没有找到完整的区域：检查“父区域”的类型（膜包围、亮或暗的对象）、膜宽与最小直径")
    elif len(areas) >= 4 and areas.max() > 4 * np.median(areas):
        warnings.append("有的区域比中位数大 4 倍以上：边界线可能有缺口，相邻区域被并在一起，请看图核对")
    if out and len(ratios) < len(out):
        warnings.append(f"{len(out) - len(ratios)} 个区域没有找到子区域（原因见每行）")
    return {"parents": out, "summary": summary, "warnings": warnings, "noise": noise}


def distribution(d: np.ndarray) -> dict[str, float]:
    """d10, d50, d90 by number and by volume (weights d³), interpolated."""
    d = np.sort(np.asarray(d, float))
    out = {}
    for name, w in (("n", np.ones_like(d)), ("v", d**3)):
        cum = np.cumsum(w) / w.sum()
        for q in (10, 50, 90):
            out[f"d{q}_{name}"] = float(np.interp(q / 100, cum - w / w.sum() / 2, d))
    return out


__all__ = ["CLASSES", "distribution", "feret", "holes", "lattice", "nested", "particles", "pores"]
