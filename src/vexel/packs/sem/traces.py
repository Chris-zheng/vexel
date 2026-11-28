"""Conductors: trace width, pads and the clearance between conductors (printed circuit boards,
metal lines and pads on a die, seen from above).

``traces`` segments the conductors (a threshold halfway between the two grey classes, as for
fibres) and labels them: one connected conductor is one net.

- **Pads** are the parts of a net much wider than its traces: pixels farther from the
  conductor's edge than 1.5 times the net's typical trace half-width (its 30th percentile on
  the centre line) and than that half-width plus 1.5 px, grown back to the inscribed discs they are centres of. Each pad's size is
  a circle fitted to its edge points (threshold crossings on rays from its centre; rays that
  run out along a trace are left out, and so, after a first fit, are the points where a trace
  joins the pad). Wide parts whose edge, without those points, is not a circle (rms off the
  circle over 8 % of its radius or 0.35 px, a four-fold swing over 4 % of it or 0.25 px, or
  edges on fewer than half the rays: square pads, sharp bends, junctions) are listed apart as wide areas with their
  area, not as pads.
- **Trace width** is measured as for fibres (``vexel.packs.sem.fibres``): across the trace at
  every centre-line pixel, from the two edge crossings of the grey profile, away from pads,
  junctions, ends and the region's edge; length-weighted. The narrowest trace is the
  smallest median width of a centre-line segment (a single sample is noisy) - among the nets
  that have one: a trace only a width or two long between pads has none, and is counted.
- **Clearance** is the edge-to-edge gap between two different nets that are neighbours (their
  zones of nearest conductor touch): the shortest distance between their outlines at the
  threshold (marching squares, interpolated like the width crossings).

There is no design to compare with, so opens and shorts are not judged: two nets that touch
are one net here, and a broken trace is two.

``vexel.synthetic.sem.pcb_board`` draws boards with exact widths, pad sizes and clearances.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vexel.packs.sem.objects import _mid_threshold


def _segment(s: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    from scipy import ndimage as ndi

    vals = s[~mask] if mask is not None and mask.any() else s.ravel()
    t, _, _ = _mid_threshold(vals)
    for _ in range(2):  # halfway between the two plateaus, not the class means: from pixels well inside each
        fg = s > t
        din, dout = ndi.distance_transform_edt(fg), ndi.distance_transform_edt(~fg)
        if mask is not None:
            din, dout = np.where(mask, 0, din), np.where(mask, 0, dout)
        core = next((din >= r for r in (3.5, 2.5, 1.5) if (din >= r).sum() > 20), None)
        far = next((dout >= r for r in (3.5, 2.5) if (dout >= r).sum() > 20), None)
        if core is None or far is None:
            break
        t = float(np.median(s[core]) + np.median(s[far])) / 2
    fg = s > t
    if mask is not None:
        fg &= ~mask
    return fg, t


def _circle_fit(p: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares circle through points (x, y): centre, radius, rms residual."""
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    res = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, res


def _boundary(region: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Outline (x, y) of a pixel region, offset to image coordinates."""
    from skimage.measure import find_contours

    c = find_contours(np.pad(region, 1).astype(float), 0.5)
    if not c:
        return np.zeros((0, 2))
    o = max(c, key=len)
    return np.column_stack([o[:, 1] - 1 + x0, o[:, 0] - 1 + y0])


def _crossing(prof: np.ndarray, t: float, i0: int, step: int) -> float | None:
    """Fractional index where ``prof`` first falls below ``t`` walking from ``i0`` by ``step``."""
    i = i0
    n = len(prof)
    if not (0 <= i < n) or prof[i] <= t:
        return None
    while 0 <= i + step < n:
        j = i + step
        if prof[j] <= t:
            return i + (prof[i] - t) / (prof[i] - prof[j] + 1e-300) * step
        i = j
    return None


def traces(a: np.ndarray, *, feature: str = "bright", method: str = "threshold", smooth: float = 1.0, min_width: float = 3.0,
           pad_factor: float = 1.5, mask: np.ndarray | None = None, sx: float = 1.0, sy: float = 1.0) -> dict[str, Any]:  # fmt: skip
    """See the module docstring. ``feature``: the conductors are brighter or darker than the
    substrate; ``method``: trace edges at the threshold crossing (``threshold``, the same edge
    as the pads and clearances) or the steepest point near it (``gradient``)."""
    from scipy import ndimage as ndi
    from skimage.measure import regionprops
    from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

    from vexel.packs.sem.fibres import _near, fibres

    a = np.asarray(a, np.float64)
    h, w = a.shape
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    s = sm if feature == "bright" else -sm
    if not np.ptp(s[~mask] if mask is not None and mask.any() else s) > 0:
        raise ValueError("the region has no contrast")
    fg, t = _segment(s, mask)
    r_min = max(min_width / 2, 0.5)
    fg = remove_small_objects(fg, max_size=int(math.pi * (2 * r_min) ** 2))
    fg = remove_small_holes(fg, max_size=int(math.pi * r_min**2) + 1)
    if not fg.any():
        raise ValueError("no conductors found: check whether they are brighter or darker than the substrate")
    nets, n_nets = ndi.label(fg, structure=np.ones((3, 3)))
    dist = ndi.distance_transform_edt(fg)
    skel = skeletonize(fg)
    warnings: list[str] = []

    # pads: much wider than the net's traces
    pad_core = np.zeros_like(fg)
    for k, sl in enumerate(ndi.find_objects(nets), start=1):
        if sl is None:
            continue
        m = nets[sl] == k
        d_sk = dist[sl][m & skel[sl]]
        if len(d_sk) < 5:
            continue
        r_trace = float(np.percentile(d_sk, 30))
        # and 1.5 px more: on thin diagonal traces the pixel distance map steps between 1.4 and 2.2
        pad_core[sl] |= m & (dist[sl] > max(pad_factor * r_trace, r_trace + 1.5, r_min * 1.5))
    pad_mask = _near(pad_core, dist + 0.5, (h, w)) & fg
    plab, n_pads = ndi.label(pad_mask, structure=np.ones((3, 3)))
    near_mask = _near(mask, np.full((h, w), 2.0), (h, w)) if mask is not None and mask.any() else None
    pads = []
    wide = []  # wide parts that are not round: square pads, sharp bends, junctions
    for p in regionprops(plab):
        r0, c0, r1, c1 = p.bbox
        cy, cx = np.unravel_index(np.argmax(np.where(plab == p.label, dist, 0)), dist.shape)
        R = float(dist[cy, cx])
        touches = r0 == 0 or c0 == 0 or r1 == h or c1 == w or (near_mask is not None and bool(near_mask[plab == p.label].any()))
        n_rays = max(64, int(2 * math.pi * R))
        th = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
        tt = np.arange(0.0, 1.6 * R + 4, 0.25)
        xs = cx + np.cos(th)[:, None] * tt[None, :]
        ys = cy + np.sin(th)[:, None] * tt[None, :]
        prof = ndi.map_coordinates(s, [ys, xs], order=1, mode="nearest")
        pts = []
        for i in range(n_rays):
            j = _crossing(prof[i], t, 0, 1)
            if j is None:
                continue
            r = j * 0.25
            if r < 1.25 * R + 1.5:  # rays that run out along a trace find no edge near the pad's rim
                pts.append((cx + r * math.cos(th[i]), cy + r * math.sin(th[i])))
        P = np.array(pts) if pts else np.zeros((0, 2))
        fx, fy, fr, res = _circle_fit(P * [sx, sy]) if len(P) >= 12 else (0.0, 0.0, 1.0, math.inf)
        square, share = math.inf, len(P) / n_rays
        px_ = math.sqrt(sx * sy)
        if math.isfinite(res):
            Q = P * [sx, sy]
            dev = np.hypot(Q[:, 0] - fx, Q[:, 1] - fy) - fr
            keep = np.abs(dev) < max(0.5 * px_, 3 * 1.4826 * float(np.median(np.abs(dev - np.median(dev)))))
            if keep.sum() >= 12:  # refit without the points where a trace joins the pad
                Q = Q[keep]
                fx, fy, fr, res = _circle_fit(Q)
                share = len(Q) / n_rays
            ang = np.arctan2(Q[:, 1] - fy, Q[:, 0] - fx)
            dev = np.hypot(Q[:, 0] - fx, Q[:, 1] - fy) - fr
            (c4, s4, _), *_ = np.linalg.lstsq(np.column_stack([np.cos(4 * ang), np.sin(4 * ang), np.ones_like(ang)]), dev, rcond=None)
            square = math.hypot(c4, s4)  # a square's edge swings 6-12 % of the radius 4 times round
        # small pads: pixel noise sets the floor
        if not (res <= max(0.08 * fr, 0.35 * px_) and square <= max(0.04 * fr, 0.25 * px_) and share >= 0.5):
            ys_, xs_ = np.nonzero(plab == p.label)
            wide.append({"area": float(p.area * sx * sy), "eq_diameter": 2 * math.sqrt(p.area * sx * sy / math.pi), "x": float(xs_.mean()),
                         "y": float(ys_.mean()), "net": int(nets[cy, cx]), "width_box": (c1 - c0) * sx, "height_box": (r1 - r0) * sy,
                         "at_border": bool(touches), "_outline": _boundary(plab[r0:r1, c0:c1] == p.label, c0, r0),
                         "_fit": (res, square, share, fr, R)})  # fmt: skip
            continue
        pads.append({"pad": len(pads) + 1, "net": int(nets[cy, cx]), "x": fx / sx, "y": fy / sy, "diameter": 2 * fr,
                     "diameter_px": 2 * fr / math.sqrt(sx * sy), "roundness_rms": res, "edge_share": len(P) / n_rays,
                     "at_border": bool(touches), "_outline": P})  # fmt: skip

    # trace width, as for fibres, away from the pads
    zone = ndi.binary_dilation(pad_mask, iterations=2)
    excl = zone if mask is None else (zone | mask)
    trace_fg = fg & ~zone
    width: dict[str, Any] = {"samples": {"x": np.zeros(0), "y": np.zeros(0), "diameter": np.zeros(0), "length": np.zeros(0)}, "segments": [], "summary": {}}
    if trace_fg.sum() > 20:
        try:
            width = fibres(sm, feature=feature, threshold="mid", method=method, smooth=0.0, min_diameter=min_width, mask=excl,
                           sx=sx, sy=sy, level=t, mask_margin=1.0)  # fmt: skip  # the board's threshold: narrow traces alone never show the copper's plateau; a trace meets a pad without a long flare
        except ValueError:
            pass
    smp = width["samples"]
    for g in width["segments"]:
        xs_, ys_ = g["_pts"]
        g["net"] = int(np.bincount(nets[ys_, xs_]).argmax()) if len(xs_) else 0

    # clearance between nets: the conductors' outlines at the threshold (marching squares, the
    # same edge as the widths), the closest points of every two neighbouring nets
    from scipy.spatial import cKDTree
    from skimage.measure import find_contours

    _, (iy, ix) = ndi.distance_transform_edt(~fg, return_indices=True)
    near_net = nets[iy, ix]  # the nearest conductor of every pixel
    verts: dict[int, list[np.ndarray]] = {}
    for c in find_contours(np.where(mask, s.min(), s) if mask is not None else s, t):
        if len(c) < 3:
            continue
        r_, c_ = np.clip(np.round(c[len(c) // 2]).astype(int), 0, [h - 1, w - 1])
        k = int(near_net[r_, c_])
        if k:
            verts.setdefault(k, []).append(c[:, ::-1])  # (x, y)
    # neighbours: nets whose nearest-conductor zones touch
    pairs = set()
    for A, B in ((near_net[:, :-1], near_net[:, 1:]), (near_net[:-1, :], near_net[1:, :])):
        diff = (A != B) & (A > 0) & (B > 0)
        pairs |= {(int(min(a, b)), int(max(a, b))) for a, b in zip(A[diff], B[diff])}
    scale = np.array([sx, sy])
    trees = {k: (np.concatenate(v), cKDTree(np.concatenate(v) * scale)) for k, v in verts.items()}
    clearances = []
    for i, j in sorted(pairs):
        if i not in trees or j not in trees:
            continue
        Vi, _ = trees[i]
        Vj, tj = trees[j]
        d, k = tj.query(Vi * scale)
        m = int(np.argmin(d))
        # refine on the polylines around the two nearest vertices, densified to 0.05 px
        def local(V: np.ndarray, q: int) -> np.ndarray:
            seg = V[max(q - 3, 0) : q + 4]
            if len(seg) < 2:
                return seg
            n = np.maximum(np.ceil(np.hypot(*np.diff(seg, axis=0).T) / 0.05).astype(int), 1)
            return np.concatenate([seg[a] + np.linspace(0, 1, n[a], endpoint=False)[:, None] * (seg[a + 1] - seg[a]) for a in range(len(seg) - 1)] + [seg[-1:]])

        Li, Lj = local(Vi, m), local(Vj, int(k[m]))
        D = np.hypot(*((Li[:, None, :] - Lj[None, :, :]) * scale).transpose(2, 0, 1))
        a, b = np.unravel_index(np.argmin(D), D.shape)
        ea, eb = Li[a], Lj[b]
        clearances.append({"nets": (i, j), "clearance": float(D[a, b]), "clearance_px": float(np.hypot(*(eb - ea))),
                           "x1": float(ea[0]), "y1": float(ea[1]), "x2": float(eb[0]), "y2": float(eb[1])})  # fmt: skip
    clearances.sort(key=lambda c: c["clearance"])

    if wide:
        warnings.append(f"{len(wide)} 处宽的导体区域不是圆形（方形焊盘、锐角拐弯或交叉处）：不计入焊盘统计，单独列出并画出轮廓")
    summary: dict[str, Any] = {"nets": int(n_nets), "pads": len(pads), "wide_areas": len(wide), "coverage": float(fg.mean()), "samples": int(len(smp["diameter"]))}
    dpad = np.array([p["diameter"] for p in pads if not p["at_border"]])
    if len(dpad):
        summary.update(pad_diameter_mean=float(dpad.mean()), pad_diameter_min=float(dpad.min()), pad_diameter_max=float(dpad.max()))
    d = smp["diameter"]
    if len(d):
        wts = smp["length"]
        mean = float(np.average(d, weights=wts))
        summary.update(width_mean=mean, width_sd=float(np.sqrt(np.average((d - mean) ** 2, weights=wts))),
                       width_median=float(np.median(d)), width_min_sample=float(d.min()))  # fmt: skip
        segs = [g for g in width["segments"] if g["samples"] >= 5]
        if segs:
            nar = min(segs, key=lambda g: g["diameter"])
            summary.update(narrowest=nar["diameter"], narrowest_x=nar["x"], narrowest_y=nar["y"])
    if clearances:
        c0 = clearances[0]
        summary.update(clearance_min=c0["clearance"], clearance_pairs=len(clearances),
                       clearance_median=float(np.median([c["clearance"] for c in clearances])))  # fmt: skip
    if len(d) and np.median(width["samples"].get("diameter_px", d)) < 5:
        warnings.append(f"走线宽度中位数只有 {np.median(width['samples'].get('diameter_px', d)):.1f} 像素：宽度误差相对较大，建议提高放大倍数（线宽至少 6 像素）")
    if not len(d):
        warnings.append("没有可测宽度的走线段（焊盘、交叉、端点和区域边界附近不测）：只给出焊盘与间距")
    else:
        unmeasured = int(n_nets) - len({g["net"] for g in width["segments"] if g["samples"] >= 5})
        if unmeasured > 0:
            summary["nets_unmeasured"] = unmeasured
            warnings.append(f"{unmeasured} 个导体没有可测宽度的走线段（走线太短，或全在焊盘、交叉与边界附近）：最窄走线只在其余导体中找")
    if n_nets == 1 and not clearances:
        warnings.append("只有一个相连的导体：没有不同网络之间的间距可测（两条走线相碰时会被当成一个网络）")
    if summary["coverage"] > 0.7:
        warnings.append(f"导体占了区域的 {summary['coverage']:.0%}：检查“导体是”的亮暗设置")
    return {"summary": summary, "pads": pads, "wide": wide, "segments": width["segments"], "samples": smp, "clearances": clearances,
            "warnings": warnings + [x for x in width.get("warnings", []) if "纤维" not in x], "mask": fg, "nets": nets}  # fmt: skip


__all__ = ["traces"]
