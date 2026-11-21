"""Fibres: diameter, orientation and the gaps between fibres (nonwovens, electrospun mats,
membranes), in the manner of DiameterJ (NIST): measurements along each fibre's centre line,
away from crossings and fibre ends.

``fibres`` segments the fibres (a global or local threshold halfway between the two grey
classes), takes their centre lines (skeleton) and, at every centre-line pixel that is not
near a crossing, a fibre end or the region's edge:

- the fibre's direction from the structure tensor of the distance map (geometry only, so
  edge brightening does not bias it);
- the diameter from the grey-level profile across the fibre: the two threshold crossings
  nearest the centre, interpolated to a quarter pixel, converted to physical units with
  the true (not the pixel-space) perpendicular when the pixels are not square.

Every sample stands for about one pixel of fibre length, so statistics over samples are
length-weighted. Samples are grouped into segments (centre line between crossings) for the
per-object table. The gaps between fibres are the background regions that are not cut by
the region's edge.

``vexel.synthetic.sem.fibre_mat`` draws fibre mats with exact ground truth for the tests
and the acceptance script.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vexel.packs.sem.objects import _mid_threshold


def _direction(fg: np.ndarray, dist: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector along the fibre (dx, dy in pixels) at every pixel."""
    from scipy import ndimage as ndi

    d = ndi.gaussian_filter(dist, 1.0)
    gy, gx = np.gradient(d)
    axx, axy, ayy = (ndi.gaussian_filter(v, rho) for v in (gx * gx, gx * gy, gy * gy))
    phi = 0.5 * np.arctan2(2 * axy, axx - ayy)  # dominant gradient: across the fibre
    return -np.sin(phi), np.cos(phi)


def _near(points: np.ndarray, radius_at: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixels closer to one of ``points`` (bool image) than that point's radius."""
    from scipy import ndimage as ndi

    if not points.any():
        return np.zeros(shape, bool)
    d, (iy, ix) = ndi.distance_transform_edt(~points, return_indices=True)
    return d < radius_at[iy, ix]


def _steepest(prof: np.ndarray, off: np.ndarray, x0: np.ndarray, sgn: int, reach: float = 2.5) -> np.ndarray:
    """Position of the steepest descent away from the fibre (``sgn``: the sign of the slope going
    outwards is -sgn... i.e. the profile falls towards larger |offset|) within ``reach`` pixels
    of each threshold crossing ``x0``, refined with a parabola."""
    step = off[1] - off[0]
    g = np.gradient(prof, step, axis=1) * sgn  # positive where the profile falls going outwards
    out = x0.copy()
    k = int(round(reach / step))
    j0 = np.clip(np.round((x0 - off[0]) / step).astype(int), 0, len(off) - 1)
    idx = j0[:, None] + np.arange(-k, k + 1)[None, :]
    idx = np.clip(idx, 1, len(off) - 2)
    rows = np.arange(len(x0))[:, None]
    gv = g[rows, idx]
    m = np.argmax(gv, axis=1)
    jm = idx[np.arange(len(x0)), m]
    r = np.arange(len(x0))
    y0, y1, y2 = g[r, jm - 1], g[r, jm], g[r, jm + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
    ok = np.isfinite(x0) & (gv.max(1) > 0)
    out[ok] = off[jm[ok]] + np.clip(frac[ok], -0.5, 0.5) * step
    return out


def fibres(a: np.ndarray, *, feature: str = "bright", threshold: str = "mid", method: str = "gradient", smooth: float = 1.0,
           min_diameter: float = 2.0, mask: np.ndarray | None = None, sx: float = 1.0, sy: float = 1.0) -> dict[str, Any]:  # fmt: skip
    """See the module docstring. ``method``: the edge is the steepest point of the grey profile
    near the threshold crossing (``gradient``: not moved by edge brightening) or the threshold
    crossing itself (``threshold``)."""
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

    a = np.asarray(a, np.float64)
    h, w = a.shape
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    sign = 1.0 if feature == "bright" else -1.0
    s = sign * sm
    vals = s[~mask] if mask is not None and mask.any() else s.ravel()
    if not np.ptp(vals) > 0:
        raise ValueError("the region has no contrast")
    if threshold == "local":
        from skimage.filters import threshold_local

        t_img = threshold_local(s, block_size=2 * (max(a.shape) // 8) + 1, offset=0)
        fg = s > t_img
    else:
        t, _, _ = _mid_threshold(vals)
        fg = s > t
        core = ndi.binary_erosion(fg, iterations=2)
        far = ~ndi.binary_dilation(fg, iterations=3)
        if core.sum() > 20 and far.sum() > 20:
            t = float(np.median(s[core]) + np.median(s[far])) / 2
            fg = s > t
        t_img = np.full_like(s, t)
    if mask is not None:
        fg &= ~mask
    fg = ndi.binary_opening(fg, iterations=1) if min_diameter >= 3 else fg
    r_min = max(min_diameter / 2, 0.5)
    fg = remove_small_objects(fg, max_size=int(math.pi * (2 * r_min) ** 2))
    fg = remove_small_holes(fg, max_size=int(math.pi * r_min**2) + 1)
    if not fg.any():
        raise ValueError("no fibres found: check whether the fibres are bright or dark")
    dist = ndi.distance_transform_edt(fg)
    skel = skeletonize(fg)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    junction = skel & (nb >= 3)
    ends = skel & (nb == 1)
    r_med = float(np.median(dist[skel])) if skel.any() else 1.0
    excl = _near(junction, 1.5 * ndi.maximum_filter(dist, 3) + 2.0, (h, w)) | _near(ends, dist + 1.5, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx])
    excl |= edge < 2 * dist + 3
    if mask is not None:
        excl |= _near(mask, 2 * ndi.maximum_filter(dist, 5) + 3, (h, w))
    ux, uy = _direction(fg, dist, max(1.5, 0.75 * r_med))
    sy_, sx_ = np.nonzero(skel & ~excl & (dist >= r_min * 0.5))
    # grey profile across the fibre through each sample, 0.25 px steps
    L = float(min(max(4 * np.percentile(dist[skel], 99) + 6, 8), max(h, w) / 2))
    off = np.arange(-L, L + 1e-9, 0.25)
    px, py = -uy[sy_, sx_], ux[sy_, sx_]  # across the fibre
    cx, cy = sx_[:, None] + off[None, :] * px[:, None], sy_[:, None] + off[None, :] * py[:, None]
    prof = ndi.map_coordinates(s, [cy.ravel(), cx.ravel()], order=1, mode="nearest").reshape(cx.shape)
    tl = ndi.map_coordinates(t_img, [cy.ravel(), cx.ravel()], order=1, mode="nearest").reshape(cx.shape)
    above = prof > tl
    c = len(off) // 2
    ok = above[:, c]
    # first index outward from the centre that is below the threshold, each side
    right = above[:, c:]
    left = above[:, : c + 1][:, ::-1]
    ir = np.argmax(~right, axis=1)
    il = np.argmax(~left, axis=1)
    ok &= (~right).any(1) & (~left).any(1)
    rows = np.arange(len(sy_))

    def cross(side: np.ndarray, idx: np.ndarray, sgn: int) -> np.ndarray:
        j1 = np.clip(c + sgn * idx, 0, len(off) - 1)
        j0 = np.clip(j1 - sgn, 0, len(off) - 1)
        v0, v1 = prof[rows, j0] - tl[rows, j0], prof[rows, j1] - tl[rows, j1]
        frac = np.where(np.abs(v0 - v1) > 1e-12, v0 / (v0 - v1 + 1e-300), 0.5)
        return off[j0] + frac * (off[j1] - off[j0])

    xr, xl = cross(right, ir, 1), cross(left, il, -1)
    if method == "gradient":
        xr, xl = _steepest(prof, off, xr, -1), _steepest(prof, off, xl, 1)
    wpx = xr - xl
    centre_off = (xr + xl) / 2
    ok &= (wpx > 0) & np.isfinite(wpx)
    # true width: the edge-to-edge vector's component perpendicular to the fibre, in physical units
    dpx, dpy = ux[sy_, sx_] * sx, uy[sy_, sx_] * sy
    dn = np.hypot(dpx, dpy)
    dpx, dpy = dpx / dn, dpy / dn
    ex, ey = wpx * px * sx, wpx * py * sy
    width = np.abs(ex * dpy - ey * dpx)
    ang = np.degrees(np.arctan2(-dpy, dpx)) % 180.0  # counter-clockwise on screen, 0..180
    step = 1.0 / np.maximum(np.abs(ux[sy_, sx_]), np.abs(uy[sy_, sx_]))  # pixels per skeleton step along the fibre
    seg_len = step * np.hypot(ux[sy_, sx_] * sx, uy[sy_, sx_] * sy)
    sel = np.nonzero(ok)[0]
    samples = {"x": sx_[sel] + centre_off[sel] * px[sel], "y": sy_[sel] + centre_off[sel] * py[sel], "diameter": width[sel],
               "diameter_px": wpx[sel], "angle": ang[sel], "length": seg_len[sel]}  # fmt: skip
    # segments: centre line between crossings
    branches = skel & ~ndi.binary_dilation(junction, iterations=1)
    lab, nseg = ndi.label(branches, structure=np.ones((3, 3)))
    seg_of = lab[sy_[sel], sx_[sel]]
    segs = []
    for k in range(1, nseg + 1):
        m = seg_of == k
        if m.sum() < 3:
            continue
        d, an, ln = samples["diameter"][m], samples["angle"][m], samples["length"][m]
        mean_dir = 0.5 * math.degrees(math.atan2(np.sum(ln * np.sin(np.radians(2 * an))), np.sum(ln * np.cos(np.radians(2 * an))))) % 180
        ys_, xs_ = np.nonzero(lab == k)
        segs.append({"segment": len(segs) + 1, "x": float(xs_.mean()), "y": float(ys_.mean()), "diameter": float(np.median(d)),
                     "diameter_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "samples": int(m.sum()),
                     "length": float(ln.sum()), "angle": float(mean_dir), "_pts": (xs_, ys_)})  # fmt: skip
    summary: dict[str, Any] = {"samples": int(len(sel)), "segments": len(segs), "coverage": float(fg.mean())}
    d = samples["diameter"]
    if len(d):
        wts = samples["length"]
        mean = float(np.average(d, weights=wts))
        summary.update(diameter_mean=mean, diameter_sd=float(np.sqrt(np.average((d - mean) ** 2, weights=wts))),
                       diameter_median=float(np.median(d)), diameter_min=float(d.min()), diameter_max=float(d.max()),
                       d10=float(np.percentile(d, 10)), d90=float(np.percentile(d, 90)))  # fmt: skip
        c2, s2 = np.sum(wts * np.cos(np.radians(2 * samples["angle"]))), np.sum(wts * np.sin(np.radians(2 * samples["angle"])))
        summary["orientation"] = float(0.5 * math.degrees(math.atan2(s2, c2)) % 180)
        summary["order"] = float(math.hypot(c2, s2) / wts.sum())  # 0: random in the plane, 1: all parallel
    # gaps between fibres
    from skimage.measure import regionprops

    bg = ~fg if mask is None else ~fg & ~mask
    glab, _ = ndi.label(bg)
    gaps = []
    for p in regionprops(glab):
        r0, c0, r1, c1 = p.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w or p.area < math.pi * r_min**2:
            continue
        gaps.append(2 * math.sqrt(p.area * sx * sy / math.pi))
    if gaps:
        g = np.array(gaps)
        summary.update(gaps=len(g), gap_mean=float(g.mean()), gap_d50=float(np.median(g)), gap_max=float(g.max()))
    else:
        summary["gaps"] = 0
    warnings = []
    if len(sel) and np.median(samples["diameter_px"]) < 6:
        warnings.append(f"纤维直径中位数只有 {np.median(samples['diameter_px']):.1f} 像素：直径误差相对较大，建议提高放大倍数（直径至少 8 像素）")
    if len(sel) < 30:
        warnings.append(f"只有 {len(sel)} 个有效测量点（交叉、端点和边界附近不测）：结果不稳定，换一个区域或降低放大倍数")
    if summary["coverage"] > 0.75:
        warnings.append(f"纤维占了区域的 {summary['coverage']:.0%}：检查“纤维是”的亮暗设置，或纤维是否铺满了整个视野（这时缝隙统计无意义）")
    return {"samples": samples, "segments": segs, "summary": summary, "warnings": warnings, "mask": fg}


__all__ = ["fibres"]
