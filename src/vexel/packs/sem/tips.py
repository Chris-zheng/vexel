"""Tips (probe needles, emitters, AFM tips, nanoneedles; M4): apex radius of curvature
and cone angle from the outline of one tip.

The outline is traced at sub-pixel precision at the grey level halfway between tip and
background. The apex is the outline's extreme point along the tip's axis (the narrow
end). The radius comes from circles fitted to the outline within a growing distance of
the apex: the fitting range matters (too small: a short arc fits poorly; too large: the
cone pulls the circle open), so the largest range before the fit residual jumps (where
the cone begins) is chosen, and the spread of the radius over the two smaller ranges is
reported as its uncertainty. The cone
angle comes from straight lines fitted to the two flanks between ``cone_from`` and
``cone_to`` (distance from the apex along the axis; default 2–10 radii).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def tip(a: np.ndarray, *, feature: str = "bright", smooth: float = 1.0, cone_from: float | None = None, cone_to: float | None = None,
        mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    from scipy import ndimage as ndi
    from skimage.measure import find_contours

    from vexel.packs.sem.measure import fit_circle
    from vexel.packs.sem.objects import _mid_threshold

    a = np.asarray(a, np.float64)
    sm = ndi.gaussian_filter(a, smooth) if smooth else a
    if feature == "dark":
        sm = -sm
    t, lo, hi = _mid_threshold(sm.ravel())
    fg = sm > t
    lab, n = ndi.label(fg)
    out: dict[str, Any] = {"warnings": []}
    if n == 0:
        out["warnings"].append("没有找到尖端：框选一个尖端，周围留出背景")
        return out
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    comp = lab == int(np.argmax(sizes)) + 1
    # outline at the mid level, the image padded with background so the outline closes at the border
    pad = np.pad(np.where(ndi.binary_dilation(comp, iterations=3), sm, lo), 2, constant_values=lo)
    cs = find_contours(pad, t)
    if not cs:
        out["warnings"].append("无法提取轮廓")
        return out
    c = max(cs, key=len)[:, ::-1] - 2  # (x, y)
    rows, cols = a.shape
    on_border = (c[:, 0] <= 0.5) | (c[:, 1] <= 0.5) | (c[:, 0] >= cols - 1.5) | (c[:, 1] >= rows - 1.5)
    # axis: principal direction of the tip's pixels
    yy, xx = np.nonzero(comp)
    pts = np.stack([xx, yy], 1).astype(float)
    centre = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - centre).T))
    axis = v[:, int(np.argmax(w))]
    proj = (c - centre) @ axis
    # the narrow end: the end of the outline that does not reach the border (else: the narrower one)
    ends = []
    for sign in (1, -1):
        k = int(np.argmax(sign * proj))
        near = sign * proj > sign * proj[k] - 0.15 * np.ptp(proj)
        width = np.ptp((c[near] - centre) @ np.array([-axis[1], axis[0]]))
        ends.append((bool(on_border[near].any()), width, sign, k))
    ends.sort(key=lambda e: (e[0], e[1]))
    _, _, sign, k_apex = ends[0]
    axis = axis * sign  # pointing from the body to the apex
    apex = c[k_apex]
    inner = c[~on_border]
    d_apex = np.hypot(*(inner - apex).T)
    # radius: fits within growing distances of the apex; take the most stable range
    span = float(np.ptp((inner - apex) @ axis))
    s_values = np.geomspace(3.0, max(span * 0.6, 6.0), 28)
    fits = []
    for s in s_values:
        sel = inner[d_apex <= s]
        if len(sel) < 7:
            fits.append((s, np.nan, np.nan, 0))
            continue
        _, _, r, rms = fit_circle(sel)
        fits.append((s, r, rms, len(sel)))
    R = np.array([f[1] for f in fits])
    rms = np.array([f[2] for f in fits])
    # the fitting range: the largest one before the fit residual jumps (the circle starts to take in
    # the cone); a circle fitted to a short arc is unreliable, one reaching into the cone too wide
    ok = [i for i, f in enumerate(fits) if f[3] >= 10 and np.isfinite(f[1])]
    if len(ok) < 3:
        out["warnings"].append("尖端太小，无法拟合曲率半径（至少需要十几个轮廓点）")
        return out
    best = None
    for prev, i in zip(ok, ok[1:]):
        if rms[i] > 0.25 and rms[i] > 1.8 * max(rms[prev], 1e-6):
            best = prev
            break
    if best is None:  # no clear jump (a small tip, noise): the range where the radius is steadiest
        steady = np.inf
        for j in range(1, len(ok) - 1):
            win = R[ok[j - 1 : j + 2]]
            rel = float(np.ptp(win) / np.mean(win))
            if rel < steady:
                best, steady = ok[j], rel
    k = ok.index(best)
    window = R[ok[max(0, k - 2) : k + 1]]
    radius = float(R[best])
    spread = float(np.ptp(window) / radius) if len(window) > 1 else 0.0
    # cone: flanks between cone_from and cone_to along the axis, each side fitted by a line
    t_ax = (inner - apex) @ (-axis)  # distance back from the apex
    lat = (inner - apex) @ np.array([-axis[1], axis[0]])
    f0 = cone_from if cone_from is not None else 2 * radius
    f1 = cone_to if cone_to is not None else min(10 * radius, 0.9 * t_ax.max())
    sides = {}
    for name, sel in (("left", lat < 0), ("right", lat > 0)):
        m = sel & (t_ax >= f0) & (t_ax <= f1)
        if m.sum() >= 5:
            slope, _ = np.polyfit(t_ax[m], lat[m], 1)
            sides[name] = math.degrees(math.atan(abs(slope)))
    out.update(apex=apex.tolist(), axis_angle=float(math.degrees(math.atan2(-axis[1], axis[0]))), radius=radius,
               radius_uncertainty=float(spread * radius / 2), fit_distance=float(s_values[best]), length_visible=float(t_ax.max()),
               cone_from=f0, cone_to=f1, outline=c.tolist())  # fmt: skip
    if len(sides) == 2:
        out.update(cone_angle=sides["left"] + sides["right"], half_angle_left=sides["left"], half_angle_right=sides["right"])
        if abs(sides["left"] - sides["right"]) > 3:
            out["warnings"].append(f"两侧半角相差 {abs(sides['left'] - sides['right']):.1f}°：尖端不对称，或轴向不在图像平面内")
    else:
        out["warnings"].append("侧面太短，无法求锥角：调整“锥角起点/终点”或框选更长的一段")
    if radius < 10:
        out["warnings"].append(f"曲率半径只有 {radius:.1f} 像素：误差可能超过 5%，建议提高放大倍数")
    return out


__all__ = ["tip"]
