"""Tilt 3D (M6): a 2.5D height map from SEM images of one place taken at different stage
tilts (stereo photogrammetry), with a confidence and an uncertainty for every pixel.

Geometry. The stage tilts about an axis in the image plane (eucentric: the axis crosses
the centre of the image). At the magnifications SEM is used at the projection is parallel,
so a surface point (y, z), measured across the axis, appears in a view tilted by φ at
``y·cos φ + z·sin φ``. Two views at φa and φb give the height

    z = (ya·cos φb − yb·cos φa) / sin(φa − φb)

where ya, yb are where the point appears in each view. The second view is first rescaled
across the axis by cos φa / cos φb (so that a flat surface would match exactly), then
aligned as a whole (phase correlation) and matched pixel by pixel (TV-L1 optical flow,
sub-pixel); what is left across the axis is the parallax, proportional to the height.

Not every pixel can be measured: a flat, textureless area has nothing to match (published
comparisons of commercial software found errors up to ~46 % there). Every pixel gets a
**confidence** (how well the two views match locally, times how much texture there is) and
an **uncertainty** (the parallax noise turned into height; with three or more views, also
how far the pairs disagree). Pixels below ``min_confidence`` are marked, not hidden: they
stay in the height map but are left out of roughness and step heights.

Heights are relative: a stage that is not exactly eucentric, or a mean height, adds a
constant and a tilt of the whole surface; levelling (``level``) removes them before
roughness is computed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# -- results ------------------------------------------------------------------------------------


@dataclass
class HeightMap:
    z: np.ndarray  # height of every pixel of the reference view (metres when calibrated, else pixels)
    confidence: np.ndarray  # 0..1
    uncertainty: np.ndarray  # 1-sigma, same unit as z
    pixel_size: float | None = None  # metres per pixel (x); None: pixels
    pixel_size_y: float | None = None
    min_confidence: float = 0.3
    pairs: list[dict[str, Any]] = field(default_factory=list)  # per pair: tilts, global shift, median confidence
    notes: list[str] = field(default_factory=list)

    @property
    def valid(self) -> np.ndarray:
        return (self.confidence >= self.min_confidence) & np.isfinite(self.z)

    @property
    def unit(self) -> str:
        return "m" if self.pixel_size else "px"

    @property
    def sx(self) -> float:
        return self.pixel_size or 1.0

    @property
    def sy(self) -> float:
        return self.pixel_size_y or self.pixel_size or 1.0


# -- matching --------------------------------------------------------------------------------------

NOISE_GAIN = 0.84  # parallax noise = gain × image noise / texture gradient (calibrated)
PARALLAX_FLOOR = 0.1  # pixels: what the matching itself leaves on noise-free views


def _norm(a: np.ndarray, highpass: float = 12.0) -> np.ndarray:
    """Zero mean, unit spread, without the slow shading (which differs between tilts)."""
    from scipy import ndimage as ndi

    a = np.asarray(a, np.float64)
    a = a - ndi.gaussian_filter(a, highpass)
    return (a - a.mean()) / (a.std() + 1e-12)


def _rescale_rows(b: np.ndarray, k: float) -> np.ndarray:
    """``b`` sampled at rows yc + (y − yc)·k: undoes the difference in foreshortening."""
    from scipy import ndimage as ndi

    rows, cols = b.shape
    yc = (rows - 1) / 2
    rr, cc = np.meshgrid((np.arange(rows) - yc) * k + yc, np.arange(cols, dtype=float), indexing="ij")
    return ndi.map_coordinates(b, [rr, cc], order=3, mode="nearest")


def _local_mean(a: np.ndarray, size: int) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.uniform_filter(a, size, mode="nearest")


def _zncc(a: np.ndarray, b: np.ndarray, size: int = 9) -> np.ndarray:
    """Local zero-mean normalised cross-correlation of two aligned images (−1..1)."""
    ma, mb = _local_mean(a, size), _local_mean(b, size)
    va = _local_mean(a * a, size) - ma * ma
    vb = _local_mean(b * b, size) - mb * mb
    cov = _local_mean(a * b, size) - ma * mb
    return cov / np.sqrt(np.maximum(va * vb, 1e-12))


def match_pair(a: np.ndarray, b: np.ndarray, tilt_a: float, tilt_b: float, *, attachment: float = 3.0,
               window: int = 9) -> dict[str, Any]:  # fmt: skip
    """Height (pixels) of every pixel of view ``a`` from views ``a`` and ``b`` (tilt axis
    horizontal, through the centre row), with confidence and parallax noise."""
    from scipy import ndimage as ndi
    from skimage.registration import optical_flow_tvl1, phase_cross_correlation

    pa, pb = math.radians(tilt_a), math.radians(tilt_b)
    if abs(math.sin(pb - pa)) < math.sin(math.radians(1.0)):
        raise ValueError("the views must differ by at least 1° of tilt")
    k = math.cos(pb) / math.cos(pa)
    A = _norm(a)
    B1 = _rescale_rows(_norm(b), k)  # a flat surface now matches with no shift
    shift = phase_cross_correlation(A, B1, upsample_factor=20, normalization=None)[0]
    B2 = ndi.shift(B1, shift, order=3, mode="nearest")
    v, u = optical_flow_tvl1(A, B2, attachment=attachment, tightness=0.3, num_warp=10, num_iter=20, prefilter=True)
    rows, cols = A.shape
    rr, cc = np.meshgrid(np.arange(rows, dtype=float), np.arange(cols, dtype=float), indexing="ij")
    Bw = ndi.map_coordinates(B2, [rr + v, cc + u], order=1, mode="nearest")
    zncc = _zncc(A, Bw, window)
    # texture across the axis: the parallax is measured along y
    gy = ndi.sobel(ndi.gaussian_filter(A, 1.0), axis=0)
    texture = np.sqrt(_local_mean(gy * gy, window))
    tex = np.clip(texture / (np.median(texture) * 0.5 + 1e-12), 0, 1)
    conf = np.clip(zncc, 0, 1) * tex
    # matching is weaker near the image's border (the views do not overlap there, the flow is extrapolated)
    m = max(8, int(0.03 * max(rows, cols)))
    ry = np.clip(np.minimum(np.arange(rows), np.arange(rows)[::-1]) / m, 0, 1)
    rx = np.clip(np.minimum(np.arange(cols), np.arange(cols)[::-1]) / m, 0, 1)
    conf = conf * np.minimum(ry[:, None], rx[None, :])
    disparity = v - shift[0]  # rows in the rescaled view b, relative to a
    scale = math.cos(pb) / math.sin(pb - pa)
    z = disparity * scale
    # parallax noise: image noise over the texture across the axis, plus the method's own floor
    # (calibrated on synthetic tilt series: tests/benchmarks/sem_tilt3d_eval.py)
    from vexel.packs.sem.process import noise_sigma

    sn = min(noise_sigma(A), noise_sigma(A.T))
    sigma_d = np.sqrt((NOISE_GAIN * sn / np.maximum(texture, 1e-6)) ** 2 + PARALLAX_FLOOR**2)
    return {"z": z, "confidence": conf, "sigma": np.abs(sigma_d * scale), "shift": [float(s) for s in shift],
            "zncc": float(np.median(zncc)), "lateral": float(np.median(u))}  # fmt: skip


# -- the reconstruction -----------------------------------------------------------------------------


def reconstruct(images: list[np.ndarray], tilts: list[float], *, axis: float = 0.0, reference: int = 0, pixel_size: float | None = None,
                pixel_size_y: float | None = None, min_confidence: float = 0.3, attachment: float = 3.0, smooth: float = 1.5) -> HeightMap:  # fmt: skip
    """A height map in the frame of ``images[reference]`` from two or more views of one place.

    ``tilts``: stage tilt of each view, degrees. ``axis``: direction of the tilt axis in the
    image, degrees counter-clockwise from horizontal (0: horizontal; 90: vertical).
    With three or more views every other view is paired with the reference and the heights
    are averaged, weighted by confidence and precision; where the pairs disagree the
    uncertainty grows.

    ``attachment``: how closely the matching follows the image detail (TV-L1; lower: smoother,
    less noise, finer detail lost). ``smooth``: a Gaussian on the heights, pixels (0: none).
    Height noise is the parallax noise times cos φ / sin Δφ (about 5.7 at a 10° difference),
    so both matter at small tilt differences."""
    from scipy import ndimage as ndi

    if len(images) < 2 or len(images) != len(tilts):
        raise ValueError("give at least two images and one tilt angle per image")
    shape = np.asarray(images[0]).shape
    if any(np.asarray(im).shape != shape for im in images):
        raise ValueError("the images differ in size")
    imgs = [np.asarray(im, np.float64) for im in images]
    if axis:
        imgs = [ndi.rotate(im, -axis, reshape=False, order=3, mode="nearest") for im in imgs]
    ref = imgs[reference]
    zs, ws, sig, confs, pairs = [], [], [], [], []
    for j, (im, t) in enumerate(zip(imgs, tilts)):
        if j == reference:
            continue
        m = match_pair(ref, im, tilts[reference], t, attachment=attachment)
        zs.append(m["z"])
        sig.append(np.maximum(m["sigma"], 1e-6))
        confs.append(m["confidence"])
        ws.append(np.maximum(m["confidence"], 1e-3) / sig[-1] ** 2)
        pairs.append({"view": j, "tilt_reference": tilts[reference], "tilt": t, "shift_px": m["shift"], "match": m["zncc"],
                      "lateral_px": m["lateral"], "confidence": float(np.median(m["confidence"]))})  # fmt: skip
    Z, W = np.stack(zs), np.stack(ws)
    wsum = W.sum(0) + 1e-30
    z = (Z * W).sum(0) / wsum
    if smooth:
        valid_w = np.maximum(W.sum(0), 1e-30)
        z = ndi.gaussian_filter(z * valid_w, smooth) / ndi.gaussian_filter(valid_w, smooth)  # weighted: unreliable pixels pull less
    conf = np.stack(confs).max(0)  # the best pair's
    unc = 1 / np.sqrt((1 / np.stack(sig) ** 2).sum(0))  # precision of the mean of the pairs
    if len(zs) > 1:  # pairs that disagree widen it
        spread = np.sqrt(((Z - z) ** 2 * W).sum(0) / wsum)
        unc = np.sqrt(unc**2 + spread**2)
    if axis:
        z = ndi.rotate(z, axis, reshape=False, order=1, mode="nearest")
        conf = ndi.rotate(conf, axis, reshape=False, order=1, mode="constant", cval=0.0)
        unc = ndi.rotate(unc, axis, reshape=False, order=1, mode="nearest")
    hm = HeightMap(z=z, confidence=conf, uncertainty=unc, pixel_size=pixel_size, pixel_size_y=pixel_size_y, min_confidence=min_confidence,
                   pairs=pairs)  # fmt: skip
    if pixel_size:
        f = pixel_size_y or pixel_size  # parallax is measured across the axis
        hm.z, hm.uncertainty = hm.z * f, hm.uncertainty * f
    low = 1 - hm.valid.mean()
    if low > 0.3:
        hm.notes.append(f"{low:.0%} 的像素匹配置信度低（表面缺少纹理或两张图差别过大），这些区域的高度不可靠，已标出")
    span = max(abs(t - tilts[reference]) for t in tilts)
    if span < 3:
        hm.notes.append(f"倾转角差只有 {span:.1f}°：视差小，高度噪声大；建议 5°–15°")
    if any(abs(p["lateral_px"]) > 2 for p in pairs):
        hm.notes.append("两张图沿倾转轴方向也有错位：倾转轴方向可能设错，或样品台不是共心倾转")
    return hm


# -- analysis --------------------------------------------------------------------------------------


def level(hm: HeightMap, order: int = 1, region: list[float] | None = None) -> np.ndarray:
    """The height map with a best-fit plane (order 1) or quadric (order 2) removed, fitted on
    the reliable pixels (of ``region`` [x0, y0, x1, y1] when given, e.g. the flat base)."""
    z = hm.z
    rows, cols = z.shape
    yy, xx = np.mgrid[:rows, :cols].astype(float)
    sel = hm.valid.copy()
    if region:
        x0, y0, x1, y1 = (int(round(v)) for v in region)
        box = np.zeros_like(sel)
        box[max(y0, 0) : y1, max(x0, 0) : x1] = True
        sel &= box
    xn, yn = xx / max(cols - 1, 1) * 2 - 1, yy / max(rows - 1, 1) * 2 - 1
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([xn[sel] ** i * yn[sel] ** j for i, j in terms], 1)
    if len(A) < len(terms) * 3:
        return z - np.nanmedian(z)
    keep = np.ones(len(A), bool)
    vals = z[sel]
    for _ in range(3):  # fit the bulk: features standing out are left out
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        res = vals - A @ coef
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-30
        keep = np.abs(res) <= 3 * sd
    return z - sum(c * xn**i * yn**j for c, (i, j) in zip(coef, terms))


def roughness(z: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """ISO 25178 height parameters of a (levelled) surface: Sa, Sq, Sp, Sv, Sz, Ssk, Sku."""
    v = np.asarray(z, np.float64)[mask if mask is not None else np.isfinite(z)]
    v = v[np.isfinite(v)]
    if v.size < 16:
        return {}
    d = v - v.mean()
    sq = float(np.sqrt(np.mean(d**2)))
    return {"Sa": float(np.mean(np.abs(d))), "Sq": sq, "Sp": float(d.max()), "Sv": float(-d.min()), "Sz": float(d.max() - d.min()),
            "Ssk": float(np.mean(d**3) / sq**3) if sq else 0.0, "Sku": float(np.mean(d**4) / sq**4) if sq else 0.0, "pixels": int(v.size)}  # fmt: skip


def profile(hm: HeightMap, z: np.ndarray, a: tuple[float, float], b: tuple[float, float], width: int = 3) -> dict[str, np.ndarray]:
    """Heights along a line a -> b ([x, y] pixels): distance, height, confidence."""
    from vexel.packs.sem.measure import sample_profile

    dist, h = sample_profile(np.nan_to_num(z), a, b, width)
    _, c = sample_profile(hm.confidence, a, b, width)
    dx, dy = (b[0] - a[0]) * hm.sx, (b[1] - a[1]) * hm.sy
    length = math.hypot(dx, dy)
    return {"distance": dist / max(dist[-1], 1e-12) * length, "height": h, "confidence": c}


def step_height(hm: HeightMap, z: np.ndarray, lower: list[float], upper: list[float]) -> dict[str, float]:
    """Height of ``upper`` above ``lower`` (regions [x0, y0, x1, y1]) as in ISO 5436-1: two
    parallel planes, one through each region (one shared tilt), fitted to the reliable pixels;
    the step is their offset. A tilt of the whole surface does not change it, so it does not
    depend on levelling. Uncertainty: the standard error of the offset in that fit combined
    with the height map's own uncertainty in the two regions (slow errors of the
    reconstruction across the field do not average out within a region)."""
    rows_all, cols_all = [], []
    vals, upper_flag, region_unc = [], [], []
    for flag, box in ((0.0, lower), (1.0, upper)):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        sel = hm.valid[y0:y1, x0:x1]
        yy, xx = np.nonzero(sel)
        if len(yy) < 10:
            raise ValueError("too few reliable pixels in a region")
        rows_all.append(yy + y0)
        cols_all.append(xx + x0)
        vals.append(np.asarray(z)[y0:y1, x0:x1][sel])
        upper_flag.append(np.full(len(yy), flag))
        region_unc.append(float(np.median(hm.uncertainty[y0:y1, x0:x1][sel])))
    yy, xx, v, u = (np.concatenate(a) for a in (rows_all, cols_all, vals, upper_flag))
    A = np.column_stack([np.ones_like(v), xx * hm.sx, yy * hm.sy, u])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    res = v - A @ coef
    dof = max(len(v) - 4, 1)
    cov = np.linalg.pinv(A.T @ A) * float(res @ res) / dof
    fit_se = math.sqrt(max(cov[3, 3], 0.0))
    return {"step": float(coef[3]), "uncertainty": float(math.sqrt(fit_se**2 + (region_unc[0] ** 2 + region_unc[1] ** 2) / 2)),
            "fit_se": fit_se, "residual_rms": float(res.std()),
            "slope_x": float(coef[1]), "slope_y": float(coef[2]), "pixels": int(len(v))}  # fmt: skip


# -- pictures and files --------------------------------------------------------------------------


def colour(z: np.ndarray, mask: np.ndarray | None = None, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """An RGB picture of a height map (dark blue low .. yellow high), low-confidence pixels grey."""
    v = np.asarray(z, np.float64)
    good = np.isfinite(v) if mask is None else mask & np.isfinite(v)
    lo = float(np.percentile(v[good], 1)) if lo is None and good.any() else (lo or 0.0)
    hi = float(np.percentile(v[good], 99)) if hi is None and good.any() else (hi or 1.0)
    t = np.clip((v - lo) / max(hi - lo, 1e-30), 0, 1)
    stops = np.array([[0.13, 0.11, 0.42], [0.13, 0.47, 0.71], [0.27, 0.75, 0.55], [0.99, 0.91, 0.15]])  # dark blue, blue, green, yellow
    pos = t * (len(stops) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    f = (pos - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    if mask is not None:
        rgb[~mask] = rgb[~mask] * 0.25 + 0.45
    return (rgb * 255 + 0.5).astype(np.uint8)


def render_3d(z: np.ndarray, *, azimuth: float = 225.0, elevation: float = 35.0, exaggeration: float = 1.0, size: int = 600,
              mask: np.ndarray | None = None, sx: float = 1.0, sy: float = 1.0) -> np.ndarray:  # fmt: skip
    """A shaded perspective-free 3D view of a height map (RGB), coloured by height: a preview
    for the person, not a measurement. ``exaggeration`` stretches the heights."""
    from scipy import ndimage as ndi

    v = np.nan_to_num(np.asarray(z, np.float64))
    rows, cols = v.shape
    step = max(1, int(math.ceil(max(rows, cols) / 400)))
    v = v[::step, ::step]
    m = None if mask is None else mask[::step, ::step]
    r, c = v.shape
    hz = (v - v.min()) / (max(c * sx, r * sy) * step) * exaggeration  # height in units of the field width
    yy, xx = np.mgrid[:r, :c].astype(float)
    X, Y = xx / max(r, c) - c / max(r, c) / 2, yy / max(r, c) - r / max(r, c) / 2
    az, el = math.radians(azimuth), math.radians(elevation)
    # rotate about the vertical axis, then tilt the view down by the elevation
    xr = X * math.cos(az) - Y * math.sin(az)
    yr = X * math.sin(az) + Y * math.cos(az)
    sxs = xr
    sys_ = -(yr * math.sin(el)) - hz * math.cos(el)
    depth = yr * math.cos(el) - hz * math.sin(el)
    # shading: light from the upper left
    gy, gx = np.gradient(hz * max(r, c))
    nrm = np.sqrt(gx**2 + gy**2 + 1)
    light = np.array([-0.5, -0.6, 0.62])
    shade = np.clip((-gx * light[0] - gy * light[1] + light[2]) / nrm, 0.15, 1.0)
    col = colour(v, m).astype(float) / 255 * (0.35 + 0.65 * shade[..., None])
    W = size
    u = ((sxs - sxs.min()) / max(np.ptp(sxs), 1e-12) * (W * 0.9) + W * 0.05).astype(int)
    h_img = int(W * max(np.ptp(sys_) / max(np.ptp(sxs), 1e-12), 0.3)) + 2
    vv = ((sys_ - sys_.min()) / max(np.ptp(sys_), 1e-12) * (h_img * 0.9) + h_img * 0.05).astype(int)
    img = np.full((h_img + 1, W + 1, 3), 1.0)
    order = np.argsort(-depth.ravel())  # far first; nearer points overwrite
    uu, vv2, dd = u.ravel()[order], vv.ravel()[order], depth.ravel()[order]
    cc = col.reshape(-1, 3)[order]
    for du in (-1, 0, 1):  # splat 3x3 so steep, stretched parts have no holes
        for dv in (-1, 0, 1):
            img[np.clip(vv2 + dv, 0, h_img), np.clip(uu + du, 0, W)] = cc
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_tiff(hm: HeightMap, path: str | Path, z: np.ndarray | None = None, meta: dict[str, Any] | None = None) -> Path:
    """Height (float32), confidence and uncertainty as a 3-page TIFF, with the scale in the description."""
    import tifffile

    path = Path(path)
    desc = {"vexel_height_map": {"unit": hm.unit, "pixel_size": hm.pixel_size, "pixel_size_y": hm.pixel_size_y, "pages": ["height", "confidence", "uncertainty"],
                                 "min_confidence": hm.min_confidence, "pairs": hm.pairs, **(meta or {})}}  # fmt: skip
    stack = np.stack([hm.z if z is None else z, hm.confidence, hm.uncertainty]).astype(np.float32)
    tifffile.imwrite(path, stack, description=json.dumps(desc), metadata=None)
    return path


def mesh(hm: HeightMap, z: np.ndarray | None = None, max_side: int = 400, exaggeration: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (x, y, z; metres or pixels) and triangles of the surface; unreliable cells left out."""
    v = hm.z if z is None else z
    rows, cols = v.shape
    step = max(1, int(math.ceil(max(rows, cols) / max_side)))
    vs = v[::step, ::step]
    ok = hm.valid[::step, ::step]
    r, c = vs.shape
    yy, xx = np.mgrid[:r, :c].astype(float)
    verts = np.stack([xx.ravel() * step * hm.sx, -yy.ravel() * step * hm.sy, np.nan_to_num(vs).ravel() * exaggeration], 1)
    idx = np.arange(r * c).reshape(r, c)
    a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    good = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    tri = np.concatenate([np.stack([a[good], cc[good], b[good]], 1), np.stack([b[good], cc[good], d[good]], 1)])
    return verts, tri


def save_mesh(hm: HeightMap, path: str | Path, z: np.ndarray | None = None, max_side: int = 400, exaggeration: float = 1.0) -> Path:
    """The surface as STL (binary) or OBJ, by the file's suffix."""
    import struct

    path = Path(path)
    verts, tri = mesh(hm, z, max_side, exaggeration)
    if path.suffix.lower() == ".obj":
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Vexel height map ({hm.unit})\n")
            fh.writelines(f"v {x:.6g} {y:.6g} {zz:.6g}\n" for x, y, zz in verts)
            fh.writelines(f"f {i + 1} {j + 1} {k + 1}\n" for i, j, k in tri)
        return path
    p = verts[tri]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = nrm, p
    with path.open("wb") as fh:
        fh.write(b"Vexel height map".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tri)))
        fh.write(rec.tobytes())
    return path


__all__ = ["HeightMap", "colour", "level", "match_pair", "mesh", "profile", "reconstruct", "render_3d", "roughness", "save_mesh", "save_tiff",
           "step_height"]  # fmt: skip


# -- kept in the measurement record --------------------------------------------------------------

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402


class TiltView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str  # path relative to the reference image's folder (or absolute)
    sha256: str = ""
    tilt: float = 0.0  # stage tilt, degrees


class Tilt3DSpec(BaseModel):
    """A tilt reconstruction, in the record of its reference image (the image the height map
    is drawn on): the other views, every angle, the settings, and what was measured on it."""

    model_config = ConfigDict(extra="forbid")

    reference_tilt: float = 0.0
    views: list[TiltView] = Field(default_factory=list)
    axis: float = 0.0  # tilt axis direction in the image, degrees counter-clockwise from horizontal
    attachment: float = 3.0
    smooth: float = 1.5
    min_confidence: float = 0.3
    level: int = 1  # 0: none, 1: plane, 2: quadric
    level_region: list[float] | None = None
    profiles: list[list[list[float]]] = Field(default_factory=list)  # [[x0, y0], [x1, y1]]
    steps: list[dict[str, list[float]]] = Field(default_factory=list)  # {"lower": box, "upper": box}


def stage_tilt(path: str | Path) -> float | None:
    """The stage tilt recorded by the instrument, degrees (None when the file does not say)."""
    from vexel.packs.sem.metadata import read_metadata

    try:
        md = read_metadata(path)
    except (OSError, ValueError):
        return None
    return float(md.tilt) if md is not None and md.tilt is not None else None


def load_views(spec: Tilt3DSpec, reference: str | Path, use_ocr: bool = False) -> tuple[list[np.ndarray], list[float], Any, list[str]]:
    """The micrographs (data bars cropped) of the reference and of every view, their tilts,
    the reference's calibration, and warnings (a view changed since it was chosen...)."""
    from vexel.packs.sem import calibrate as C

    reference = Path(reference)
    cal = C.for_file(reference, use_ocr=use_ocr)
    images, tilts, warnings = [C.micrograph(reference, cal)], [spec.reference_tilt], []
    for v in spec.views:
        p = Path(v.file)
        p = p if p.is_absolute() else reference.parent / p
        if not p.is_file():
            raise FileNotFoundError(f"tilt view not found: {p}")
        if v.sha256 and C.file_sha256(p) != v.sha256:
            warnings.append(f"{p.name} 在选定后被改动过")
        vc = C.for_file(p, use_ocr=use_ocr)
        img = C.micrograph(p, vc)
        if img.shape != images[0].shape:
            raise ValueError(f"{p.name} 的大小（{img.shape[1]}×{img.shape[0]}）与参考图不同")
        if cal.ok and vc.ok and abs(vc.pixel_size / cal.pixel_size - 1) > 0.01:
            warnings.append(f"{p.name} 的像素尺寸与参考图相差 {vc.pixel_size / cal.pixel_size - 1:+.1%}：放大倍数不同？")
        images.append(img)
        tilts.append(v.tilt)
    return images, tilts, cal, warnings


@dataclass
class Tilt3DResult:
    hm: HeightMap
    z: np.ndarray  # levelled heights
    roughness: dict[str, float]
    profiles: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    warnings: list[str]


def run_spec(spec: Tilt3DSpec, reference: str | Path, *, use_ocr: bool = False) -> Tilt3DResult:
    images, tilts, cal, warnings = load_views(spec, reference, use_ocr)
    hm = reconstruct(images, tilts, axis=spec.axis, pixel_size=cal.pixel_size if cal.ok else None,
                     pixel_size_y=cal.pixel_size_y if cal.ok else None, min_confidence=spec.min_confidence,
                     attachment=spec.attachment, smooth=spec.smooth)  # fmt: skip
    return analyse(hm, spec, warnings + hm.notes)


def analyse(hm: HeightMap, spec: Tilt3DSpec, warnings: list[str] | None = None) -> Tilt3DResult:
    z = level(hm, spec.level, spec.level_region) if spec.level else hm.z - np.nanmedian(hm.z[hm.valid]) if hm.valid.any() else hm.z
    profs = []
    for a, b in spec.profiles:
        pr = profile(hm, z, tuple(a), tuple(b))
        profs.append({"from": a, "to": b, **{k: v.tolist() for k, v in pr.items()}})
    steps = []
    for st in spec.steps:
        try:
            steps.append({**st, **step_height(hm, z, st["lower"], st["upper"])})
        except ValueError as e:
            steps.append({**st, "error": str(e)})
    return Tilt3DResult(hm, z, roughness(z, hm.valid), profs, steps, list(warnings or []))
