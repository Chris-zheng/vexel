"""Segmentation operators. Method A: threshold + marker-controlled watershed."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op


class ThresholdWatershedParams(Params):
    smooth_sigma: float = Field(1.0, ge=0, description="Gaussian smoothing before thresholding, pixels; 0 disables")
    method: Literal["otsu", "triangle", "fixed"] = "otsu"
    threshold: float | None = Field(None, description="threshold value when method='fixed'")
    min_threshold: float = Field(
        0.0, description="floor for automatic thresholds, in image units; keeps empty frames from being segmented as noise"
    )
    polarity: Literal["bright", "dark"] = "bright"
    fill_holes: bool = Field(True, description="fill enclosed regions (turns ring-like outlines into solid objects)")
    opening_radius: int = Field(1, ge=0, description="binary opening radius in pixels; removes specks and thin bridges")
    min_area: int = Field(50, ge=0, description="discard objects smaller than this many pixels")
    split_touching: bool = Field(True, description="split touching objects with a distance-transform watershed")
    min_distance: int = Field(8, ge=1, description="minimum distance between object centres for splitting, pixels")
    clear_border: bool = Field(False, description="remove objects touching the image border")


@register_op(
    "segment.threshold_watershed",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE},
    outputs={"labels": Kind.LABELS},
    params=ThresholdWatershedParams,
)
def threshold_watershed(params: ThresholdWatershedParams, ctx: Context, *, image: np.ndarray) -> OpResult:
    """Threshold, clean up, and split touching objects with a marker-controlled watershed."""
    img = image.astype(np.float32)
    if params.polarity == "dark":
        img = -img
    if params.smooth_sigma > 0:
        img = ndi.gaussian_filter(img, params.smooth_sigma)

    if params.method == "fixed":
        if params.threshold is None:
            raise ValueError("segment.threshold_watershed: method='fixed' requires 'threshold'")
        thr = params.threshold if params.polarity == "bright" else -params.threshold
    else:
        if float(img.max()) == float(img.min()):
            thr = float("inf")
        elif params.method == "otsu":
            thr = float(filters.threshold_otsu(img))
        else:
            thr = float(filters.threshold_triangle(img))
        thr = max(thr, params.min_threshold)

    mask = img > thr
    if params.fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if params.opening_radius > 0:
        mask = ndi.binary_opening(mask, morphology.disk(params.opening_radius))
    if params.min_area > 1:
        mask = morphology.remove_small_objects(mask, max_size=params.min_area - 1)

    if params.split_touching and mask.any():
        distance = ndi.distance_transform_edt(mask)
        components = measure.label(mask)
        peaks = peak_local_max(distance, min_distance=params.min_distance, labels=components, exclude_border=False)
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = segmentation.watershed(-distance, markers, mask=mask)
    else:
        labels = measure.label(mask)

    if params.clear_border:
        labels = segmentation.clear_border(labels)
    if params.min_area > 1 and labels.max() > 0:
        if labels.max() == 1:  # a single label is a mask; skimage warns when given it as labels
            labels = labels * morphology.remove_small_objects(labels > 0, max_size=params.min_area - 1)
        else:
            labels = morphology.remove_small_objects(labels, max_size=params.min_area - 1)
    labels, _, _ = segmentation.relabel_sequential(labels)

    qc = {
        "threshold": float(abs(thr)) if np.isfinite(thr) else 0.0,
        "n_objects": float(labels.max()),
        "foreground_fraction": float((labels > 0).mean()),
    }
    return OpResult({"labels": labels.astype(np.int32)}, qc)


# -- method B: circle detection ------------------------------------------------------------


class CirclesParams(Params):
    min_diameter_px: float | None = Field(
        None, gt=0, description="smallest droplet diameter to look for, pixels; None = automatic (from a coarse first pass)"
    )
    max_diameter_px: float | None = Field(None, gt=0, description="largest droplet diameter to look for, pixels; None = automatic")
    auto_max_diameter_px: float = Field(400.0, gt=4, description="upper bound of the automatic size estimate, pixels")
    edge_sigma: float = Field(1.0, gt=0, description="Gaussian sigma of the Canny edge detector, pixels")
    motion_threshold: float = Field(
        8.0, ge=0, description="edges count only where the contrast image exceeds this (ignores static walls and text); "
        "0 = no motion gate, for droplets that do not move (sampled frames, stopped flow, static arrays)"
    )
    min_score: float = Field(0.35, gt=0, le=1, description="minimum fraction of a circle's perimeter that must lie on edges")
    min_separation: float = Field(
        1.5, gt=0, description="minimum centre distance between two droplets, in mean radii (2 = touching, no overlap)"
    )
    size_tolerance: float | None = Field(
        0.3, gt=0, description="drop circles whose radius differs from the frame's median by more than this fraction; None keeps all"
    )
    refine: bool = Field(True, description="refine each radius to the outer edge of the droplet in the contrast image")


def _hough(edges: np.ndarray, radii: np.ndarray, min_score: float, min_dist: int, max_peaks: int):
    """Circle candidates: for every pixel the best-supported radius, then local maxima of
    that best score at least ``min_dist`` apart (one maximum filter instead of one per
    radius)."""
    from skimage.transform import hough_circle

    acc = hough_circle(edges, radii, normalize=True)
    best = acc.max(axis=0)
    which = acc.argmax(axis=0)
    peaks = peak_local_max(best, min_distance=max(1, min_dist), threshold_abs=min_score, num_peaks=max_peaks, exclude_border=False)
    if len(peaks) == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    cy, cx = peaks[:, 0], peaks[:, 1]
    return best[cy, cx].astype(float), cx.astype(float), cy.astype(float), radii[which[cy, cx]].astype(float)


def _suppress(scores, cx, cy, r, min_separation: float) -> np.ndarray:
    """Greedy non-maximum suppression: strongest circles first; a circle is dropped when
    its centre is closer than ``min_separation`` mean radii to an accepted one."""
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(np.hypot(cx[i] - cx[j], cy[i] - cy[j]) > min_separation * (r[i] + r[j]) / 2 for j in keep):
            keep.append(int(i))
    return np.array(keep, dtype=int)


def _refine_centre(edge_yx: np.ndarray, x: float, y: float, r: float) -> tuple[float, float]:
    """Least-squares (Kasa) circle fit to the edge pixels near a detected circle; Hough
    centres are whole pixels. Iterated with a narrowing band so that a droplet's inner
    and outer edges are not mixed unevenly. Kept only when the edge points surround the
    centre and the fit moves it less than a third of the radius."""
    x0, y0 = x, y
    for band in (max(2.0, 0.2 * r), max(1.5, 0.12 * r), 1.5):
        d = np.hypot(edge_yx[:, 0] - y, edge_yx[:, 1] - x)
        pts = edge_yx[np.abs(d - r) <= band]
        if len(pts) < 12:
            break
        angles = np.arctan2(pts[:, 0] - y, pts[:, 1] - x)
        if np.unique(np.floor((angles + np.pi) / (np.pi / 4))).size < 6:  # needs 6 of 8 octants
            break
        py, px = pts[:, 0].astype(float), pts[:, 1].astype(float)
        a = np.column_stack([px, py, np.ones_like(px)])
        sol, *_ = np.linalg.lstsq(a, px**2 + py**2, rcond=None)
        nx, ny = sol[0] / 2, sol[1] / 2
        nr2 = sol[2] + nx**2 + ny**2
        if nr2 <= 0 or np.hypot(nx - x0, ny - y0) > r / 3:
            break
        x, y, r = float(nx), float(ny), float(np.sqrt(nr2))
    return x, y


def _refine_radius(contrast: np.ndarray, x: float, y: float, r: float) -> float:
    """Outer edge of the droplet: the radius where the angular median of the contrast
    profile falls fastest (robust to neighbours touching on a few sides)."""
    radii = np.arange(max(1.0, 0.6 * r), 1.5 * r, 0.25)
    angles = np.linspace(0, 2 * np.pi, max(24, int(2 * np.pi * r)), endpoint=False)
    yy = y + radii[:, None] * np.sin(angles)[None, :]
    xx = x + radii[:, None] * np.cos(angles)[None, :]
    inside = (yy >= 0) & (yy <= contrast.shape[0] - 1) & (xx >= 0) & (xx <= contrast.shape[1] - 1)
    if inside.mean() < 0.5:
        return r
    samples = ndi.map_coordinates(contrast, [yy.ravel(), xx.ravel()], order=1, mode="nearest").reshape(yy.shape)
    samples = np.where(inside, samples, np.nan)
    profile = np.nanmedian(samples, axis=1)
    slope = np.gradient(profile, radii)
    k = int(np.nanargmin(slope))
    if 0 < k < len(slope) - 1 and np.isfinite(slope[k - 1 : k + 2]).all():  # parabolic sub-sample peak
        a, b, c = slope[k - 1 : k + 2]
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom > 0 else 0.0
        return float(radii[k] + np.clip(offset, -1, 1) * 0.25)
    return float(radii[k])


def _paint(shape: tuple[int, int], cx, cy, r) -> np.ndarray:
    """Disks as labels; a pixel covered by two disks goes to the one it is relatively deeper in."""
    labels = np.zeros(shape, dtype=np.int32)
    best = np.full(shape, np.inf, dtype=np.float32)
    for k, (x, y, rr) in enumerate(zip(cx, cy, r), 1):
        y0, y1 = max(int(y - rr) - 1, 0), min(int(y + rr) + 2, shape[0])
        x0, x1 = max(int(x - rr) - 1, 0), min(int(x + rr) + 2, shape[1])
        if y0 >= y1 or x0 >= x1:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(gy - y, gx - x) / rr
        win, cur = best[y0:y1, x0:x1], labels[y0:y1, x0:x1]
        take = (d <= 1.0) & (d < win)
        win[take] = d[take]
        cur[take] = k
    return labels


@register_op(
    "segment.circles",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE, "contrast": Kind.IMAGE},
    outputs={"labels": Kind.LABELS},
    params=CirclesParams,
)
def circles(params: CirclesParams, ctx: Context, *, image: np.ndarray, contrast: np.ndarray) -> OpResult:
    """Round droplets by circle detection. Edges come from the raw ``image`` (sharp, either
    polarity), restricted to where the ``contrast`` image (e.g. difference from the static
    background) shows something; a Hough transform finds circles, which are de-duplicated,
    checked against the frame's typical size, and refined to the droplet's outer edge.
    Handles droplets that appear as thin dark rings or with a bright centre, which a
    threshold cannot fill."""
    from skimage.feature import canny

    raw = image.astype(np.float32)
    empty_qc = {"n_objects": 0.0, "radius_median_px": 0.0, "score_median": 0.0, "foreground_fraction": 0.0}
    empty = OpResult({"labels": np.zeros(raw.shape, np.int32)}, empty_qc)
    if params.motion_threshold == 0:
        # No motion gate (droplets that do not move: sampled frames, stopped flow, static
        # arrays): edges count everywhere, and outlines are measured on the image's own
        # deviation from its typical level instead of the difference from the background.
        con = np.abs(ndi.gaussian_filter(raw, 0.5) - float(np.median(raw)))
        present = np.ones(raw.shape, dtype=bool)
    else:
        con = ndi.gaussian_filter(contrast.astype(np.float32), 0.5)
        # Where something differs from the static scene. The floor adapts to the noise of
        # the contrast image (mostly background pixels, so its median and MAD describe the noise).
        smooth = ndi.gaussian_filter(con, 1.0)
        med = float(np.median(smooth))
        mad = 1.4826 * float(np.median(np.abs(smooth - med)))
        floor = max(params.motion_threshold, med + 6.0 * mad)
        present = smooth > floor
    if not present.any():
        return empty
    moving = ndi.binary_dilation(present, morphology.disk(3))

    grad = filters.sobel(ndi.gaussian_filter(raw, params.edge_sigma))[moving]
    # Gated, the edge thresholds come from where things move (mostly droplet outlines);
    # ungated, from the whole frame, which is mostly flat background: only its strongest
    # gradients are outlines.
    if params.motion_threshold == 0:
        # The whole frame, mostly flat background whose gradient median is the noise level:
        # an outline is at least the top 5% and 4x that level (2x to continue an edge).
        # Expressed as quantiles, which canny applies on its own gradient scale.
        typical = float(np.median(grad))
        if typical <= 0:
            return empty
        q_low = max(0.85, float(np.mean(grad < 2.0 * typical)))
        q_high = max(0.95, float(np.mean(grad < 4.0 * typical)))
        edges = canny(raw, sigma=params.edge_sigma, low_threshold=min(q_low, q_high), high_threshold=q_high, use_quantiles=True)
    else:
        low, high = np.percentile(grad, 50), np.percentile(grad, 80)
        if high <= 0:
            return empty
        edges = canny(raw, sigma=params.edge_sigma, low_threshold=low, high_threshold=high) & moving

    if params.min_diameter_px is None or params.max_diameter_px is None:
        # Typical size from a coarse fill of the contrast image: rings closed and filled,
        # specks removed. The inscribed radius (distance-transform peaks) measures each
        # droplet even inside a chain of touching ones; broken outlines give smaller
        # peaks, so the upper quartile decides.
        blobs = ndi.binary_fill_holes(ndi.binary_closing(present, morphology.disk(2)))
        blobs = ndi.binary_opening(blobs, morphology.disk(1))
        dist = ndi.distance_transform_edt(blobs)
        peaks = peak_local_max(dist, min_distance=3, threshold_abs=2.5, labels=measure.label(blobs), exclude_border=False)
        if len(peaks) == 0:
            return empty
        typical = float(np.percentile(dist[tuple(peaks.T)], 75))
        typical = min(typical, params.auto_max_diameter_px / 2)
    r_min = params.min_diameter_px / 2 if params.min_diameter_px else 0.7 * typical
    r_max = params.max_diameter_px / 2 if params.max_diameter_px else 1.4 * typical
    radii = np.arange(max(2.0, np.floor(r_min)), np.ceil(r_max) + 0.5, 1.0)
    s, x, y, r = _hough(edges, radii, params.min_score, max(2, int(r_min * params.min_separation / 2)), 500)
    keep = _suppress(s, x, y, r, params.min_separation)
    s, x, y, r = s[keep], x[keep], y[keep], r[keep]

    if params.refine and len(r):
        edge_yx = np.argwhere(edges)
        centres = [_refine_centre(edge_yx, xi, yi, ri) for xi, yi, ri in zip(x, y, r)]
        x = np.array([c[0] for c in centres])
        y = np.array([c[1] for c in centres])
        r = np.array([_refine_radius(con, xi, yi, ri) for xi, yi, ri in zip(x, y, r)])
    if params.size_tolerance is not None and len(r):
        med = float(np.median(r))
        ok = np.abs(r - med) <= params.size_tolerance * med
        s, x, y, r = s[ok], x[ok], y[ok], r[ok]

    labels = _paint(raw.shape, x, y, r)
    qc = {
        "n_objects": float(len(r)),
        "radius_median_px": float(np.median(r)) if len(r) else 0.0,
        "score_median": float(np.median(s)) if len(s) else 0.0,
        "foreground_fraction": float((labels > 0).mean()),
    }
    return OpResult({"labels": labels}, qc)
