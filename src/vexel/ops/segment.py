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
        labels = morphology.remove_small_objects(labels, max_size=params.min_area - 1)
    labels, _, _ = segmentation.relabel_sequential(labels)

    qc = {
        "threshold": float(abs(thr)) if np.isfinite(thr) else 0.0,
        "n_objects": float(labels.max()),
        "foreground_fraction": float((labels > 0).mean()),
    }
    return OpResult({"labels": labels.astype(np.int32)}, qc)
