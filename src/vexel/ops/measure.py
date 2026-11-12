"""Measurement operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op

_PROPS = ("label", "centroid", "area", "perimeter", "axis_major_length", "axis_minor_length", "eccentricity", "bbox")
_COLUMNS = [
    "label",
    "centroid_y_px",
    "centroid_x_px",
    "area_px",
    "diameter_px",
    "centroid_y",
    "centroid_x",
    "area",
    "diameter",
    "axis_major",
    "axis_minor",
    "eccentricity",
    "circularity",
    "touches_border",
    "mean_intensity",
]


@register_op(
    "measure.regionprops",
    scope=Scope.FRAME,
    inputs={"labels": Kind.LABELS, "intensity": Kind.IMAGE},
    optional_inputs=("intensity",),
    outputs={"table": Kind.TABLE},
)
def regionprops(params: Params, ctx: Context, *, labels: np.ndarray, intensity: np.ndarray | None = None) -> OpResult:
    """Per-object position, size and shape. ``*_px`` columns are in pixels; the others are in
    the calibrated length unit. ``diameter`` is the equivalent-area diameter."""
    if labels.max() == 0:
        return OpResult({"table": pd.DataFrame(columns=_COLUMNS)}, {"n_objects": 0.0})
    sy, sx = ctx.spacing_yx
    props = _PROPS + (("intensity_mean",) if intensity is not None else ())
    raw = measure.regionprops_table(labels, intensity_image=intensity, properties=props)
    area_px = raw["area"].astype(np.float64)
    perimeter = np.maximum(raw["perimeter"], 1e-9)
    h, w = labels.shape
    # Length-like shape descriptors are scaled with the mean in-plane spacing; exact for
    # square pixels, an approximation otherwise.
    s = float(np.sqrt(sy * sx))
    table = pd.DataFrame(
        {
            "label": raw["label"].astype(np.int64),
            "centroid_y_px": raw["centroid-0"],
            "centroid_x_px": raw["centroid-1"],
            "area_px": area_px,
            "diameter_px": np.sqrt(4.0 * area_px / np.pi),
            "centroid_y": raw["centroid-0"] * sy,
            "centroid_x": raw["centroid-1"] * sx,
            "area": area_px * sy * sx,
            "diameter": np.sqrt(4.0 * area_px * sy * sx / np.pi),
            "axis_major": raw["axis_major_length"] * s,
            "axis_minor": raw["axis_minor_length"] * s,
            "eccentricity": raw["eccentricity"],
            "circularity": np.minimum(1.0, 4.0 * np.pi * area_px / perimeter**2),
            "touches_border": (raw["bbox-0"] == 0) | (raw["bbox-1"] == 0) | (raw["bbox-2"] == h) | (raw["bbox-3"] == w),
            "mean_intensity": raw["intensity_mean"] if intensity is not None else np.nan,
        },
        columns=_COLUMNS,
    )
    return OpResult({"table": table}, {"n_objects": float(len(table))})
