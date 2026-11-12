"""Preprocessing operators."""

from __future__ import annotations

from typing import Literal

import dask.array as da
import numpy as np
from pydantic import Field
from scipy import ndimage as ndi
from skimage import morphology

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op


class StaticBackgroundParams(Params):
    mode: Literal["absdiff", "subtract", "divide"] = Field(
        "absdiff", description="absdiff: |I-B| (objects of either polarity become bright); subtract: I-B; divide: I/B"
    )
    max_frames: int = Field(64, ge=3, description="frames (evenly spaced) used for the temporal median")


@register_op(
    "preprocess.static_background",
    scope=Scope.WINDOW,
    inputs={"image": Kind.IMAGE},
    outputs={"image": Kind.IMAGE},
    params=StaticBackgroundParams,
)
def static_background(params: StaticBackgroundParams, ctx: Context, *, image: da.Array) -> OpResult:
    """Remove the static scene (channel walls, illumination) using a temporal median projection."""
    n = image.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, params.max_frames)).round().astype(int))
    sample = image[idx].astype(np.float32).compute()
    background = np.median(sample, axis=0).astype(np.float32)
    frames = image.astype(np.float32)
    if params.mode == "absdiff":
        out = abs(frames - background)
    elif params.mode == "subtract":
        out = frames - background
    else:
        out = frames / np.maximum(background, 1e-6)
    qc = {
        "frames_sampled": float(len(idx)),
        "background_mean": float(background.mean()),
        # How much sampled frames deviate from the model: high values mean the scene is
        # not static or objects cover a pixel more than half of the time.
        "residual_mean": float(np.abs(sample - background).mean()),
    }
    return OpResult({"image": out.astype(np.float32)}, qc)


class GaussianParams(Params):
    sigma: float = Field(1.0, gt=0, description="standard deviation in pixels")


@register_op(
    "preprocess.gaussian",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE},
    outputs={"image": Kind.IMAGE},
    params=GaussianParams,
)
def gaussian(params: GaussianParams, ctx: Context, *, image: np.ndarray) -> OpResult:
    """Gaussian smoothing."""
    return OpResult({"image": ndi.gaussian_filter(image.astype(np.float32), params.sigma)})


class MedianParams(Params):
    radius: int = Field(1, ge=1, description="filter radius in pixels (window = 2*radius+1)")


@register_op(
    "preprocess.median",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE},
    outputs={"image": Kind.IMAGE},
    params=MedianParams,
)
def median(params: MedianParams, ctx: Context, *, image: np.ndarray) -> OpResult:
    """Edge-preserving median denoising."""
    return OpResult({"image": ndi.median_filter(image, size=2 * params.radius + 1)})


class BackgroundCorrectionParams(Params):
    method: Literal["tophat", "gaussian"] = "tophat"
    radius: float = Field(25.0, gt=0, description="tophat disk radius, or Gaussian sigma, in pixels; larger than the objects")
    polarity: Literal["bright", "dark"] = Field("bright", description="whether objects are brighter or darker than background")


@register_op(
    "preprocess.background_correction",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE},
    outputs={"image": Kind.IMAGE},
    params=BackgroundCorrectionParams,
)
def background_correction(params: BackgroundCorrectionParams, ctx: Context, *, image: np.ndarray) -> OpResult:
    """Per-frame background correction (white/black top-hat or large-scale Gaussian subtraction)."""
    img = image.astype(np.float32)
    if params.method == "tophat":
        footprint = morphology.disk(int(round(params.radius)))
        out = morphology.white_tophat(img, footprint) if params.polarity == "bright" else morphology.black_tophat(img, footprint)
    else:
        bg = ndi.gaussian_filter(img, params.radius)
        out = img - bg if params.polarity == "bright" else bg - img
    return OpResult({"image": out.astype(np.float32)})
