"""Pixel classification trained from brush strokes (segmentation for images that
thresholds and circle detection do not handle).

A person paints a few strokes on example frames, one class per brush (class 1 = the
objects, 2 = background, further classes allowed). Training data is a small file:
for every painted frame, the image patch around the strokes and the stroke classes,
plus the intensity range used to normalise images. A random forest on multi-scale
pixel features (intensity, edges, texture) is fitted with a fixed seed whenever the
file is first used, so recipe + training file always give the same segmentation; no
fitted model is stored. The recipe records the file's SHA-256 and refuses a changed file.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from skimage.feature import multiscale_basic_features, peak_local_max

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op

FEATURES = {"sigma_min": 0.7, "sigma_max": 4.0, "num_sigma": 4}
PATCH_MARGIN = 24  # pixels kept around strokes: features near the patch edge stay valid


# -- training data ------------------------------------------------------------------------------


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_training(path: str | Path, frames: dict[int, np.ndarray], strokes: dict[int, np.ndarray], classes: list[str]) -> str:
    """Write a training file from whole frames and their stroke images (0 = unpainted,
    1..k = class). Only patches around the strokes are kept. Returns the SHA-256."""
    painted = [t for t, s in strokes.items() if (s > 0).any()]
    if not painted:
        raise ValueError("no strokes painted")
    stack = np.concatenate([frames[t][strokes[t] > 0].ravel() for t in painted]).astype(np.float64)
    all_px = np.concatenate([frames[t].ravel() for t in painted]).astype(np.float64)
    lo, hi = np.percentile(all_px, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(stack.min()), float(stack.max()) + 1.0
    arrays: dict[str, Any] = {"classes": np.array(classes), "range": np.array([lo, hi], dtype=np.float64),
                              "frames": np.array(painted, dtype=np.int64)}  # fmt: skip
    for t in painted:
        ys, xs = np.nonzero(strokes[t])
        y0, y1 = max(0, ys.min() - PATCH_MARGIN), min(frames[t].shape[0], ys.max() + PATCH_MARGIN + 1)
        x0, x1 = max(0, xs.min() - PATCH_MARGIN), min(frames[t].shape[1], xs.max() + PATCH_MARGIN + 1)
        arrays[f"image_{t}"] = frames[t][y0:y1, x0:x1].astype(np.float32)
        arrays[f"strokes_{t}"] = strokes[t][y0:y1, x0:x1].astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    return file_digest(path)


def _features(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    img = (image.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return multiscale_basic_features(img, intensity=True, edges=True, texture=True, **FEATURES)


class Trained:
    def __init__(self, model: Any, lo: float, hi: float, classes: list[str]) -> None:
        self.model, self.lo, self.hi, self.classes = model, lo, hi, classes

    def probabilities(self, image: np.ndarray) -> np.ndarray:
        """(H, W, n_classes) class probabilities; columns follow class numbers 1..k."""
        feats = _features(image, self.lo, self.hi)
        proba = self.model.predict_proba(feats.reshape(-1, feats.shape[-1]))
        out = np.zeros((proba.shape[0], len(self.classes)), dtype=np.float32)
        for col, cls in enumerate(self.model.classes_):
            out[:, int(cls) - 1] = proba[:, col]
        return out.reshape(*image.shape, len(self.classes))


def train(path: str | Path | None = None, *, arrays: dict[str, Any] | None = None, seed: int = 0, n_trees: int = 60) -> Trained:
    """Fit the classifier on a training file (or the same arrays held in memory)."""
    from sklearn.ensemble import RandomForestClassifier

    data = arrays if arrays is not None else dict(np.load(path, allow_pickle=False))
    lo, hi = (float(v) for v in data["range"])
    xs, ys = [], []
    for t in np.asarray(data["frames"]).tolist():
        feats = _features(np.asarray(data[f"image_{t}"]), lo, hi)
        s = np.asarray(data[f"strokes_{t}"])
        xs.append(feats[s > 0])
        ys.append(s[s > 0])
    x, y = np.concatenate(xs), np.concatenate(ys)
    if np.unique(y).size < 2:
        raise ValueError("paint at least two classes (for example objects and background)")
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=1)
    model.fit(x, y)
    return Trained(model, lo, hi, [str(c) for c in np.asarray(data["classes"]).tolist()])


_CACHE: dict[tuple[str, int], Trained] = {}
_LOCK = threading.Lock()


def trained(path: str, sha256: str, seed: int) -> Trained:
    """The classifier for a training file, fitted once per process and checked against
    the digest the recipe recorded."""
    key = (sha256, seed)
    with _LOCK:
        if key not in _CACHE:
            if not Path(path).exists():
                raise FileNotFoundError(f"segment.pixel_classifier: training file not found: {path}")
            actual = file_digest(path)
            if actual != sha256:
                raise ValueError(f"segment.pixel_classifier: {path} has changed since the recipe was made (SHA-256 {actual[:12]}…, recipe {sha256[:12]}…)")
            _CACHE[key] = train(path, seed=seed)
        return _CACHE[key]


# -- operator -------------------------------------------------------------------------------------


class PixelClassifierParams(Params):
    training: str = Field(description="training file written by the brush tool (.npz)")
    training_sha256: str = Field(description="SHA-256 of the training file; a changed file is refused")
    object_class: int = Field(1, ge=1, description="class number of the objects (the first brush)")
    probability: float = Field(0.5, gt=0, lt=1, description="minimum object probability of a pixel")
    min_area: int = Field(30, ge=0, description="discard objects smaller than this many pixels")
    opening_radius: int = Field(1, ge=0, description="binary opening radius in pixels; removes specks and thin bridges")
    split_touching: bool = Field(True, description="split touching objects with a distance-transform watershed")
    min_distance: int = Field(8, ge=1, description="minimum distance between object centres for splitting, pixels")


def labels_from_probability(
    prob: np.ndarray, threshold: float = 0.5, min_area: int = 30, opening_radius: int = 1, split_touching: bool = True, min_distance: int = 8
) -> np.ndarray:
    """Objects from an object-probability map: threshold, fill holes, open, drop specks,
    split touching objects with a distance-transform watershed."""
    mask = ndi.binary_fill_holes(prob >= threshold)
    if opening_radius > 0:
        mask = ndi.binary_opening(mask, morphology.disk(opening_radius))
    if min_area > 1:
        mask = morphology.remove_small_objects(mask, max_size=min_area - 1)
    if split_touching and mask.any():
        distance = ndi.distance_transform_edt(mask)
        peaks = peak_local_max(distance, min_distance=min_distance, labels=measure.label(mask), exclude_border=False)
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = segmentation.watershed(-distance, markers, mask=mask)
    else:
        labels = measure.label(mask)
    if min_area > 1 and labels.max() > 0:  # splitting can leave pieces too small to be objects
        sizes = np.bincount(labels.ravel())
        small = np.nonzero(sizes < min_area)[0]
        labels[np.isin(labels, small[small > 0])] = 0
    return segmentation.relabel_sequential(labels)[0].astype(np.int32)


@register_op(
    "segment.pixel_classifier",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE},
    outputs={"labels": Kind.LABELS, "probability": Kind.IMAGE},
    params=PixelClassifierParams,
)
def pixel_classifier(params: PixelClassifierParams, ctx: Context, *, image: np.ndarray) -> OpResult:
    """Objects from a pixel classifier trained on brush strokes: the object-class
    probability is thresholded, holes filled, specks removed, touching objects split."""
    clf = trained(params.training, params.training_sha256, ctx.seed)
    if params.object_class > len(clf.classes):
        raise ValueError(f"segment.pixel_classifier: object_class {params.object_class} but the training has {len(clf.classes)} classes")
    prob = clf.probabilities(image)[..., params.object_class - 1]
    labels = labels_from_probability(prob, params.probability, params.min_area, params.opening_radius,
                                     params.split_touching, params.min_distance)  # fmt: skip
    qc = {"n_objects": float(labels.max()), "foreground_fraction": float((labels > 0).mean()), "probability_mean": float(prob.mean())}
    return OpResult({"labels": labels.astype(np.int32), "probability": prob.astype(np.float32)}, qc)


def use_in_recipe(recipe: Any, training: str | Path, sha256: str, **params: Any) -> Any:
    """A copy of ``recipe`` whose segmentation (the step producing the review labels, or a
    step with id ``seg``) is the pixel classifier trained from ``training``. The step keeps
    its id, so later steps and overrides of other steps still apply."""
    from vexel.engine.recipe import Recipe

    data = recipe.model_dump(mode="json")
    target = (recipe.review.labels.split(".")[0] if recipe.review else None) or "seg"
    for step in data["steps"]:
        if step["id"] == target:
            image = step["inputs"].get("image", "source")
            if image != "source" and not image.startswith("source."):
                image = "source"  # classify the recorded image, not a processed one
            step.update(op="segment.pixel_classifier", inputs={"image": image},
                        params={"training": str(Path(training).resolve()), "training_sha256": sha256, **params}, qc=[])  # fmt: skip
            break
    else:
        raise ValueError(f"recipe has no step {target!r} to replace with the pixel classifier")
    data["description"] = (data.get("description") or "") + " Segmentation: pixel classifier trained from brush strokes."
    return Recipe.model_validate(data)
