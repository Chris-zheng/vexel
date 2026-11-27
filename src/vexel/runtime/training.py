"""Human-in-the-loop active learning and local multi-class model training.

Turns brush strokes, interactive prompt masks, and human review corrections
(``corrections.json``) into a validated, SHA-256-fingerprinted local model artifact with
hold-out evaluation metrics (IoU, F1, per-class accuracy) and an automatic
:class:`vexel.plugins.ModelCard` ready for registration into any Vexel workflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vexel.ops.classify import _features, save_training, train
from vexel.plugins import Dataset, ModelCard, models_dir


@dataclass(frozen=True)
class HoldoutMetrics:
    n_train_pixels: int
    n_val_pixels: int
    accuracy: float
    macro_f1: float
    mean_iou: float
    per_class_iou: dict[str, float]


@dataclass(frozen=True)
class TrainedArtifact:
    card: ModelCard
    training_path: Path
    metrics: HoldoutMetrics
    metadata_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "card": self.card.to_json(),
            "training_path": str(self.training_path),
            "metadata_path": str(self.metadata_path),
            "metrics": asdict(self.metrics),
        }


def evaluate_holdout(
    frames: dict[int, np.ndarray],
    strokes: dict[int, np.ndarray],
    classes: list[str],
    *,
    val_fraction: float = 0.25,
    seed: int = 0,
    n_trees: int = 60,
) -> HoldoutMetrics:
    """Stratified pixel hold-out evaluation before promoting a locally trained model."""
    from sklearn.ensemble import RandomForestClassifier

    painted = [t for t, s in strokes.items() if (s > 0).any()]
    if not painted:
        raise ValueError("no annotated strokes available for evaluation")
    all_px = np.concatenate([frames[t].ravel() for t in painted]).astype(np.float64)
    lo, hi = np.percentile(all_px, (0.5, 99.5))
    xs, ys = [], []
    for t in painted:
        feats = _features(frames[t], lo, hi)
        s = strokes[t]
        xs.append(feats[s > 0])
        ys.append(s[s > 0])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    rng = np.random.default_rng(seed)

    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        if len(idx) < 2:
            train_idx.append(idx)
            val_idx.append(idx)
        else:
            val_idx.append(idx[:n_val])
            train_idx.append(idx[n_val:])
    ti = np.concatenate(train_idx)
    vi = np.concatenate(val_idx)

    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=1)
    clf.fit(x[ti], y[ti])
    pred = clf.predict(x[vi])
    truth = y[vi]

    acc = float((pred == truth).mean())
    ious: dict[str, float] = {}
    f1s: list[float] = []
    for idx_cls, name in enumerate(classes, start=1):
        tp = int(((pred == idx_cls) & (truth == idx_cls)).sum())
        fp = int(((pred == idx_cls) & (truth != idx_cls)).sum())
        fn = int(((pred != idx_cls) & (truth == idx_cls)).sum())
        if tp + fp + fn == 0:
            continue
        iou = tp / max(1, tp + fp + fn)
        f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
        ious[name] = round(float(iou), 4)
        f1s.append(float(f1))

    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return HoldoutMetrics(
        n_train_pixels=int(len(ti)),
        n_val_pixels=int(len(vi)),
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        mean_iou=round(mean_iou, 4),
        per_class_iou=ious,
    )


def train_and_publish_local_model(
    model_id: str,
    version: str,
    frames: dict[int, np.ndarray],
    strokes: dict[int, np.ndarray],
    classes: list[str],
    *,
    task: str = "segment.classes",
    description: str = "Locally fine-tuned multi-class pixel classifier from Vexel annotations & review corrections.",
    seed: int = 0,
    min_iou: float = 0.0,
) -> TrainedArtifact:
    """Save verified training archive, compute hold-out metrics, and publish a ModelCard."""
    metrics = evaluate_holdout(frames, strokes, classes, seed=seed)
    if metrics.mean_iou < min_iou:
        raise ValueError(f"hold-out mean IoU {metrics.mean_iou:.3f} is below required gate {min_iou:.3f}")

    dest = models_dir() / model_id / version
    dest.mkdir(parents=True, exist_ok=True)
    training_path = dest / "training.npz"
    sha256 = save_training(training_path, frames, strokes, classes)
    # Verify deterministic fit succeeds cleanly before writing card metadata
    train(training_path, seed=seed)

    card = ModelCard(
        id=model_id,
        version=version,
        task=task,
        framework="sklearn",
        license="Local-User-Owned",
        training_data=(Dataset(name=f"Local Vexel annotations ({len(frames)} frames)", license="Local-User-Owned"),),
        file="training.npz",
        sha256=sha256,
        input={"classes": classes, "normalise": "percentile-0.5-99.5"},
        output={"mean_iou": metrics.mean_iou, "macro_f1": metrics.macro_f1, "accuracy": metrics.accuracy},
        description=description,
    )
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "card": card.to_json(),
        "metrics": asdict(metrics),
        "seed": seed,
    }
    meta_path = dest / "model_card.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return TrainedArtifact(card=card, training_path=training_path, metrics=metrics, metadata_path=meta_path)


def hash_corrections_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
