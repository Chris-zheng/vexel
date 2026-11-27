"""Verification tests for Stage 3 Trained Model runtime: tiled inference and active learning.

Checks:
1. Cosine-feathered tiled inference matches full-image linear predictions without seam artefacts.
2. Active learning ``evaluate_holdout`` and ``train_and_publish_local_model`` create a
   SHA-256 verified training archive and ModelCard under ``VEXEL_HOME/models``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vexel.runtime.inference import TileSpec, run_tiled_probability
from vexel.runtime.training import evaluate_holdout, train_and_publish_local_model


def test_tiled_inference_seam_free_identity() -> None:
    rng = np.random.default_rng(42)
    img = rng.uniform(0.0, 1.0, size=(300, 340)).astype(np.float32)

    def linear_predictor(patch: np.ndarray) -> np.ndarray:
        return np.clip(0.25 + 0.5 * patch, 0.0, 1.0)

    full = linear_predictor(img)
    tiled = run_tiled_probability(img, linear_predictor, TileSpec(tile_size=128, overlap=32))
    assert tiled.shape == full.shape
    np.testing.assert_allclose(tiled, full, atol=1e-5)


def test_active_learning_holdout_and_model_card_publish(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path))
    img = np.zeros((96, 96), dtype=np.float32)
    img[16:48, 16:48] = 220.0
    img[56:84, 52:84] = 120.0
    strokes = np.zeros((96, 96), dtype=np.uint8)
    strokes[20:44, 20:44] = 1  # bright phase
    strokes[60:80, 56:80] = 2  # mid phase
    strokes[2:12, 2:90] = 3  # background

    metrics = evaluate_holdout({0: img}, {0: strokes}, ["bright_phase", "mid_phase", "background"], seed=0)
    assert metrics.accuracy >= 0.9
    assert metrics.mean_iou >= 0.85

    artifact = train_and_publish_local_model(
        "sem-phase-demo",
        "1.0",
        {0: img},
        {0: strokes},
        ["bright_phase", "mid_phase", "background"],
        min_iou=0.8,
    )
    assert artifact.training_path.exists()
    assert artifact.metadata_path.exists()
    assert artifact.card.ref == "sem-phase-demo@1.0"
    assert len(artifact.card.sha256) == 64
