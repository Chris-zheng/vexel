"""Unified trained-model inference runtime with tiled large-image stitching.

Provides hardware-routed ONNX Runtime execution (CUDA / CoreML-MPS / DirectML / CPU),
SHA-256 weight verification via :mod:`vexel.plugins`, and overlap-tile stitching with
cosine feathering so 4K/8K SEM micrographs and high-resolution microscopy frames run
deterministically on 16 GB CPU-only workstations without tile seam artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vexel.plugins import ModelCard, file_status


@dataclass(frozen=True)
class TileSpec:
    tile_size: int = 512
    overlap: int = 64

    def __post_init__(self) -> None:
        if self.tile_size < 32:
            raise ValueError("tile_size must be >= 32")
        if not (0 <= self.overlap < self.tile_size // 2):
            raise ValueError("overlap must be in [0, tile_size // 2)")


def select_onnx_providers(preferred: str | None = None) -> list[str]:
    """Choose available ONNX Runtime execution providers in priority order."""
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    order = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    if preferred == "cpu":
        return ["CPUExecutionProvider"]
    chosen = [p for p in order if p in avail]
    return chosen or ["CPUExecutionProvider"]


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(1, length - tile), step))
    last = length - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _feather_window(h: int, w: int, overlap: int) -> np.ndarray:
    """2-D cosine-tapered blending weight so overlapping tiles merge without seams."""
    def _1d(n: int) -> np.ndarray:
        win = np.ones(n, dtype=np.float32)
        ramp = min(overlap, n // 2)
        if ramp > 1:
            t = np.linspace(0.0, np.pi / 2.0, ramp, dtype=np.float32)
            edge = np.sin(t) ** 2
            win[:ramp] = np.maximum(edge, 1e-3)
            win[-ramp:] = np.maximum(edge[::-1], 1e-3)
        return win

    return np.outer(_1d(h), _1d(w)).astype(np.float32)


def run_tiled_probability(
    image: np.ndarray,
    predict_tile: Callable[[np.ndarray], np.ndarray],
    spec: TileSpec = TileSpec(),
) -> np.ndarray:
    """Apply a tile predictor returning ``(H_t, W_t)`` or ``(C, H_t, W_t)`` probabilities
    over an arbitrarily large 2-D image with cosine-feathered overlap blending."""
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]
    if h <= spec.tile_size and w <= spec.tile_size:
        return np.asarray(predict_tile(arr), dtype=np.float32)

    ys = _tile_starts(h, spec.tile_size, spec.overlap)
    xs = _tile_starts(w, spec.tile_size, spec.overlap)
    acc: np.ndarray | None = None
    weights = np.zeros((h, w), dtype=np.float32)

    for y0 in ys:
        y1 = min(h, y0 + spec.tile_size)
        for x0 in xs:
            x1 = min(w, x0 + spec.tile_size)
            patch = arr[y0:y1, x0:x1]
            pred = np.asarray(predict_tile(patch), dtype=np.float32)
            win = _feather_window(y1 - y0, x1 - x0, spec.overlap)
            if acc is None:
                if pred.ndim == 2:
                    acc = np.zeros((h, w), dtype=np.float32)
                else:
                    acc = np.zeros((pred.shape[0], h, w), dtype=np.float32)
            if pred.ndim == 2:
                acc[y0:y1, x0:x1] += pred * win
            else:
                acc[:, y0:y1, x0:x1] += pred * win[None, :, :]
            weights[y0:y1, x0:x1] += win

    assert acc is not None
    denom = np.maximum(weights, 1e-6)
    return (acc / denom) if acc.ndim == 2 else (acc / denom[None, :, :])


class TiledOnnxModel:
    """Verified ONNX Runtime model wrapper implementing the :class:`vexel.plugins.Model` protocol."""

    def __init__(self, card: ModelCard, *, tile_spec: TileSpec = TileSpec(), preferred_device: str | None = None) -> None:
        self.card = card
        self.tile_spec = tile_spec
        status = file_status(card)
        if status not in ("ok", "built in"):
            raise ValueError(f"cannot load {card.ref}: {status}")
        path = card.path()
        if path is None:
            raise ValueError(f"model {card.ref} has no weight file path")
        import onnxruntime as ort  # type: ignore[import-not-found]

        self.providers = select_onnx_providers(preferred_device)
        self.session = ort.InferenceSession(str(path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _normalize(self, patch: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(patch, (0.5, 99.5))
        norm = (patch.astype(np.float32) - float(lo)) / max(float(hi - lo), 1e-6)
        return np.clip(norm, 0.0, 1.0)[None, None, :, :]

    def predict_tile(self, patch: np.ndarray) -> np.ndarray:
        inp = self._normalize(patch)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        return np.squeeze(out, axis=0).astype(np.float32)

    def predict(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        return run_tiled_probability(image, self.predict_tile, self.tile_spec)
