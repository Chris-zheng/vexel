"""Recipe memory: what worked before on similar data.

Every planned run adds an entry: the data's fingerprint, the recipe used (built-in name
plus the settings that differ from it), and the outcome (QC rules passed or not, share
of flagged objects, number of human corrections). A query applies hard filters on the
metadata (same channel layout and analysis family), then ranks the remaining entries
by distance on normalised fingerprint features, and reports each hit with its history
score: how often runs with that recipe on similar data passed QC.

Stored as JSON lines under ``$VEXEL_HOME/memory/recipes.jsonl`` (default ``~/.vexel``).

Any module can use it: a fingerprint that has ``meta()`` (the hard filters) and a
``SCALES`` table (the feature spreads) brings its own; SEM keeps its runs apart, in
``memory/sem.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# typical spread of each feature, to put them on one scale
SCALES = {
    "log_snr": 0.5, "skewness": 1.0, "bimodality": 1.0, "static_fraction": 0.2, "moving_fraction": 0.1, "speed_px": 5.0,
    "jitter_px": 1.0, "objects_per_frame": 3.0, "diameter_px": 8.0, "circularity": 0.1, "interior_minus_rim": 30.0,
    "spots_per_object": 0.5,
}  # fmt: skip


def home() -> Path:
    return Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel"))


def overrides_key(base: str, overrides: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({"base": base, "overrides": overrides}, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Hit:
    entry: dict[str, Any]
    distance: float
    runs: int  # runs on record with this recipe (same base and settings) on matching data
    passed: int  # of those, runs that passed every QC rule

    @property
    def score(self) -> float:
        return self.passed / self.runs if self.runs else 0.0


class RecipeMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else home() / "memory" / "recipes.jsonl"

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a damaged line does not lose the rest
        return out

    def add(self, fingerprint: Any, base: str, overrides: dict[str, Any], outcome: dict[str, Any], run_dir: str | Path | None = None) -> dict[str, Any]:
        meta = _meta(fingerprint)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "meta": meta,
            "features": fingerprint.features(),
            "base": base,
            "overrides": overrides,
            "key": overrides_key(base, overrides),
            "outcome": outcome,
            "run": str(run_dir) if run_dir else None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def query(self, fingerprint: Any, family: str | None = None, k: int = 3) -> list[Hit]:
        """Nearest past runs on data with the same layout (and analysis family, if given)."""
        meta, feats = _meta(fingerprint), fingerprint.features()
        scales = getattr(fingerprint, "SCALES", SCALES)
        pool = [e for e in self.entries() if _compatible(e["meta"], meta) and (family is None or _family(e["base"]) == family)]
        if not pool:
            return []
        best: dict[str, tuple[float, dict[str, Any]]] = {}
        for e in pool:
            d = _distance(feats, e.get("features", {}), scales)
            if e["key"] not in best or d < best[e["key"]][0]:
                best[e["key"]] = (d, e)
        hits = []
        for key, (d, e) in best.items():
            same = [x for x in pool if x["key"] == key and _distance(feats, x.get("features", {}), scales) <= max(2 * d, 1.0)]
            hits.append(Hit(e, d, len(same), sum(1 for x in same if x["outcome"].get("qc_passed"))))
        hits.sort(key=lambda h: (h.distance, -h.score))
        return hits[:k]


def _meta(fp: Any) -> dict[str, Any]:
    if hasattr(fp, "meta"):
        return fp.meta()
    return {
        "n_channels": fp.shape.get("C", 1),
        "has_fluorescence": fp.fluorescence_channel is not None,
        "has_spots": bool((fp.channels[fp.droplet_channel or 0].spots_per_object or 0) >= 0.15) if fp.channels else False,
        "reader": fp.reader,
        "calibrated": fp.calibrated,
    }


def _compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("module") or b.get("module"):
        return a.get("module") == b.get("module") and a.get("kind") == b.get("kind") and a.get("calibrated") == b.get("calibrated")
    return a.get("n_channels") == b.get("n_channels") and a.get("has_fluorescence") == b.get("has_fluorescence")


def _family(base: str) -> str:
    return base


def _distance(a: dict[str, float], b: dict[str, float], scales: dict[str, float] | None = None) -> float:
    scales = SCALES if scales is None else scales
    keys = [k for k in scales if k in a and k in b]
    if not keys:
        return float("inf")
    diff = np.array([(a[k] - b[k]) / scales[k] for k in keys])
    return float(np.sqrt(np.mean(diff**2)))
