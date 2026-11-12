"""Compare a droplet run against ground truth (synthetic ``truth/`` folder, or hand
annotations in the same format)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vexel.ops.stats import describe_values
from vexel.packs.droplet.ops import crossing_frame


@dataclass(frozen=True)
class Tolerances:
    """Acceptance thresholds. Defaults are placeholders until the thresholds agreed with
    the design partner (Phase 0 sign-off) replace them."""

    count_abs: int = 0
    diameter_mean_rel: float = 0.02
    diameter_cv_abs: float = 1.0  # percentage points
    frequency_rel: float = 0.02


@dataclass(frozen=True)
class Check:
    name: str
    measured: float | None
    truth: float | None
    error: float | None
    tolerance: float
    kind: str  # "abs" or "rel"
    passed: bool


def truth_summary(truth_dir: Path, count_line_px: float, n_frames: int) -> dict[str, float | int | None]:
    """Truth for the droplets that cross ``count_line_px`` within the first ``n_frames``."""
    droplets = pd.read_csv(truth_dir / "droplets.csv").set_index("droplet_id")
    positions = pd.read_csv(truth_dir / "positions.csv")
    positions = positions[positions["frame"] < n_frames]
    config = json.loads((truth_dir / "config.json").read_text())
    crossings: dict[int, float] = {}
    for droplet_id, g in positions.sort_values("frame").groupby("droplet_id"):
        c = crossing_frame(g["x_px"].to_numpy(float), g["frame"].to_numpy(float), count_line_px)
        if c is not None:
            crossings[int(droplet_id)] = c
    ids = sorted(crossings, key=crossings.get)
    stats = describe_values(droplets.loc[ids, "diameter_um"].to_numpy(float))
    times = np.array([crossings[i] for i in ids]) * config["frame_interval_s"]
    frequency = (len(ids) - 1) / (times[-1] - times[0]) if len(ids) >= 2 else None
    return {
        "n_droplets": len(ids),
        "diameter_mean": stats["mean"],
        "diameter_cv_percent": stats["cv_percent"],
        "frequency": frequency,
    }


def _check(name: str, measured, truth, tolerance: float, kind: str) -> Check:
    if measured is None or truth is None:
        return Check(name, measured, truth, None, tolerance, kind, False)
    error = abs(measured - truth)
    if kind == "rel":
        error = error / abs(truth) if truth else float("inf")
    return Check(name, float(measured), float(truth), float(error), tolerance, kind, bool(error <= tolerance))


def validate_run(run_dir: str | Path, truth_dir: str | Path, tol: Tolerances = Tolerances()) -> list[Check]:
    run_dir, truth_dir = Path(run_dir), Path(truth_dir)
    summaries = json.loads((run_dir / "summary.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())
    measured = next((s for s in summaries.values() if "n_droplets" in s and "count_line_px" in s), None)
    if measured is None:
        raise ValueError(f"{run_dir} contains no droplet.summary output")
    if measured["length_unit"] != "um":
        raise ValueError(f"run is in {measured['length_unit']!r}; truth diameters are in um - run with calibration")
    truth = truth_summary(truth_dir, measured["count_line_px"], provenance["input"]["frames_used"])
    return [
        _check("n_droplets", measured["n_droplets"], truth["n_droplets"], tol.count_abs, "abs"),
        _check("diameter_mean", measured["diameter_mean"], truth["diameter_mean"], tol.diameter_mean_rel, "rel"),
        _check("diameter_cv_percent", measured["diameter_cv_percent"], truth["diameter_cv_percent"], tol.diameter_cv_abs, "abs"),
        _check("frequency", measured["frequency"], truth["frequency"], tol.frequency_rel, "rel"),
    ]
