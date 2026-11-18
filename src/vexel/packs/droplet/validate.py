"""Compare a droplet run against ground truth (synthetic ``truth/`` folder, or hand
annotations in the same format)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vexel.ops.stats import describe_values
from vexel.packs.droplet.ops import crossing_frame, droplet_volume_um3


@dataclass(frozen=True)
class Tolerances:
    """Acceptance thresholds. Defaults are placeholders until the thresholds agreed with
    the design partner (Phase 0 sign-off) replace them."""

    count_abs: int = 0
    diameter_mean_rel: float = 0.02
    diameter_cv_abs: float = 1.0  # percentage points
    frequency_rel: float = 0.02
    volume_rel: float = 0.06  # volume scales with diameter cubed: ~3x the diameter tolerance
    # encapsulation (checked only when the run has droplet.poisson output and the truth
    # has an ``n_cells`` column)
    inclusion_mismatch: float = 0.05  # share of droplets whose cell count is wrong
    lambda_rel: float = 0.05
    fraction_abs: float = 0.03  # empty / single / multiple shares
    # fluorescence (checked when the run has droplet.positivity output and the truth has
    # a ``positive`` column)
    positive_mismatch: float = 0.02  # share of droplets called wrongly
    positive_fraction_abs: float = 0.03


@dataclass(frozen=True)
class Check:
    name: str
    measured: float | None
    truth: float | None
    error: float | None
    tolerance: float
    kind: str  # "abs", "rel" (relative error) or "share" (a fraction, shown as %)
    passed: bool


def truth_crossings(truth_dir: Path, count_line_px: float, n_frames: int) -> dict[int, float]:
    """Droplet id -> frame at which it crosses ``count_line_px``, within the first ``n_frames``."""
    positions = pd.read_csv(truth_dir / "positions.csv")
    positions = positions[positions["frame"] < n_frames]
    crossings: dict[int, float] = {}
    for droplet_id, g in positions.sort_values("frame").groupby("droplet_id"):
        c = crossing_frame(g["x_px"].to_numpy(float), g["frame"].to_numpy(float), count_line_px)
        if c is not None:
            crossings[int(droplet_id)] = c
    return crossings


def truth_summary(
    truth_dir: Path, count_line_px: float, n_frames: int, channel_height_um: float | None = None
) -> dict[str, float | int | None]:
    """Truth for the droplets that cross ``count_line_px`` within the first ``n_frames``."""
    droplets = pd.read_csv(truth_dir / "droplets.csv").set_index("droplet_id")
    config = json.loads((truth_dir / "config.json").read_text(encoding="utf-8"))
    crossings = truth_crossings(truth_dir, count_line_px, n_frames)
    ids = sorted(crossings, key=crossings.get)
    stats = describe_values(droplets.loc[ids, "diameter_um"].to_numpy(float))
    volumes = droplet_volume_um3(droplets.loc[ids, "diameter_um"].to_numpy(float), channel_height_um) / 1000.0
    times = np.array([crossings[i] for i in ids]) * config["frame_interval_s"]
    frequency = (len(ids) - 1) / (times[-1] - times[0]) if len(ids) >= 2 else None
    return {
        "n_droplets": len(ids),
        "diameter_mean": stats["mean"],
        "diameter_cv_percent": stats["cv_percent"],
        "frequency": frequency,
        "volume_mean_pl": float(volumes.mean()) if len(ids) else None,
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
    summaries = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    measured = next((s for s in summaries.values() if "n_droplets" in s and "count_line_px" in s), None)
    if measured is None:
        raise ValueError(f"{run_dir} contains no droplet.summary output")
    if measured["length_unit"] != "um":
        raise ValueError(f"run is in {measured['length_unit']!r}; truth diameters are in um - run with calibration")
    truth = truth_summary(
        truth_dir, measured["count_line_px"], provenance["input"]["frames_used"], measured.get("channel_height_um")
    )
    checks = [
        _check("n_droplets", measured["n_droplets"], truth["n_droplets"], tol.count_abs, "abs"),
        _check("diameter_mean", measured["diameter_mean"], truth["diameter_mean"], tol.diameter_mean_rel, "rel"),
        _check("diameter_cv_percent", measured["diameter_cv_percent"], truth["diameter_cv_percent"], tol.diameter_cv_abs, "abs"),
        _check("frequency", measured["frequency"], truth["frequency"], tol.frequency_rel, "rel"),
    ]
    if measured.get("volume_mean_pl") is not None:
        checks.append(_check("volume_mean", measured["volume_mean_pl"], truth["volume_mean_pl"], tol.volume_rel, "rel"))
    encapsulation = next((s for s in summaries.values() if "lambda" in s and "frac_empty" in s), None)
    truth_droplets = pd.read_csv(truth_dir / "droplets.csv")
    frames = provenance["input"]["frames_used"]
    if encapsulation is not None and "n_cells" in truth_droplets.columns:
        checks += encapsulation_checks(run_dir, truth_dir, measured["count_line_px"], frames, encapsulation, tol)
    positivity = next((s for s in summaries.values() if "positive_fraction" in s), None)
    if positivity is not None and "positive" in truth_droplets.columns:
        checks += positivity_checks(run_dir, truth_dir, measured["count_line_px"], frames, positivity, tol)
    return checks


def _droplet_table(run_dir: Path, column: str = "n_inclusions") -> pd.DataFrame:
    for path in sorted((run_dir / "tables").glob("*.csv")):
        header = path.open(encoding="utf-8").readline().strip().split(",")
        if {"crossing_frame", column} <= set(header):
            return pd.read_csv(path)
    raise ValueError(f"{run_dir} has no per-droplet table with {column}")


def match_droplets(measured_frames: np.ndarray, truth: dict[int, float], max_gap: float = 1.0) -> dict[int, int]:
    """Measured row index -> truth droplet id, pairing crossings closest in time first."""
    pairs = sorted(
        (abs(mf - tf), i, tid) for i, mf in enumerate(measured_frames) for tid, tf in truth.items() if abs(mf - tf) <= max_gap
    )
    used_m: set[int] = set()
    used_t: set[int] = set()
    out: dict[int, int] = {}
    for _, i, tid in pairs:
        if i not in used_m and tid not in used_t:
            out[i] = tid
            used_m.add(i)
            used_t.add(tid)
    return out


def encapsulation_checks(
    run_dir: Path, truth_dir: Path, count_line_px: float, n_frames: int, measured: dict, tol: Tolerances
) -> list[Check]:
    crossings = truth_crossings(truth_dir, count_line_px, n_frames)
    cells = pd.read_csv(truth_dir / "droplets.csv").set_index("droplet_id")["n_cells"]
    truth_counts = np.array([cells[i] for i in crossings], dtype=np.int64)
    table = _droplet_table(run_dir)
    pairs = match_droplets(table["crossing_frame"].to_numpy(float), crossings)
    wrong = sum(int(table["n_inclusions"].iloc[i]) != int(cells[tid]) for i, tid in pairs.items())
    mismatch = wrong / len(pairs) if pairs else None
    n = truth_counts.size
    truth_frac = {
        "empty": float(np.mean(truth_counts == 0)) if n else None,
        "single": float(np.mean(truth_counts == 1)) if n else None,
        "multi": float(np.mean(truth_counts >= 2)) if n else None,
    }
    return [
        _check("inclusion_mismatch", mismatch, 0.0, tol.inclusion_mismatch, "share"),
        _check("lambda", measured["lambda"], float(truth_counts.mean()) if n else None, tol.lambda_rel, "rel"),
        _check("frac_empty", measured["frac_empty"], truth_frac["empty"], tol.fraction_abs, "share"),
        _check("frac_single", measured["frac_single"], truth_frac["single"], tol.fraction_abs, "share"),
        _check("frac_multi", measured["frac_multi"], truth_frac["multi"], tol.fraction_abs, "share"),
    ]


def positivity_checks(
    run_dir: Path, truth_dir: Path, count_line_px: float, n_frames: int, measured: dict, tol: Tolerances
) -> list[Check]:
    crossings = truth_crossings(truth_dir, count_line_px, n_frames)
    truth_pos = pd.read_csv(truth_dir / "droplets.csv").set_index("droplet_id")["positive"].astype(bool)
    table = _droplet_table(run_dir, "positive")
    pairs = match_droplets(table["crossing_frame"].to_numpy(float), crossings)
    calls = table["positive"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    wrong = sum(bool(calls[i]) != bool(truth_pos[tid]) for i, tid in pairs.items())
    truth_fraction = float(np.mean([truth_pos[i] for i in crossings])) if crossings else None
    return [
        _check("positive_mismatch", wrong / len(pairs) if pairs else None, 0.0, tol.positive_mismatch, "share"),
        _check("positive_fraction", measured["positive_fraction"], truth_fraction, tol.positive_fraction_abs, "share"),
    ]
