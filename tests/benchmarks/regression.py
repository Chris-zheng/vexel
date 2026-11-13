"""Regression benchmarks on fixed synthetic data.

Each case in cases.json is generated, analysed with its recipe, and must

1. pass ``validate_run`` against its exact ground truth, and
2. reproduce the metrics stored in baseline.json (counts exactly, other numbers to a
   small relative tolerance that absorbs floating-point differences between platforms).

Run with ``uv run pytest -m benchmark``. After a deliberate change in results:
``uv run python tests/benchmarks/regression.py --update`` and review the baseline diff.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.json"
BASELINE = HERE / "baseline.json"
REL_TOL = 1e-3

# summary fields recorded per case, by summary kind (found by their fields)
SUMMARY_FIELDS = {
    "count_line_px": ["n_droplets", "diameter_mean", "diameter_cv_percent", "frequency", "volume_mean_pl"],
    "frac_empty": ["n_droplets", "lambda", "frac_empty", "frac_single", "frac_multi", "chi2_p"],
    "positive_fraction": ["n_droplets", "n_positive", "positive_fraction", "copies_per_droplet", "copies_per_ul", "threshold_method"],
    "n_flagged": ["n_flagged"],
}


def load_cases() -> dict[str, dict[str, Any]]:
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


def run_case(spec: dict[str, Any], work: Path) -> dict[str, Any]:
    from vexel.engine import Recipe, run_recipe
    from vexel.packs.droplet.validate import validate_run
    from vexel.readers import open_volume
    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    video = write_dataset(work, DropletSynthConfig(**spec["synth"]))
    recipe = Recipe.load(spec["recipe"]).with_overrides(spec.get("overrides", {}))
    result = run_recipe(recipe, open_volume(video), work / "run", hash_input=False, report=False)
    checks = validate_run(work / "run", work / "truth")
    metrics: dict[str, Any] = {}
    for summary in result.summary.values():
        for key, fields in SUMMARY_FIELDS.items():
            if key in summary:
                prefix = {"count_line_px": "droplets", "frac_empty": "cells", "positive_fraction": "fluo", "n_flagged": "flags"}[key]
                metrics.update({f"{prefix}.{f}": summary.get(f) for f in fields})
    return {
        "metrics": metrics,
        "checks": {c.name: {"measured": c.measured, "truth": c.truth, "passed": c.passed} for c in checks},
    }


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Differences between a baseline entry and a fresh result, as readable lines."""
    problems = []
    for key in sorted(set(expected) | set(actual)):
        a, b = expected.get(key), actual.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if isinstance(a, int) and isinstance(b, int):
                ok = a == b
            else:
                ok = math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-9)
        else:
            ok = a == b
        if not ok:
            problems.append(f"{key}: baseline {a!r}, now {b!r}")
    return problems


def run_all(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for name, spec in load_cases().items():
        if names and name not in names:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out[name] = run_case(spec, Path(tmp))
        failed = [k for k, c in out[name]["checks"].items() if not c["passed"]]
        print(f"{name:<22} {'ok' if not failed else 'FAILED ' + ', '.join(failed)}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="rewrite baseline.json from a fresh run")
    parser.add_argument("cases", nargs="*", help="only these cases")
    args = parser.parse_args()
    results = run_all(args.cases or None)
    if args.update:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        for name, r in results.items():
            if not all(c["passed"] for c in r["checks"].values()):
                print(f"refusing to store {name}: it does not pass validation", file=sys.stderr)
                return 1
            baseline[name] = r["metrics"]
        BASELINE.write_text(json.dumps(dict(sorted(baseline.items())), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE}")
        return 0
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    bad = 0
    for name, r in results.items():
        for line in compare(baseline.get(name, {}), r["metrics"]):
            print(f"  {name}: {line}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
