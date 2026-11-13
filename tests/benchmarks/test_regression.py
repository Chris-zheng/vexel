"""Regression benchmarks (slow): `uv run pytest -m benchmark`. See regression.py."""

from __future__ import annotations

import json

import pytest

from regression import BASELINE, compare, load_cases, run_case

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("name", sorted(load_cases()))
def test_benchmark(name, tmp_path):
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert name in baseline, f"no baseline for {name}: run `python tests/benchmarks/regression.py --update {name}`"
    result = run_case(load_cases()[name], tmp_path)
    failed = {k: v for k, v in result["checks"].items() if not v["passed"]}
    assert not failed, f"{name} no longer matches its ground truth: {failed}"
    drift = compare(baseline[name], result["metrics"])
    assert not drift, f"{name} drifted from baseline:\n" + "\n".join(drift)
