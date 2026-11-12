"""QC metric aggregation and gate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vexel.engine.recipe import QCRule


class QCGateError(RuntimeError):
    """A QC rule with ``on_fail='stop'`` was violated."""


@dataclass(frozen=True)
class Violation:
    step: str
    metric: str
    value: float | None
    min: float | None
    max: float | None
    on_fail: str

    def message(self) -> str:
        bounds = ", ".join(f"{k} {v:g}" for k, v in (("min", self.min), ("max", self.max)) if v is not None)
        value = "missing" if self.value is None else f"{self.value:g}"
        return f"QC {self.step}.{self.metric} = {value} ({bounds})"

    def to_json(self) -> dict:
        return {"metric": self.metric, "value": self.value, "min": self.min, "max": self.max, "on_fail": self.on_fail}


def aggregate_frames(per_frame: list[dict[str, float]]) -> dict[str, float]:
    """Collapse per-frame metrics into ``name.mean`` / ``name.min`` / ``name.max``."""
    out: dict[str, float] = {}
    names = sorted({k for row in per_frame for k in row})
    for name in names:
        values = [row[name] for row in per_frame if name in row and math.isfinite(row[name])]
        if not values:
            continue
        out[f"{name}.mean"] = sum(values) / len(values)
        out[f"{name}.min"] = min(values)
        out[f"{name}.max"] = max(values)
    return out


def evaluate(step: str, metrics: dict[str, float], rules: list[QCRule]) -> list[Violation]:
    violations = []
    for rule in rules:
        value = metrics.get(rule.metric)
        ok = (
            value is not None
            and math.isfinite(value)
            and (rule.min is None or value >= rule.min)
            and (rule.max is None or value <= rule.max)
        )
        if not ok:
            clean = value if (value is not None and math.isfinite(value)) else None
            violations.append(Violation(step, rule.metric, clean, rule.min, rule.max, rule.on_fail))
    return violations
