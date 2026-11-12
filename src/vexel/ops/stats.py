"""Generic statistics operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op


def describe_values(values: np.ndarray) -> dict[str, float | int | None]:
    """n / mean / sample std / CV% / median / min / max. ``None`` where undefined."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "cv_percent": None, "median": None, "min": None, "max": None}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv_percent": (100.0 * std / mean) if (std is not None and mean != 0) else None,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


class DescribeParams(Params):
    column: str = Field(description="numeric column to summarise")


@register_op(
    "stats.describe",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE},
    outputs={"summary": Kind.SUMMARY},
    params=DescribeParams,
)
def describe(params: DescribeParams, ctx: Context, *, table: pd.DataFrame) -> OpResult:
    """Summary statistics (n, mean, std, CV, median, range) of one table column."""
    if params.column not in table.columns:
        raise KeyError(f"stats.describe: column {params.column!r} not in table (have {list(table.columns)})")
    stats = describe_values(table[params.column].to_numpy(dtype=np.float64))
    return OpResult({"summary": {"column": params.column, **stats}}, {"n": float(stats["n"])})
