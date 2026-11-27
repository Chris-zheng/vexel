"""SEM operators on Vexel's operator registry: the SEM processing steps
(``packs.sem.process``), the saturation mask and every automatic measurement
(``packs.sem.analysis``) run inside the recipe engine, so SEM images get provenance,
``vexel batch`` statistics and the HTTP API like the droplet module.

The operators' parameters are generated from the definitions the SEM window uses
(``process.OPS`` and ``analysis.RECIPES``): names, choices, ranges and defaults are the
same, so a recipe run here and the same measurement in the window give the same numbers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, create_model

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op
from vexel.packs.sem import analysis as sem_analysis
from vexel.packs.sem import process as sem_process

_UNIT_TO_METRES = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12}


class _Scale:
    """The pixel size as ``packs.sem.analysis`` expects it (metres, or pixels when uncalibrated)."""

    def __init__(self, sx: float, sy: float, calibrated: bool) -> None:
        self.sx, self.sy, self.calibrated = sx, sy, calibrated


def _scale_from_ctx(ctx: Context) -> tuple[_Scale, str, float]:
    sy_raw, sx_raw = ctx.spacing_yx
    unit = ctx.space_unit or "px"
    if unit in _UNIT_TO_METRES:
        factor = _UNIT_TO_METRES[unit]
        return _Scale(float(sx_raw) * factor, float(sy_raw) * factor, True), unit, factor
    return _Scale(float(sx_raw), float(sy_raw), False), "px", 1.0


def _params_model(name: str, params: list[Any], extra: dict[str, Any] | None = None) -> type[Params]:
    """A frozen pydantic model from the window's ``Param`` definitions."""
    fields: dict[str, Any] = {}
    for p in params:
        if p.kind == "choice":
            fields[p.name] = (Literal[tuple(c[0] for c in p.choices)], Field(p.default, description=p.help or p.label))  # type: ignore[misc]
        elif p.kind == "bool":
            fields[p.name] = (bool, Field(bool(p.default), description=p.help or p.label))
        elif p.kind == "int":
            fields[p.name] = (int, Field(int(p.default), ge=int(p.lo), le=int(min(p.hi, 2**31 - 1)), description=p.help or p.label))
        else:
            fields[p.name] = (float, Field(float(p.default), ge=p.lo, le=p.hi, description=p.help or p.label))
    fields.update(extra or {})
    return create_model(name, __base__=Params, **fields)


# -- processing steps --------------------------------------------------------------------------


def _make_process_op(op: Any) -> None:
    model = _params_model(f"Sem{op.name.title().replace('_', '')}Params", op.params)

    def fn(params: Any, ctx: Context, *, image: np.ndarray) -> OpResult:
        out = op.func(np.asarray(image, dtype=np.float64), **params.model_dump())
        return OpResult({"image": out.astype(np.float32)}, qc={"std_after": float(np.std(out))})

    fn.__name__ = f"op_sem_{op.name}"
    fn.__doc__ = op.help
    register_op(f"sem.{op.name}", scope=Scope.FRAME, inputs={"image": Kind.IMAGE}, outputs={"image": Kind.IMAGE}, params=model)(fn)


for _op in sem_process.OPS.values():
    if _op.role == "correct":  # display-only steps (contrast, sharpen) never feed measurements
        _make_process_op(_op)


# -- saturation ---------------------------------------------------------------------------------


class SaturationParams(Params):
    high: float | None = Field(None, description="grey level at or above which a pixel is saturated; None: the dtype's limit or a plateau at the maximum")
    low: float | None = Field(None, description="with dark: grey level at or below which a pixel is clipped; None: found as for high")
    dark: bool = Field(False, description="also mark pixels clipped at the bottom")
    dilate: int = Field(1, ge=0, le=20, description="widen the mask by this many pixels")


@register_op("sem.saturation_mask", scope=Scope.FRAME, inputs={"image": Kind.IMAGE}, outputs={"mask": Kind.MASK}, params=SaturationParams)
def op_saturation_mask(params: SaturationParams, ctx: Context, *, image: np.ndarray) -> OpResult:
    """Saturated pixels (True), as the SEM window marks them: measurements leave them out;
    they are never repaired."""
    mask, info = sem_process.saturation_mask(np.asarray(image), sem_process.Saturation(**params.model_dump()))
    return OpResult({"mask": mask}, qc={"saturated_fraction": float(info["fraction"])})


# -- measurements -------------------------------------------------------------------------------

_REGION = {
    "region": (list[float] | None, Field(None, description="region [x0, y0, x1, y1] in image pixels; None: the whole micrograph")),
    "crop_rows": (int | None, Field(None, ge=1, description="rows of the micrograph above the instrument's data bar")),
}


def _summary_dict(an: sem_analysis.Analysis, unit: str, factor: float, calibrated: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"recipe": an.spec.recipe, "calibrated": calibrated, "length_unit": unit, "n_warnings": len(an.warnings),
                           "warnings": list(an.warnings)}  # fmt: skip
    if an.error:
        out["error"] = an.error
        return out
    for key, _label, value, kind in an.summary:
        if value is None:
            out[key] = None
        elif kind == "length" and calibrated:
            out[key] = float(value) / factor
        elif kind == "area" and calibrated:
            out[key] = float(value) / factor**2
        else:
            out[key] = value
    return out


def _shapes(an: sem_analysis.Analysis) -> list[dict[str, Any]]:
    return [{"kind": kind, "points": pts, "color": color, "anchor": list(anchor), "label": label} for kind, pts, color, anchor, label in an.shapes]


def run_measurement(recipe: str, params: dict[str, Any], ctx: Context, image: np.ndarray, mask: np.ndarray | None = None) -> OpResult:
    scale, unit, factor = _scale_from_ctx(ctx)
    raw = dict(params)
    region, crop_rows = raw.pop("region", None), raw.pop("crop_rows", None)
    spec = sem_analysis.AnalysisSpec(id=1, recipe=recipe, region=region, params=raw)
    an = sem_analysis.run(spec, np.asarray(image, dtype=np.float64), scale, mask=None if mask is None else np.asarray(mask, bool),
                          crop_rows=crop_rows)  # fmt: skip
    table = an.objects.copy() if isinstance(an.objects, pd.DataFrame) else pd.DataFrame()
    qc = {"n_objects": float(len(table)), "n_warnings": float(len(an.warnings)), "has_error": 1.0 if an.error else 0.0}
    return OpResult({"table": table, "summary": _summary_dict(an, unit, factor, scale.calibrated), "geometry": _shapes(an)}, qc=qc)


def _make_measurement_op(rec: Any) -> None:
    model = _params_model(f"Sem{rec.name.title()}Params", rec.params, _REGION)

    def fn(params: Any, ctx: Context, *, image: np.ndarray, mask: np.ndarray | None = None) -> OpResult:
        return run_measurement(rec.name, params.model_dump(), ctx, image, mask)

    fn.__name__ = f"op_sem_{rec.name}"
    fn.__doc__ = rec.help
    register_op(f"sem.{rec.name}", scope=Scope.FRAME, inputs={"image": Kind.IMAGE, "mask": Kind.MASK}, optional_inputs=("mask",),
                outputs={"table": Kind.TABLE, "summary": Kind.SUMMARY, "geometry": Kind.GEOMETRY}, params=model)(fn)  # fmt: skip


for _rec in sem_analysis.RECIPES.values():
    _make_measurement_op(_rec)


__all__ = ["SaturationParams", "op_saturation_mask", "run_measurement"]
