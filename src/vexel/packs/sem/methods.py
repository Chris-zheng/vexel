"""Saved methods ("方案") and batches that show only what needs a person.

A method is an automatic measurement worth repeating: its recipe, settings, region and the
image processing it ran on, under a name (``M1 线宽``). It is applied to new images as it
is - no planning, no guessing - and one method can be set to apply automatically to every
image opened without measurements.

``run_files`` measures many images with one method (each with its own calibration) and
``triage`` sorts the results so that a person looks only at the exceptions: errors,
missing or unconfirmed calibration, warnings, and values far from the rest of the batch.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Method(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    recipe: str
    params: dict[str, Any] = Field(default_factory=dict)
    region: list[float] | None = None  # image pixels; None: the whole micrograph
    steps: list[dict[str, Any]] = Field(default_factory=list)  # processing.Step as dicts
    note: str = ""
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _path() -> Path:
    return Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel")) / "sem" / "methods.json"


def _load() -> dict[str, Any]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return {"methods": [Method(**m) for m in data.get("methods", [])], "auto": data.get("auto")}
    except (OSError, ValueError, TypeError):
        return {"methods": [], "auto": None}


def _save(data: dict[str, Any]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"methods": [m.model_dump() for m in data["methods"]], "auto": data["auto"]}, ensure_ascii=False, indent=2),
                 encoding="utf-8")  # fmt: skip


def list_methods() -> list[Method]:
    return list(_load()["methods"])


def get(name: str) -> Method | None:
    return next((m for m in list_methods() if m.name == name), None)


def save(method: Method) -> Method:
    """Add the method, or replace the one with the same name."""
    data = _load()
    data["methods"] = [m for m in data["methods"] if m.name != method.name] + [method]
    _save(data)
    return method


def delete(name: str) -> None:
    data = _load()
    data["methods"] = [m for m in data["methods"] if m.name != name]
    if data["auto"] == name:
        data["auto"] = None
    _save(data)


def auto() -> Method | None:
    """The method applied to every image opened without measurements, if one is set."""
    data = _load()
    return next((m for m in data["methods"] if m.name == data["auto"]), None)


def set_auto(name: str | None) -> None:
    data = _load()
    if name is not None and not any(m.name == name for m in data["methods"]):
        raise KeyError(f"no method called {name!r}")
    data["auto"] = name
    _save(data)


def from_record(ms: Any, spec: Any, name: str, note: str = "") -> Method:
    """A method from one automatic measurement of a record and the record's processing."""
    if not name.strip():
        raise ValueError("a method needs a name")
    return Method(name=name.strip(), recipe=spec.recipe, params=dict(spec.params), region=list(spec.region) if spec.region else None,
                  steps=[s.model_dump() for s in ms.processing.steps], note=note)  # fmt: skip


def apply(method: Method, ms: Any) -> Any:
    """Use the method on a record: its processing replaces the record's, and its measurement
    is added (named after the method). Returns the new AnalysisSpec."""
    from vexel.packs.sem.process import Step

    ms.processing.steps = [Step(**s) for s in method.steps]
    spec = ms.add_analysis(method.recipe, method.region, **method.params)
    spec.name = method.name
    return spec


# -- batches ------------------------------------------------------------------------------------


@dataclass
class FileResult:
    path: Path
    analysis: Any = None
    calibration: str = ""  # describe() of the calibration
    calibrated: bool = False
    confirmed: bool = False
    error: str | None = None
    status: str = "ok"  # ok | check | error
    reasons: list[str] = field(default_factory=list)

    @property
    def main(self) -> tuple[str, float | None, str]:
        """(label, value, kind) of the measurement's first length (or, without one, first number)."""
        an = self.analysis
        if an is None or an.error:
            return "", None, ""
        for _k, label, v, kind in an.summary:
            if kind in ("length", "angle") and v is not None:
                return label, float(v), kind
        for _k, label, v, kind in an.summary:
            if isinstance(v, (int, float)) and v is not None:
                return label, float(v), kind
        return "", None, ""


def run_files(method: Method, paths: list[str | Path], *, use_ocr: bool = True, progress: Any = None) -> list[FileResult]:
    """Measure every file with ``method`` (each with its own calibration and record; the
    records are not changed)."""
    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.measure import scale_from

    out = []
    for i, p in enumerate(paths):
        p = Path(p)
        r = FileResult(p)
        try:
            cal = semcal.for_file(p, use_ocr=use_ocr)
            img = semcal.load_image(p)
            ms = measure_io.new_set(p, cal)
            ms.scale = scale_from(cal, 0.0)
            apply(method, ms)
            r.analysis = measure_io.analyses_of(ms, img)[-1]
            r.calibration, r.calibrated, r.confirmed = cal.describe(), cal.ok, bool(cal.confirmed)
            r.error = r.analysis.error
        except (ValueError, OSError) as e:
            r.error = f"{type(e).__name__}: {e}"
        out.append(r)
        if progress is not None:
            progress(i + 1, len(paths))
    return triage(out)


def triage(results: list[FileResult], z: float = 3.5) -> list[FileResult]:
    """Status and reasons for each result; exceptions first. A value is far from the batch
    when its robust z-score (median and MAD of the batch) is over ``z`` (needs 5 results)."""
    vals = np.array([r.main[1] if r.main[1] is not None else np.nan for r in results], float)
    ok = np.isfinite(vals)
    med = float(np.median(vals[ok])) if ok.sum() >= 5 else math.nan
    mad = float(1.4826 * np.median(np.abs(vals[ok] - med))) if ok.sum() >= 5 else math.nan
    for r, v in zip(results, vals):
        r.reasons = []
        if r.error:
            r.status, r.reasons = "error", [r.error]
            continue
        if not r.calibrated:
            r.reasons.append("未标定：结果以像素为单位")
        elif not r.confirmed:
            r.reasons.append("标定待确认")
        r.reasons += list(r.analysis.warnings)
        if np.isfinite(v) and math.isfinite(mad) and mad > 0 and abs(v - med) / mad > z:
            r.reasons.append(f"{r.main[0]} 与这批的中位数相差 {abs(v - med) / mad:.1f} 个稳健标准差")
        r.status = "check" if r.reasons else "ok"
    order = {"error": 0, "check": 1, "ok": 2}
    return sorted(results, key=lambda r: (order[r.status], r.path.name))


def table(results: list[FileResult], unit: str, f: float) -> Any:
    """One row per file: status, reasons, calibration, the measurement's summary values."""
    import pandas as pd

    rows = []
    for r in results:
        row: dict[str, Any] = {"file": r.path.name, "status": r.status, "reasons": "; ".join(r.reasons), "calibration": r.calibration}
        if r.analysis is not None and not r.analysis.error:
            for k, _label, v, kind in r.analysis.summary:
                row[k] = v / f if kind == "length" and v is not None and r.calibrated else v
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["FileResult", "Method", "apply", "auto", "delete", "from_record", "get", "list_methods", "run_files", "save", "set_auto", "table", "triage"]
