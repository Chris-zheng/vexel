"""Run one recipe over every row of a design table, unattended.

- Each file runs into ``<out>/runs/<row id>/`` with the ordinary single-file engine, so
  every file has its own tables, summary, review pictures, report and provenance.
- A failure (unreadable file, wrong channel, QC stop) is recorded and the batch goes on.
- Resumable: a file whose run folder already holds a completed run made with the same
  recipe, settings, calibration and input file (path, size, modification time) is not
  run again. Interrupt at any time and start the same command again.
- After the last file: ``results.csv`` (one row per file), the replicate and group tables,
  comparisons, ``report.html``, methods text and ``reproduce.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from vexel.batch.design import Design, Row
from vexel.engine import QCGateError, Recipe, run_recipe

KEY_FILE = ".batch_key"


@dataclass
class FileOutcome:
    row: int
    id: str
    file: str
    group: str
    replicate: str
    status: str  # completed | resumed | failed | stopped_by_qc | missing
    run_dir: str
    duration_s: float = 0.0
    error: str = ""
    qc_warnings: list[str] = field(default_factory=list)
    result_digest: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("completed", "resumed")


@dataclass
class BatchResult:
    out_dir: Path
    outcomes: list[FileOutcome]
    stats: Any = None  # vexel.batch.stats.Analysis

    @property
    def n_ok(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def failed(self) -> list[FileOutcome]:
        return [o for o in self.outcomes if not o.ok]


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """``{"droplets.summary": {"n_droplets": 3}}`` -> ``{"droplets.n_droplets": 3}``."""
    out: dict[str, Any] = {}
    for section, values in summary.items():
        if not isinstance(values, dict):
            continue
        prefix = section.split(".")[0]
        for key, value in values.items():
            if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
                out[f"{prefix}.{key}"] = value
    return out


def _input_identity(path: Path) -> dict[str, Any]:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return {"path": str(path.resolve()), "n_files": len(files), "bytes": sum(p.stat().st_size for p in files),
                "mtime_ns": max((p.stat().st_mtime_ns for p in files), default=0)}  # fmt: skip
    st = path.stat()
    return {"path": str(path.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _row_key(recipe: Recipe, row: Row, max_frames: int | None) -> str:
    payload = {"recipe": recipe.resolve().digest(), "input": _input_identity(row.file), "pixel_size_um": row.pixel_size_um,
               "frame_interval_s": row.frame_interval_s, "max_frames": max_frames}  # fmt: skip
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _resumable(run_dir: Path, key: str) -> dict[str, Any] | None:
    try:
        if (run_dir / KEY_FILE).read_text(encoding="utf-8").strip() != key:
            return None
        prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        if prov.get("status") != "completed":
            return None
        return prov
    except (OSError, json.JSONDecodeError):
        return None


def results_frame(design: Design, outcomes: list[FileOutcome]) -> pd.DataFrame:
    by_row = {r.index: r for r in design.rows}
    recs = []
    for o in outcomes:
        r = by_row[o.row]
        rec = {"row": o.row, "id": o.id, "file": o.file, "group": o.group, "replicate": o.replicate, **r.meta,
               "status": o.status, "error": o.error, "qc_warnings": len(o.qc_warnings), "duration_s": o.duration_s,
               "run_dir": o.run_dir, "result_digest": o.result_digest}  # fmt: skip
        rec.update(o.metrics)
        recs.append(rec)
    return pd.DataFrame(recs)


def run_batch(
    design: Design,
    recipe: Recipe,
    out_dir: str | Path,
    *,
    settings: dict[str, Any] | None = None,
    resume: bool = True,
    hash_input: bool = True,
    max_frames: int | None = None,
    workers: int | None = None,
    progress: Callable[[str], None] | None = None,
    on_file: Callable[[int, int, FileOutcome], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    reports: bool = True,
) -> BatchResult:
    """Run ``recipe`` (plus ``settings`` for every file, plus each row's own settings) on
    every file of ``design`` into ``out_dir``; returns the outcomes and the statistics."""
    from vexel.batch import report as batch_report
    from vexel.batch import stats
    from vexel.readers import open_volume

    say = progress or (lambda _m: None)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not (out / "batch.json").exists():
        raise FileExistsError(f"{out} is not empty and is not a batch folder; choose another output folder")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    base = recipe.with_overrides(settings) if settings else recipe
    _write_atomic(out / "recipe.json", base.resolve().model_dump_json(indent=2))
    if design.path is not None:
        shutil.copyfile(design.path, out / "design.csv")
    else:
        design.to_frame().to_csv(out / "design.csv", index=False)

    started = datetime.now(timezone.utc)
    outcomes: list[FileOutcome] = []
    record: dict[str, Any] = {
        "kind": "vexel.batch",
        "status": "running",
        "started_at": started.isoformat(timespec="seconds"),
        "recipe": {"name": base.name, "sha256": base.resolve().digest()},
        "settings": settings or {},
        "design": {"sha256": design.sha256, "n_rows": len(design.rows), "source": str(design.path) if design.path else None},
        "max_frames": max_frames,
        "files": [],
    }

    def save(status: str) -> None:
        from vexel.engine.provenance import environment

        record["status"] = status
        record["environment"] = environment()
        record["files"] = [asdict(o) | {"metrics": None} for o in outcomes]
        record["n_completed"] = sum(1 for o in outcomes if o.ok)
        record["n_failed"] = sum(1 for o in outcomes if not o.ok)
        _write_atomic(out / "batch.json", json.dumps(record, indent=2, ensure_ascii=False, default=str))
        results_frame(design, outcomes).to_csv(out / "results.csv", index=False, float_format="%.10g")

    total = len(design.rows)
    try:
        for i, row in enumerate(design.rows, 1):
            if should_stop is not None and should_stop():
                say("stopped by request")
                save("interrupted")
                return BatchResult(out, outcomes)
            run_dir = out / "runs" / row.id
            o = FileOutcome(row.index, row.id, str(row.file), row.group, row.replicate, "failed", run_dir.relative_to(out).as_posix())
            t0 = time.perf_counter()
            say(f"[{i}/{total}] {row.file.name}")
            if not row.file.exists():
                o.status, o.error = "missing", f"file not found: {row.file}"
            else:
                try:
                    r = base.with_overrides(row.settings) if row.settings else base
                    key = _row_key(r, row, max_frames)
                    prov = _resumable(run_dir, key) if resume else None
                    if prov is not None:
                        o.status = "resumed"
                        o.result_digest = prov.get("result_digest", "")
                        o.qc_warnings = [f"{s['id']}.{v['metric']}" for s in prov.get("steps", []) for v in s.get("qc_violations", [])]
                    else:
                        if run_dir.exists():
                            shutil.rmtree(run_dir)
                        volume = open_volume(row.file, pixel_size=row.pixel_size_um, frame_interval=row.frame_interval_s)
                        try:
                            result = run_recipe(r, volume, run_dir, max_frames=max_frames, workers=workers,
                                                hash_input=hash_input, report=reports)  # fmt: skip
                            o.status = "completed"
                            o.result_digest = result.result_digest
                            o.qc_warnings = [v.message() for v in result.violations]
                            (run_dir / KEY_FILE).write_text(key, encoding="utf-8")
                        except QCGateError as e:
                            o.status, o.error = "stopped_by_qc", str(e)
                    if o.ok:
                        o.metrics = flatten_summary(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
                except Exception as e:  # noqa: BLE001  one bad file must not stop the batch
                    o.status = "failed"
                    o.error = f"{type(e).__name__}: {e}"
                    (out / "runs").mkdir(exist_ok=True)
                    (out / "runs" / f"{row.id}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            o.duration_s = round(time.perf_counter() - t0, 2)
            if not o.ok:
                say(f"    ! {o.status}: {o.error}")
            outcomes.append(o)
            save("running")
            if on_file is not None:
                on_file(i, total, o)
    except KeyboardInterrupt:
        save("interrupted")
        raise

    from vexel.modules import batch_metrics

    analysis = stats.analyse(results_frame(design, outcomes), batch_metrics(base))
    stats.write_tables(analysis, out)
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["duration_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    save("completed")
    batch_report.write_all(out, design, analysis)
    return BatchResult(out, outcomes, analysis)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
