"""Batch executor: runs a frozen recipe over a recorded acquisition with bounded memory.

Execution is staged. Each step's array outputs are computed block by block (block size
from the memory watermark) and written to a scratch Zarr store, then handed to later
steps as lazy arrays. Nothing is computed twice, memory stays bounded regardless of
video length, and intermediates can be kept for inspection.

Phase 1 handles 2-D time series (one channel and z-plane selected by the recipe);
``volume``-scope operators are accepted by the registry but not executed yet.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dask
import dask.array as da
import imageio.v3 as iio
import numpy as np
import pandas as pd
import zarr
from skimage.segmentation import mark_boundaries

from vexel.core.device import plan_block_frames, select_device
from vexel.core.volume import SpatialVolume
from vexel.engine import provenance as prov
from vexel.engine.qc import QCGateError, Violation, aggregate_frames, evaluate
from vexel.engine.recipe import SOURCE, Recipe, Step
from vexel.ops.registry import Context, Kind, OpResult, OpSpec, Scope, get_op

_CHUNK_BYTES = 16 * 1024**2
_RUN_MARKERS = ("provenance.json", "recipe.json")
_ARRAY_KINDS = (Kind.IMAGE, Kind.LABELS)
Progress = Callable[[str], None]


@dataclass
class RunResult:
    out_dir: Path
    recipe: Recipe
    status: str
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    provenance: dict[str, Any]
    violations: list[Violation] = field(default_factory=list)

    @property
    def result_digest(self) -> str:
        return self.provenance["result_digest"]


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        is_run = any((out_dir / m).exists() for m in _RUN_MARKERS)
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty; choose another output directory or pass overwrite")
        if not is_run:
            raise FileExistsError(f"{out_dir} is not empty and is not a previous vexel run; refusing to overwrite it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


class _ArrayWriter:
    """Writes a ``(T, ...)`` array block by block into a scratch Zarr store."""

    def __init__(self, path: Path, n_frames: int) -> None:
        self.path = path
        self.n_frames = n_frames
        self._array: Any = None

    @staticmethod
    def chunk_frames(frame_nbytes: int) -> int:
        return max(1, _CHUNK_BYTES // max(1, frame_nbytes))

    def write(self, start: int, block: np.ndarray) -> None:
        if self._array is None:
            frame_shape = block.shape[1:]
            ct = min(self.n_frames, self.chunk_frames(block[0].nbytes))
            self._array = zarr.create_array(
                store=str(self.path), shape=(self.n_frames, *frame_shape), chunks=(ct, *frame_shape), dtype=block.dtype, overwrite=True
            )
        self._array[start : start + len(block)] = block

    def lazy(self) -> da.Array:
        if self._array is None:
            raise RuntimeError(f"nothing was written to {self.path}")
        return da.from_zarr(str(self.path))


def _block_frames(frame_shape: tuple[int, ...], n_arrays: int) -> int:
    frame_nbytes = int(np.prod(frame_shape)) * 4  # sized for float32 working copies
    return plan_block_frames(frame_nbytes, n_arrays, multiple=_ArrayWriter.chunk_frames(frame_nbytes))


def _check_outputs(spec: OpSpec, result: OpResult) -> None:
    if not isinstance(result, OpResult):
        raise TypeError(f"{spec.name} must return OpResult, got {type(result).__name__}")
    if set(result.outputs) != set(spec.outputs):
        raise ValueError(f"{spec.name} returned {sorted(result.outputs)} but declares {sorted(spec.outputs)}")


def _run_frame_step(
    spec: OpSpec, params: Any, inputs: dict[str, da.Array], ctx: Context, scratch: Path, step_id: str, pool: ThreadPoolExecutor
) -> tuple[dict[str, Any], dict[str, float]]:
    names = list(inputs)
    n = ctx.n_frames
    writers = {o: _ArrayWriter(scratch / f"{step_id}.{o}.zarr", n) for o, k in spec.outputs.items() if k in _ARRAY_KINDS}
    table_parts: dict[str, list[pd.DataFrame]] = {o: [] for o, k in spec.outputs.items() if k is Kind.TABLE}
    unsupported = [o for o, k in spec.outputs.items() if k is Kind.SUMMARY]
    if unsupported:
        raise ValueError(f"{spec.name}: frame-scope operators cannot output summaries ({unsupported})")
    qc_rows: list[dict[str, float]] = []
    block = _block_frames(ctx.frame_shape, len(inputs) + len(spec.outputs) + 4)

    for start in range(0, n, block):
        stop = min(n, start + block)
        blocks = dask.compute(*[inputs[k][start:stop] for k in names])

        def one(i: int) -> OpResult:
            result = spec.fn(params, ctx.for_frame(start + i), **{k: b[i] for k, b in zip(names, blocks)})
            _check_outputs(spec, result)
            return result

        results = list(pool.map(one, range(stop - start)))
        for o, writer in writers.items():
            writer.write(start, np.stack([r.outputs[o] for r in results]))
        for o, parts in table_parts.items():
            for i, r in enumerate(results):
                t = r.outputs[o]
                if len(t):
                    t = t.copy()
                    t.insert(0, "frame", start + i)
                    parts.append(t)
                elif not parts:
                    empty = t.copy()
                    empty.insert(0, "frame", pd.Series(dtype=np.int64))
                    parts.append(empty)
        qc_rows.extend(r.qc for r in results)

    outputs: dict[str, Any] = {o: w.lazy() for o, w in writers.items()}
    for o, parts in table_parts.items():
        non_empty = [p for p in parts if len(p)]
        outputs[o] = pd.concat(non_empty, ignore_index=True) if non_empty else parts[0]
    return outputs, aggregate_frames(qc_rows)


def _run_window_step(
    spec: OpSpec, params: Any, inputs: dict[str, Any], ctx: Context, scratch: Path, step_id: str
) -> tuple[dict[str, Any], dict[str, float]]:
    result = spec.fn(params, ctx, **inputs)
    _check_outputs(spec, result)
    outputs: dict[str, Any] = {}
    for o, kind in spec.outputs.items():
        value = result.outputs[o]
        if kind in _ARRAY_KINDS:
            lazy = value if isinstance(value, da.Array) else da.from_array(np.asarray(value), chunks=(1, -1, -1))
            if lazy.shape[0] != ctx.n_frames:
                raise ValueError(f"{spec.name}: output {o!r} has {lazy.shape[0]} frames, expected {ctx.n_frames}")
            writer = _ArrayWriter(scratch / f"{step_id}.{o}.zarr", ctx.n_frames)
            block = _block_frames(lazy.shape[1:], 4)
            for start in range(0, ctx.n_frames, block):
                writer.write(start, np.asarray(lazy[start : start + block].compute()))
            outputs[o] = writer.lazy()
        else:
            outputs[o] = value
    return outputs, dict(result.qc)


def _write_review(image: da.Array, labels: da.Array, n_frames: int, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    total = image.shape[0]
    picks = np.unique(np.linspace(0, total - 1, min(total, n_frames)).round().astype(int))
    written = []
    for t in picks:
        img, lab = dask.compute(image[t], labels[t])
        img = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(img, (1, 99))
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        overlay = mark_boundaries(np.dstack([gray] * 3), np.asarray(lab), color=(1.0, 0.15, 0.15), mode="thick")
        path = folder / f"frame_{int(t):06d}.png"
        iio.imwrite(path, (overlay * 255).round().astype(np.uint8))
        written.append(path)
    return written


def _finite(metrics: dict[str, float]) -> dict[str, float | None]:
    return {k: (float(v) if math.isfinite(v) else None) for k, v in sorted(metrics.items())}


def run_recipe(
    recipe: Recipe,
    volume: SpatialVolume,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    keep_intermediates: bool = False,
    max_frames: int | None = None,
    workers: int | None = None,
    hash_input: bool = True,
    progress: Progress | None = None,
) -> RunResult:
    """Resolve and execute ``recipe`` on ``volume``; write tables, summary, review overlays
    and provenance into ``out_dir``."""
    say = progress or (lambda _msg: None)
    resolved = recipe.resolve()
    specs = {s.id: get_op(s.op) for s in resolved.steps}
    not_yet = [s.id for s in resolved.steps if specs[s.id].scope is Scope.VOLUME]
    if not_yet:
        raise NotImplementedError(f"volume-scope operators are not executed by the Phase 1 batch executor (steps: {not_yet})")

    out_dir = Path(out_dir)
    _prepare_out_dir(out_dir, overwrite)
    scratch = out_dir / "scratch"
    scratch.mkdir()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    source = volume.frames(resolved.input.channel, resolved.input.z)
    if max_frames is not None:
        source = source[:max_frames]
    if source.shape[0] == 0:
        raise ValueError("input has no frames")
    ctx = Context(
        frame_shape=(int(source.shape[1]), int(source.shape[2])),
        n_frames=int(source.shape[0]),
        spacing_yx=(volume.spacing["y"], volume.spacing["x"]),
        space_unit=volume.space_unit,
        frame_interval=volume.spacing["t"],
        time_unit=volume.time_unit,
        seed=resolved.seed,
    )
    prov.write_json(out_dir / "recipe.json", resolved.model_dump(mode="json"))

    values: dict[str, Any] = {SOURCE: source}
    kinds = resolved.kinds()
    step_records: list[dict[str, Any]] = []
    violations: list[Violation] = []
    status = "completed"
    stop_error: QCGateError | None = None
    workers = workers or min(8, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, step in enumerate(resolved.steps, 1):
            spec = specs[step.id]
            say(f"[{i}/{len(resolved.steps)}] {step.id}: {spec.name} ({spec.scope.value})")
            params = spec.params.model_validate(step.params)
            inputs = {name: values[ref] for name, ref in step.inputs.items()}
            ts = time.perf_counter()
            if spec.scope is Scope.FRAME:
                outputs, metrics = _run_frame_step(spec, params, inputs, ctx, scratch, step.id, pool)
            else:
                outputs, metrics = _run_window_step(spec, params, inputs, ctx, scratch, step.id)
            for name, value in outputs.items():
                values[f"{step.id}.{name}"] = value
            failed = evaluate(step.id, metrics, step.qc)
            violations.extend(failed)
            step_records.append(
                {
                    "id": step.id,
                    "op": spec.name,
                    "op_version": spec.version,
                    "scope": spec.scope.value,
                    "duration_s": round(time.perf_counter() - ts, 4),
                    "qc": _finite(metrics),
                    "qc_violations": [v.to_json() for v in failed],
                }
            )
            for v in failed:
                say(f"    ! {v.message()}")
            stops = [v for v in failed if v.on_fail == "stop"]
            if stops:
                status = "stopped_by_qc"
                stop_error = QCGateError("; ".join(v.message() for v in stops))
                break

    # -- deliverables ----------------------------------------------------------------
    tables = {ref: v for ref, v in values.items() if kinds.get(ref) is Kind.TABLE}
    summary = {ref: v for ref, v in values.items() if kinds.get(ref) is Kind.SUMMARY}
    files: dict[str, Path] = {}
    (out_dir / "tables").mkdir()
    for ref, table in tables.items():
        path = out_dir / "tables" / f"{ref}.csv"
        table.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        files[f"tables/{ref}.csv"] = path
    prov.write_json(out_dir / "summary.json", _json_safe(summary))
    files["summary.json"] = out_dir / "summary.json"

    review_files: list[Path] = []
    if resolved.review and resolved.review.image in values and resolved.review.labels in values:
        review_files = _write_review(values[resolved.review.image], values[resolved.review.labels], resolved.review.n_frames, out_dir / "review")

    if keep_intermediates:
        intermediates = sorted(p.name for p in scratch.iterdir())
    else:
        intermediates = []
        values.clear()
        shutil.rmtree(scratch, ignore_errors=True)

    per_file, result_digest = prov.digest_files(files)
    input_record: dict[str, Any] = volume.describe()
    input_record.update({"channel": resolved.input.channel, "z": resolved.input.z, "frames_used": ctx.n_frames})
    if hash_input and volume.source and Path(volume.source).exists():
        say("hashing input")
        digest, size = prov.sha256_path(Path(volume.source))
        input_record.update({"sha256": digest, "bytes": size})
    record = {
        "status": status,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 3),
        "recipe": {"name": resolved.name, "sha256": resolved.digest()},
        "seed": resolved.seed,
        "device": select_device(),
        "input": input_record,
        "environment": prov.environment(),
        "steps": step_records,
        "outputs": per_file,
        "review": [p.name for p in review_files],
        "intermediates": intermediates,
        "result_digest": result_digest,
    }
    prov.write_json(out_dir / "provenance.json", record)
    if stop_error is not None:
        raise stop_error
    return RunResult(out_dir, resolved, status, summary, tables, record, violations)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value
