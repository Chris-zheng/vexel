"""Stream executor: runs a recipe on frames as they arrive (Phase 3).

Frame-scope operators run unchanged, one frame at a time. The two window operators the
droplet recipes need while frames arrive have online counterparts here:

* ``preprocess.static_background``: the temporal median is first taken over the opening
  frames (a warm-up buffer), then refreshed from a ring of recent frames;
* ``track.link``: detections are linked frame by frame with the same flow estimate,
  cutoff and gap closing as the batch tracker; a track is closed once it has not been
  seen for longer than the gap allowance.

Window operators that only read tables (droplet summary, Poisson, positivity, flags) are
the batch operators themselves: they run periodically on the tracks closed within a
rolling window (rolling statistics, alarms) and once at the end on all of them, which
gives the same deliverables as a batch run (tables, summary, review, provenance).

Frames are taken from the source through a bounded queue. When processing falls behind
a paced source, frames that find the queue full are dropped and counted; the drop count
and the share of frames processed are recorded. With an unpaced source the queue blocks
instead, which measures the software's own throughput.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from vexel.core.device import select_device
from vexel.engine import provenance as prov
from vexel.engine.batch_executor import _check_outputs, _finite, _json_safe, _overlay, _prepare_out_dir
from vexel.engine.qc import Violation, aggregate_frames, evaluate
from vexel.engine.recipe import SOURCE, Recipe
from vexel.ops.registry import Context, Kind, Scope, get_op
from vexel.ops.track import LinkParams, displacements, global_flow, pair_shift
from vexel.plugins import model_record
from vexel.sources.replay import Frame, StreamSource

_BACKGROUND = "preprocess.static_background"
_TRACK = "track.link"
_END = object()
Progress = Callable[[str], None]


@dataclass
class StreamSettings:
    queue_size: int = 64  # frames waiting to be processed
    max_stride: int = 16  # when behind a paced source, process every n-th frame (n doubles up to this)
    update_every: int = 50  # processed frames between rolling updates
    window: int = 500  # frames covered by the rolling statistics
    background_stride: int = 2  # every n-th frame enters the background ring
    background_refresh: int = 32  # frames between background recomputations
    flow_warmup: int = 32  # frames buffered to estimate the flow before linking starts
    drift_min_droplets: int = 20  # droplets in the window before drift is judged
    drift_diameter: float = 0.05  # relative change of the mean diameter that raises an alarm
    drift_frequency: float = 0.15  # relative change of the generation frequency
    max_event_frames: int = 20  # overlays saved for alarms


@dataclass
class StreamResult:
    out_dir: Path
    recipe: Recipe
    status: str
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    provenance: dict[str, Any]
    rolling: pd.DataFrame
    alarms: pd.DataFrame
    violations: list[Violation] = field(default_factory=list)


# -- online counterparts of the two window operators ----------------------------------------


class _Background:
    def __init__(self, params: Any, stride: int, refresh: int) -> None:
        self.params = params
        self.input_ref = SOURCE
        self.stride = max(1, stride)
        self.refresh = max(1, refresh)
        self.samples: deque[np.ndarray] = deque(maxlen=params.max_frames)
        self.background: np.ndarray | None = None
        self._since = 0
        self._residuals: list[float] = []

    def seed(self, frames: list[np.ndarray]) -> None:
        for f in frames[-self.params.max_frames :]:
            self.samples.append(f)
        self._compute()

    def _compute(self) -> None:
        stack = np.stack(self.samples).astype(np.float32)
        self.background = np.median(stack, axis=0).astype(np.float32)
        self._residuals.append(float(np.abs(stack - self.background).mean()))
        self._since = 0

    def update(self, index: int, frame: np.ndarray) -> None:
        if index % self.stride == 0:
            self.samples.append(frame)
            self._since += 1
            if self._since * self.stride >= self.refresh:
                self._compute()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        if self.params.mode == "absdiff":
            return np.abs(f - self.background)
        if self.params.mode == "subtract":
            return f - self.background
        return f / np.maximum(self.background, 1e-6)

    def qc(self) -> dict[str, float]:
        return {
            "frames_sampled": float(len(self.samples)),
            "background_mean": float(self.background.mean()) if self.background is not None else float("nan"),
            "residual_mean": float(np.mean(self._residuals)) if self._residuals else float("nan"),
        }


class _Tracker:
    """Frame-by-frame linking with the batch tracker's flow estimate, cutoff and gaps."""

    def __init__(self, params: LinkParams) -> None:
        self.params = params
        self.cutoff = params.max_distance_px**2
        self.flow: np.ndarray | None = None if params.estimate_shift else np.asarray(params.expected_shift_px, float)
        self.offset = np.zeros(2)
        self.prev_t: int | None = None
        self.prev_pos = np.empty((0, 2))
        self.recent: deque[np.ndarray] = deque(maxlen=3)
        self.active: dict[int, dict[str, Any]] = {}
        self.closed: list[pd.DataFrame] = []
        self.next_id = 0
        self.step = 0  # frames processed so far; gaps are counted in processed frames
        self.shifts: list[np.ndarray] = []


    def learn_flow(self, tables: list[pd.DataFrame]) -> None:
        """Global flow from the warm-up frames (needs at least one consecutive pair)."""
        if not self.params.estimate_shift:
            return
        search = 2.0 * self.params.max_distance_px
        vectors = []
        for a, b in zip(tables, tables[1:]):
            if len(a) and len(b) and int(b["frame"].iloc[0]) == int(a["frame"].iloc[0]) + 1:
                vectors.append(displacements(_pos(a), _pos(b), search))
        found = global_flow(np.concatenate(vectors), search) if vectors else None
        self.flow = found if found is not None else np.zeros(2)

    def push(self, t: int, table: pd.DataFrame) -> list[pd.DataFrame]:
        pos = _pos(table)
        if self.prev_t is not None:
            gap = t - self.prev_t
            if self.params.estimate_shift:
                vectors = displacements(self.prev_pos, pos, 2.0 * self.params.max_distance_px * gap)
                self.recent.append(pair_shift(vectors, self.flow * gap, self.params.max_distance_px) / gap)
                step = np.median(np.stack(self.recent), axis=0) * gap
            else:
                step = self.flow * gap
            self.offset = self.offset + step
            self.shifts.append(step / gap)
        moving = pos - self.offset

        self.step += 1
        oldest = self.step - 1 - self.params.max_gap_frames
        ids = [k for k, tr in self.active.items() if tr["last_step"] >= oldest]
        matched: dict[int, int] = {}
        if ids and len(moving):
            ends = np.array([self.active[k]["pos"] for k in ids])
            cost = ((ends[:, None, :] - moving[None, :, :]) ** 2).sum(-1)
            big = self.cutoff * 1e6 + 1.0
            rows, cols = linear_sum_assignment(np.where(cost <= self.cutoff, cost, big))
            matched = {int(c): ids[r] for r, c in zip(rows, cols) if cost[r, c] <= self.cutoff}
        records = table.to_dict("records")
        for j, rec in enumerate(records):
            k = matched.get(j)
            if k is None:
                k = self.next_id
                self.next_id += 1
                self.active[k] = {"rows": [], "first_t": t}

            tr = self.active[k]
            tr["rows"].append({**rec, "track_id": k})
            tr["pos"] = moving[j]
            tr["last_step"] = self.step
        self.prev_t, self.prev_pos = t, pos
        return self._close(lambda tr: tr["last_step"] < self.step - self.params.max_gap_frames)

    def finish(self) -> list[pd.DataFrame]:
        return self._close(lambda tr: True)

    def _close(self, done: Callable[[dict[str, Any]], bool]) -> list[pd.DataFrame]:
        out = []
        for k in [k for k, tr in self.active.items() if done(tr)]:
            tr = self.active.pop(k)
            df = pd.DataFrame(tr["rows"])
            df.attrs["first_t"] = tr["first_t"]
            out.append(df)
        self.closed.extend(out)
        return out

    def table(self, since: int | None = None, template: pd.DataFrame | None = None) -> pd.DataFrame:
        parts = [d for d in self.closed if since is None or d.attrs.get("first_t", 0) >= since]
        if not parts:
            empty = (template.iloc[:0] if template is not None else pd.DataFrame()).copy()
            empty["track_id"] = pd.Series(dtype=np.int64)
            return empty
        return pd.concat(parts, ignore_index=True).sort_values(["frame", "label"], kind="stable").reset_index(drop=True)

    def qc(self) -> dict[str, float]:
        lengths = [len(d) for d in self.closed]
        out = {"n_tracks": float(len(lengths)), "mean_track_length": float(np.mean(lengths)) if lengths else 0.0}
        if self.params.estimate_shift and self.shifts:
            med = np.median(np.stack(self.shifts), axis=0)
            out.update({"flow_dy_px": float(med[0]), "flow_dx_px": float(med[1])})
        return out


def _pos(table: pd.DataFrame) -> np.ndarray:
    if not len(table):
        return np.empty((0, 2))
    return np.column_stack([table["centroid_y_px"].to_numpy(float), table["centroid_x_px"].to_numpy(float)])


# -- the executor ----------------------------------------------------------------------------


class _Plan:
    """Splits a resolved recipe into per-frame work, the tracker, and table steps."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.kinds = recipe.kinds()
        self.frame_steps, self.table_steps = [], []
        self.background: list[str] = []
        self.track: str | None = None
        self.summary_min_frames = 0
        streamed: set[str] = set(recipe.input.sources())  # values that exist per frame
        for step in recipe.steps:
            spec = get_op(step.op)
            in_kinds = [self.kinds[ref] for ref in step.inputs.values()]
            if spec.name == _BACKGROUND:
                if step.inputs["image"] not in recipe.input.sources():
                    raise NotImplementedError(f"step {step.id!r}: a streamed background must read a source channel")
                self.background.append(step.id)
                self.frame_steps.append(step)
                streamed.update(f"{step.id}.{o}" for o in spec.outputs)
            elif spec.name == _TRACK:
                if self.track is not None:
                    raise NotImplementedError("the stream executor supports one tracking step")
                self.track = step.id
            elif spec.scope is Scope.FRAME:
                missing = [ref for ref in step.inputs.values() if ref not in streamed]
                if missing:
                    raise NotImplementedError(f"step {step.id!r} reads {missing}, which are not available frame by frame")
                self.frame_steps.append(step)
                streamed.update(f"{step.id}.{o}" for o in spec.outputs)
            elif spec.scope is Scope.WINDOW and all(k in (Kind.TABLE, Kind.SUMMARY) for k in in_kinds):
                self.table_steps.append(step)
                if spec.name == "droplet.summary":
                    self.summary_min_frames = int(step.params.get("min_track_frames", 3))
            else:
                raise NotImplementedError(f"step {step.id!r} ({spec.name}) cannot run on a stream")


def _concat(parts: list[pd.DataFrame], template: pd.DataFrame | None) -> pd.DataFrame:
    non_empty = [p for p in parts if len(p)]
    if non_empty:
        return pd.concat(non_empty, ignore_index=True)
    return template.iloc[:0].copy() if template is not None else pd.DataFrame({"frame": pd.Series(dtype=np.int64)})


def _numeric(summary: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)}


def run_stream(
    recipe: Recipe,
    source: StreamSource,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    settings: StreamSettings | None = None,
    progress: Progress | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    report: bool = True,
    workers: int | None = None,
    on_frame: Callable[[int, np.ndarray, np.ndarray, np.ndarray | None], None] | None = None,
) -> StreamResult:
    """Run ``recipe`` on the frames ``source`` delivers; write the same deliverables as a
    batch run plus ``stream/rolling.csv`` (rolling statistics), ``stream/alarms.csv`` and
    ``stream/events/`` (an overlay of the frame at each alarm).

    ``on_update`` receives every rolling snapshot; ``on_frame`` every processed frame's
    review image, labels and marks (called from the executor thread: keep it cheap)."""
    cfg = settings or StreamSettings()
    say = progress or (lambda _m: None)
    resolved = recipe.resolve()
    plan = _Plan(resolved)
    volume = source.volume
    out_dir = Path(out_dir)
    _prepare_out_dir(out_dir, overwrite)
    (out_dir / "stream" / "events").mkdir(parents=True)
    prov.write_json(out_dir / "recipe.json", resolved.model_dump(mode="json"))
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    frames_in = iter(source)
    try:
        first_frame = next(frames_in)
    except StopIteration:
        raise ValueError("input has no frames") from None
    first = first_frame.images[SOURCE]
    base_ctx = Context(
        frame_shape=(int(first.shape[0]), int(first.shape[1])),
        n_frames=source.n_frames or 1,
        spacing_yx=(volume.spacing["y"], volume.spacing["x"]),
        space_unit=volume.space_unit,
        frame_interval=volume.spacing["t"],
        time_unit=volume.time_unit,
        seed=resolved.seed,
    )
    dt = source.frame_interval_s

    specs = {s.id: get_op(s.op) for s in resolved.steps}
    params = {s.id: specs[s.id].params.model_validate(s.params) for s in resolved.steps}
    backgrounds = {sid: _Background(params[sid], cfg.background_stride, cfg.background_refresh) for sid in plan.background}
    for sid, bg in backgrounds.items():
        bg.input_ref = next(s for s in resolved.steps if s.id == sid).inputs["image"]
    track_step = next((s for s in resolved.steps if s.id == plan.track), None)
    tracker = _Tracker(params[plan.track]) if track_step is not None else None

    warmup = max([params[sid].max_frames for sid in plan.background] + [cfg.flow_warmup if tracker else 0, 1])

    frame_tables: dict[str, list[pd.DataFrame]] = {}
    templates: dict[str, pd.DataFrame] = {}
    frame_qc: dict[str, list[dict[str, float]]] = {s.id: [] for s in plan.frame_steps if s.id not in backgrounds}
    step_time: dict[str, float] = {s.id: 0.0 for s in resolved.steps}
    rv = resolved.review
    known = source.n_frames
    picks = set(np.unique(np.linspace(0, known - 1, min(known, rv.n_frames)).round().astype(int))) if rv and known else set()
    review_frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    # Length unknown (a watched folder): keep an evenly thinned set of frames, from which
    # the review frames are chosen at the end.
    sampled: list[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]]] = []
    sample_step = 1
    last_view: tuple[int, np.ndarray, np.ndarray, np.ndarray | None] | None = None

    def compute(fr: Frame, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, float]]:
        """The frame-scope operators on one frame (thread-safe: touches only its own values)."""
        ctx = base_ctx.for_frame(fr.index)
        qcs: dict[str, dict[str, float]] = {}
        times: dict[str, float] = {}
        for step in plan.frame_steps:
            if step.id in backgrounds:
                continue
            ts = time.perf_counter()
            spec = specs[step.id]
            result = spec.fn(params[step.id], ctx, **{k: values[ref] for k, ref in step.inputs.items()})
            _check_outputs(spec, result)
            for name, value in result.outputs.items():
                if spec.outputs[name] is Kind.TABLE:
                    value = value.copy()
                    value.insert(0, "frame", fr.index)
                values[f"{step.id}.{name}"] = value
            qcs[step.id] = result.qc
            times[step.id] = time.perf_counter() - ts
        return values, qcs, times

    def process(batch: list[Frame], learn: bool = True) -> list[pd.DataFrame | None]:
        """Backgrounds in frame order, the frame operators in parallel, bookkeeping in order.
        Returns, per frame, the table the tracker reads."""
        nonlocal last_view, sample_step
        prepared = []
        for fr in batch:
            values: dict[str, Any] = dict(fr.images)
            for sid, bg in backgrounds.items():
                ts = time.perf_counter()
                image = values[bg.input_ref]
                values[f"{sid}.image"] = bg.apply(image)
                if learn:  # warm-up frames are already in the background
                    bg.update(fr.index, image)
                step_time[sid] += time.perf_counter() - ts
            prepared.append(values)
        if len(batch) == 1:
            results = [compute(batch[0], prepared[0])]
        else:
            results = list(pool.map(compute, batch, prepared))
        out = []
        for fr, (values, qcs, times) in zip(batch, results):
            for step in plan.frame_steps:
                if step.id in backgrounds:
                    continue
                spec = specs[step.id]
                for name, kind in spec.outputs.items():
                    if kind is Kind.TABLE:
                        ref = f"{step.id}.{name}"
                        t = values[ref]
                        templates.setdefault(ref, t.iloc[:0].copy())
                        if len(t):
                            frame_tables.setdefault(ref, []).append(t)
                frame_qc[step.id].append(qcs[step.id])
                step_time[step.id] += times[step.id]
            if rv and rv.image in values and rv.labels in values:
                view = (values[rv.image], values[rv.labels], values.get(rv.marks) if rv.marks else None)
                last_view = (fr.index, *view)
                if known:
                    due_pick = next((p for p in sorted(picks) if p <= fr.index and p not in review_frames), None)
                    if due_pick is not None:
                        review_frames[due_pick] = view
                elif fr.index % sample_step == 0:
                    sampled.append((fr.index, view))
                    if len(sampled) > 4 * rv.n_frames:
                        del sampled[1::2]
                        sample_step *= 2
                if on_frame is not None:
                    on_frame(fr.index, *view)
            out.append(values.get(track_step.inputs["table"]) if track_step is not None else None)
        return out

    def link(t: int, table: pd.DataFrame | None) -> None:
        if tracker is None or table is None:
            return
        ts = time.perf_counter()
        tracker.push(t, table)
        step_time[plan.track] += time.perf_counter() - ts

    def table_values(since: int | None) -> dict[str, Any]:
        vals: dict[str, Any] = {}
        for ref, template in templates.items():
            df = _concat(frame_tables.get(ref, []), template)
            vals[ref] = df if since is None else df[df["frame"] >= since].reset_index(drop=True)
        if tracker is not None:
            vals[f"{plan.track}.table"] = tracker.table(since, templates.get(track_step.inputs["table"]))
        return vals

    def run_tables(vals: dict[str, Any], n_frames: int) -> dict[str, dict[str, float]]:
        ctx = dataclasses.replace(base_ctx, n_frames=max(1, n_frames), frame=None)
        metrics: dict[str, dict[str, float]] = {}
        for step in plan.table_steps:
            spec = specs[step.id]
            ts = time.perf_counter()
            result = spec.fn(params[step.id], ctx, **{k: vals[ref] for k, ref in step.inputs.items()})
            _check_outputs(spec, result)
            for name, value in result.outputs.items():
                vals[f"{step.id}.{name}"] = value
            metrics[step.id] = dict(result.qc)
            step_time[step.id] += time.perf_counter() - ts
        return metrics

    workers = workers or min(8, os.cpu_count() or 1)
    pool = ThreadPoolExecutor(max_workers=workers)

    # -- producer ------------------------------------------------------------------------
    q: queue.Queue = queue.Queue(maxsize=max(1, cfg.queue_size))
    stop = threading.Event()
    counts = {"received": 0, "dropped": 0, "sampled_out": 0, "overflow": 0, "stride": 1}
    stride_log: list[dict[str, int]] = []
    errors: list[BaseException] = []

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        frames = itertools.chain([first_frame], frames_in)
        # Paced: when the queue fills up, keep every n-th frame (n doubles while the queue
        # stays over half full, halves once it is nearly empty), so frames are thinned out
        # evenly rather than lost in bursts; a frame that still finds the queue full is lost.
        stride, since_change, hold = 1, 0, max(1, cfg.queue_size // 2)
        try:
            for fr in frames:
                if stop.is_set():
                    break
                counts["received"] += 1
                if not source.live:
                    if not put(fr):
                        break
                    continue
                since_change += 1
                fill = q.qsize() / q.maxsize
                if since_change >= hold and fill > 0.5 and stride < cfg.max_stride:
                    stride, since_change = stride * 2, 0
                    counts["stride"] = stride
                    stride_log.append({"frame": fr.index, "stride": stride})
                elif since_change >= hold and fill < 0.1 and stride > 1:
                    stride, since_change = stride // 2, 0
                    counts["stride"] = stride
                    stride_log.append({"frame": fr.index, "stride": stride})
                if fr.index % stride:
                    counts["sampled_out"] += 1
                    counts["dropped"] += 1
                    continue
                try:
                    q.put_nowait(fr)
                except queue.Full:
                    counts["overflow"] += 1
                    counts["dropped"] += 1
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            errors.append(e)
        finally:
            put(_END)

    producer = threading.Thread(target=produce, name="vexel-replay", daemon=True)
    producer.start()

    # -- consumer ------------------------------------------------------------------------
    rolling: list[dict[str, Any]] = []
    alarms: list[dict[str, Any]] = []
    active_alarms: set[tuple[str, str]] = set()
    reference: dict[str, float] = {}
    throughput = {"lost": 0}
    latencies: list[float] = []
    processed = 0
    last_index = -1
    events_written = 0
    status = "completed"
    say(f"warming up on {warmup} frame(s)")

    def alarm(t: int, kind: str, step: str, metric: str, value: float | None, limit: str, message: str) -> None:
        nonlocal events_written
        key = (kind, f"{step}.{metric}")
        if key in active_alarms:
            return
        active_alarms.add(key)
        image = ""
        if last_view is not None and events_written < cfg.max_event_frames:
            vt, img, lab, mk = last_view
            img = np.asarray(img, dtype=np.float32)
            lo, hi = np.percentile(img, (1, 99))
            path = out_dir / "stream" / "events" / f"frame_{vt:06d}_{len(alarms):03d}.png"
            iio.imwrite(path, (_overlay(img, np.asarray(lab), None if mk is None else np.asarray(mk), lo, hi) * 255).round().astype(np.uint8))
            image = f"events/{path.name}"
            events_written += 1
        row = {"frame": t, "time_s": t * dt if dt else None, "kind": kind, "step": step, "metric": metric,
               "value": value, "limit": limit, "message": message, "image": image}  # fmt: skip
        alarms.append(row)
        say(f"    ! {message}")

    def clear(kind: str, step: str, metric: str) -> None:
        active_alarms.discard((kind, f"{step}.{metric}"))

    def update(t: int) -> None:
        since = max(0, t + 1 - cfg.window)
        vals = table_values(since)
        metrics = run_tables(vals, t + 1 - since)
        wall = time.perf_counter() - t_stream
        snap: dict[str, Any] = {
            "frame": t,
            "time_s": t * dt if dt else None,
            "window_start": since,
            "received": counts["received"],
            "processed": processed,
            "dropped": counts["dropped"],
            "fps": processed / wall if wall > 0 else None,
            "latency_ms": float(np.median(latencies[-cfg.update_every :]) * 1e3) if latencies else None,
            "final": final_update,
        }
        for step in plan.table_steps:
            for name, kind in specs[step.id].outputs.items():
                if kind is Kind.SUMMARY:
                    snap.update({f"{step.id}.{k}": v for k, v in _numeric(vals[f"{step.id}.{name}"]).items()})
        rolling.append(snap)
        # recipe QC rules, once the window holds its full length of frames
        full = t + 1 - since >= cfg.window
        for step in plan.table_steps if full else []:
            failed = {v.metric: v for v in evaluate(step.id, metrics[step.id], step.qc)}
            for rule in step.qc:
                if rule.metric in failed:
                    v = failed[rule.metric]
                    alarm(t, "qc", step.id, rule.metric, v.value, f"min {rule.min} max {rule.max}", v.message())
                else:
                    clear("qc", step.id, rule.metric)
        # throughput: frames lost outright, or thinned so far that droplets can no longer be
        # told apart between processed frames (moving more than half their spacing)
        lost = counts["overflow"]
        if lost > throughput["lost"]:
            alarm(t, "throughput", "stream", "frames_lost", float(lost), "0",
                  f"processing behind: {lost} frame(s) lost with the queue full; counts may be incomplete")  # fmt: skip
        else:
            clear("throughput", "stream", "frames_lost")
        throughput["lost"] = lost
        # thinned so far that a droplet crossing the field is seen in too few processed frames
        if tracker is not None and tracker.shifts and plan.summary_min_frames:
            per_frame = float(np.max(np.abs(np.median(np.stack(tracker.shifts[-50:]), axis=0))))
            axis = int(np.argmax(np.abs(np.median(np.stack(tracker.shifts[-50:]), axis=0))))
            field = base_ctx.frame_shape[axis]
            sightings = field / max(per_frame * counts["stride"], 1e-9)
            if per_frame > 0 and sightings < plan.summary_min_frames:
                alarm(t, "throughput", "stream", "sightings", sightings, f">= {plan.summary_min_frames}",
                      f"processing behind: every {counts['stride']} frame(s) processed, so a droplet is seen about "
                      f"{sightings:.1f} times while crossing the field; counts may be incomplete")  # fmt: skip
            else:
                clear("throughput", "stream", "sightings")
        # drift of size and frequency against the first well-populated window
        n = snap.get("droplets.n_droplets", 0) or 0
        if n >= cfg.drift_min_droplets:
            for metric, tol in (("diameter_mean", cfg.drift_diameter), ("frequency", cfg.drift_frequency)):
                value = snap.get(f"droplets.{metric}")
                if value is None or not math.isfinite(value):
                    continue
                ref = reference.setdefault(metric, value)
                change = (value - ref) / ref if ref else 0.0
                if abs(change) > tol:
                    alarm(t, "drift", "droplets", metric, value, f"{ref:.4g} ± {tol:.0%}",
                          f"drift droplets.{metric} = {value:.4g}, {change:+.1%} from {ref:.4g}")  # fmt: skip
                else:
                    clear("drift", "droplets", metric)
        if on_update is not None:
            on_update({**snap, "alarms": [dict(r) for r in alarms]})  # alarms so far, for live displays

    final_update = False
    t_stream = time.perf_counter()
    buffer: list[Frame] = []
    warm = True
    next_update = cfg.update_every

    def advance(batch: list[Frame], tables: list[pd.DataFrame | None], timed: bool) -> None:
        nonlocal processed, last_index, next_update
        for fr, table in zip(batch, tables):
            link(fr.index, table)
            if timed and source.live:  # unpaced, the wait is only queueing
                latencies.append(time.monotonic() - fr.due)
            processed += 1
            last_index = fr.index
            if processed >= next_update:
                next_update += cfg.update_every
                update(last_index)

    try:
        ended = False
        while not ended:
            item = q.get()
            if item is _END:
                break
            if should_stop is not None and should_stop():
                status = "stopped"
                break
            batch = [item]
            while len(batch) < workers:  # take whatever else is already waiting
                try:
                    extra = q.get_nowait()
                except queue.Empty:
                    break
                if extra is _END:
                    ended = True
                    break
                batch.append(extra)
            if warm:
                buffer.extend(batch)
                if len(buffer) < warmup and not ended:
                    continue
                batch, buffer, warm = buffer, [], False
                tables = _start(batch, backgrounds, tracker, process)
                advance(batch, tables, timed=False)
                continue
            advance(batch, process(batch), timed=True)
        if warm and buffer:  # shorter than the warm-up
            advance(buffer, _start(buffer, backgrounds, tracker, process), timed=False)
    finally:
        stop.set()
        source.request_stop()
        producer.join(timeout=5)
        pool.shutdown(wait=True)
    if errors:
        raise errors[0]
    if processed == 0:
        raise ValueError("input has no frames")
    wall = time.perf_counter() - t_stream

    # -- final results: the batch table operators on everything --------------------------
    if tracker is not None:
        tracker.finish()
    vals = table_values(None)
    final_metrics = run_tables(vals, last_index + 1)
    if rolling and rolling[-1]["frame"] == last_index:
        rolling.pop()  # taken before the last tracks were closed
    final_update = True
    update(last_index)

    kinds = plan.kinds
    tables = {ref: v for ref, v in vals.items() if kinds.get(ref) is Kind.TABLE}
    summary = {ref: v for ref, v in vals.items() if kinds.get(ref) is Kind.SUMMARY}
    files: dict[str, Path] = {}
    (out_dir / "tables").mkdir()
    for ref, table in tables.items():
        path = out_dir / "tables" / f"{ref}.csv"
        table.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        files[f"tables/{ref}.csv"] = path
    prov.write_json(out_dir / "summary.json", _json_safe(summary))
    files["summary.json"] = out_dir / "summary.json"
    rolling_df = pd.DataFrame(rolling)
    alarms_df = pd.DataFrame(alarms, columns=["frame", "time_s", "kind", "step", "metric", "value", "limit", "message", "image"])
    rolling_df.to_csv(out_dir / "stream" / "rolling.csv", index=False, float_format="%.6g", lineterminator="\n")
    alarms_df.to_csv(out_dir / "stream" / "alarms.csv", index=False, lineterminator="\n")

    if not known and sampled:
        chosen = np.unique(np.linspace(0, len(sampled) - 1, min(len(sampled), rv.n_frames)).round().astype(int))
        review_frames = {sampled[i][0]: sampled[i][1] for i in chosen}
    review_files = []
    if review_frames:
        (out_dir / "review").mkdir()
        for t, (img, lab, mk) in sorted(review_frames.items()):
            img = np.asarray(img, dtype=np.float32)
            lo, hi = np.percentile(img, (1, 99))
            path = out_dir / "review" / f"frame_{t:06d}.png"
            iio.imwrite(path, (_overlay(img, np.asarray(lab), None if mk is None else np.asarray(mk), lo, hi) * 255).round().astype(np.uint8))
            review_files.append(path)

    violations: list[Violation] = []
    step_records = []
    for step in resolved.steps:
        spec = specs[step.id]
        if step.id in backgrounds:
            metrics = backgrounds[step.id].qc()
        elif step.id == plan.track:
            metrics = tracker.qc()
        elif step.id in final_metrics:
            metrics = final_metrics[step.id]
        else:
            metrics = aggregate_frames(frame_qc.get(step.id, []))
        failed = evaluate(step.id, metrics, step.qc)
        violations.extend(failed)
        step_records.append({
            "id": step.id, "op": spec.name, "op_version": spec.version, "scope": spec.scope.value,
            "streamed": step.id in backgrounds or step.id == plan.track,
            "duration_s": round(step_time[step.id], 4), "qc": _finite(metrics), "qc_violations": [v.to_json() for v in failed],
            **({"model": m} if (m := model_record(params[step.id], spec)) else {}),
        })  # fmt: skip

    lat = np.array(latencies) * 1e3
    stream_record = {
        "source": source.describe(),
        "frames_received": counts["received"],
        "frames_processed": processed,
        "frames_dropped": counts["dropped"],
        "frames_sampled_out": counts["sampled_out"],
        "frames_lost_queue_full": counts["overflow"],
        "sampling_changes": stride_log,
        "processed_fraction": processed / counts["received"] if counts["received"] else None,
        "processing_fps": processed / wall if wall > 0 else None,
        "wall_s": round(wall, 3),
        "warmup_frames": warmup,
        "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())} if lat.size else None,
        "settings": cfg.__dict__,
        "n_rolling_updates": len(rolling),
        "n_alarms": len(alarms),
        "note": "replay timing excludes camera readout and transfer; it is the software's speed, not end-to-end latency",
    }
    per_file, result_digest = prov.digest_files(files)
    input_record: dict[str, Any] = source.volume.describe()  # a watched folder knows its size only now
    # frames_used is the span covered (as in batch runs); thinned-out frames inside it are in "stream"
    input_record.update({"channel": resolved.input.channel, "channels": dict(resolved.input.channels), "z": resolved.input.z,
                         "frames_used": last_index + 1, "frames_processed": processed})  # fmt: skip
    record = {
        "mode": "stream",
        "status": status,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 3),
        "recipe": {"name": resolved.name, "sha256": resolved.digest()},
        "seed": resolved.seed,
        "device": select_device(),
        "input": input_record,
        "environment": prov.environment(),
        "steps": step_records,
        "stream": stream_record,
        "outputs": per_file,
        "review": [p.name for p in review_files],
        "review_flags": [],
        "intermediates": [],
        "result_digest": result_digest,
    }
    prov.write_json(out_dir / "provenance.json", _json_safe(record))
    if report:
        from vexel.report import write_report

        try:
            write_report(out_dir)
        except Exception as e:  # noqa: BLE001
            say(f"report not written: {type(e).__name__}: {e}")
    return StreamResult(out_dir, resolved, status, summary, tables, record, rolling_df, alarms_df, violations)


def _start(buffer: list[Frame], backgrounds: dict[str, _Background], tracker: _Tracker | None, process) -> list[pd.DataFrame | None]:
    """End of warm-up: seed the backgrounds, process the buffered frames and learn the flow
    from their detections; returns the tables for linking."""
    for bg in backgrounds.values():
        bg.seed([fr.images[bg.input_ref] for fr in buffer])
    tables = process(buffer, learn=False)
    if tracker is not None:
        tracker.learn_flow([t for t in tables if t is not None])
    return tables
