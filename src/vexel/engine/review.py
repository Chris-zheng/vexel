"""Review support: look at any frame of a finished run, and re-run it with corrections.

Runs keep tables and summaries but not their label images (they are scratch). The
:class:`FrameInspector` recomputes one frame through the recipe's frame steps (window
steps such as the static background are evaluated lazily, once), with the corrections
made so far applied, which is what a review screen needs. :func:`rerun_with_corrections`
re-executes the whole recipe with a correction file, so the corrected results are
exactly reproducible from recipe + input + corrections.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.engine.batch_executor import RunResult, run_recipe
from vexel.engine.corrections import Corrections, apply_label_edits
from vexel.engine.recipe import Recipe
from vexel.ops.registry import Context, Kind, Scope, get_op

CORRECTIONS_FILE = "corrections.json"
CORRECTED_DIR = "corrected"


def run_input(run_dir: str | Path) -> SpatialVolume:
    """The data a run analysed, opened with the calibration the run used."""
    from vexel.readers import open_volume

    run_dir = Path(run_dir)
    prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    inp = prov["input"]
    source = inp.get("source")
    stored = run_dir / "frames.zarr"  # a watched acquisition keeps its frames here
    if prov.get("stream", {}).get("source", {}).get("kind") == "watch" and stored.exists():
        source = str(stored)
    if not source or not Path(source).exists():
        raise FileNotFoundError(f"the input of this run is no longer at {source!r}")
    volume = open_volume(source)
    sp = inp.get("spacing", {})
    for axis in ("y", "x", "t"):
        if axis in sp:
            volume.spacing[axis] = float(sp[axis])
    volume.space_unit = inp.get("space_unit", volume.space_unit)
    volume.time_unit = inp.get("time_unit", volume.time_unit)
    return volume


def base_record(run_dir: str | Path) -> dict[str, Any]:
    prov = json.loads((Path(run_dir) / "provenance.json").read_text(encoding="utf-8"))
    return {"run": str(Path(run_dir).resolve()), "recipe_sha256": prov["recipe"]["sha256"], "result_digest": prov.get("result_digest")}


def load_corrections(run_dir: str | Path) -> Corrections:
    path = Path(run_dir) / CORRECTIONS_FILE
    return Corrections.load(path) if path.exists() else Corrections(base=base_record(run_dir))


def rerun_with_corrections(
    run_dir: str | Path,
    corrections: Corrections | None = None,
    out_dir: str | Path | None = None,
    *,
    progress: Any = None,
    tick: Any = None,
) -> RunResult:
    """Re-execute a run's recipe on its input with ``corrections`` (default: the run's
    ``corrections.json``) into ``out_dir`` (default ``<run>/corrected``), replacing a
    previous corrected run there. The corrections are saved with the run and copied
    into the corrected run."""
    run_dir = Path(run_dir)
    corrections = corrections if corrections is not None else load_corrections(run_dir)
    if not corrections.base:
        corrections.base = base_record(run_dir)
    corrections.save(run_dir / CORRECTIONS_FILE)
    recipe = Recipe.model_validate_json((run_dir / "recipe.json").read_text(encoding="utf-8"))
    prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    out = Path(out_dir) if out_dir is not None else run_dir / CORRECTED_DIR
    result = run_recipe(
        recipe, run_input(run_dir), out, overwrite=True, max_frames=prov["input"].get("frames_used"),
        hash_input=False, progress=progress, tick=tick, corrections=corrections,
    )  # fmt: skip
    corrections.save(out / CORRECTIONS_FILE)
    return result


class FrameInspector:
    """Recomputes single frames of a run through its frame steps, with label corrections."""

    def __init__(self, run_dir: str | Path, cache_frames: int = 8) -> None:
        self.run_dir = Path(run_dir)
        self.recipe = Recipe.model_validate_json((self.run_dir / "recipe.json").read_text(encoding="utf-8")).resolve()
        prov = json.loads((self.run_dir / "provenance.json").read_text(encoding="utf-8"))
        self.volume = run_input(self.run_dir)
        n = int(prov["input"].get("frames_used") or self.volume.sizes["T"])
        self.sources = {ref: self.volume.frames(c, self.recipe.input.z)[:n] for ref, c in self.recipe.input.sources().items()}
        first = self.sources["source"]
        self.n_frames = int(first.shape[0])
        self.ctx = Context(
            frame_shape=(int(first.shape[1]), int(first.shape[2])), n_frames=self.n_frames,
            spacing_yx=(self.volume.spacing["y"], self.volume.spacing["x"]), space_unit=self.volume.space_unit,
            frame_interval=self.volume.spacing["t"], time_unit=self.volume.time_unit, seed=self.recipe.seed,
        )  # fmt: skip
        self.kinds = self.recipe.kinds()
        self._lazy: dict[str, Any] = dict(self.sources)  # values defined over all frames (sources, window images)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._cache_frames = cache_frames
        self.corrections = load_corrections(self.run_dir)

    @property
    def review(self):
        return self.recipe.review

    def invalidate(self, frame: int | None = None) -> None:
        if frame is None:
            self._cache.clear()
        else:
            self._cache.pop(frame, None)

    def frame(self, t: int) -> dict[str, Any]:
        """Every per-frame value of frame ``t`` (images, labels, per-frame tables)."""
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t]
        values: dict[str, Any] = {}
        label_edits = self.corrections.label_edits()
        for step in self.recipe.steps:
            spec = get_op(step.op)
            in_kinds = [self.kinds[ref] for ref in step.inputs.values()]
            if spec.scope is Scope.FRAME:
                args = {k: self._value(ref, t, values) for k, ref in step.inputs.items()}
                result = spec.fn(spec.params.model_validate(step.params), self.ctx.for_frame(t), **args)
                for name, value in result.outputs.items():
                    ref = f"{step.id}.{name}"
                    if spec.outputs[name] is Kind.LABELS and t in label_edits.get(ref, {}):
                        value, _ = apply_label_edits(value, label_edits[ref][t])
                    values[ref] = value
            elif all(k is Kind.IMAGE for k in in_kinds) and all(k is Kind.IMAGE for k in spec.outputs.values()):
                if f"{step.id}.{next(iter(spec.outputs))}" not in self._lazy:  # e.g. the static background
                    result = spec.fn(spec.params.model_validate(step.params), self.ctx, **{k: self._lazy[ref] for k, ref in step.inputs.items()})
                    for name, value in result.outputs.items():
                        self._lazy[f"{step.id}.{name}"] = value
        for ref, lazy in self._lazy.items():
            values.setdefault(ref, np.asarray(lazy[t]))
        self._cache[t] = values
        if len(self._cache) > self._cache_frames:
            self._cache.popitem(last=False)
        return values

    def _value(self, ref: str, t: int, values: dict[str, Any]) -> Any:
        if ref in values:
            return values[ref]
        if ref in self._lazy:
            values[ref] = np.asarray(self._lazy[ref][t])
            return values[ref]
        raise KeyError(f"{ref} is not available frame by frame")

    def view(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Review image, labels and marks of frame ``t`` (as the recipe's review section names them)."""
        rv = self.review
        v = self.frame(t)
        return v[rv.image], v[rv.labels], (v.get(rv.marks) if rv.marks else None)


# -- what a review screen may edit: generic description, filled in by each pack ----------------



@dataclass(frozen=True)
class EditableField:
    """A per-object value a person may set (written with a ``table.set`` edit)."""

    key: str  # column written by the edit
    label: str  # shown to the user
    target: str  # table the edit applies to
    kind: str  # "int" | "bool"
    shown: str = ""  # column holding the current value, if different from ``key``
    help: str = ""


@dataclass(frozen=True)
class ReviewAdapter:
    """How a run's objects are reviewed: where the objects are listed, which label image
    the outline tools edit, how an object is excluded, which values can be set."""

    objects: str  # table with one row per reviewed object (and track_id or frame/centroid)
    labels: str  # labels output the outline tools edit
    exclude: str  # table a ``table.exclude`` edit targets (the tracking table removes whole tracks)
    flags: str | None = None  # table listing objects that need a look (frame, centroid, reasons)
    fields: list[EditableField] = field(default_factory=list)
    object_name: str = "目标"
    reasons: dict[str, str] = field(default_factory=dict)  # flag reason code -> text shown

    def describe(self, row: dict[str, Any]) -> str:
        return ""


def review_adapter(recipe: Recipe) -> ReviewAdapter:
    """The pack's adapter for this recipe, or a generic one built from its review section."""
    from vexel.packs import review_adapter as from_packs

    found = from_packs(recipe)
    if found is not None:
        return found
    kinds = recipe.kinds()
    tables = [ref for ref, k in kinds.items() if k is Kind.TABLE]
    tracked = next((s.id + ".table" for s in recipe.steps if s.op == "track.link"), None)
    objects = tracked or (tables[0] if tables else "")
    return ReviewAdapter(objects=objects, labels=recipe.review.labels if recipe.review else "", exclude=objects)
