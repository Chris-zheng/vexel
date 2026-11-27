"""Strongly typed operator registry.

Every operator declares:

* a **scope** - what it needs to see at once:
  ``frame`` (one 2-D image), ``window`` (the time axis), ``volume`` (a 3-D stack).
  The stream executor will only accept ``frame`` and ``window`` operators.
* typed **inputs/outputs** (:class:`Kind`), checked when a recipe is resolved.
* a pydantic **params** model; unknown parameters are rejected.

Operator function signature::

    fn(params, ctx, **inputs) -> OpResult

For ``frame`` operators array inputs are 2-D numpy arrays and ``ctx.frame`` is set;
the executor maps the function over time. ``window`` operators receive lazy
``(T, Y, X)`` dask arrays and whole tables, and may return lazy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class Scope(str, Enum):
    FRAME = "frame"
    WINDOW = "window"
    VOLUME = "volume"


class Kind(str, Enum):
    IMAGE = "image"  # intensity array
    LABELS = "labels"  # integer instance labels, 0 = background
    TABLE = "table"  # pandas DataFrame; per-frame tables carry a ``frame`` column
    SUMMARY = "summary"  # flat dict of JSON scalars
    MASK = "mask"  # boolean array, True = marked (e.g. saturated pixels, left out of measurements)
    HEIGHTMAP = "heightmap"  # float heights per pixel
    GEOMETRY = "geometry"  # JSON-able shapes (outlines, edges, labels) to draw
    ROI = "roi"  # regions of interest


class Params(BaseModel):
    """Base class for operator parameters: frozen, unknown keys rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class Context:
    """Calibration and run state handed to every operator."""

    frame_shape: tuple[int, int]
    n_frames: int
    spacing_yx: tuple[float, float] = (1.0, 1.0)
    space_unit: str = "px"
    frame_interval: float = 1.0
    time_unit: str = "frame"
    seed: int = 0
    frame: int | None = None

    def for_frame(self, frame: int) -> "Context":
        return replace(self, frame=frame)


@dataclass
class OpResult:
    outputs: dict[str, Any]
    qc: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OpSpec:
    name: str
    scope: Scope
    inputs: dict[str, Kind]
    outputs: dict[str, Kind]
    params: type[Params]
    fn: Callable[..., OpResult]
    optional_inputs: frozenset[str] = frozenset()
    version: str = "1"
    doc: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "version": self.version,
            "doc": self.doc,
            "inputs": {k: v.value for k, v in self.inputs.items()},
            "optional_inputs": sorted(self.optional_inputs),
            "outputs": {k: v.value for k, v in self.outputs.items()},
            "params": self.params.model_json_schema().get("properties", {}),
        }


_REGISTRY: dict[str, OpSpec] = {}


def register_op(
    name: str,
    *,
    scope: Scope,
    inputs: dict[str, Kind],
    outputs: dict[str, Kind],
    params: type[Params] = Params,
    optional_inputs: tuple[str, ...] = (),
    version: str = "1",
) -> Callable[[Callable[..., OpResult]], Callable[..., OpResult]]:
    def decorator(fn: Callable[..., OpResult]) -> Callable[..., OpResult]:
        if name in _REGISTRY:
            raise ValueError(f"operator {name!r} is already registered")
        unknown = set(optional_inputs) - set(inputs)
        if unknown:
            raise ValueError(f"{name}: optional inputs {sorted(unknown)} are not declared inputs")
        if scope is Scope.FRAME and Kind.TABLE in inputs.values():
            raise ValueError(f"{name}: frame-scope operators take arrays only")
        doc = (fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        _REGISTRY[name] = OpSpec(
            name=name,
            scope=Scope(scope),
            inputs=dict(inputs),
            outputs=dict(outputs),
            params=params,
            fn=fn,
            optional_inputs=frozenset(optional_inputs),
            version=version,
            doc=" ".join(doc.split()),
        )
        return fn

    return decorator


def load_builtin_ops() -> None:
    """Import every module that registers operators (idempotent)."""
    import importlib

    from vexel.modules import packs
    from vexel.ops import classify, measure, preprocess, segment, stats, track  # noqa: F401

    for pack in packs():  # each module's operators
        try:
            importlib.import_module(f"vexel.packs.{pack}.ops")
        except ModuleNotFoundError as e:
            if e.name != f"vexel.packs.{pack}.ops":
                raise  # a real import error inside the pack
    from vexel.plugins import load_plugins

    load_plugins()  # operators from installed plugins (entry point group "vexel.plugins")


def get_op(name: str) -> OpSpec:
    load_builtin_ops()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown operator {name!r}; run `vexel ops` to list available operators") from None


def list_ops() -> list[OpSpec]:
    load_builtin_ops()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def allowed_params_from_schema(step_id: str, spec: OpSpec) -> dict[str, tuple[Any, ...]]:
    """Derive agent allowlist rules ``{"<step_id>.<field>": ("enum", [...]) | ("float", lo, hi)}``
    directly from an operator's Pydantic ``Params`` JSON schema."""
    schema = spec.params.model_json_schema()
    out: dict[str, tuple[Any, ...]] = {}
    for field_name, meta in schema.get("properties", {}).items():
        key = f"{step_id}.{field_name}"
        if "enum" in meta:
            out[key] = ("enum", list(meta["enum"]))
            continue
        t = meta.get("type")
        if t == "boolean":
            out[key] = ("enum", [True, False])
        elif t in ("number", "integer"):
            lo = meta.get("minimum", meta.get("exclusiveMinimum", -1e12))
            hi = meta.get("maximum", meta.get("exclusiveMaximum", 1e12))
            out[key] = ("float", float(lo), float(hi))
        elif "anyOf" in meta:
            for branch in meta["anyOf"]:
                if "enum" in branch:
                    out[key] = ("enum", list(branch["enum"]))
                    break
                if branch.get("type") in ("number", "integer"):
                    lo = branch.get("minimum", branch.get("exclusiveMinimum", -1e12))
                    hi = branch.get("maximum", branch.get("exclusiveMaximum", 1e12))
                    out[key] = ("float", float(lo), float(hi))
                    break
    return out

