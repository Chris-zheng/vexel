"""Recipe: a frozen, validated description of an analysis.

A recipe is generated once (by a person now, by the planner later), resolved against
the operator registry, and then executed by deterministic operators. Resolution fills
in every default parameter, so the frozen recipe is a complete record: same recipe +
same data must give the same result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vexel.ops.registry import Kind, get_op

SOURCE = "source"


class RecipeError(ValueError):
    """The recipe is not executable (unknown operator, bad wiring, invalid parameters)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QCRule(_Model):
    """Bound on a QC metric. Frame-scope metrics are aggregated over time and addressed as
    ``<metric>.mean``, ``<metric>.min`` or ``<metric>.max``."""

    metric: str
    min: float | None = None
    max: float | None = None
    on_fail: Literal["warn", "stop"] = "warn"

    @model_validator(mode="after")
    def _has_bound(self) -> "QCRule":
        if self.min is None and self.max is None:
            raise ValueError("a QC rule needs 'min' and/or 'max'")
        return self


class Step(_Model):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    op: str
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    qc: list[QCRule] = Field(default_factory=list)


class InputSelect(_Model):
    """Which planes of the data a recipe reads.

    ``channel`` is ``source``. ``channels`` names further channels, each addressable as
    ``source.<name>`` (e.g. ``{"fluo": 1}`` -> ``source.fluo``), so a recipe can segment on
    one channel and measure another. Channel numbers are 0-based and can be overridden
    (``input.channel``, ``input.channels.<name>``) without editing the recipe.
    """

    channel: int = Field(0, ge=0)
    z: int = Field(0, ge=0)
    channels: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_names(self) -> "InputSelect":
        import re

        for name, index in self.channels.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"channel name {name!r} must be lower-case letters, digits or '_' and start with a letter")
            if index < 0:
                raise ValueError(f"channel {name!r} must be >= 0")
        return self

    def sources(self) -> dict[str, int]:
        """Value reference -> channel number, for every channel the recipe reads."""
        return {SOURCE: self.channel, **{f"{SOURCE}.{k}": v for k, v in self.channels.items()}}


class Review(_Model):
    """Overlay images written for human review: ``labels`` outlines in red on top of
    ``image``, and optional ``marks`` (e.g. detected cells) outlined in yellow."""

    image: str = SOURCE
    labels: str
    marks: str | None = None
    n_frames: int = Field(4, ge=1, le=64)


class Recipe(_Model):
    schema_version: Literal[1] = 1
    name: str
    description: str = ""
    seed: int = 0
    input: InputSelect = Field(default_factory=InputSelect)
    steps: list[Step] = Field(min_length=1)
    review: Review | None = None

    # -- loading -----------------------------------------------------------------

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        try:
            return cls.model_validate_json(text)
        except ValidationError as e:
            raise RecipeError(f"invalid recipe: {e}") from e

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Recipe":
        """Load a recipe file, or a built-in recipe by name."""
        from vexel.packs import builtin_recipes

        path = Path(name_or_path)
        if path.is_file():
            return cls.from_json(path.read_text(encoding="utf-8"))
        builtin = builtin_recipes()
        if str(name_or_path) in builtin:
            return cls.from_json(builtin[str(name_or_path)])
        raise RecipeError(f"recipe {str(name_or_path)!r} is neither a file nor a built-in recipe ({', '.join(builtin)})")

    # -- resolution --------------------------------------------------------------

    def kinds(self) -> dict[str, Kind]:
        """Every addressable value (``source`` and ``step.output``) and its kind."""
        out = {ref: Kind.IMAGE for ref in self.input.sources()}
        for step in self.steps:
            for name, kind in get_op(step.op).outputs.items():
                out[f"{step.id}.{name}"] = kind
        return out

    def resolve(self) -> "Recipe":
        """Validate against the registry and return a copy with all parameters explicit."""
        available: dict[str, Kind] = {ref: Kind.IMAGE for ref in self.input.sources()}
        seen: set[str] = set()
        steps: list[Step] = []
        for step in self.steps:
            where = f"step {step.id!r}"
            if step.id in seen or step.id == SOURCE:
                raise RecipeError(f"{where}: duplicate or reserved step id")
            seen.add(step.id)
            try:
                spec = get_op(step.op)
            except KeyError as e:
                raise RecipeError(f"{where}: {e.args[0]}") from None
            unknown = set(step.inputs) - set(spec.inputs)
            if unknown:
                raise RecipeError(f"{where}: {spec.name} has no input(s) {sorted(unknown)} (expects {sorted(spec.inputs)})")
            missing = set(spec.inputs) - set(step.inputs) - spec.optional_inputs
            if missing:
                raise RecipeError(f"{where}: {spec.name} is missing required input(s) {sorted(missing)}")
            for name, ref in step.inputs.items():
                if ref not in available:
                    raise RecipeError(f"{where}: input {name!r} refers to {ref!r}, which no earlier step produces")
                if available[ref] is not spec.inputs[name]:
                    raise RecipeError(
                        f"{where}: input {name!r} expects {spec.inputs[name].value} but {ref!r} is {available[ref].value}"
                    )
            try:
                params = spec.params.model_validate(step.params).model_dump(mode="json")
            except ValidationError as e:
                raise RecipeError(f"{where}: invalid parameters for {spec.name}: {e}") from e
            steps.append(step.model_copy(update={"params": params}))
            for name, kind in spec.outputs.items():
                available[f"{step.id}.{name}"] = kind
        if self.review is not None:
            refs = [(self.review.image, Kind.IMAGE), (self.review.labels, Kind.LABELS)]
            if self.review.marks is not None:
                refs.append((self.review.marks, Kind.LABELS))
            for ref, kind in refs:
                if available.get(ref) is not kind:
                    raise RecipeError(f"review: {ref!r} is not an available {kind.value}")
        return self.model_copy(update={"steps": steps})

    def with_overrides(self, overrides: dict[str, Any]) -> "Recipe":
        """Apply ``{"step.param": value}`` overrides (validated on the next ``resolve``).

        ``input.channel``, ``input.z`` and ``input.channels.<name>`` select data planes.
        """
        steps = {s.id: s for s in self.steps}
        patched: dict[str, dict[str, Any]] = {}
        selection = self.input.model_dump()
        for key, value in overrides.items():
            if key.startswith("input."):
                field_name = key[len("input.") :]
                if field_name in ("channel", "z"):
                    selection[field_name] = value
                elif field_name.startswith("channels.") and field_name[len("channels.") :] in selection["channels"]:
                    selection["channels"][field_name[len("channels.") :]] = value
                else:
                    known = ["input.channel", "input.z"] + [f"input.channels.{k}" for k in selection["channels"]]
                    raise RecipeError(f"override {key!r}: unknown input setting (have {', '.join(known)})")
                continue
            step_id, _, param = key.partition(".")
            if step_id not in steps or not param:
                raise RecipeError(f"override {key!r}: expected '<step>.<param>' with step in {sorted(steps)}")
            patched.setdefault(step_id, dict(steps[step_id].params))[param] = value
        new_steps = [s.model_copy(update={"params": patched[s.id]}) if s.id in patched else s for s in self.steps]
        try:
            new_input = InputSelect.model_validate(selection)
        except ValidationError as e:
            raise RecipeError(f"invalid input override: {e}") from e
        return self.model_copy(update={"steps": new_steps, "input": new_input})

    # -- identity ----------------------------------------------------------------

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
