"""Plugins and models: the contract for operators backed by trained models.

A plugin is a Python package that declares an entry point in the group ``vexel.plugins``::

    [project.entry-points."vexel.plugins"]
    droplet_cells = "vexel_droplet_cells:register"

``register()`` is called once, the first time Vexel looks up operators. It registers
operators with :func:`vexel.ops.registry.register_op` and the models they use with
:func:`register_model`. A plugin that fails to load is reported by ``vexel plugins`` and
otherwise ignored: it never stops Vexel from starting.

Model-backed operators follow three rules, so that results stay reproducible and the
licence of every result is known:

1. **The model is named in the recipe.** The operator's parameters have a field
   ``model`` holding ``"<id>@<version>"``; no "latest". The executor records the model
   card of every step that names a model in ``provenance.json``.
2. **Weights are verified.** A model card carries the SHA-256 of its weight file;
   :func:`load_model` refuses a file that does not match. Weights are not downloaded
   behind the user's back: ``vexel plugins`` shows where a missing file should go.
3. **The training data is declared.** A card lists its training datasets with their
   licences; :attr:`ModelCard.commercial_use` is ``False`` when any of them (or the
   weights) are non-commercial, and ``vexel plugins`` says so.

Operators keep the scope rules of the registry: ``frame`` operators can run in the
stream executor, so a model used live must work frame by frame.

Each task fixes what the operator takes and gives, so a model-backed operator can
replace a classical one in a recipe without touching the steps after it:

=====================  ================================  ==================================================
task                   inputs                            outputs
=====================  ================================  ==================================================
``segment.instances``  ``image``                         ``labels`` (instances, 0 = background); optional
                                                         ``probability`` image
``segment.classes``    ``image``                         ``labels`` (class index per pixel, 0 = background)
``count.inclusions``   ``image``, ``labels`` (droplets)  ``table`` with ``frame``, ``label``,
                                                         ``n_inclusions`` (+ optional ``confidence``);
                                                         ``labels`` (marks of what was counted)
``classify.objects``   ``image``, ``labels``             ``table`` with ``frame``, ``label``, ``class``,
                                                         ``confidence``
=====================  ================================  ==================================================
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

ENTRY_POINT_GROUP = "vexel.plugins"
TASKS = {
    "segment.instances": ({"image"}, {"labels"}),
    "segment.classes": ({"image"}, {"labels"}),
    "count.inclusions": ({"image", "labels"}, {"table", "labels"}),
    "classify.objects": ({"image", "labels"}, {"table"}),
}
NON_COMMERCIAL = ("nc", "non-commercial", "noncommercial", "research only", "academic")


@dataclass(frozen=True)
class Dataset:
    name: str
    license: str
    url: str = ""


@dataclass(frozen=True)
class ModelCard:
    id: str
    version: str
    task: str
    framework: str  # "onnx", "sklearn", "torch", "numpy", ...
    license: str  # licence of the weights
    training_data: tuple[Dataset, ...] = ()
    file: str = ""  # weight file name under models_dir()/<id>/<version>/, or an absolute path
    sha256: str = ""
    url: str = ""  # where to download the weights by hand
    input: dict[str, Any] = field(default_factory=dict)  # channels, expected pixel size, normalisation
    output: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    citation: str = ""

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"model {self.id}: unknown task {self.task!r}; one of {sorted(TASKS)}")
        if not self.version or "@" in self.id:
            raise ValueError(f"model {self.id!r}: needs an id without '@' and a version")

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def commercial_use(self) -> bool | None:
        """False if the weights or any training dataset are non-commercial; None if a licence is unknown."""
        licences = [self.license] + [d.license for d in self.training_data]
        if any(any(t in lic.lower() for t in NON_COMMERCIAL) for lic in licences):
            return False
        if any(not lic or lic.lower() in ("unknown", "待核实") for lic in licences):
            return None
        return True

    def path(self) -> Path | None:
        if not self.file:
            return None
        p = Path(self.file)
        return p if p.is_absolute() else models_dir() / self.id / self.version / self.file

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["ref"] = self.ref
        out["commercial_use"] = self.commercial_use
        return out


class Model(Protocol):
    """What a loader returns: the card and a predictor (its signature depends on the task)."""

    card: ModelCard

    def predict(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class PluginInfo:
    name: str
    target: str
    distribution: str = ""
    version: str = ""
    status: str = "loaded"  # loaded | failed
    error: str = ""
    ops: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)


_MODELS: dict[str, tuple[ModelCard, Callable[[ModelCard], Model]]] = {}
_LOADED: dict[str, Model] = {}
_PLUGINS: list[PluginInfo] | None = None


def models_dir() -> Path:
    from vexel.agent.memory import home

    return home() / "models"


def register_model(card: ModelCard, loader: Callable[[ModelCard], Model]) -> None:
    if card.ref in _MODELS:
        raise ValueError(f"model {card.ref} is already registered")
    _MODELS[card.ref] = (card, loader)


def model_cards() -> list[ModelCard]:
    load_plugins()
    return [c for c, _ in _MODELS.values()]


def get_card(ref: str) -> ModelCard:
    load_plugins()
    if "@" not in ref:
        raise ValueError(f"model {ref!r}: name the version, e.g. {ref}@1.0 (see `vexel plugins`)")
    try:
        return _MODELS[ref][0]
    except KeyError:
        raise KeyError(f"unknown model {ref!r}; installed: {sorted(_MODELS) or 'none'} (see `vexel plugins`)") from None


def file_status(card: ModelCard) -> str:
    p = card.path()
    if p is None:
        return "built in"
    if not p.exists():
        return f"missing: put the weights at {p}" + (f" (download: {card.url})" if card.url else "")
    if card.sha256 and _sha256(p) != card.sha256:
        return "checksum mismatch"
    return "ok"


def load_model(ref: str) -> Model:
    """The model named ``ref`` (``id@version``), weights verified, loaded once per process."""
    card = get_card(ref)
    if ref not in _LOADED:
        status = file_status(card)
        if status not in ("ok", "built in"):
            raise FileNotFoundError(f"model {ref}: {status}") if status.startswith("missing") else ValueError(f"model {ref}: {status}")
        model = _MODELS[ref][1](card)
        _LOADED[ref] = model
    return _LOADED[ref]


def model_record(params: Any, spec: Any = None) -> dict[str, Any] | None:
    """Provenance entry for a step whose parameters name a model (``params.model``); also
    checks that the operator has the inputs and outputs its model's task requires."""
    ref = getattr(params, "model", None)
    if not isinstance(ref, str):
        return None
    card = get_card(ref)
    if spec is not None:
        need_in, need_out = TASKS[card.task]
        if not need_in <= set(spec.inputs) or not need_out <= set(spec.outputs):
            raise ValueError(f"operator {spec.name} uses model {card.ref} (task {card.task}), which needs inputs {sorted(need_in)} "
                             f"and outputs {sorted(need_out)}")  # fmt: skip
    return {"ref": card.ref, "task": card.task, "framework": card.framework, "sha256": card.sha256, "license": card.license,
            "training_data": [asdict(d) for d in card.training_data], "commercial_use": card.commercial_use}  # fmt: skip


def load_plugins(force: bool = False) -> list[PluginInfo]:
    """Call every installed plugin's ``register()`` once; failures are recorded, not raised."""
    global _PLUGINS
    if _PLUGINS is not None and not force:
        return _PLUGINS
    from vexel.ops import registry

    _PLUGINS = []
    for ep in md.entry_points(group=ENTRY_POINT_GROUP):
        info = PluginInfo(ep.name, ep.value)
        if ep.dist is not None:
            info.distribution, info.version = ep.dist.name, ep.dist.version
        ops_before, models_before = set(registry._REGISTRY), set(_MODELS)
        try:
            ep.load()()
        except Exception as e:  # noqa: BLE001  a broken plugin must not break Vexel
            info.status, info.error = "failed", f"{type(e).__name__}: {e}"
        info.ops = sorted(set(registry._REGISTRY) - ops_before)
        info.models = sorted(set(_MODELS) - models_before)
        _PLUGINS.append(info)
    return _PLUGINS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
