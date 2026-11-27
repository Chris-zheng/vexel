"""Application modules: independent analysis areas on one platform.

A module (microfluidic droplets, SEM, ...) brings its own operators and recipes (its
packs), desktop window, run report, batch metrics and review adapter. Everything else is
shared: data model, readers, operator registry, recipe engine, provenance, QC, batch
runs and group statistics, plugins, planning models, diagnostics.

The shared code never imports a module directly: it asks this registry. Entries name
their parts as ``"package.module:attribute"`` strings, imported only when used, so a
module's dependencies (and any failure in them) stay inside that module.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Module:
    id: str
    name: str  # shown in the interface (Chinese)
    name_en: str
    description: str
    packs: tuple[str, ...]  # packages under vexel.packs: operators (ops.py) and recipes (recipes/*.json)
    window: str  # "module:Class" of the desktop window; called with no arguments
    status: str = "ready"  # "ready" or "preview" (shown, not yet usable)
    detect: str | None = None  # "module:function(path) -> float in 0..1": how likely a file belongs here
    report: str | None = None  # "module:write_report(run_dir) -> Path" for this module's runs
    batch_metrics: str | None = None  # "module:LIST" of vexel.batch.stats.Metric
    review: str | None = None  # "module:adapter(recipe) -> ReviewAdapter | None"
    op_prefixes: tuple[str, ...] = field(default_factory=tuple)  # operators that mark a recipe as this module's

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def resolve(ref: str) -> Any:
    """Import ``"package.module:attribute"``."""
    module, _, attr = ref.partition(":")
    obj: Any = importlib.import_module(module)
    for part in attr.split(".") if attr else []:
        obj = getattr(obj, part)
    return obj


MODULES: tuple[Module, ...] = (
    Module(
        id="droplet",
        name="微流控液滴",
        name_en="Microfluidic droplets",
        description="液滴粒径与频率、包裹计数与泊松统计、荧光阴阳判读；录像分析、在线监控、批量与分组统计。",
        packs=("droplet",),
        window="vexel.ui.main_window:MainWindow",
        detect="vexel.packs.droplet.detect:score",
        report="vexel.report.html_reporter:write_report",
        batch_metrics="vexel.batch.stats:DROPLET_METRICS",
        review="vexel.packs.droplet.review:adapter",
        op_prefixes=("droplet.",),
    ),
    Module(
        id="sem",
        name="SEM 扫描电镜",
        name_en="SEM",
        description="自动标定与信息栏裁剪、手动测量、图像处理、尺寸计量与分割、倾转 2.5D 重建。",
        packs=("sem",),
        window="vexel.ui.sem_window:SemWindow",
        status="preview",  # until the acceptance on real images (plan M7)
        detect="vexel.packs.sem.detect:score",
        batch_metrics="vexel.batch.stats:SEM_METRICS",
        op_prefixes=("sem.",),
    ),
)


def modules(include_preview: bool = True) -> list[Module]:
    return [m for m in MODULES if include_preview or m.ready]


def get(module_id: str) -> Module:
    for m in MODULES:
        if m.id == module_id:
            return m
    raise KeyError(f"unknown module {module_id!r}; available: {', '.join(m.id for m in MODULES)}")


def packs() -> list[str]:
    out: list[str] = []
    for m in MODULES:
        out += [p for p in m.packs if p not in out]
    return out


def for_recipe(recipe: Any) -> Module:
    """The module a recipe belongs to: the one whose operators it uses (droplet by default,
    for recipes made only of shared operators)."""
    ops = [s.op for s in recipe.steps] if hasattr(recipe, "steps") else [s["op"] for s in recipe.get("steps", [])]
    for m in MODULES:
        if any(op.startswith(p) for op in ops for p in m.op_prefixes):
            return m
    return MODULES[0]


def for_path(path: str | Path) -> tuple[Module | None, dict[str, float]]:
    """The module most likely to handle this file, with every module's score. None when no
    module is reasonably sure (the person chooses)."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    scores: dict[str, float] = {}
    for m in MODULES:
        if m.detect is None:
            continue
        try:
            fn: Callable[[Path], float] = resolve(m.detect)
            scores[m.id] = float(fn(Path(path)))
        except Exception:  # noqa: BLE001  a detector failing must not stop the others
            scores[m.id] = 0.0
    if not scores:
        return None, scores
    best = max(scores, key=scores.__getitem__)
    ranked = sorted(scores.values(), reverse=True)
    sure = scores[best] >= 0.6 and (len(ranked) < 2 or ranked[0] - ranked[1] >= 0.2)
    return (get(best) if sure else None), scores


def batch_metrics(recipe: Any) -> list[Any]:
    """The group-statistics metrics of the module this recipe belongs to."""
    m = for_recipe(recipe)
    return list(resolve(m.batch_metrics)) if m.batch_metrics else []
