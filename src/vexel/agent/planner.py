"""Planner: from data and a goal to a recipe draft, checked on a trial run.

Offline and deterministic (the rule engine of the model abstraction layer). Steps:

1. **Goal**: what the user wants measured (cells per droplet? positive droplets?),
   read from free text in Chinese or English; unspecified parts are left to the data.
2. **Draft**: a built-in recipe and the settings that differ from it, from the probe's
   fingerprint (droplet and fluorescence channels, flow direction, cells seen), each
   decision with its reason. A close enough past run in recipe memory that passed QC
   lends its settings.
3. **Trial**: the draft runs on the first few hundred frames. QC rule violations and
   plain failures (no droplets counted, outlines not round) trigger the next rule-based
   adjustment, up to a few rounds; the best attempt is kept.
4. **Result**: a plan with the recipe, the reasons, every attempt and its metrics, and a
   status: ``ready`` (passed the trial), ``check`` (runs, but some QC warnings remain) or
   ``needs_user`` (no usable segmentation: brush training suggested).

The plan is a draft: the recipe is validated here, but it is the user who accepts it.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vexel.agent.memory import Hit, RecipeMemory
from vexel.agent.probe import Fingerprint, probe
from vexel.core.volume import SpatialVolume
from vexel.engine.recipe import Recipe

TRIAL_FRAMES = 300
MAX_ROUNDS = 3
MEMORY_DISTANCE = 0.6  # a past run this close (normalised feature distance) lends its settings

_CELLS_YES = re.compile(r"细胞|包裹|珠子|微球|cell|bead|encapsulat|poisson|泊松", re.I)
_CELLS_NO = re.compile(r"(不|无需|不用|不要|没有)(数|统计|计)?(细胞|包裹)|no cells?|without cells?", re.I)
_FLUO_YES = re.compile(r"荧光|阳性|阴性|阴阳|数字\s*pcr|dpcr|ddpcr|digital|fluoresc|positive", re.I)
_FLUO_NO = re.compile(r"(不|无需|不用|不要|没有)(看|判读|测)?荧光|no fluoresc|without fluoresc", re.I)


@dataclass
class Goal:
    text: str = ""
    cells: bool | None = None  # count cells per droplet
    fluorescence: bool | None = None  # call droplets positive / negative

    @classmethod
    def parse(cls, text: str | None) -> "Goal":
        t = text or ""
        cells = False if _CELLS_NO.search(t) else (True if _CELLS_YES.search(t) else None)
        fluo = False if _FLUO_NO.search(t) else (True if _FLUO_YES.search(t) else None)
        return cls(t, cells, fluo)


@dataclass
class Attempt:
    overrides: dict[str, Any]
    change: str  # what was changed for this attempt, and why
    metrics: dict[str, Any]
    violations: list[str]
    failure: str | None = None

    @property
    def problems(self) -> tuple[int, float]:
        """Failures, then the number of QC violations, then how far over their limits they are."""
        excess = 0.0
        for v in self.violations:
            m = re.search(r"= ([-\d.e]+) \((?:min [-\d.e]+, )?max ([-\d.e]+)\)", v)
            if m:
                value, limit = float(m.group(1)), float(m.group(2))
                excess += (value - limit) / max(abs(limit), 1e-9)
        return (10 if self.failure else 0) + len(self.violations), excess


@dataclass
class Plan:
    goal: Goal
    base: str  # built-in recipe the draft starts from
    overrides: dict[str, Any]
    reasons: list[str]
    status: str = "draft"  # draft | ready | check | needs_user
    attempts: list[Attempt] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    planner: str = "rules"  # tier that answered: rules | claude | local
    model: str = ""  # model name, for language-model tiers
    skipped: list[str] = field(default_factory=list)  # tiers passed over, and why

    def recipe(self) -> Recipe:
        return Recipe.load(self.base).with_overrides(self.overrides).resolve()

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["recipe"] = self.recipe().model_dump(mode="json")
        return out

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan.json").write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        recipe_path = folder / "recipe.json"
        recipe_path.write_text(self.recipe().model_dump_json(indent=2) + "\n", encoding="utf-8")
        return recipe_path


def _base(cells: bool, fluo: bool) -> str:
    return {(False, False): "droplet_basic", (True, False): "droplet_encapsulation",
            (False, True): "droplet_fluorescence", (True, True): "droplet_fluorescence_encapsulation"}[(cells, fluo)]  # fmt: skip


def draft(fp: Fingerprint, goal: Goal, memory: RecipeMemory | None = None) -> Plan:
    reasons: list[str] = []
    d = fp.channels[fp.droplet_channel or 0] if fp.channels else None
    spots = bool(d and (d.spots_per_object or 0) >= 0.15)

    if goal.cells is None:
        cells = spots
        if spots:
            reasons.append(f"液滴里检测到稳定的小深色斑点（每滴约 {d.spots_per_object:.1f} 个），按细胞/珠子计数；不需要时在目标里写“不数细胞”")
    else:
        cells = goal.cells
        reasons.append("按目标统计每个液滴里的细胞" if cells else "按目标不统计细胞")
        if cells and not spots:
            reasons.append("注意：数据里没有检测到明显的细胞斑点，计数可能全为 0")
    if goal.fluorescence is None:
        fluo = fp.fluorescence_channel is not None
        if fluo:
            reasons.append(f"通道 {fp.fluorescence_channel} 是暗背景上的亮目标，按荧光通道判读阴阳性；不需要时在目标里写“不看荧光”")
    else:
        fluo = goal.fluorescence and fp.fluorescence_channel is not None
        if goal.fluorescence and fp.fluorescence_channel is None:
            reasons.append("目标要求荧光判读，但没有找到荧光通道（暗背景上的亮目标）；先不做荧光判读")
        elif goal.fluorescence:
            reasons.append(f"按目标判读阴阳性，荧光通道为 {fp.fluorescence_channel}")

    base = _base(cells, fluo)
    reasons.insert(0, f"选用内置配方 {base}")
    overrides: dict[str, Any] = {}
    if fp.droplet_channel:
        overrides["input.channel"] = fp.droplet_channel
        reasons.append(f"液滴在通道 {fp.droplet_channel} 中最清楚（圆形运动目标最多）")
    if fluo:
        overrides["input.channels.fluo"] = fp.fluorescence_channel
    if cells:
        overrides["input.channels.cells"] = fp.droplet_channel or 0
    if fp.flow_axis == "y":
        overrides["droplets.flow_axis"] = "y"
        reasons.append("液滴沿竖直方向流动")
    elif fp.flow_axis == "x" and d is not None:
        direction = "从左向右" if d.flow_px[1] > 0 else "从右向左"
        reasons.append(f"液滴沿水平方向流动（{direction}，每帧约 {abs(d.flow_px[1]):.0f} 像素）")
    if d is not None and d.seg_hint:
        overrides.update(d.seg_hint)
        reasons.append(
            f"默认的液滴识别不可信（液滴在帧间几乎不动，或找到的是很多小颗粒）：按尺寸扫描找到直径约 "
            f"{0.5 * (d.seg_hint['seg.min_diameter_px'] + d.seg_hint['seg.max_diameter_px']) / 1.025:.0f} 像素的液滴，"
            f"识别范围设为 {d.seg_hint['seg.min_diameter_px']:g}–{d.seg_hint['seg.max_diameter_px']:g} 像素，并关闭运动门控"
        )
        if fp.flow_axis is None:
            reasons.append("没有测出流动：若液滴确实不流动（静态阵列、停流），跨帧去重计数不适用，请以逐帧结果为准")
    if d is not None and d.object_diameter_px:
        reasons.append(f"液滴直径约 {d.object_diameter_px:.0f} 像素，圆度 {d.object_circularity:.2f}")

    plan = Plan(goal, base, _prune(base, overrides), reasons, fingerprint=fp.to_json())
    if d is None or d.objects_per_frame < 0.5:
        plan.status = "needs_user"
        plan.reasons.append("没有找到圆形液滴：请用“画笔训练”标出液滴和背景，或检查所选数据")
    if memory is not None:
        hits = memory.query(fp, family=base)
        plan.memory = [{"distance": h.distance, "runs": h.runs, "passed": h.passed, "base": h.entry["base"],
                        "overrides": h.entry["overrides"], "run": h.entry.get("run"), "time": h.entry["time"]} for h in hits]  # fmt: skip
        good = [h for h in hits if h.distance <= MEMORY_DISTANCE and h.score >= 0.5]
        if good:
            h = good[0]
            lent = {k: v for k, v in h.entry["overrides"].items() if k not in overrides}
            if lent:
                plan.overrides.update(lent)
                plan.reasons.append(
                    f"沿用相似数据上的历史配方设置 {lent}（距离 {h.distance:.2f}，{h.passed}/{h.runs} 次通过质控，来源 {h.entry.get('run') or h.entry['time']}）"
                )
    return plan


def _prune(base: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Drop settings equal to the built-in recipe's own (they say nothing)."""
    recipe = Recipe.load(base).resolve()
    params = {s.id: s.params for s in recipe.steps}
    out = {}
    for key, value in overrides.items():
        if key == "input.channel":
            same = recipe.input.channel == value
        elif key.startswith("input.channels."):
            same = recipe.input.channels.get(key.split(".", 2)[2]) == value
        else:
            step, _, name = key.partition(".")
            same = params.get(step, {}).get(name, object()) == value
        if not same:
            out[key] = value
    return out


# -- trial and refinement ---------------------------------------------------------------------


def _trial(recipe: Recipe, volume: SpatialVolume, frames: int) -> tuple[dict[str, Any], list[str], str | None]:
    from vexel.engine import run_recipe
    from vexel.engine.qc import QCGateError

    out = Path(tempfile.mkdtemp(prefix="vexel-plan-"))
    try:
        result = run_recipe(recipe, volume, out / "run", max_frames=frames, hash_input=False, report=False, overwrite=True)
    except QCGateError as e:
        return {}, [str(e)], "stopped by a QC rule"
    except Exception as e:  # noqa: BLE001 - a failed trial is information for the next attempt
        return {}, [], f"{type(e).__name__}: {e}"
    finally:
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    s = result.summary
    steps = {r["id"]: r["qc"] for r in result.provenance["steps"]}
    metrics = {
        "n_droplets": s.get("droplets.summary", {}).get("n_droplets"),
        "diameter_cv_percent": s.get("droplets.summary", {}).get("diameter_cv_percent"),
        "circularity_median": steps.get("droplets", {}).get("circularity_median"),
        "objects_per_frame": steps.get("seg", {}).get("n_objects.mean"),
        "flagged_fraction": s.get("flags.summary", {}).get("flagged_fraction"),
        "tracks_too_short": s.get("droplets.summary", {}).get("n_tracks_too_short"),
        "tracks_not_crossing": s.get("droplets.summary", {}).get("n_tracks_not_crossing"),
    }
    failure = None
    if not metrics["objects_per_frame"]:
        failure = "no objects segmented"
    elif not metrics["n_droplets"]:
        failure = "no droplet crossed the counting line"
    return metrics, [v.message() for v in result.violations], failure


def _next_change(plan: Plan, last: Attempt, tried: set[str]) -> tuple[dict[str, Any], str] | None:
    """The next rule-based adjustment after a trial, or None when nothing is left to try."""
    o = dict(last.overrides)
    if last.failure == "no objects segmented" and "loose_circles" not in tried:
        tried.add("loose_circles")
        o.update({"seg.min_score": 0.25, "seg.motion_threshold": 4.0})
        return o, "没有分割出目标：放宽圆检测（轮廓完整度 0.25，最低对比度 4）"
    if last.failure == "no droplet crossed the counting line":
        axis = o.get("droplets.flow_axis", "x")
        if "swap_axis" not in tried:
            tried.add("swap_axis")
            o["droplets.flow_axis"] = "y" if axis == "x" else "x"
            return o, f"没有液滴越过计数线：改按{'竖直' if axis == 'x' else '水平'}方向计数"
        for line in (0.3, 0.7):
            key = f"line_{line}"
            if key not in tried:
                tried.add(key)
                o["droplets.flow_axis"] = axis
                o["droplets.count_line"] = line
                return o, f"没有液滴越过计数线：把计数线移到视场 {line:.0%} 处"
    text = " ".join(last.violations)
    if "low_agreement_fraction" in text and "noise" not in tried and "inclusions.noise_factor" not in o:
        tried.add("noise")
        o["inclusions.noise_factor"] = 8.0
        return o, "细胞计数帧间不稳定：提高细胞检测阈值（噪声的 8 倍）"
    return None


def refine(plan: Plan, volume: SpatialVolume, *, frames: int = TRIAL_FRAMES, rounds: int = MAX_ROUNDS) -> Plan:
    if plan.status == "needs_user":
        return plan
    tried: set[str] = set()
    overrides, change = dict(plan.overrides), "初稿"
    for _ in range(rounds):
        recipe = Recipe.load(plan.base).with_overrides(overrides)
        metrics, violations, failure = _trial(recipe, volume, frames)
        attempt = Attempt(dict(overrides), change, metrics, violations, failure)
        plan.attempts.append(attempt)
        if attempt.problems == (0, 0.0):
            break
        nxt = _next_change(plan, attempt, tried)
        if nxt is None:
            break
        overrides, change = nxt[0], nxt[1]
    best = min(plan.attempts, key=lambda a: (a.problems, plan.attempts.index(a)))
    plan.overrides = _prune(plan.base, best.overrides)
    if best is not plan.attempts[0]:
        plan.reasons.append(f"试运行后调整：{best.change}")
    if best.failure:
        plan.status = "needs_user"
        plan.reasons.append(f"试运行仍然失败（{_failure_text(best.failure)}）：建议用“画笔训练”或手动调整")
    elif best.violations:
        plan.status = "check"
        plan.reasons.append("试运行有质控提示：" + "；".join(best.violations))
    else:
        plan.status = "ready"
        n = best.metrics.get("n_droplets")
        plan.reasons.append(f"试运行（前 {frames} 帧）通过全部质控规则，计到 {n} 个液滴")
    return plan


def _failure_text(code: str) -> str:
    return {"no objects segmented": "没有分割出目标", "no droplet crossed the counting line": "没有液滴越过计数线"}.get(code, code)


def valid_keys(base: str) -> set[str]:
    """``step.param`` keys the built-in recipe has (what settings may name)."""
    recipe = Recipe.load(base).resolve()
    return {f"{s.id}.{k}" for s in recipe.steps for k in s.params}


def _summary(fp: Fingerprint) -> dict[str, Any]:
    out = {k: v for k, v in fp.to_json().items() if k not in ("source",)}
    return out


def _context(stage: str, plan: Plan, fp: Fingerprint, goal_text: str) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "stage": stage, "goal": goal_text, "fingerprint": _summary(fp),
        "draft": {"recipe": plan.base, "settings": plan.overrides, "reasons": plan.reasons},
        "valid_keys": sorted(valid_keys(plan.base)),
    }  # fmt: skip
    if plan.attempts:
        ctx["trials"] = [{"change": a.change, "settings": a.overrides, "metrics": a.metrics, "qc_warnings": a.violations,
                          "failure": a.failure} for a in plan.attempts]  # fmt: skip
    return ctx


def _apply(plan: Plan, proposal: Any, label: str) -> bool:
    """Validated settings from a model proposal; returns whether anything changed."""
    from vexel.agent.models import check_settings

    kept, dropped = check_settings(proposal.settings, valid_keys(plan.base))
    for note in proposal.notes:
        plan.reasons.append(f"{label}：{note}")
    if dropped:
        plan.reasons.append(f"{label}的建议中这些没有采用：" + "；".join(dropped))
    changed = {k: v for k, v in kept.items() if plan.overrides.get(k) != v}
    plan.overrides.update(changed)
    plan.overrides = _prune(plan.base, plan.overrides)
    return bool(changed)


def make_plan(
    volume: SpatialVolume, goal: str | Goal | None = None, *, memory: RecipeMemory | None = None, trial: bool = True,
    frames: int = TRIAL_FRAMES, chain: list[Any] | None = None, image: bool = True,
) -> Plan:  # fmt: skip
    """Probe, draft (rules, then the first available language-model tier), and (by default)
    check the draft on a trial run; a model gets one more look when the trial fails."""
    from vexel.agent import models

    goal_text = goal.text if isinstance(goal, Goal) else (goal or "")
    rules_goal = goal if isinstance(goal, Goal) else Goal.parse(goal)
    fp = probe(volume)
    plan = draft(fp, rules_goal, memory)
    picture = None
    if image:
        import tempfile as _tf

        from vexel.agent.probe import montage

        with _tf.TemporaryDirectory() as tmp:
            try:
                picture = models.montage_bytes(montage(volume, fp, Path(tmp) / "m.png"))
            except Exception:  # noqa: BLE001 - the picture is optional
                picture = None
    proposal, skipped = models.propose(_context("draft", plan, fp, goal_text), picture, chain)
    plan.planner, plan.model, plan.skipped = proposal.tier, proposal.model, skipped
    if proposal.tier != "rules":
        label = f"模型 {proposal.model or proposal.tier}"
        wanted = Goal(goal_text, proposal.cells if proposal.cells is not None else rules_goal.cells,
                      proposal.fluorescence if proposal.fluorescence is not None else rules_goal.fluorescence)  # fmt: skip
        if (wanted.cells, wanted.fluorescence) != (rules_goal.cells, rules_goal.fluorescence):
            redo = draft(fp, wanted, memory)
            redo.planner, redo.model, redo.skipped = plan.planner, plan.model, plan.skipped
            redo.reasons.insert(1, f"{label} 对目标的理解：数细胞 {wanted.cells}，荧光判读 {wanted.fluorescence}")
            plan = redo
        _apply(plan, proposal, label)
    if trial:
        refine(plan, volume, frames=frames)
        if plan.status in ("check", "needs_user") and plan.planner != "rules" and plan.attempts:
            second, more = models.propose(_context("revise", plan, fp, goal_text), None, chain)
            plan.skipped += [s for s in more if s not in plan.skipped]
            if second.tier != "rules":
                before = plan.overrides.copy()
                if _apply(plan, second, f"模型 {second.model or second.tier} 看过试运行后"):
                    metrics, violations, failure = _trial(Recipe.load(plan.base).with_overrides(plan.overrides), volume, frames)
                    attempt = Attempt(plan.overrides.copy(), "模型建议的调整", metrics, violations, failure)
                    plan.attempts.append(attempt)
                    best = min(plan.attempts, key=lambda a: (a.problems, plan.attempts.index(a)))
                    if best is attempt:
                        plan.status = "ready" if not attempt.problems[0] else ("check" if not attempt.failure else "needs_user")
                        plan.reasons.append("采用模型建议后试运行" + ("通过全部质控规则" if plan.status == "ready" else "有改善"))
                    else:
                        plan.overrides = before
                        plan.reasons.append("模型建议的调整没有改善试运行，未采用")
    plan._fp = fp  # type: ignore[attr-defined]  # kept for remembering the outcome
    return plan


def remember(memory: RecipeMemory, plan: Plan, run: Any) -> dict[str, Any]:
    """Record a run made from a plan (``run`` is a RunResult or StreamResult)."""
    fp = getattr(plan, "_fp", None)
    if fp is None:
        raise ValueError("this plan was not made in this session (no fingerprint object); re-plan to remember it")
    flags = run.summary.get("flags.summary", {})
    outcome = {"qc_passed": not run.violations, "n_violations": len(run.violations),
               "flagged_fraction": flags.get("flagged_fraction"), "status": run.status,
               "n_droplets": run.summary.get("droplets.summary", {}).get("n_droplets")}  # fmt: skip
    return memory.add(fp, plan.base, plan.overrides, outcome, run.out_dir)


def hits_text(hits: list[Hit]) -> list[str]:
    return [f"{h.entry['base']} {h.entry['overrides']}：距离 {h.distance:.2f}，{h.passed}/{h.runs} 次通过" for h in hits]
