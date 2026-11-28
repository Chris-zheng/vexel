"""SEM planner (M5): from an image and a goal to a measurement plan, tried before it is
offered - the SEM side of Vexel's agent (``vexel.agent``).

1. **Goal**: what the user wants measured, read from free text (Chinese or English):
   line width or roughness, holes, particles or defects, layer thickness, sidewall angle,
   tip radius. Unsaid: the image decides.
2. **Probe** (``recognize.probe``): what the image shows (lines, holes, particles,
   layers, profile, tip, other) and its state (noise, stripes, uneven brightness,
   saturation).
3. **Draft**: the recipe, its settings (which features are bright, the edge definition,
   sizes), the region and the processing steps, each decision with its reason. A close
   enough past plan in recipe memory that a person accepted lends its settings. A
   language model, when configured and allowed, may then choose the recipe and adjust
   settings from an allowlist (``vexel.agent.models``); the rules tier is always there.
4. **Trial**: the plan runs on the image. Each recipe has its checks (lines found, edges
   missing, holes round, a radius that does not depend on the fitting range...). A
   failed check triggers the next adjustment (the other polarity, the other edge
   definition...), a few rounds; the best attempt is kept.
5. **Result**: ``ready`` (passed every check), ``check`` (runs, warnings remain) or
   ``needs_user`` (no usable result, or the image type is unclear). The plan is a
   proposal: a person accepts it (``apply``) before it enters the record.

The planner never produces a measured value itself: it only chooses and configures the
validated recipes; every value comes from them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vexel.packs.sem.process import IMAGE_DTYPE

from vexel.packs.sem.recognize import KIND_NAMES, SemFingerprint, probe

MAX_ROUNDS = 4
MEMORY_DISTANCE = 0.6

GOALS = [
    ("lines", re.compile(r"线宽|线条|间隔|周期|粗糙度|\bCD\b|LER|LWR|line ?width|line/space|pitch|roughness|critical dimension", re.I)),
    ("pores", re.compile(r"孔隙|多孔|气孔|porosity|porous|\bpores?\b", re.I)),
    ("fibres", re.compile(r"纤维|无纺|静电纺|熔喷|纺粘|fib(?:er|re)|nonwoven|electrospun|meltblown|spunbond", re.I)),
    ("holes", re.compile(r"接触孔|通孔|孔径|孔阵列|contact|via|hole", re.I)),
    ("particles", re.compile(r"颗粒|粒径|粒度|缺陷|粉末|particle|defect|powder|grain size", re.I)),
    ("layers", re.compile(r"膜厚|层厚|厚度|多层|layer|thickness|film|stack", re.I)),
    ("profile", re.compile(r"侧壁|沟槽|截面轮廓|台阶高|刻蚀深度|sidewall|trench|profile|etch depth|step height", re.I)),
    ("tip", re.compile(r"尖端|针尖|曲率半径|锥角|探针|tip|apex|cone|needle|emitter", re.I)),
]


def goal_recipe(text: str | None) -> str | None:
    """The recipe a goal names (the first one mentioned), or None."""
    if not text:
        return None
    hits = [(m.start(), name) for name, rx in GOALS for m in [rx.search(text)] if m]
    return min(hits)[1] if hits else None


@dataclass
class SemAttempt:
    params: dict[str, Any]
    steps: list[str]
    change: str
    summary: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class SemPlan:
    recipe: str | None
    params: dict[str, Any]
    region: list[float] | None
    steps: list[dict[str, Any]]  # processing to add: {"op", "params", "reason"}
    reasons: list[str]
    goal: str = ""
    status: str = "needs_user"
    attempts: list[SemAttempt] = field(default_factory=list)
    planner: str = "rules"
    model: str = ""
    skipped: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    fingerprint: SemFingerprint | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        if self.fingerprint is not None:
            d["fingerprint"] = {k: v for k, v in self.fingerprint.to_json().items() if k not in ("objects",)} | {"objects": {
                k: v for k, v in self.fingerprint.objects.items() if not isinstance(v, (list, np.ndarray)) or k in ("main_box", "main_sides")}}
        return d

    @property
    def problems(self) -> list[str]:
        best = self.best
        return best.problems if best else ["没有试运行"]

    @property
    def best(self) -> SemAttempt | None:
        if not self.attempts:
            return None
        return min(self.attempts, key=lambda a: (len(a.problems) + (1 if a.error else 0), len(a.warnings), self.attempts.index(a)))

    def apply(self, ms: Any) -> Any:
        """Add the plan to a measurement record: its processing steps (those the record does not
        have yet) and the automatic measurement. Returns the new AnalysisSpec."""
        from vexel.packs.sem.process import OPS, step

        if self.recipe is None:
            raise ValueError("the plan has no measurement (the image type is unclear): choose one")
        have = {s.op for s in ms.processing.steps if s.enabled}
        for st in self.steps:
            if st["op"] in have:
                continue
            new = step(st["op"], **st.get("params", {}))
            index = next((i for i, s in enumerate(ms.processing.steps) if OPS.get(s.op) and OPS[s.op].role == "display"), len(ms.processing.steps))
            ms.processing.steps.insert(index, new)
        spec = ms.add_analysis(self.recipe, self.region, **self.params)
        spec.name = "自动规划"
        return spec


# -- draft ----------------------------------------------------------------------------------------


def _region(fp: SemFingerprint, recipe: str) -> list[float] | None:
    """Whole micrograph, except for single objects: their box with a margin of background."""
    box = fp.objects.get("main_box")
    if recipe not in ("profile", "tip") or not box:
        return None
    x0, y0, x1, y1 = box
    rows, cols = fp.shape
    mx, my = 0.3 * (x1 - x0) + 8, 0.3 * (y1 - y0) + 8
    if recipe == "tip":  # keep the whole length that enters from the border
        mx, my = max(mx, 8), max(my, 8)
    return [max(0.0, x0 - mx), max(0.0, y0 - my), min(float(cols), x1 + mx), min(float(rows), y1 + my)]


def draft(fp: SemFingerprint, goal: str = "", memory: Any = None) -> SemPlan:
    wanted = goal_recipe(goal)
    reasons: list[str] = []
    recipe = wanted or fp.recipe
    if wanted:
        reasons.append(f"目标指明了测量：{wanted}" + ("" if fp.recipe in (None, wanted) else f"（图像识别为“{KIND_NAMES[fp.kind]}”，按目标测量）"))
    if fp.kind != "unknown":
        why = "；".join(fp.evidence.get(fp.kind, []))
        reasons.append(f"图像识别为“{KIND_NAMES[fp.kind]}”（把握 {fp.confidence:.2f}）" + (f"：{why}" if why else ""))
    else:
        reasons += fp.notes[:1]
    plan = SemPlan(recipe=recipe, params={}, region=None, steps=list(fp.suggested_steps), reasons=reasons, goal=goal or "", fingerprint=fp)
    if recipe is None:
        plan.reasons.append("没有可用的测量方法：请在“自动测量”中选择方法" if fp.kind != "other" else "图像不属于已有的测量类型")
        return plan
    p: dict[str, Any] = {}
    pol = fp.polarity
    if recipe == "lines":
        if fp.orientation in ("vertical", "horizontal"):
            p["orientation"] = fp.orientation
        if pol == "dark":
            p["feature"] = "dark"
        if fp.edge_peaks:
            p["method"] = "peak"
            plan.reasons.append("线条边缘有亮峰：用“亮边峰值”定义边缘（结果写明定义）")
        plan.reasons.append(f"较窄的{'亮' if pol != 'dark' else '暗'}条按线条计（另一种请改“线条是”）")
    elif recipe == "holes":
        if pol == "bright":
            p["feature"] = "bright"
        d = fp.objects.get("diameter")
        if d:
            p["min_diameter"] = round(max(2.0, 0.5 * d), 1)
            plan.reasons.append(f"孔直径约 {d:.1f} 像素：最小直径设为其一半")
    elif recipe == "particles":
        if pol == "dark":
            p["feature"] = "dark"
        d = fp.objects.get("diameter")
        da = fp.objects.get("area_diameter")
        if d and da and da >= 4 * d and da >= 20:
            # most objects are tiny (noise, stain or debris) but the area is on much larger particles
            p["min_diameter"] = round(0.2 * da, 1)
            p["smooth"] = round(min(max(0.05 * da, 0.5), 10.0) * 2) / 2
            plan.reasons.append(f"大多数目标很小（中位约 {d:.0f} 像素，多半是噪点、染色颗粒或碎屑），面积却主要在约 {da:.0f} 像素的颗粒上："
                                f"最小直径设为 {p['min_diameter']:g} 像素、平滑 {p['smooth']:g} 像素；小颗粒也要计入时把最小直径改回 3")  # fmt: skip
        elif d and d > 60:  # only for large particles: keep the default (3 px) so a broad distribution keeps its small ones
            p["min_diameter"] = round(0.05 * d, 1)
        if d and d < 15 and "smooth" not in p:
            plan.reasons.append(f"颗粒较小（约 {d:.0f} 像素）：保持 0.5 像素的轻度平滑")
    elif recipe in ("profile", "tip", "fibres"):
        if pol == "dark":
            p["feature"] = "dark"
    plan.params = p
    if recipe in ("lines", "layers") and any(st["op"] == "denoise" for st in plan.steps):
        plan.steps = [st for st in plan.steps if st["op"] != "denoise"]
        plan.reasons.append("去噪会改变边缘位置的起伏（粗糙度）：不去噪，粗糙度中的噪声部分单独扣除")
    if recipe in ("lines", "layers") and fp.orientation == "horizontal":
        plan.steps = [st for st in plan.steps if st["op"] not in ("line_level", "destripe")]
    plan.region = _region(fp, recipe)
    if plan.region:
        plan.reasons.append("单个目标：区域取目标外框并留出背景")
    for st in plan.steps:
        plan.reasons.append(f"处理：{st['op']}（{st['reason']}）")
    plan.reasons += [n for n in fp.notes if n not in plan.reasons]
    # recipe memory: a close past plan a person accepted
    if memory is not None and fp.kind != "unknown":
        hits = [h for h in memory.query(fp, family=recipe) if h.passed]
        if hits and hits[0].distance <= MEMORY_DISTANCE:
            lent = hits[0].entry.get("overrides", {})
            params = {k: v for k, v in lent.get("params", {}).items()}
            if params != plan.params:
                plan.params = params
                plan.reasons.append(f"记忆中有相近的图像（距离 {hits[0].distance:.2f}，{hits[0].passed}/{hits[0].runs} 次被采用）：沿用其设置")
        plan.memory = [f"{h.entry['base']} {h.entry['overrides'].get('params')}：距离 {h.distance:.2f}，{h.passed}/{h.runs} 次被采用" for h in hits[:3]]
    return plan


# -- trial ----------------------------------------------------------------------------------------


def checks(recipe: str, an: Any) -> tuple[list[str], list[str]]:
    """Problems (the result is not usable) and warnings of one analysis."""
    if an.error:
        return [an.error], []
    v = an.value
    problems: list[str] = []
    if recipe == "lines":
        if (v("lines") or 0) < 2:
            problems.append("找到的完整线条少于 2 条")
        else:
            cd, sd = v("cd_mean"), v("cd_sd")
            if cd and sd is not None and sd / cd > 0.15:
                problems.append(f"各线线宽相差 {sd / cd:.0%}：线条与间隔可能被弄反，或边缘找错")
    elif recipe == "holes":
        n = v("holes") or 0
        if n < 1:
            problems.append("没有找到孔")
        elif (v("merged") or 0) > 0.1 * n:
            problems.append("超过 10% 的孔疑似相连")
        elif (v("roundness_mean") or 0) < 0.7:
            problems.append("孔的圆度很低：可能把孔与表面弄反")
    elif recipe == "particles":
        if (v("particles") or 0) < 1:
            problems.append("没有找到完整的颗粒")
    elif recipe == "pores":
        if v("porosity") is None or (v("pores") or 0) + (v("at_border") or 0) < 1:
            problems.append("没有找到孔")
    elif recipe == "fibres":
        if (v("segments") or 0) < 1 or (v("samples") or 0) < 30:
            problems.append("找到的纤维太少（有效测量点少于 30）")
    elif recipe == "layers":
        if (v("interfaces") or 0) < 2:
            problems.append("找到的界面少于 2 个")
    elif recipe == "profile":
        if v("height") is None:
            problems.append("没有找到完整的特征")
    elif recipe == "tip":
        r, u = v("radius"), v("radius_uncertainty")
        if r is None:
            problems.append("没有求出曲率半径")
        elif u is not None and u > 0.1 * r:
            problems.append(f"曲率半径随拟合范围变化 {u / r:.0%}")
        if v("cone_angle") is None:
            problems.append("没有求出锥角")
    return problems, [w for w in an.warnings]


ALTERNATIVES: dict[str, list[tuple[str, dict[str, Any], list[str]]]] = {
    # (why, settings to change, processing steps to add)
    "lines": [("换成另一种灰度的条纹作为线条", {"feature": "flip"}, []), ("边缘定义改为最大梯度", {"method": "gradient"}, []),
              ("每 2 行平均后再找边缘", {"average": 2}, [])],
    "holes": [("孔的灰度反过来", {"feature": "flip"}, []), ("边缘定义改为最大梯度", {"method": "gradient"}, []), ("加强平滑", {"smooth": 2.0}, [])],
    "particles": [("颗粒的灰度反过来", {"feature": "flip"}, []), ("改用局部阈值", {"threshold": "local"}, []), ("先去噪", {}, ["denoise"])],
    "pores": [("孔的灰度反过来", {"feature": "flip"}, []), ("改用局部阈值", {"threshold": "local"}, [])],
    "fibres": [("纤维的灰度反过来", {"feature": "flip"}, []), ("改用局部阈值", {"threshold": "local"}, []), ("边缘定义改为阈值", {"method": "threshold"}, [])],
    "layers": [("边缘定义改为阈值", {"method": "threshold"}, []), ("加强平滑", {"smooth": 3.0}, [])],
    "profile": [("特征的灰度反过来", {"feature": "flip"}, []), ("加强平滑", {"smooth": 2.0}, [])],
    "tip": [("尖端的灰度反过来", {"feature": "flip"}, []), ("加强平滑", {"smooth": 2.0}, [])],
}  # fmt: skip


def _run(plan: SemPlan, params: dict[str, Any], steps: list[dict[str, Any]], image: np.ndarray, scale: Any, crop_rows: int | None,
         base_steps: list[Any]) -> tuple[Any, list[str]]:  # fmt: skip
    from vexel.packs.sem.analysis import AnalysisSpec, run
    from vexel.packs.sem.process import Processing, Processor, step

    have = {s.op for s in base_steps if s.enabled}
    chain = list(base_steps) + [step(s["op"], **s.get("params", {})) for s in steps if s["op"] not in have]
    res = Processor(image, crop_rows).run(Processing(steps=chain))
    spec = AnalysisSpec(id=0, recipe=plan.recipe, region=plan.region, params=params)  # type: ignore[arg-type]
    return run(spec, res.analysis, scale, res.mask, crop_rows), [s.op for s in chain]


def trial(plan: SemPlan, image: np.ndarray, scale: Any, crop_rows: int | None = None, base_steps: list[Any] | None = None,
          rounds: int = MAX_ROUNDS) -> SemPlan:  # fmt: skip
    """Run the draft; while a check fails, try the recipe's adjustments, each on its own (a
    variant of the draft), up to ``rounds`` runs. The best attempt becomes the plan."""
    from vexel.packs.sem.analysis import RECIPES

    if plan.recipe is None:
        plan.status = "needs_user"
        return plan
    base_steps = list(base_steps or [])
    draft_params, draft_steps = dict(plan.params), list(plan.steps)
    variants: list[tuple[str, dict[str, Any], list[str]]] = [("初稿", {}, []), *ALTERNATIVES.get(plan.recipe, [])]
    tried_steps: list[list[dict[str, Any]]] = []
    for why, delta, add in variants[: max(1, rounds)]:
        params = dict(draft_params)
        for k, v in delta.items():
            if v == "flip":
                cur = params.get(k, RECIPES[plan.recipe].resolve({})[k])
                params[k] = "dark" if cur == "bright" else "bright"
            else:
                params[k] = v
        steps = draft_steps + [{"op": op, "params": {}, "reason": why} for op in add if op not in {s["op"] for s in draft_steps}]
        an, ops = _run(plan, params, steps, image, scale, crop_rows, base_steps)
        problems, warnings = checks(plan.recipe, an)
        summary = {k: v for k, _, v, kind in an.summary if kind != "text"}
        plan.attempts.append(SemAttempt(params, ops, why, summary, problems, warnings, an.error))
        tried_steps.append(steps)
        if not problems and not an.error:
            break
    best = plan.best
    assert best is not None
    plan.params = best.params
    plan.steps = tried_steps[plan.attempts.index(best) - (len(plan.attempts) - len(tried_steps))]
    if best.problems or best.error:
        plan.status = "needs_user"
        plan.reasons.append("试运行没有得到可用结果：" + "；".join(best.problems or [best.error or ""]))
    elif best.warnings:
        plan.status = "check"
        plan.reasons.append(f"试运行通过检查（{best.change}），但有提示，请核对")
    else:
        plan.status = "ready"
        plan.reasons.append(f"试运行通过全部检查（{best.change}）")
    return plan


def _better(a: SemAttempt, b: SemAttempt) -> bool:
    return (len(a.problems), len(a.warnings)) < (len(b.problems), len(b.warnings))


# -- the whole plan ----------------------------------------------------------------------------------


def _context(stage: str, plan: SemPlan) -> dict[str, Any]:
    from vexel.agent.models import sem_allowed

    fp = plan.fingerprint
    allowed = sem_allowed()
    keys = [k for k in allowed if k.startswith("process.") or (plan.recipe and k.startswith(plan.recipe + "."))] if plan.recipe else list(allowed)
    return {"task": "sem", "stage": stage, "goal": plan.goal,
            "image": None if fp is None else {"kind": fp.kind, "confidence": fp.confidence, "scores": fp.scores, "orientation": fp.orientation,
                                              "period_px": fp.period_px, "polarity": fp.polarity, "noise": round(fp.noise, 4), "snr": round(fp.snr, 1),
                                              "stripes": round(fp.stripes, 2), "gradient": round(fp.gradient, 3), "saturated": fp.saturated,
                                              "objects": {k: v for k, v in fp.objects.items() if isinstance(v, (int, float, str)) or v is None}},
            "draft": {"recipe": plan.recipe, "settings": plan.params, "processing": [s["op"] for s in plan.steps], "reasons": plan.reasons},
            "attempts": [{"change": a.change, "settings": a.params, "problems": a.problems, "warnings": a.warnings[:3]} for a in plan.attempts],
            "valid_keys": keys}  # fmt: skip


def _apply_proposal(plan: SemPlan, proposal: Any, label: str) -> bool:
    from vexel.agent.models import check_settings, sem_allowed

    changed = False
    if proposal.recipe and proposal.recipe != plan.recipe and not goal_recipe(plan.goal):
        plan.reasons.append(f"{label}选择了测量：{proposal.recipe}（原为 {plan.recipe}）")
        plan.recipe, plan.params, changed = proposal.recipe, {}, True
    ctx = _context("check", plan)
    kept, dropped = check_settings(proposal.settings, set(ctx["valid_keys"]), sem_allowed())
    for key, value in kept.items():
        group, _, name = key.partition(".")
        if group == "process":
            ops = [s["op"] for s in plan.steps]
            if value == "on" and name not in ops:
                if name == "denoise" and plan.recipe == "lines":
                    plan.reasons.append(f"{label}建议去噪：会改变线边缘粗糙度，未采用")
                    continue
                plan.steps.append({"op": name, "params": {}, "reason": f"{label}建议"})
                changed = True
            elif value == "off" and name in ops:
                plan.steps = [s for s in plan.steps if s["op"] != name]
                changed = True
        elif group == plan.recipe and plan.params.get(name) != value:
            plan.params[name] = value
            changed = True
    plan.reasons += [f"{label}：{n}" for n in proposal.notes[:6]]
    plan.reasons += [f"{label}的建议未采用：{d}" for d in dropped]
    return changed


def _picture(image: np.ndarray) -> bytes:
    import io

    from PIL import Image

    from vexel.packs.sem.process import _to_u8

    a = np.asarray(image, IMAGE_DTYPE)
    img = Image.fromarray(_to_u8(a))
    img.thumbnail((768, 768))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MIN_SIDE = 16  # px: smaller images are not recognised (the planner leaves the choice to the person)


def make_plan(image: np.ndarray, *, cal: Any = None, goal: str = "", scale: Any = None, crop_rows: int | None = None,
              base_steps: list[Any] | None = None, memory: Any = None, trial_run: bool = True, chain: list[Any] | None = None,
              models: bool = True) -> SemPlan:  # fmt: skip
    """Plan the measurement of ``image`` (the whole file; ``crop_rows`` the micrograph's rows).

    ``models``: let the configured language-model tiers look (numbers only, unless the
    configuration and the organisation policy allow sending a picture)."""
    from vexel.packs.sem.measure import Scale

    crop = crop_rows if crop_rows and crop_rows < image.shape[0] else None
    micro = image[:crop] if crop else image
    if min(micro.shape[:2]) < MIN_SIDE:  # nothing to recognise in a handful of pixels
        return SemPlan(recipe=None, params={}, region=None, steps=[], goal=goal, status="needs_user",
                       reasons=[f"图像太小（{micro.shape[1]}×{micro.shape[0]} 像素）：无法识别，请手动选择测量方法"])  # fmt: skip
    fp = probe(micro, raw=micro, cal=cal)
    plan = draft(fp, goal, memory)
    if models:
        from vexel.agent import models as M

        try:
            picture = _picture(micro)
        except Exception:  # noqa: BLE001 - the picture is optional
            picture = None
        proposal, skipped = M.propose(_context("draft", plan), picture, chain)
        plan.planner, plan.model, plan.skipped = proposal.tier, proposal.model, skipped
        if proposal.tier != "rules":
            _apply_proposal(plan, proposal, f"模型 {proposal.model or proposal.tier} ")
    if trial_run and plan.recipe:
        trial(plan, image, scale or Scale(), crop, base_steps)
        if models and plan.status != "ready" and plan.planner != "rules":
            from vexel.agent import models as M

            second, more = M.propose(_context("revise", plan), None, chain)
            plan.skipped += [s for s in more if s not in plan.skipped]
            if second.tier != "rules" and _apply_proposal(plan, second, f"模型 {second.model or second.tier} 看过试运行后"):
                before = list(plan.attempts)
                plan.attempts = []
                trial(plan, image, scale or Scale(), crop, base_steps, rounds=1)
                plan.attempts = before + plan.attempts
    elif not plan.recipe:
        plan.status = "needs_user"
    return plan


def plan_file(path: str | Path, goal: str = "", *, use_ocr: bool = True, memory: Any = None, trial_run: bool = True, chain: list[Any] | None = None,
              models: bool = True) -> tuple[SemPlan, Any, Any]:  # fmt: skip
    """Plan for an image file: its calibration and record are used. Returns (plan, calibration, record)."""
    from vexel.packs.sem import calibrate as C
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.measure import scale_from

    cal = C.for_file(path, use_ocr=use_ocr)
    ms = measure_io.load_record(cal.sha256) or measure_io.new_set(path, cal)
    ms.scale = scale_from(cal, ms.scale.tilt)
    img = C.load_image(path)
    plan = make_plan(img, cal=cal, goal=goal, scale=ms.scale, crop_rows=cal.crop_rows, base_steps=ms.processing.steps, memory=memory,
                     trial_run=trial_run, chain=chain, models=models)  # fmt: skip
    return plan, cal, ms


def memory_store() -> Any:
    from vexel.agent.memory import RecipeMemory, home

    return RecipeMemory(home() / "memory" / "sem.jsonl")


def remember(memory: Any, plan: SemPlan, accepted: bool = True, where: str | None = None) -> dict[str, Any] | None:
    """Record a plan a person accepted (or turned down), for similar images later."""
    if plan.fingerprint is None or plan.recipe is None or plan.fingerprint.kind == "unknown":
        return None
    outcome = {"qc_passed": bool(accepted and plan.status in ("ready", "check")), "status": plan.status, "accepted": accepted}
    return memory.add(plan.fingerprint, plan.recipe, {"params": plan.params, "steps": [s["op"] for s in plan.steps]}, outcome, where)


__all__ = ["GOALS", "SemAttempt", "SemPlan", "checks", "draft", "goal_recipe", "make_plan", "memory_store", "plan_file", "remember", "trial"]
