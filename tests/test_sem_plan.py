"""SEM probe and planner (M5): recognition of what an image shows, plans tried before they
are offered, model proposals checked against the allowlist, recipe memory, the record,
the command line and the MCP tools."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vexel.packs.sem.measure import MeasurementSet, Scale
from vexel.packs.sem.plan import draft, goal_recipe, make_plan, remember, trial
from vexel.packs.sem.recognize import probe
from vexel.synthetic import sem as S


def test_goals_in_chinese_and_english():
    assert goal_recipe("测线宽和线边缘粗糙度") == "lines"
    assert goal_recipe("LER/LWR of the resist lines") == "lines"
    assert goal_recipe("接触孔孔径均匀性") == "holes"
    assert goal_recipe("缺陷颗粒的粒度分布") == "particles"
    assert goal_recipe("多层膜的膜厚") == "layers"
    assert goal_recipe("沟槽侧壁角") == "profile"
    assert goal_recipe("探针针尖曲率半径") == "tip"
    assert goal_recipe("先测孔径，再看颗粒") == "holes"  # the first one named
    assert goal_recipe("") is None and goal_recipe("看看这张图") is None


IMAGES = {
    "lines": lambda: S.rough_lines(seed=1)[0],
    "lines (horizontal, dark)": lambda: 1 - S.rough_lines(seed=2)[0].T,
    "holes": lambda: S.hole_array(seed=1)[0],
    "particles": lambda: S.particle_field(seed=1)[0],
    "layers": lambda: S.layer_stack(seed=1)[0],
    "profile": lambda: S.xsection_feature(seed=1)[0],
    "profile (trench)": lambda: S.xsection_feature(seed=1, feature="dark")[0],
    "tip": lambda: np.rot90(S.tip(seed=2)[0], -1),
    "other": lambda: np.random.default_rng(0).normal(0.5, 0.1, (300, 400)),
}


@pytest.mark.parametrize("name", list(IMAGES))
def test_probe_recognises_each_kind(name):
    fp = probe(IMAGES[name]())
    assert fp.kind == name.split(" ")[0], (fp.kind, fp.scores, fp.evidence)
    if fp.kind != "other":
        assert fp.confidence >= 0.55 and fp.evidence[fp.kind]


def test_probe_details():
    fp = probe(1 - S.rough_lines(seed=2)[0].T)
    assert fp.orientation == "horizontal" and fp.polarity == "dark" and abs(fp.period_px - 36.7) < 1.5
    assert probe(S.rough_lines(seed=3, edge_peaks=True)[0]).edge_peaks and not probe(S.rough_lines(seed=3)[0]).edge_peaks
    holes = probe(S.hole_array(seed=1)[0])
    assert holes.polarity == "dark" and abs(holes.lattice["pitch"] - 48.3) < 1.5 and abs(holes.objects["diameter"] - 24.6) < 2
    assert probe(S.tip(seed=1)[0]).objects["main_box"]


def test_a_few_regular_horizontal_bands_are_not_guessed():
    img, _ = S.rough_lines(rows=300, cols=300, width=30, pitch=75, seed=4)  # four periods, then turned horizontal
    fp = probe(img.T)
    assert fp.kind in ("unknown", "lines")  # never a confident "layers" for a regular pattern
    assert fp.scores["layers"] <= 0.5


def test_textures_are_not_objects():
    from scipy import ndimage as ndi

    rng = np.random.default_rng(1)
    blobs = ndi.gaussian_filter(rng.normal(0, 1, (300, 400)), 4)
    assert probe(blobs).kind == "other"


def test_processing_is_suggested_only_where_it_helps():
    parts = S.add_stripes(S.particle_field(seed=1)[0], 0.08)
    assert "line_level" in [s["op"] for s in probe(parts).suggested_steps]
    lines_h = S.add_stripes(S.rough_lines(seed=1)[0].T, 0.08)
    fp = probe(lines_h)
    assert fp.kind == "lines" and "line_level" not in [s["op"] for s in fp.suggested_steps] and any("水平" in n for n in fp.notes)
    noisy = probe(S.rough_lines(seed=1, noise=0.3)[0])
    assert "denoise" not in [s["op"] for s in noisy.suggested_steps]
    grad = probe(S.add_gradient(S.particle_field(seed=1)[0], 0.5))
    assert "background" in [s["op"] for s in grad.suggested_steps]


# -- plans --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "key", "truth", "tol"), [
    ("lines", "cd_mean", 14.3, 0.5), ("holes", "diameter_mean", 24.6, 1.5), ("layers", "layer1", 40.3, 0.5),
    ("profile", "angle_left", 84.0, 0.5), ("tip", "radius", 20.0, 1.0),
])  # fmt: skip
def test_plans_are_ready_and_measure_correctly(name, key, truth, tol):
    plan = make_plan(IMAGES[name](), scale=Scale(), models=False)
    assert plan.status == "ready", plan.reasons
    assert plan.recipe == name and len(plan.attempts) == 1 and plan.planner == "rules"
    assert abs(plan.best.summary[key] - truth) < tol
    assert any("试运行通过" in r for r in plan.reasons)


def test_plan_settings_follow_the_image():
    dark = make_plan(1 - S.rough_lines(seed=2)[0].T, scale=Scale(), models=False)
    assert dark.params == {"orientation": "horizontal", "feature": "dark"}
    peaks = make_plan(S.rough_lines(seed=3, edge_peaks=True)[0], scale=Scale(), models=False)
    assert peaks.params.get("method") == "peak" and peaks.status == "ready"
    tip = make_plan(S.tip(seed=1)[0], scale=Scale(), models=False)
    x0, y0, x1, y1 = tip.region
    assert x0 < 150.3 < x1 and y0 <= 80.6 and y1 == 400  # the box of the tip with background around it
    trench = make_plan(S.xsection_feature(seed=1, feature="dark")[0], goal="沟槽侧壁角", scale=Scale(), models=False)
    assert trench.recipe == "profile" and trench.params == {"feature": "dark"}


def test_unclear_images_and_failed_trials_need_the_person():
    noise = np.random.default_rng(0).normal(0.5, 0.1, (300, 400))
    plan = make_plan(noise, scale=Scale(), models=False)
    assert plan.recipe is None and plan.status == "needs_user" and not plan.attempts
    forced = make_plan(noise, goal="测线宽", scale=Scale(), models=False)
    assert forced.recipe == "lines" and forced.status == "needs_user" and len(forced.attempts) >= 2
    assert "denoise" not in [s["op"] for s in forced.steps]  # never before roughness
    with pytest.raises(ValueError):
        plan.apply(MeasurementSet())


def test_a_failed_check_tries_the_next_adjustment():
    img = S.hole_array(seed=1)[0]
    plan = draft(probe(img))
    plan.params["feature"] = "bright"  # wrong: the holes are dark
    trial(plan, img, Scale())
    assert [a.change for a in plan.attempts][:2] == ["初稿", "孔的灰度反过来"]
    assert plan.attempts[0].problems and not plan.attempts[1].problems
    assert plan.status == "ready" and plan.params["feature"] == "dark"


def test_model_proposals_are_checked_against_the_allowlist():
    from vexel.agent.models import Proposal

    class Fake:
        name = "fake"

        def available(self):
            return True, ""

        def propose(self, context, image=None):
            assert context["task"] == "sem" and "lines.average" in context["valid_keys"] and "holes.level" not in context["valid_keys"]
            return Proposal(settings={"lines.average": 2, "lines.bogus": 1, "process.denoise": "on", "holes.level": 0.3, "lines.level": 7},
                            notes=["线条清晰"], tier="fake", model="fake-1")  # fmt: skip

    plan = make_plan(S.rough_lines(seed=1)[0], scale=Scale(), chain=[Fake()])
    assert plan.planner == "fake" and plan.model == "fake-1"
    assert plan.params["average"] == 2 and "bogus" not in plan.params and "level" not in plan.params
    assert "denoise" not in [s["op"] for s in plan.steps] and any("去噪" in r and "未采用" in r for r in plan.reasons)
    assert sum("未采用" in r for r in plan.reasons) >= 3  # bogus, another recipe's key, out of range
    assert plan.status == "ready"


def test_a_model_may_choose_the_recipe_unless_the_goal_did():
    from vexel.agent.models import Proposal

    class Picky:
        name = "picky"

        def available(self):
            return True, ""

        def propose(self, context, image=None):
            return Proposal(recipe="particles", tier="picky", model="p")

    img = S.hole_array(seed=1)[0]
    assert make_plan(img, scale=Scale(), chain=[Picky()], trial_run=False).recipe == "particles"
    assert make_plan(img, goal="孔径", scale=Scale(), chain=[Picky()], trial_run=False).recipe == "holes"


def test_recipe_memory_lends_settings_a_person_accepted(tmp_path):
    from vexel.agent.memory import RecipeMemory

    memory = RecipeMemory(tmp_path / "sem.jsonl")
    img = S.hole_array(seed=1)[0]
    first = make_plan(img, scale=Scale(), memory=memory, models=False)
    first.params = {**first.params, "method": "gradient"}
    remember(memory, first, accepted=True)
    again = make_plan(S.hole_array(seed=2)[0], scale=Scale(), memory=memory, models=False)
    assert again.params.get("method") == "gradient" and any("记忆" in r for r in again.reasons)
    other_kind = make_plan(S.rough_lines(seed=1)[0], scale=Scale(), memory=memory, models=False)
    assert "method" not in other_kind.params  # memory is per image kind
    entry = json.loads((tmp_path / "sem.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["meta"]["module"] == "sem" and entry["meta"]["kind"] == "holes" and entry["outcome"]["accepted"]


def test_applying_a_plan_fills_the_record():
    img = S.add_stripes(S.particle_field(seed=1)[0], 0.08)
    plan = make_plan(img, scale=Scale(), models=False)
    ms = MeasurementSet()
    spec = plan.apply(ms)
    assert spec.recipe == "particles" and spec.name == "自动规划" and [s.op for s in ms.processing.steps] == ["line_level"]
    plan.apply(ms)  # applied twice: the step is not added twice
    assert [s.op for s in ms.processing.steps] == ["line_level"] and len(ms.analyses) == 2


# -- command line and MCP ---------------------------------------------------------------------------


@pytest.fixture
def sem_files(tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VEXEL_OFFLINE", "1")
    out = {}
    for name, img in {"lines": S.rough_lines(seed=1)[0], "holes": S.hole_array(seed=1)[0], "noise": np.random.default_rng(0).normal(0.5, 0.1, (300, 400))}.items():
        p = tmp_path / f"{name}.tif"
        tifffile.imwrite(p, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(p, pixel_size=1.5e-9).confirm())
        out[name] = p
    return out


def test_cli_plan_and_analyze_auto(sem_files):
    run = lambda *a: subprocess.run([sys.executable, "-m", "vexel.cli", "sem", *a], capture_output=True, text=True, env=dict(os.environ))  # noqa: E731
    r = run("plan", str(sem_files["lines"]), str(sem_files["noise"]), "--offline", "--json")
    assert r.returncode == 0, r.stderr
    plans = json.loads(r.stdout)
    assert [p["recipe"] for p in plans] == ["lines", None] and plans[0]["status"] == "ready" and plans[1]["status"] == "needs_user"
    r = run("plan", str(sem_files["holes"]), "--offline", "--apply")
    assert r.returncode == 0 and "applied: A1" in r.stdout
    r = run("analyze", str(sem_files["holes"]), str(sem_files["lines"]), "--auto", "--offline", "--json")
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)["results"]
    assert [x["recipe"] for x in res] == ["holes", "lines"]
    # bright rims: the threshold definition measures these holes ~1.2 px small (an edge-definition offset)
    assert res[0]["diameter_mean"] == pytest.approx(24.6 * 1.5, abs=2.5) and res[1]["cd_mean"] == pytest.approx(14.3 * 1.5, rel=0.05)
    r = run("analyze", str(sem_files["noise"]), "--auto", "--offline")
    assert "could not plan" in r.stderr and r.returncode == 1


def test_mcp_sem_tools(sem_files):
    import asyncio

    pytest.importorskip("mcp")
    from vexel.agent.mcp_server import build_server

    server = build_server()

    def call(name, **args):
        result = asyncio.run(server.call_tool(name, args))
        assert not result.is_error, result.content
        return json.loads(result.content[0].text)

    cal = call("sem_calibrate", path=str(sem_files["holes"]))
    assert cal["ok"] and cal["pixel_size_m"] == pytest.approx(1.5e-9)
    plan = call("sem_plan", path=str(sem_files["holes"]))
    assert plan["kind"] == "holes" and plan["status"] == "ready" and plan["reasons"]
    assert call("sem_analyze", path=str(sem_files["holes"]))["analyses"] == []  # sem_plan changed nothing
    done = call("sem_analyze", path=str(sem_files["holes"]), apply=True)
    values = {v["key"]: v["value"] for v in done["analyses"][0]["values"]}
    assert values["holes"] == 77 and values["diameter_mean"] == pytest.approx(24.6 * 1.5e-9, abs=2.5e-9)  # rims: see above
    named = call("sem_analyze", path=str(sem_files["lines"]), recipe="lines", settings={"average": 2})
    assert named["analyses"][0]["settings"] == {"average": 2}
    refused = call("sem_analyze", path=str(sem_files["noise"]), apply=True)
    assert refused["applied"] is False and refused["status"] == "needs_user"
