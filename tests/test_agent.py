"""Agent layer: probe, goal parsing, planner with trial refinement, recipe memory, CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vexel.agent.memory import RecipeMemory
from vexel.agent.planner import Goal, draft, make_plan, refine, remember
from vexel.agent.probe import montage, probe
from vexel.core.volume import SpatialVolume
from vexel.engine import run_recipe
from vexel.readers import open_volume


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))


@pytest.fixture(scope="module")
def fps(droplet_dataset, cells_dataset, fluo_dataset):
    plain = open_volume(droplet_dataset / "droplets.ome.tif")
    vertical = SpatialVolume.from_array(np.swapaxes(np.asarray(plain.frames(0, 0)), 1, 2), "TYX")
    noise = SpatialVolume.from_array(np.random.default_rng(0).normal(100, 3, (40, 64, 96)).astype(np.float32), "TYX")
    vols = {"plain": plain, "cells": open_volume(cells_dataset / "droplets.ome.tif"), "fluo": open_volume(fluo_dataset / "droplets.ome.tif"),
            "vertical": vertical, "noise": noise}  # fmt: skip
    return vols, {k: probe(v) for k, v in vols.items()}


def test_probe_reads_motion_objects_and_channel_roles(fps):
    _, fp = fps
    d = fp["plain"].channels[0]
    assert fp["plain"].flow_axis == "x" and d.flow_px[1] == pytest.approx(10, abs=1)  # 20 mm/s at 2 um, 1 ms
    assert d.object_diameter_px == pytest.approx(30, rel=0.08) and d.object_circularity > 0.9 and d.objects_per_frame > 4
    assert fp["plain"].calibrated and fp["plain"].frame_interval_s == pytest.approx(0.001)
    assert (fp["plain"].channels[0].spots_per_object or 0) < 0.15 and fp["cells"].channels[0].spots_per_object > 0.5
    assert fp["fluo"].droplet_channel == 0 and fp["fluo"].fluorescence_channel == 1
    assert fp["vertical"].flow_axis == "y"
    assert fp["noise"].channels[0].objects_per_frame == 0 and any("没有找到" in n for n in fp["noise"].notes)


def test_montage(fps, tmp_path):
    import imageio.v3 as iio

    vols, fp = fps
    path = montage(vols["plain"], fp["plain"], tmp_path / "m.png", size=400)
    img = iio.imread(path)
    assert img.ndim == 3 and img.shape[1] <= 404


@pytest.mark.parametrize(
    "text, cells, fluo",
    [
        ("数每个液滴里的细胞", True, None),
        ("count positive droplets (ddPCR)", None, True),
        ("包裹率和荧光阳性", True, True),
        ("只要粒径，不数细胞，不看荧光", False, False),
        ("", None, None),
    ],
)
def test_goal_parsing(text, cells, fluo):
    g = Goal.parse(text)
    assert (g.cells, g.fluorescence) == (cells, fluo)


def test_drafts_follow_data_and_goal(fps):
    _, fp = fps
    assert draft(fp["plain"], Goal()).base == "droplet_basic"
    cells = draft(fp["cells"], Goal())
    assert cells.base == "droplet_encapsulation" and cells.overrides == {}  # defaults are not repeated
    assert draft(fp["cells"], Goal.parse("不数细胞")).base == "droplet_basic"
    assert draft(fp["fluo"], Goal()).base == "droplet_fluorescence"
    assert draft(fp["vertical"], Goal()).overrides == {"droplets.flow_axis": "y"}
    noise = draft(fp["noise"], Goal())
    assert noise.status == "needs_user" and any("画笔训练" in r for r in noise.reasons)


def test_trial_run_marks_a_good_draft_ready(fps):
    vols, _ = fps
    plan = make_plan(vols["cells"], "数细胞", frames=200)
    assert plan.status == "ready" and plan.attempts[0].metrics["n_droplets"] >= 15
    recipe = plan.recipe()
    assert recipe.name == "droplet_encapsulation"


def test_trial_run_repairs_a_wrong_setting(fps):
    vols, fp = fps
    plan = draft(fp["plain"], Goal())
    plan.overrides["droplets.flow_axis"] = "y"  # wrong: droplets move along x
    refine(plan, vols["plain"], frames=200)
    assert plan.attempts[0].failure == "no droplet crossed the counting line"
    assert plan.status == "ready" and "droplets.flow_axis" not in plan.overrides
    assert any("试运行后调整" in r for r in plan.reasons)


def test_memory_filters_ranks_and_lends_settings(fps, tmp_path):
    vols, fp = fps
    mem = RecipeMemory()
    plan = make_plan(vols["plain"], "", memory=mem, frames=200)
    plan.overrides["droplets.count_line"] = 0.45  # a setting a person found useful
    run = run_recipe(plan.recipe(), vols["plain"], tmp_path / "r", hash_input=False, report=False)
    entry = remember(mem, plan, run)
    assert entry["outcome"]["qc_passed"] and len(mem.entries()) == 1
    hits = mem.query(fp["plain"], family="droplet_basic")
    assert hits and hits[0].distance == pytest.approx(0) and hits[0].score == 1.0
    assert mem.query(fp["fluo"]) == []  # another channel layout never matches
    again = draft(fp["plain"], Goal(), mem)
    assert again.overrides.get("droplets.count_line") == 0.45 and any("历史配方" in r for r in again.reasons)
    with mem.path.open("a") as fh:
        fh.write("{not json\n")
    assert len(mem.entries()) == 1  # a damaged line is skipped


def test_cli_probe_and_plan(droplet_dataset, tmp_path, capsys):
    from vexel.cli import main

    video = str(droplet_dataset / "droplets.ome.tif")
    assert main(["probe", video, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["flow_axis"] == "x"
    out = tmp_path / "plan"
    assert main(["plan", video, "-g", "粒径", "-o", str(out), "--run", "--no-input-hash"]) == 0
    text = capsys.readouterr().out
    assert "ready" in text and (out / "recipe.json").exists() and (out / "run" / "summary.json").exists()
    saved = json.loads((out / "plan.json").read_text())
    assert saved["status"] == "ready" and saved["recipe"]["name"] == "droplet_basic"
    assert len(RecipeMemory().entries()) == 1  # the accepted run is remembered


# -- model tiers ------------------------------------------------------------------------------


class StubTier:
    """A language-model tier that answers with a fixed proposal (or fails)."""

    def __init__(self, name="claude", proposal=None, error=None, ok=True):
        self.name, self._proposal, self._error, self._ok = name, proposal, error, ok
        self.contexts = []

    def available(self):
        return self._ok, "" if self._ok else "not configured"

    def propose(self, context, image=None):
        from vexel.agent.models import ModelUnavailable

        self.contexts.append(context)
        if self._error:
            raise ModelUnavailable(self._error)
        return self._proposal


def test_settings_are_checked_against_the_allowlist():
    from vexel.agent.models import check_settings
    from vexel.agent.planner import valid_keys

    keys = valid_keys("droplet_encapsulation")
    kept, dropped = check_settings(
        {"droplets.count_line": 0.4, "droplets.flow_axis": "z", "inclusions.noise_factor": 99, "seg.bogus": 1,
         "positivity.threshold": 5, "track.max_distance_px": "far"}, keys)  # fmt: skip
    assert kept == {"droplets.count_line": 0.4}
    assert len(dropped) == 5  # bad enum, out of range, not allowlisted, not in this recipe, not a number


def test_fallback_order_and_offline_switch(monkeypatch, tmp_path):
    from vexel.agent import models

    good = models.Proposal(cells=True, tier="local", model="m")
    proposal, skipped = models.propose({"goal": ""}, None, [StubTier("claude", error="offline"), StubTier("local", good), models.RulesTier()])
    assert proposal is good and skipped == ["claude: offline"]
    proposal, skipped = models.propose({"goal": "数细胞"}, None, [StubTier("claude", ok=False)])
    assert proposal.tier == "rules" and proposal.cells is True and skipped == ["claude: not configured"]
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "home" / "config.json").write_text(json.dumps({"models": {"order": ["local", "rules"], "local": {"model": "x"}}}))
    cfg = models.load_config()
    assert cfg["order"] == ["local", "rules"] and cfg["local"]["model"] == "x" and cfg["claude"]["model"] == "claude-opus-5-5"
    monkeypatch.setenv("VEXEL_OFFLINE", "1")
    assert [t.name for t in models.tiers()] == ["rules"]


def test_model_reading_of_the_goal_and_its_settings_shape_the_plan(fps):
    from vexel.agent.models import Proposal

    vols, _ = fps
    stub = StubTier("claude", Proposal(cells=True, fluorescence=None, settings={"droplets.count_line": 0.45, "seg.nope": 3},
                                       notes=["用户想知道珠子的装载分布"], tier="claude", model="claude-opus-5-5"))  # fmt: skip
    plan = make_plan(vols["plain"], "看看珠子的装载是不是随机的", chain=[stub], trial=False, image=False)
    assert plan.planner == "claude" and plan.model == "claude-opus-5-5"
    assert plan.base == "droplet_encapsulation"  # the model read "beads loading" as counting cells
    assert plan.overrides == {"droplets.count_line": 0.45}
    assert any("没有采用" in r and "seg.nope" in r for r in plan.reasons)
    ctx = stub.contexts[0]
    assert ctx["stage"] == "draft" and "fingerprint" in ctx and "droplets.count_line" in ctx["valid_keys"]


def test_claude_tier_request_and_answer(monkeypatch):
    import types

    anthropic = pytest.importorskip("anthropic")

    from vexel.agent import models

    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            text = json.dumps({"cells": None, "fluorescence": True, "settings": [{"key": "droplets.count_line", "value": 0.4, "reason": "r"}],
                               "notes": ["ok"]})  # fmt: skip
            return types.SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                                         content=[types.SimpleNamespace(type="text", text=text)])  # fmt: skip

    class FakeClient:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=FakeMessages())

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    tier = models.ClaudeTier(models.DEFAULT_CONFIG["claude"])
    p = tier.propose({"stage": "draft", "goal": "ddPCR", "valid_keys": ["droplets.count_line"]}, image=b"png")
    assert p.fluorescence is True and p.settings == {"droplets.count_line": 0.4} and p.tier == "claude"
    assert sent["model"] == "claude-opus-5-5" and sent["fallbacks"] == "default"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert all(c["type"] != "image" for c in sent["messages"][0]["content"])  # pictures only when allowed

    class Refusing(FakeMessages):
        def create(self, **kw):
            return types.SimpleNamespace(stop_reason="refusal", model="m", content=[])

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace(beta=types.SimpleNamespace(messages=Refusing())))
    with pytest.raises(models.ModelUnavailable, match="declined"):
        tier.propose({"stage": "draft", "valid_keys": []})


def test_local_tier_talks_to_an_ollama_style_server():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from vexel.agent import models

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            answer = {"cells": True, "fluorescence": False, "settings": [], "notes": []}
            body = json.dumps({"message": {"role": "assistant", "content": json.dumps(answer)}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        tier = models.LocalTier({**models.DEFAULT_CONFIG["local"], "url": f"http://127.0.0.1:{server.server_port}", "model": "vl"})
        p = tier.propose({"stage": "draft", "valid_keys": []})
        assert (p.cells, p.fluorescence, p.tier, p.model) == (True, False, "local", "vl")
        assert seen["format"]["required"] == ["cells", "fluorescence", "settings", "notes"] and seen["model"] == "vl"
    finally:
        server.shutdown()
    unreachable = models.LocalTier({**models.DEFAULT_CONFIG["local"], "url": "http://127.0.0.1:9", "model": "vl", "timeout_s": 2})
    with pytest.raises(models.ModelUnavailable):
        unreachable.propose({"stage": "draft", "valid_keys": []})


def test_compatible_tier_talks_to_an_openai_style_server(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from vexel.agent import models

    seen = {}
    replies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, payload = replies.pop(0)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = {**models.DEFAULT_CONFIG["deepseek"], "base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key_env": "VEXEL_TEST_KEY"}
        tier = models.CompatibleTier("deepseek", cfg)
        monkeypatch.delenv("VEXEL_TEST_KEY", raising=False)
        assert tier.available()[0] is False and "VEXEL_TEST_KEY" in tier.available()[1]
        monkeypatch.setenv("VEXEL_TEST_KEY", "sk-test")
        assert tier.available() == (True, "")

        answer = {"cells": True, "fluorescence": False, "settings": [{"key": "droplets.count_line", "value": 0.4, "reason": "x"}], "notes": []}
        fenced = "```json\n" + json.dumps(answer) + "\n```"
        replies.append((200, {"model": "deepseek-chat", "choices": [{"message": {"content": fenced}, "finish_reason": "stop"}]}))
        p = tier.propose({"stage": "draft", "valid_keys": ["droplets.count_line"]})
        assert (p.cells, p.tier, p.model) == (True, "deepseek", "deepseek-chat")
        assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
        assert seen["body"]["response_format"] == {"type": "json_object"} and seen["body"]["temperature"] == 0
        assert isinstance(seen["body"]["messages"][1]["content"], str)  # numbers only unless images are allowed

        replies.append((200, {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]}))
        with pytest.raises(models.ModelUnavailable, match="cut off"):
            tier.propose({"stage": "draft", "valid_keys": []})
        replies.append((401, {"error": "bad key"}))
        with pytest.raises(models.ModelUnavailable, match="401"):
            tier.propose({"stage": "draft", "valid_keys": []})
        # in the planner chain a failing service falls back to the rules
        replies.append((500, {}))
        proposal, skipped = models.propose({"stage": "draft", "goal": "", "valid_keys": []}, chain=[tier, models.RulesTier()])
        assert proposal.tier == "rules" and skipped and skipped[0].startswith("deepseek")
    finally:
        server.shutdown()


def test_model_config_round_trip(tmp_path, monkeypatch):
    from vexel.agent import models

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path))
    cfg = models.load_config()
    assert set(models.PROVIDERS) >= {"claude", "deepseek", "qwen", "gemini", "custom", "local", "rules"}
    cfg["order"] = ["qwen", "rules"]
    cfg["qwen"]["model"] = "qwen-max"
    models.save_config(cfg)
    again = models.load_config()
    assert again["order"] == ["qwen", "rules"] and again["qwen"]["model"] == "qwen-max"
    assert again["qwen"]["base_url"] == models.COMPATIBLE["qwen"]["base_url"]  # untouched presets survive
    assert [t.name for t in models.tiers(again)] == ["qwen", "rules"]
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "sk-" not in json.dumps(saved)  # keys are never written


# -- MCP ----------------------------------------------------------------------------------------


def test_mcp_tools(droplet_dataset, tmp_path):
    import asyncio

    pytest.importorskip("mcp")
    from vexel.agent.mcp_server import build_server

    server = build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"list_recipes", "show_recipe", "list_operators", "probe_data", "plan_analysis", "run_analysis", "read_results"}

    def call(name, **args):
        result = asyncio.run(server.call_tool(name, args))
        assert not result.is_error, result.content
        return json.loads(result.content[0].text)

    video = str(droplet_dataset / "droplets.ome.tif")
    assert call("probe_data", path=video)["flow_axis"] == "x"
    plan = call("plan_analysis", path=video, goal="粒径", out_dir=str(tmp_path / "plan"), trial=False)
    assert plan["recipe"] == "droplet_basic" and plan["planner"] == "rules" and Path(plan["recipe_path"]).exists()
    run = call("run_analysis", path=video, recipe=plan["recipe_path"], out_dir=str(tmp_path / "run"), max_frames=120)
    assert run["status"] == "completed" and run["summary"]["droplets.summary"]["n_droplets"] > 5
    assert call("read_results", run_dir=run["run_dir"])["result_digest"]


def test_mcp_over_stdio(droplet_dataset, tmp_path):
    import asyncio
    import sys

    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def session():
        params = StdioServerParameters(command=sys.executable, args=["-m", "vexel.cli", "mcp"],
                                       env={**os.environ, "VEXEL_OFFLINE": "1"})  # fmt: skip
        async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            result = await s.call_tool("list_recipes", {})
            return {t.name for t in tools.tools}, result

    names, result = asyncio.run(asyncio.wait_for(session(), 60))
    assert "plan_analysis" in names and "droplet_basic" in result.content[0].text
