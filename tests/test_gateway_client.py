"""The desktop's gateway tier against a running gateway (simulated provider): sign in, plan an
SEM image and a droplet video through it, fall back to the offline rules when refused or
unreachable, and keep the token out of diagnostics."""

from __future__ import annotations

import json
import socket
import threading
import time
import zipfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    from vexel.gateway.app import create_app
    from vexel.gateway.store import Store

    monkeypatch.setenv("VEXEL_GATEWAY_MOCK", "1")
    store = Store(tmp_path / "gw.db")
    store.add_org("lab", region="intl")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(store), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", store
    server.should_exit = True
    t.join(5)


def test_sign_in_plan_and_fall_back(gateway, tmp_path, monkeypatch):
    from vexel.agent import models
    from vexel.packs.sem.plan import make_plan
    from vexel.synthetic import sem as S

    url, store = gateway
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VEXEL_PREVIEW", raising=False)
    cfg = models.load_config(policy=False)
    assert models.load_config()["order"] == ["rules"]  # not signed in: offline rules only
    cfg["gateway"]["url"] = url
    got = models.gateway_request(cfg["gateway"], "/v1/auth/redeem", {"invite": store.new_invite("lab"), "label": "pc"})
    cfg["gateway"]["token"] = got["token"]
    path = models.save_config(cfg)
    assert oct(path.stat().st_mode)[-3:] == "600"
    assert models.load_config()["order"] == ["gateway", "rules"]  # signed in: the service, then the rules
    q = models.gateway_request(models.load_config()["gateway"], "/v1/quota")
    assert q["org"] == "lab" and q["requests_left_today"] == 30

    img, _ = S.particle_field(rows=300, cols=400, n=30, seed=2)
    plan = make_plan(img, goal="颗粒粒径分布", trial_run=False)
    assert plan.planner == "gateway" and plan.model.startswith("mock/") and plan.recipe == "particles"
    # refused by the service: the offline rules plan, and the reason is kept
    plan = make_plan(img, goal="帮我写一封邮件", trial_run=False)
    assert plan.planner == "rules" and any("gateway" in s and ("邮件" in s or "只能规划" in s) for s in plan.skipped)
    # the service unreachable: the same
    bad = models.load_config()
    bad["gateway"]["url"] = "http://127.0.0.1:9"
    models.save_config(bad)
    plan = make_plan(img, goal="颗粒粒径", trial_run=False)
    assert plan.planner == "rules" and any("not reachable" in s for s in plan.skipped)
    # diagnostics: whether signed in, never the token
    from vexel import bugreport

    zf = zipfile.ZipFile(bugreport.write_bundle(tmp_path / "bug.zip"))
    env = json.loads(zf.read("environment.json"))
    assert env["planning_models"]["gateway"]["signed_in"] is True
    assert got["token"] not in "".join(zf.read(n).decode("utf-8", "replace") for n in zf.namelist())


def test_droplet_plan_through_the_gateway(gateway, tmp_path, monkeypatch):
    from vexel.agent import models
    from vexel.agent.planner import make_plan
    from vexel.readers import open_volume
    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    url, store = gateway
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    cfg = models.load_config(policy=False)
    cfg["gateway"].update(url=url, token=store.redeem(store.new_invite("lab"), "pc"))
    models.save_config(cfg)
    vol = open_volume(write_dataset(tmp_path / "d", DropletSynthConfig(seed=3, n_frames=80, cell_lambda=1.0)))
    plan = make_plan(vol, "数每个液滴里的细胞", trial=False)
    assert plan.planner == "gateway" and plan.base == "droplet_encapsulation"
    rows = store._q("SELECT task, status FROM usage")
    assert [(r["task"], r["status"]) for r in rows] == [("droplet", "ok")]


def test_sign_in_window(gateway, tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    from vexel.agent import models
    from vexel.ui.ai_service import AiServiceDialog

    url, store = gateway
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    app = QApplication.instance() or QApplication([])
    d = AiServiceDialog()
    assert "未登录" in d.status.text()
    d.url.setText(url)
    d.invite.setText("VX-WRONG")
    assert not d.sign_in() and "登录失败" in d.status.text()
    d.invite.setText(store.new_invite("lab"))
    assert d.sign_in() and "已登录：lab" in d.status.text() and "30" in d.status.text()
    assert models.load_config()["order"] == ["gateway", "rules"]
    d.sign_out()
    assert models.load_config()["order"] == ["rules"] and "未登录" in d.status.text()
    d.close()
    app.processEvents()


def test_agent_loop_revises_up_to_three_times(monkeypatch):
    """The agent loop: after a failed trial the model revises, at most MAX_REVISIONS times,
    and stops as soon as it proposes nothing new."""
    from vexel.agent import models
    from vexel.packs.sem.plan import make_plan
    from vexel.synthetic import sem as S

    calls = []

    class Eager:
        name = "eager"

        def available(self):
            return True, ""

        def propose(self, context, image=None):
            calls.append(context["stage"])
            n = len(calls)
            return models.Proposal(settings={"particles.min_diameter": 3.0 + n}, tier="eager", model="m", recipe="particles")

    img = np.random.default_rng(0).normal(0.5, 0.2, (300, 400))  # noise: no trial ever passes
    plan = make_plan(img, goal="颗粒", chain=[Eager(), models.RulesTier()])
    assert calls == ["draft"] + ["revise"] * models.MAX_REVISIONS
    calls.clear()

    class Quiet(Eager):
        def propose(self, context, image=None):
            calls.append(context["stage"])
            return models.Proposal(settings={}, tier="eager", model="m", recipe="particles")

    make_plan(img, goal="颗粒", chain=[Quiet(), models.RulesTier()])
    assert calls == ["draft", "revise"]  # nothing new proposed: the loop stops
