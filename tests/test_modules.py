"""Platform modules (droplet, SEM), the module launcher, and the organisation policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from vexel import modules
from vexel.engine import Recipe

# -- registry -------------------------------------------------------------------------------------


def test_registry_lists_both_modules():
    ids = [m.id for m in modules.modules()]
    assert ids == ["droplet", "sem"]
    assert modules.get("droplet").ready and not modules.get("sem").ready
    assert [m.id for m in modules.modules(include_preview=False)] == ["droplet"]
    with pytest.raises(KeyError, match="unknown module"):
        modules.get("tem")


def test_droplet_recipes_and_operators_come_through_the_registry():
    from vexel.ops import list_ops
    from vexel.packs import builtin_recipes

    droplet = {n for n in builtin_recipes() if n.startswith("droplet_")}
    assert droplet == {"droplet_basic", "droplet_encapsulation", "droplet_fluorescence", "droplet_fluorescence_encapsulation"}
    assert any(o.name == "droplet.summary" for o in list_ops())
    for name in builtin_recipes():
        assert modules.for_recipe(Recipe.load(name)).id == ("droplet" if name in droplet else "sem")
    assert [m.key for m in modules.batch_metrics(Recipe.load("droplet_basic"))][0] == "droplets.diameter_mean"


def _sem_tiff(path: Path, tag: int = 34682) -> Path:
    text = "[Scan]\r\nPixelWidth=2.5e-09\r\nPixelHeight=2.5e-09\r\n"
    tifffile.imwrite(path, np.zeros((64, 80), np.uint8), extratags=[(tag, "s", 0, text, True)])
    return path


def test_files_are_routed_to_their_module(tmp_path, droplet_dataset):
    import imageio.v3 as iio

    m, scores = modules.for_path(droplet_dataset / "droplets.ome.tif")
    assert m.id == "droplet" and scores["droplet"] > scores["sem"]
    for tag in (34682, 34118):
        m, _ = modules.for_path(_sem_tiff(tmp_path / f"fei{tag}.tif", tag))
        assert m.id == "sem"
    img = tmp_path / "plain.png"
    iio.imwrite(img, np.zeros((32, 32), np.uint8))
    assert modules.for_path(img)[0] is None  # a single picture: the person chooses
    hitachi = tmp_path / "h.tif"
    tifffile.imwrite(hitachi, np.zeros((32, 32), np.uint8))
    (tmp_path / "h.txt").write_text("[SemImageFile]\nPixelSize=1.25\n", encoding="utf-8")
    assert modules.for_path(hitachi)[0].id == "sem"
    with pytest.raises(FileNotFoundError):
        modules.for_path(tmp_path / "missing.tif")


def test_cli_modules(capsys, droplet_dataset):
    from vexel.cli import main

    assert main(["modules"]) == 0 and "SEM" in capsys.readouterr().out
    assert main(["modules", str(droplet_dataset / "droplets.ome.tif")]) == 0
    assert capsys.readouterr().out.strip().endswith("-> droplet")


# -- organisation policy --------------------------------------------------------------------------


@pytest.fixture
def policy_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VEXEL_OFFLINE", raising=False)
    path = tmp_path / "policy.json"
    monkeypatch.setenv("VEXEL_POLICY", str(path))
    return path


def test_no_policy_by_default(monkeypatch, tmp_path):
    from vexel.policy import load_policy

    monkeypatch.setenv("VEXEL_POLICY", "")
    monkeypatch.setattr("vexel.policy.default_path", lambda: tmp_path / "none.json")
    p = load_policy()
    assert p.source is None and not p.offline and p.host_allowed("https://api.deepseek.com")


def test_offline_policy_limits_models_and_images(policy_file):
    from vexel.agent import models
    from vexel.policy import load_policy

    policy_file.write_text(json.dumps({"offline": True, "allowed_hosts": ["localhost"], "allow_images": False}), encoding="utf-8")
    p = load_policy()
    assert p.offline and p.host_allowed("http://localhost:11434") and not p.host_allowed("https://api.deepseek.com")
    user = models.load_config(policy=False)
    user["order"] = ["deepseek", "local", "claude", "rules"]
    user["deepseek"]["send_images"] = True
    models.save_config(user)
    cfg = models.load_config()
    assert cfg["order"] == ["local", "rules"] and cfg["deepseek"]["send_images"] is False
    assert models.load_config(policy=False)["order"][0] == "deepseek"  # the person's own settings are kept

    class Remote:
        name, cfg = "deepseek", {"base_url": "https://api.deepseek.com"}

        def available(self):
            return True, ""

        def propose(self, context, image=None):
            raise AssertionError("must not be called")

    proposal, skipped = models.propose({"stage": "draft", "goal": "", "valid_keys": []}, chain=[Remote(), models.RulesTier()])
    assert proposal.tier == "rules" and "organisation policy" in skipped[0]


def test_unreadable_or_missing_policy_is_strictest(policy_file):
    from vexel.agent import models
    from vexel.policy import load_policy

    assert load_policy().offline and load_policy().error == "file not found"  # VEXEL_POLICY names a missing file
    policy_file.write_text("{not json", encoding="utf-8")
    p = load_policy()
    assert p.offline and not p.allow_images and p.error and models.load_config()["order"] == ["rules"]


def test_policy_in_cli_and_bugreport(policy_file, tmp_path, capsys):
    import zipfile

    from vexel.cli import main

    policy_file.write_text(json.dumps({"offline": True, "note": "Fab policy FP-12"}), encoding="utf-8")
    assert main(["policy"]) == 0
    out = capsys.readouterr().out
    assert "FP-12" in out and "rules" in out.splitlines()[-1]
    main(["bugreport", "-o", str(tmp_path / "b.zip")])
    env = json.loads(zipfile.ZipFile(tmp_path / "b.zip").read("environment.json"))
    assert env["policy"]["offline"] is True
