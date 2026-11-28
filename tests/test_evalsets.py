"""The evaluation sets of the 0.8 plan (``evalsets/``) stay well formed: every planning case
names a known generator and a known measurement, every red-team prompt has a category and
an expectation."""

from __future__ import annotations

import json
from pathlib import Path

from vexel.packs.sem.analysis import RECIPES
from vexel.synthetic import sem as S
from vexel.synthetic.droplets import DropletSynthConfig

ROOT = Path(__file__).resolve().parents[1] / "evalsets"
DROPLET_RECIPES = {"droplet_basic", "droplet_encapsulation", "droplet_fluorescence", "droplet_fluorescence_encapsulation"}


def test_planning_cases_are_well_formed():
    cases = json.loads((ROOT / "planning" / "cases.json").read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) and len(cases) >= 20
    for c in cases:
        src, want = c["source"], c["expect"]["recipe"]
        assert c["module"] in ("sem", "droplet") and ("synthetic" in src) != ("file" in src), c["id"]
        if c["module"] == "sem":
            assert want is None or want in RECIPES, c["id"]
            assert all(a in RECIPES for a in c.get("accept", [])), c["id"]
            if "synthetic" in src:
                assert src["synthetic"] == "noise" or callable(getattr(S, src["synthetic"], None)), c["id"]
        else:
            assert want in DROPLET_RECIPES, c["id"]
            DropletSynthConfig(**src.get("args", {}))  # every argument exists


def test_redteam_prompts_are_well_formed():
    prompts = json.loads((ROOT / "redteam" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    cats = {"allowed", "off_topic", "injection", "exfiltration", "abuse"}
    assert len({p["id"] for p in prompts}) == len(prompts)
    for p in prompts:
        assert p["category"] in cats and p["expect"] in ("allow", "refuse") and p["lang"] in ("zh", "en"), p["id"]
        assert (p["expect"] == "allow") == (p["category"] == "allowed"), p["id"]
    assert {p["category"] for p in prompts} == cats
    assert any(len(p["text"]) > 200 for p in prompts)  # the goal length limit is exercised
