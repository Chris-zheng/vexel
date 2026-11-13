"""Droplet volume and concentration, flagged droplets, and the HTML report."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from vexel.cli import main
from vexel.engine import Recipe, run_recipe
from vexel.ops import Context, get_op
from vexel.packs.droplet.ops import droplet_volume_um3, per_microlitre
from vexel.packs.droplet.validate import validate_run
from vexel.readers import open_volume
from vexel.report import build_report, write_report
from vexel.report.notes import run_notes


# -- volume ----------------------------------------------------------------------------------


def test_volume_model_sphere_and_flattened():
    sphere = droplet_volume_um3(np.array([30.0]), None)[0]
    assert sphere == pytest.approx(math.pi * 30**3 / 6)
    # continuous at D = h, smaller than a sphere when flattened, larger than the bare cylinder
    h = 40.0
    at_h = droplet_volume_um3(np.array([h]), h)[0]
    assert at_h == pytest.approx(math.pi * h**3 / 6)
    assert droplet_volume_um3(np.array([h + 1e-6]), h)[0] == pytest.approx(at_h, rel=1e-6)
    d = 100.0
    flat = droplet_volume_um3(np.array([d]), h)[0]
    assert math.pi * ((d - h) / 2) ** 2 * h < flat < math.pi * d**3 / 6
    # a disc much wider than it is tall approaches a cylinder of diameter D minus the rim loss
    wide = droplet_volume_um3(np.array([2000.0]), 10.0)[0]
    assert wide == pytest.approx(math.pi * (2000 / 2) ** 2 * 10, rel=0.01)
    assert droplet_volume_um3(np.array([20.0]), h)[0] == pytest.approx(math.pi * 20**3 / 6)  # smaller than h: sphere


def test_concentration_from_mean_volume():
    table = pd.DataFrame({"volume_pl": [100.0, 100.0]})
    assert per_microlitre(1.0, table) == pytest.approx(1e4)  # 1 per 100 pL = 10,000 per uL
    assert per_microlitre(1.0, pd.DataFrame({"x": [1]})) is None
    assert per_microlitre(None, table) is None


def test_volume_in_runs_and_validation(droplet_dataset, tmp_path):
    vol = open_volume(droplet_dataset / "droplets.ome.tif")
    r = run_recipe(Recipe.load("droplet_basic").with_overrides({"droplets.channel_height_um": 40.0}), vol, tmp_path / "r", hash_input=False)
    s = r.summary["droplets.summary"]
    assert s["volume_model"] == "flattened" and s["n_flattened"] == s["n_droplets"]
    assert "volume_pl" in r.tables["droplets.droplets"].columns
    checks = {c.name: c for c in validate_run(tmp_path / "r", droplet_dataset / "truth")}
    assert checks["volume_mean"].passed
    sphere = run_recipe(Recipe.load("droplet_basic"), vol, tmp_path / "s", hash_input=False).summary["droplets.summary"]
    assert sphere["volume_model"] == "sphere" and sphere["volume_mean_pl"] > s["volume_mean_pl"]


def test_uncalibrated_runs_have_no_volume(tmp_path):
    from vexel.core.volume import SpatialVolume
    from vexel.synthetic.droplets import DropletSynthConfig, generate

    video, _, _ = generate(DropletSynthConfig(n_frames=40, seed=3))
    r = run_recipe(Recipe.load("droplet_basic"), SpatialVolume.from_array(video, "TYX"), tmp_path / "r")
    s = r.summary["droplets.summary"]
    assert s["volume_mean_pl"] is None and s["volume_model"] is None
    assert "volume_pl" not in r.tables["droplets.droplets"].columns


def test_fluorescence_concentration(fluo_dataset, tmp_path):
    r = run_recipe(Recipe.load("droplet_fluorescence"), open_volume(fluo_dataset / "droplets.ome.tif"), tmp_path / "r", hash_input=False)
    s = r.summary["positivity.summary"]
    mean_volume = r.tables["positivity.droplets"]["volume_pl"].mean()
    assert s["copies_per_ul"] == pytest.approx(s["copies_per_droplet"] / (mean_volume * 1e-6))
    assert s["copies_per_ul_ci_low"] < s["copies_per_ul"] < s["copies_per_ul_ci_high"]


# -- flags -------------------------------------------------------------------------------------


def _flag(table: pd.DataFrame, tracks: pd.DataFrame, **params):
    op = get_op("droplet.flags")
    return op.fn(op.params(**params), Context(frame_shape=(100, 100), n_frames=10), table=table, tracks=tracks)


def test_flag_rules_and_inspection_frame():
    n = 20
    table = pd.DataFrame(
        {
            "track_id": np.arange(n),
            "circularity": np.r_[0.6, np.full(n - 1, 0.95)],
            "diameter": np.r_[60.0, 90.0, np.full(n - 2, 60.0) + np.linspace(-1, 1, n - 2)],
            "n_frames": np.r_[30, 30, 5, np.full(n - 3, 30)],
            "crossing_frame": np.full(n, 10.4),
            "inclusion_agreement": np.r_[1, 1, 1, 0.5, np.ones(n - 4)],
            "call_margin": np.r_[1, 1, 1, 1, 0.02, np.ones(n - 5)],
        }
    )
    tracks = pd.DataFrame(
        [{"track_id": t, "frame": f, "centroid_y_px": 50.0, "centroid_x_px": 5.0 * f, "diameter_px": 30.0, "touches_border": f == 10} for t in range(n) for f in range(8, 13)]
    )
    out = _flag(table, tracks)
    flagged = out.outputs["table"].set_index("track_id")
    assert flagged.loc[0, "reasons"] == "low_circularity"
    assert flagged.loc[1, "reasons"] == "size_outlier"
    assert flagged.loc[2, "reasons"] == "few_frames"
    assert flagged.loc[3, "reasons"] == "count_unstable"
    assert flagged.loc[4, "reasons"] == "near_threshold"
    assert len(flagged) == 5 and out.outputs["summary"]["n_flagged"] == 5
    # the frame closest to the counting line where the droplet is whole (frame 10 touches the border)
    assert flagged.loc[0, "frame"] in (9, 11) and flagged.loc[0, "centroid_x_px"] == 5.0 * flagged.loc[0, "frame"]
    assert "粘连" in flagged.loc[0, "reason_text"]


def test_flags_empty_inputs():
    empty = pd.DataFrame(columns=["track_id", "circularity", "diameter", "n_frames", "crossing_frame"])
    out = _flag(empty, pd.DataFrame(columns=["track_id", "frame"]))
    assert out.outputs["table"].empty and out.outputs["summary"]["flagged_fraction"] is None


def test_flagged_run_writes_crops_and_qc_warning(hard_run):
    flags = pd.read_csv(hard_run / "tables" / "flags.table.csv")
    prov = json.loads((hard_run / "provenance.json").read_text())
    assert len(flags) > 0 and "count_unstable" in ";".join(flags["reasons"])
    crops = sorted((hard_run / "review" / "flags").glob("*.png"))
    assert len(crops) == min(len(flags), 48) and prov["review_flags"] == [p.name for p in crops]
    assert crops[0].name.startswith("track") and "_frame" in crops[0].name


def test_clean_run_has_no_flags(droplet_dataset, tmp_path):
    r = run_recipe(Recipe.load("droplet_basic"), open_volume(droplet_dataset / "droplets.ome.tif"), tmp_path / "r", hash_input=False)
    assert r.summary["flags.summary"]["n_flagged"] == 0
    assert not (tmp_path / "r" / "review" / "flags").exists()


# -- report ------------------------------------------------------------------------------------


def test_report_is_written_with_every_section(hard_run):
    page = (hard_run / "report.html").read_text(encoding="utf-8")
    for section in ("<h2>结果</h2>", "<h2>图表</h2>", "<h2>复核图</h2>", "<h2>可疑液滴</h2>", "<h2>方法</h2>", "<h2>溯源</h2>"):
        assert section in page
    assert page.count("<svg") == 2  # diameter + cell-count distribution
    assert "data:image/png;base64," in page and 'src="http' not in page  # self-contained
    assert "以线性指派（laptrack）跨帧关联液滴" in page
    for note in run_notes(json.loads((hard_run / "summary.json").read_text()), json.loads((hard_run / "provenance.json").read_text())):
        assert note.replace("“", "").split("（")[0][:12] in page.replace("“", "")


def test_report_escapes_text_and_cli(hard_run, tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["report", str(hard_run), "--out", str(out)]) == 0
    assert out.exists() and "wrote" in capsys.readouterr().out
    assert main(["report", str(tmp_path)]) == 2
    assert not re.search(r"<script", out.read_text(encoding="utf-8"))


def test_report_for_fluorescence_has_scatter(fluo_dataset, tmp_path):
    run_recipe(Recipe.load("droplet_fluorescence"), open_volume(fluo_dataset / "droplets.ome.tif"), tmp_path / "r", hash_input=False)
    page = build_report(tmp_path / "r")
    assert page.count("<svg") == 2 and "阈值" in page and "拷贝/µL" in page


def test_report_failure_does_not_lose_results(droplet_dataset, tmp_path, monkeypatch):
    import vexel.report as report

    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(report, "write_report", boom)
    messages = []
    run_recipe(Recipe.load("droplet_basic"), open_volume(droplet_dataset / "droplets.ome.tif"), tmp_path / "r", hash_input=False, progress=messages.append)
    assert (tmp_path / "r" / "summary.json").exists() and not (tmp_path / "r" / "report.html").exists()
    assert any("report not written" in m for m in messages)
    assert write_report(tmp_path / "r").exists()
