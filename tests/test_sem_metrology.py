"""SEM automatic measurements (M4) on synthetic images with exact geometry: line/space CD
and roughness, contact holes, particles, cross-section layers and profiles, tips; the
record, exports and the command line."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

from vexel.packs.sem import measure_io
from vexel.packs.sem import metrology as M
from vexel.packs.sem import objects as O
from vexel.packs.sem.analysis import RECIPES, AnalysisSpec, run
from vexel.packs.sem.measure import MeasurementSet, Scale
from vexel.packs.sem.tips import tip
from vexel.synthetic import sem as S


def _cd_true(tr: dict, slant: float = 0.0) -> float:
    return float(np.mean([np.mean(ln["right"] - ln["left"]) for ln in tr["lines"]])) * math.cos(math.atan(slant))


# -- line / space --------------------------------------------------------------------------------


@pytest.mark.parametrize(("kw", "feature"), [({}, "bright"), ({"noise": 0.08}, "bright"), ({"slant": -0.05}, "bright"), ({}, "dark")])
def test_cd_and_pitch(kw, feature):
    img, tr = S.rough_lines(rows=256, cols=400, seed=3, **kw)
    if feature == "dark":
        img = 1 - img
    r = M.lines_spaces(img, feature=feature)
    across = math.cos(math.atan(kw.get("slant", 0.0)))
    assert r.summary["lines"] == len(tr["lines"])
    assert abs(r.summary["cd_mean"] - _cd_true(tr, kw.get("slant", 0.0))) < 0.5
    assert abs(r.summary["pitch"] - tr["pitch"] * across) < 0.2
    assert abs(r.summary["slant_deg"] - math.degrees(math.atan(kw.get("slant", 0.0)))) < 0.1


def test_edge_definitions_agree_on_pitch_and_peaks_pair_into_lines():
    img, tr = S.rough_lines(rows=256, cols=400, seed=3, edge_peaks=True)
    r = M.lines_spaces(img, method="peak")
    assert r.summary["lines"] == len(tr["lines"]) and abs(r.summary["pitch"] - tr["pitch"]) < 0.2
    assert abs(r.summary["cd_mean"] - _cd_true(tr)) < 1.2  # peaks sit slightly inside the line: a property of the definition
    plain, _ = S.rough_lines(rows=256, cols=400, seed=3)
    g = M.lines_spaces(plain, method="gradient").summary
    assert abs(g["cd_mean"] - _cd_true(tr)) < 0.5


def test_roughness_is_corrected_for_noise():
    ref = M.lines_spaces(S.rough_lines(seed=5, corr=8, noise=0.0)[0]).summary
    noisy = M.lines_spaces(S.rough_lines(seed=5, corr=8, noise=0.08, noise_seed=9)[0]).summary
    assert abs(noisy["ler_3sigma_unbiased"] / ref["ler_3sigma"] - 1) < 0.1
    assert abs(noisy["lwr_3sigma_unbiased"] / ref["lwr_3sigma"] - 1) < 0.1
    assert noisy["ler_3sigma_unbiased"] <= noisy["ler_3sigma"] and noisy["ler_noise_3sigma"] > 0
    assert 4 < noisy["corr_length_lines"] < 14


def test_roughness_of_known_series():
    rng = np.random.default_rng(0)
    edge = S.rough_edge(4096, 1.0, 10.0, rng)
    noise = rng.normal(0, 0.5, 4096)
    clean = M.roughness(edge)
    both = M.roughness(edge + noise)
    assert clean["sigma"] == pytest.approx(np.std(edge - np.polyval(np.polyfit(np.arange(4096), edge, 1), np.arange(4096))), rel=1e-6)
    assert both["sigma"] > clean["sigma"] * 1.05
    assert both["sigma_unbiased"] == pytest.approx(clean["sigma"], rel=0.05)
    assert both["noise_sigma"] == pytest.approx(0.5, rel=0.15)
    assert clean["corr_length"] == pytest.approx(10.0, rel=0.3)
    assert M.roughness(np.full(10, 1.0))["sigma"] is None  # too short


def test_too_few_lines_and_missing_edges_are_reported():
    img, _ = S.rough_lines(rows=40, cols=400, seed=1)
    assert any("行" in w for w in M.lines_spaces(img).warnings)
    assert M.lines_spaces(np.full((100, 100), 0.5)).warnings


# -- holes and particles ---------------------------------------------------------------------------


def test_contact_holes():
    img, tr = S.hole_array(seed=4, sd=0.8, rim=False)
    r = O.holes(img)
    tc = np.array([[x - 0.5, y - 0.5] for x, y, _ in tr["holes"]])
    td = np.array([d for *_, d in tr["holes"]])
    mc = np.array([[h["x"], h["y"]] for h in r["holes"]])
    dist, idx = cKDTree(tc).query(mc)
    err = np.array([h["diameter"] for h in r["holes"]]) - td[idx]
    assert len(r["holes"]) == len(tr["holes"]) and dist.max() < 0.3
    assert abs(err.mean()) < 0.3 and np.abs(err).max() < 1.0
    assert r["summary"]["missing"] == len(tr["missing"])
    found = np.array(r["lattice"]["missing"])
    assert cKDTree(found).query(np.array(tr["missing"]) - 0.5)[0].max() < 0.5
    assert abs(r["summary"]["pitch_a"] - 48.3) < 0.05 and abs(r["summary"]["lattice_angle"] - 90) < 0.1
    assert r["summary"]["roundness_mean"] > 0.98


def test_holes_with_bright_rims_depend_on_the_edge_definition():
    img, tr = S.hole_array(seed=4, rim=True)
    t = O.holes(img, method="threshold")["summary"]["diameter_mean"]
    g = O.holes(img, method="gradient")["summary"]["diameter_mean"]
    assert abs(t - tr["diameter"]) < 1.5 and abs(g - tr["diameter"]) < 1.5 and t != g


@pytest.mark.parametrize(("kw", "need"), [({"touching": 0.0}, 2.0), ({"touching": 0.25}, 2.0), ({"d_mean": 12, "d_sd": 2, "touching": 0.0}, 2.0)])
def test_particles(kw, need):
    img, tr = S.particle_field(seed=6, **kw)
    r = O.particles(img)
    tc = np.array([[x - 0.5, y - 0.5] for x, y, _ in tr["particles"]])
    td = np.array([d for *_, d in tr["particles"]])
    mc = np.array([[p["x"], p["y"]] for p in r["particles"]])
    dist, idx = cKDTree(tc).query(mc)
    ok = dist < 2
    rel = np.abs(np.array([p["eq_diameter"] for p in r["particles"]])[ok] / td[idx[ok]] - 1) * 100
    assert np.median(rel) < need
    assert abs(len(r["particles"]) + r["summary"]["at_border"] - len(td)) <= 0.05 * len(td)
    if kw.get("touching"):
        assert r["summary"]["agglomerated"] > 0.1
    assert r["summary"]["d10_n"] < r["summary"]["d50_n"] < r["summary"]["d90_n"] and r["summary"]["d50_v"] >= r["summary"]["d50_n"]


def test_particles_cut_by_the_border_are_left_out():
    img, _ = S.particle_field(seed=6, touching=0.0)
    r = O.particles(img[40:-40, 40:-40])
    assert r["summary"]["at_border"] > 0 and any("ISO" in w for w in r["warnings"])


def test_feret_of_a_rectangle():
    pts = np.array([[0, 0], [30, 0], [30, 10], [0, 10]], float)
    fmax, fmin, _ = O.feret(pts)
    assert fmax == pytest.approx(math.hypot(30, 10), rel=1e-3) and fmin == pytest.approx(10, rel=1e-3)


# -- cross-sections and tips ----------------------------------------------------------------------


def test_layer_thickness():
    img, tr = S.layer_stack(seed=1)
    for method in ("gradient", "threshold"):
        r = M.layers(img, method=method)
        assert len(r["layers"]) == len(tr["thicknesses"])
        for ly, t in zip(r["layers"], tr["thicknesses"]):
            assert abs(ly["thickness"] - t) < 0.5


@pytest.mark.parametrize("feature", ["bright", "dark"])
@pytest.mark.parametrize("angle", [80.0, 90.0, 95.0])
def test_sidewall_profile(feature, angle):
    img, tr = S.xsection_feature(angle=angle, feature=feature, seed=2)
    f = M.feature_profile(img, feature=feature)
    assert abs(f["angle_left"] - angle) < 0.5 and abs(f["angle_right"] - angle) < 0.5
    assert abs(f["height"] - tr["height"]) <= 1 and abs(f["width_middle"] - tr["width_middle"]) < 0.5


@pytest.mark.parametrize(("radius", "half"), [(8.0, 10.0), (25.0, 20.0)])
def test_tip_radius_and_cone(radius, half):
    r = tip(S.tip(radius=radius, half_angle=half, seed=1)[0])
    assert abs(r["radius"] / radius - 1) < 0.05 and abs(r["cone_angle"] - 2 * half) < 0.5
    assert r["radius_uncertainty"] < 0.05 * radius and abs(r["axis_angle"] - 90) < 1


def test_a_tip_pointing_sideways():
    img, _ = S.tip(radius=15, half_angle=12, seed=2)
    r = tip(np.rot90(img, -1))  # apex now on the right
    assert abs(r["radius"] / 15 - 1) < 0.05 and abs(r["axis_angle"]) < 1


# -- analyses in the record ---------------------------------------------------------------------


def test_analyses_are_kept_in_the_record_and_calibrated(tmp_path):
    img, tr = S.hole_array(seed=4, rim=False)
    ms = MeasurementSet(scale=Scale(pixel_size=2e-9), image={"shape": list(img.shape)})
    ms.add_analysis("holes", [10, 10, 490, 390], min_diameter=6)
    with pytest.raises(ValueError):
        ms.add_analysis("nothing")
    with pytest.raises(ValueError):
        ms.add_analysis("holes", feature="purple")
    back = MeasurementSet.load(ms.save(tmp_path / "r.json"))
    assert back.analyses == ms.analyses
    an = measure_io.analyses_of(back, img)[0]
    assert an.error is None and an.value("diameter_mean") == pytest.approx(tr["diameter"] * 2e-9, rel=0.02)
    assert an.shapes[0][4] == "A1 holes" and len(an.objects) == an.value("holes")
    t = measure_io.tables(back, img)
    row = t["auto"][t["auto"]["key"] == "diameter_mean"].iloc[0]
    assert row["unit"] == "nm" and row["value"] == pytest.approx(tr["diameter"] * 2, rel=0.02)
    assert "diameter (nm)" in t["auto_objects"].columns
    import openpyxl

    assert {"自动测量", "自动测量对象"} <= set(openpyxl.load_workbook(measure_io.export_xlsx(back, img, tmp_path / "m.xlsx")).sheetnames)
    svg = measure_io.export_svg(img, back, tmp_path / "m.svg").read_text(encoding="utf-8")
    assert svg.count("<polygon") > 50


def test_regions_and_errors():
    img = np.random.default_rng(0).normal(0.5, 0.01, (100, 100))
    spec = AnalysisSpec(id=1, recipe="lines", region=[10, 10, 14, 90])
    assert "too small" in run(spec, img, Scale()).error
    spec = AnalysisSpec(id=2, recipe="lines", region=[90, 95, 10, 5])  # corners in any order; clipped to the micrograph
    an = run(spec, img, Scale(), crop_rows=80)
    assert an.extra["region"] == [10, 5, 90, 80]
    assert run(AnalysisSpec(id=3, recipe="nope"), img, Scale()).error


def test_every_recipe_runs_with_its_defaults():
    pics = {"lines": S.rough_lines(rows=200, cols=300)[0], "holes": S.hole_array(rows=200, cols=300)[0],
            "particles": S.particle_field(rows=200, cols=300, n=20)[0], "layers": S.layer_stack()[0],
            "profile": S.xsection_feature()[0], "tip": S.tip()[0]}  # fmt: skip
    assert set(pics) == set(RECIPES)
    for name, img in pics.items():
        an = run(AnalysisSpec(id=1, recipe=name), img, Scale(pixel_size=1e-9))
        assert an.error is None and an.summary, name
        assert all(len(s) == 4 for s in an.summary)


def test_cli_analyze_and_repeatability(tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    env = dict(os.environ)
    spot = tmp_path / "spot1"
    spot.mkdir()
    for k in range(3):
        img, _ = S.rough_lines(rows=200, cols=300, seed=7, noise=0.05, noise_seed=k)
        p = spot / f"s{k}.tif"
        tifffile.imwrite(p, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(p, pixel_size=1e-9).confirm())
    run_ = lambda *a: subprocess.run([sys.executable, "-m", "vexel.cli", "sem", "analyze", *a], capture_output=True, text=True, env=env)  # noqa: E731
    r = run_("--list")
    assert r.returncode == 0 and all(name in r.stdout for name in RECIPES)
    r = run_(*[str(p) for p in sorted(spot.glob("*.tif"))], "-r", "lines", "-p", "average=2", "--json", "--save")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert len(out["results"]) == 3 and out["results"][0]["cd_mean"] == pytest.approx(14.3, abs=0.6)
    cd = next(x for x in out["repeatability"] if x["quantity"] == "cd_mean")
    assert cd["n"] == 3 and cd["group"] == "spot1" and cd["3sigma"] < 0.1
    r = run_(str(spot / "s0.tif"), "-o", str(tmp_path / "a.xlsx"))  # the saved analysis is used
    assert r.returncode == 0 and (tmp_path / "a.xlsx").is_file()
