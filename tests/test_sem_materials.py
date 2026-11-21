"""SEM materials measurements on synthetic images with exact geometry: particle shape and
rule-based classes, pores and porosity, fibre diameter and orientation, profile roughness
(ISO 21920), statistics pooled over many images, the HTML report, the command line."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

from vexel.packs.sem import objects as O
from vexel.packs.sem.analysis import AnalysisSpec, run
from vexel.packs.sem.fibres import fibres
from vexel.packs.sem.measure import Scale
from vexel.packs.sem.population import pooled
from vexel.packs.sem.tilt3d import profile_roughness
from vexel.synthetic import sem as S


def _field(extra: list[tuple[float, float, float]] = (), *, rows: int = 300, cols: int = 400, seed: int = 0) -> tuple[np.ndarray, list]:
    """Twelve separate discs (r 9..11 px) on a grid, plus ``extra`` discs."""
    rng = np.random.default_rng(seed)
    discs = [(50 + 70 * (i % 5), 50 + 70 * (i // 5), float(rng.uniform(9, 11))) for i in range(12)]
    return np.asarray(discs + list(extra)), discs


def _image(cov: np.ndarray, seed: int = 0, noise: float = 0.02) -> np.ndarray:
    return ndi.gaussian_filter(0.2 + 0.6 * cov, 0.8) + np.random.default_rng(seed).normal(0, noise, cov.shape)


# -- particles ----------------------------------------------------------------------------------


def test_particle_shape_and_rule_based_classes():
    rows, cols = 300, 420
    discs, _ = _field()
    cov = S._disks(rows, cols, [tuple(d) for d in discs])
    # a "broken" particle: a disc with a bite out of it (low solidity)
    yy, xx = np.mgrid[0:rows, 0:cols]
    broken = ((xx - 260) ** 2 + (yy - 260) ** 2 <= 14**2) & ~((xx - 270) ** 2 + (yy - 260) ** 2 <= 9**2)
    # two touching discs: an agglomerate
    pair = S._disks(rows, cols, [(370.0, 60.0, 10.0), (370.0, 79.0, 10.0)])
    img = _image(np.clip(cov + broken + pair, 0, 1))
    # a foreign particle: same size, much brighter
    img += 0.5 * ndi.gaussian_filter(S._disks(rows, cols, [(370.0, 200.0, 10.0)]), 0.8)
    r = O.particles(img)
    by_pos = {(round(p["x"] / 10), round(p["y"] / 10)): p for p in r["particles"]}
    normal = [p for p in r["particles"] if p["class"] == "normal"]
    assert len(normal) >= 11
    assert all(p["solidity"] > 0.93 and p["convexity"] > 0.93 for p in normal)
    near = lambda x, y: min(r["particles"], key=lambda p: math.hypot(p["x"] - x, p["y"] - y))  # noqa: E731
    assert near(258, 260)["class"] == "irregular" and near(258, 260)["solidity"] < 0.85
    assert near(370, 60)["class"] == "agglomerate" and near(370, 79)["class"] == "agglomerate"
    assert near(370, 200)["class"] == "outlier" and "grey" in near(370, 200)["reason"]
    assert r["summary"]["class_agglomerate"] == 2 and sum(r["summary"][f"class_{c}"] for c in O.CLASSES) == len(r["particles"])
    assert by_pos  # coordinates are filled in
    an = run(AnalysisSpec(id=1, recipe="particles"), img, Scale(pixel_size=1e-9))
    assert {"solidity", "convexity", "class", "reason"} <= set(an.objects.columns)
    assert an.value("class_outlier") == 1


def test_classes_need_enough_particles_to_call_outliers():
    cov = S._disks(120, 200, [(40.0, 60.0, 10.0), (100.0, 60.0, 10.0), (160.0, 60.0, 4.0)])
    r = O.particles(_image(cov))
    assert all(p["class"] == "normal" for p in r["particles"])


# -- pores --------------------------------------------------------------------------------------


def test_porosity_and_pore_sizes():
    rows, cols = 300, 400
    rng = np.random.default_rng(3)
    discs = [(float(rng.uniform(0, cols)), float(rng.uniform(0, rows)), float(rng.uniform(6, 14))) for _ in range(40)]
    cov = S._disks(rows, cols, discs)
    img = ndi.gaussian_filter(0.8 - 0.6 * cov, 0.8) + rng.normal(0, 0.02, cov.shape)
    r = O.pores(img)
    assert r["summary"]["porosity"] == pytest.approx(cov.mean(), abs=0.01)
    assert r["summary"]["at_border"] > 0 and r["summary"]["particles"] > 10
    an = run(AnalysisSpec(id=1, recipe="pores"), img, Scale(pixel_size=1e-9))
    assert an.error is None and an.value("porosity") == pytest.approx(cov.mean(), abs=0.01)
    assert "pore" in an.objects.columns and an.value("d50_n") > 0
    # bright "pores" set wrongly: the warning says so
    assert any("孔隙率" in w for w in O.pores(img, feature="bright")["warnings"])


# -- fibres -------------------------------------------------------------------------------------


def _fibre_errors(img: np.ndarray, truth: list, **kw) -> np.ndarray:
    r = fibres(img, **kw)
    s = r["samples"]
    pts = [t["points"] for t in truth]
    rel = []
    for x, y, d in zip(s["x"], s["y"], s["diameter"]):
        k = int(np.argmin([np.min(np.hypot(p[:, 0] - x, p[:, 1] - y)) for p in pts]))
        rel.append(d / truth[k]["diameter"] - 1)
    return np.asarray(rel)


@pytest.mark.parametrize(("dia", "edge", "method", "bias"), [((8, 12), 0.0, "gradient", 0.02), ((16, 24), 0.0, "threshold", 0.015),
                                                              ((16, 24), 0.25, "gradient", 0.04)])  # fmt: skip
def test_fibre_diameter(dia, edge, method, bias):
    img, truth = S.fibre_mat((400, 400), n=7, diameter=dia, seed=2, edge=edge)
    rel = _fibre_errors(img, truth, method=method)
    assert len(rel) > 300 and abs(np.median(rel)) < bias


def test_edge_brightening_moves_the_threshold_edge_more_than_the_steepest_point():
    img, truth = S.fibre_mat((400, 400), n=7, diameter=(8, 12), seed=4, edge=0.25)
    g = np.median(_fibre_errors(img, truth, method="gradient"))
    t = np.median(_fibre_errors(img, truth, method="threshold"))
    assert 0 < g < t


def test_fibre_orientation():
    img, truth = S.fibre_mat((400, 400), n=8, diameter=(8, 12), seed=5, angle=(28.0, 32.0))
    sm = fibres(img)["summary"]
    assert sm["order"] > 0.95 and sm["orientation"] == pytest.approx(np.mean([t["angle"] for t in truth]), abs=3)
    img, _ = S.fibre_mat((400, 400), n=14, diameter=(8, 12), seed=6)
    assert fibres(img)["summary"]["order"] < 0.6


def test_fibre_diameter_with_rectangular_pixels():
    # one horizontal fibre 12 px thick; pixels twice as tall as wide: 24 nm thick
    img, _ = S.fibre_mat((200, 300), n=1, diameter=(12, 12), seed=1, angle=(0.0, 0.0), edge=0.0)
    sm = fibres(img, sx=1e-9, sy=2e-9)["summary"]
    assert sm["diameter_median"] == pytest.approx(24e-9, rel=0.03) and min(sm["orientation"], 180 - sm["orientation"]) < 1


def test_fibre_recipe_gaps_and_crossings():
    img, _ = S.fibre_mat((400, 400), n=10, diameter=(8, 12), seed=7)
    an = run(AnalysisSpec(id=1, recipe="fibres"), img, Scale(pixel_size=2e-9))
    assert an.error is None and an.value("segments") > 5 and an.value("gaps") > 0
    assert an.value("diameter_mean") == pytest.approx(20e-9, rel=0.2) and 0 < an.value("coverage") < 0.7
    assert {"diameter", "angle", "length_along"} <= set(an.objects.columns) and an.extra["samples"]["diameter"]
    with pytest.raises(ValueError):
        fibres(np.zeros((50, 50)))


# -- profile roughness --------------------------------------------------------------------------


def test_profile_roughness_of_a_sine_with_waviness():
    dx, lam, A = 1.0, 20.0, 3.0
    x = np.arange(2000) * dx
    z = A * np.sin(2 * np.pi * x / lam) + 40 * np.sin(2 * np.pi * x / 2000) + 0.01 * x  # roughness + waviness + tilt
    r = profile_roughness(z, dx, cutoff=100.0)
    assert r["Ra"] == pytest.approx(2 * A / math.pi, rel=0.03) and r["Rq"] == pytest.approx(A / math.sqrt(2), rel=0.03)
    assert r["Rz"] == pytest.approx(2 * A, rel=0.03) and r["sections"] == 5 and abs(r["Rsk"]) < 0.05
    short = profile_roughness(z[:150], dx, cutoff=100.0)
    assert short["sections"] == 1 and short["notes"]
    auto = profile_roughness(z, dx)
    assert auto["cutoff"] == pytest.approx(x[-1] / 5)
    assert profile_roughness(np.full(8, np.nan), 1.0) == {}


# -- pooled statistics, report, command line ------------------------------------------------------


def test_pooled_statistics_over_images():
    ans = []
    for seed in range(3):
        img, _ = S.particle_field(rows=250, cols=300, n=20, seed=seed, touching=0.0)
        ans.append(("g", run(AnalysisSpec(id=1, recipe="particles"), img, Scale(pixel_size=1e-9)), True))
    t = pooled(ans, "nm", 1e-9)
    assert len(t) == 1 and t.iloc[0]["images"] == 3 and t.iloc[0]["objects"] == sum(len(a.objects) for _, a, _ in ans)
    allv = np.concatenate([a.objects["eq_diameter"].to_numpy() for _, a, _ in ans]) / 1e-9
    assert t.iloc[0]["mean"] == pytest.approx(allv.mean()) and t.iloc[0]["d50_v"] >= t.iloc[0]["d50"]
    # an uncalibrated image is not mixed with calibrated ones
    t2 = pooled([*ans, ("g", ans[0][1], False)], "nm", 1e-9)
    assert t2.iloc[0]["images"] == 3


def test_html_report(tmp_path):
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.measure import MeasurementSet
    from vexel.packs.sem.report import build_report

    img, _ = S.fibre_mat((300, 300), n=6, seed=3)
    ms = MeasurementSet(image={"file": "f.tif", "sha256": "0" * 64}, scale=Scale(pixel_size=2e-9))
    ms.add_analysis("fibres", None)
    ms.add_analysis("particles", [10, 10, 200, 200])
    html = build_report(ms, img)
    assert html.startswith("<!doctype html>") and "纤维直径（按长度）" in html and html.count("<svg") >= 2
    assert "data:image/png;base64," in html and "A2" in html and measure_io.display_unit(ms)[0] in html


def test_cli_fibres_pooled_and_report(tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    env = dict(os.environ)
    files = []
    for k in range(2):
        img, _ = S.fibre_mat((300, 300), n=6, seed=k)
        p = tmp_path / f"f{k}.tif"
        tifffile.imwrite(p, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(p, pixel_size=2e-9).confirm())
        files.append(str(p))
    sem = lambda *a: subprocess.run([sys.executable, "-m", "vexel.cli", "sem", *a], capture_output=True, text=True, env=env)  # noqa: E731
    r = sem("analyze", *files, "-r", "fibres", "--json", "--save", "--no-ocr")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["pooled"][0]["images"] == 2 and out["pooled"][0]["samples"] > 300
    assert not any(x["quantity"] == "orientation" for x in out["repeatability"])
    r = sem("report", files[0], "--no-ocr", "-o", str(tmp_path / "r.html"))
    assert r.returncode == 0, r.stderr
    assert "纤维" in (tmp_path / "r.html").read_text(encoding="utf-8")


def test_goals_plan_pores_and_fibres():
    from vexel.packs.sem.plan import goal_recipe, make_plan

    assert goal_recipe("纤维直径和取向") == "fibres" and goal_recipe("electrospun fibre diameter") == "fibres"
    assert goal_recipe("孔隙率") == "pores" and goal_recipe("porosity of the membrane") == "pores" and goal_recipe("接触孔孔径") == "holes"
    img, _ = S.fibre_mat((300, 300), n=6, seed=2)
    plan = make_plan(img, goal="纤维直径", scale=Scale(pixel_size=2e-9), models=False)
    assert plan.recipe == "fibres" and plan.status != "needs_user" and plan.best is not None and not plan.problems

