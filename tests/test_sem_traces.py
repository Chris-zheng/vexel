"""Conductors from above (PCB, metal on a die): trace width, round pads, clearance between
conductors, against synthetic boards with exact geometry (``vexel.synthetic.sem.pcb_board``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vexel.packs.sem.analysis import AnalysisSpec, objects_table, run
from vexel.packs.sem.measure import Scale
from vexel.packs.sem.traces import traces
from vexel.synthetic import sem as S


def _net_of(tr: dict, x: float, y: float) -> dict:
    p = np.array([x, y])
    return min(tr["nets"], key=lambda n: min(float(S._seg_dist(p, np.array(a), np.array(b))) for a, b in zip(n["centre_line"][:-1], n["centre_line"][1:])))


@pytest.mark.parametrize("seed", [1, 4])
def test_board_widths_pads_and_clearance(seed):
    img, tr = S.pcb_board(seed=seed, nets=8, clearance=4.0)
    r = traces(img)
    sm = r["summary"]
    assert sm["nets"] == len(tr["nets"])
    errs = [g["diameter"] - _net_of(tr, g["x"], g["y"])["width"] for g in r["segments"] if g["samples"] >= 5]
    assert len(errs) >= 5 and abs(np.mean(errs)) < 0.15 and np.max(np.abs(errs)) < 0.5
    truth_pads = [p for n in tr["nets"] for p in n["pads"]]
    assert sm["pads"] == len(truth_pads)  # sharp (45°) bends may be listed as wide areas: they are wide copper
    for p in r["pads"]:
        x, y, d = min(truth_pads, key=lambda q: math.hypot(q[0] - p["x"], q[1] - p["y"]))
        assert math.hypot(x - p["x"], y - p["y"]) < 1.0 and p["diameter"] == pytest.approx(d, abs=0.5)
    assert sm["clearance_min"] == pytest.approx(tr["clearance_min"], abs=0.5)
    assert sm["coverage"] == pytest.approx(tr["coverage"], abs=0.003)


def test_square_pads_are_not_counted_as_round():
    img, tr = S.pcb_board(seed=2, nets=8, clearance=4.0, square=1.0)
    r = traces(img)
    assert r["summary"]["pads"] == 0 and r["summary"]["wide_areas"] >= 2 * len(tr["nets"]) - 1
    assert any("不是圆形" in w for w in r["warnings"])


def test_dark_conductors_and_thin_traces():
    img, tr = S.pcb_board(seed=3, nets=8, clearance=4.0, width=(3.0, 5.0))
    r = traces(1.0 - img, feature="dark", smooth=0.5, min_width=2.0)
    errs = [g["diameter"] - _net_of(tr, g["x"], g["y"])["width"] for g in r["segments"] if g["samples"] >= 5]
    assert abs(np.mean(errs)) < 0.15 and r["summary"]["pads"] == sum(len(n["pads"]) for n in tr["nets"])


def test_one_conductor_has_no_clearance():
    img, tr = S.pcb_board(seed=1, nets=1)
    r = traces(img)
    assert r["summary"]["nets"] == 1 and "clearance_min" not in r["summary"]
    assert any("只有一个相连的导体" in w for w in r["warnings"])


def test_recipe_reports_calibrated_values_and_shapes():
    img, tr = S.pcb_board(seed=1, nets=8, clearance=4.0, square=0.3)
    an = run(AnalysisSpec(id=1, recipe="traces"), img, Scale(pixel_size=2e-6, source="manual", confirmed=True))
    assert an.error is None
    vals = {k: v for k, _, v, _ in an.summary}
    assert vals["clearance_min"] == pytest.approx(tr["clearance_min"] * 2e-6, abs=1e-6)  # corners of square pads: see the benchmark
    assert vals["narrowest"] > 0 and vals["width_mean"] > vals["narrowest"]
    df = objects_table([an], "µm", 1e-6)
    assert {"pad", "trace", "clearance"} <= set(df["kind"]) and "clearance (µm)" in df.columns
    labels = [s[4] for s in an.shapes]
    assert "S min" in labels and "W min" in labels
    noise = run(AnalysisSpec(id=1, recipe="traces"), np.random.default_rng(0).normal(0.5, 0.05, (300, 300)), Scale())
    assert noise.error and "噪声" in noise.error


def test_sem_api_lists_and_measures_traces():
    from fastapi.testclient import TestClient

    from vexel.server.api import create_app
    from vexel.server.sem_api import MeasureRequest, measure

    client = TestClient(create_app())
    names = [r["name"] for r in client.get("/api/v1/sem/recipes", params={"lang": "en"}).json()["recipes"]]
    assert "traces" in names
    img, tr = S.pcb_board(seed=4, nets=8, clearance=4.0)
    import base64

    a = np.ascontiguousarray(img, dtype="<f4")
    res = measure(MeasureRequest(recipe="traces", pixels=base64.b64encode(a.tobytes()).decode(), width=a.shape[1], height=a.shape[0],
                                 pixel_size=1.0, unit="µm", lang="en"))  # fmt: skip
    s = {x["key"]: x for x in res["summary"]}
    assert res["error"] is None and s["clearance_min"]["unit"] == "µm"
    assert s["clearance_min"]["value"] == pytest.approx(tr["clearance_min"], abs=0.5)
    assert not any("一" <= ch <= "鿿" for x in res["summary"] for ch in x["label"])
