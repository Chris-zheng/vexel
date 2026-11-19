"""SEM module, M2: manual measurements on synthetic geometry with exact truth, records, exports, CLI.

Acceptance (plan §2.2): errors no larger than one pixel's worth; a record exported and
reloaded gives identical values.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from vexel.packs.sem import measure_io
from vexel.packs.sem.measure import MeasurementSet, Scale, compute, fit_circle, fit_ellipse, profile_edges, sample_profile
from vexel.synthetic.sem import line_space, write_sem

PX = 2.5e-9  # metres per pixel in these tests


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))


def _set(**kw) -> MeasurementSet:
    return MeasurementSet(scale=Scale(pixel_size=PX, **kw))


# -- geometry ---------------------------------------------------------------------------------


def test_lengths_areas_and_angles_are_exact():
    ms = _set()
    line = compute(ms.add("line", [(10, 20), (310, 420)]), ms.scale)
    assert line["length"] == pytest.approx(500 * PX) and line["dx"] == pytest.approx(300 * PX) and line["length_px"] == pytest.approx(500)
    assert line["angle"] == pytest.approx(180 - math.degrees(math.atan2(400, 300)))  # y points down on screen
    assert compute(ms.add("polyline", [(0, 0), (30, 40), (30, 0)]), ms.scale)["length"] == pytest.approx(90 * PX)
    rect = compute(ms.add("polygon", [(0, 0), (40, 0), (40, 25), (0, 25)]), ms.scale)
    assert rect["area"] == pytest.approx(1000 * PX**2) and rect["perimeter"] == pytest.approx(130 * PX)
    assert (rect["cx"], rect["cy"]) == (pytest.approx(20 * PX), pytest.approx(12.5 * PX))
    tri = [(0, 0), (60, 0), (0, 60 * math.tan(math.radians(35)))]
    assert compute(ms.add("angle", [tri[0], tri[1], tri[2]]), ms.scale)["angle"] == pytest.approx(35)
    pts = compute(ms.add("point", [(1, 1), (2, 2), (3, 3)], category="缺陷"), ms.scale)
    assert pts["count"] == 3 and pts["category"] == "缺陷"


def test_anisotropic_pixels_and_tilt_correction():
    ms = MeasurementSet(scale=Scale(pixel_size=2e-9, pixel_size_y=3e-9))
    item = ms.add("line", [(0, 0), (0, 10)])
    assert compute(item, ms.scale)["length"] == pytest.approx(30e-9)
    ms.scale = Scale(pixel_size=2e-9, tilt=60)  # tilted 60°: vertical distances appear halved
    assert compute(item, ms.scale)["length"] == pytest.approx(40e-9)
    assert compute(ms.add("line", [(0, 0), (10, 0)]), ms.scale)["length"] == pytest.approx(20e-9)  # along the tilt axis: unchanged


def test_uncalibrated_measurements_are_in_pixels():
    ms = MeasurementSet()
    v = compute(ms.add("line", [(0, 0), (3, 4)]), ms.scale)
    assert v["length"] == pytest.approx(5) and v["unit"] == "px"


@pytest.mark.parametrize("r", [12.0, 35.5, 80.2])
def test_circle_fit_within_a_pixel(r):
    rng = np.random.default_rng(int(r))
    th = rng.uniform(0, 2 * np.pi, 7)
    pts = np.c_[200 + r * np.cos(th), 150 + r * np.sin(th)] + rng.normal(0, 0.5, (7, 2))  # clicks: half a pixel off
    cx, cy, rr, rms = fit_circle(pts)
    assert abs(2 * rr - 2 * r) <= 1 and math.hypot(cx - 200, cy - 150) <= 1 and rms < 1


@pytest.mark.parametrize("deg", [0, 25, 90, 150])
def test_ellipse_fit_within_a_pixel(deg):
    rng = np.random.default_rng(deg)
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    a, b, t = 40.0, 16.0, math.radians(deg)
    ex, ey = a * np.cos(th), b * np.sin(th)
    pts = np.c_[100 + ex * math.cos(t) - ey * math.sin(t), 80 - (ex * math.sin(t) + ey * math.cos(t))] + rng.normal(0, 0.3, (10, 2))
    cx, cy, major, minor, orient = fit_ellipse(pts)
    assert abs(major - 2 * a) <= 1 and abs(minor - 2 * b) <= 1 and abs(cx - 100) <= 0.5 and abs(cy - 80) <= 0.5
    assert min(abs(orient - deg), 180 - abs(orient - deg)) < 1.5


@pytest.mark.parametrize("r", [6.3, 14.8, 41.2])
def test_magic_wand_size_within_a_pixel(r):
    yy, xx = np.mgrid[:220, :220]
    disc = 0.2 + 0.6 * np.clip(r - np.hypot(xx - 110.4, yy - 109.7) + 0.5, 0, 1)
    img = ndi.gaussian_filter(disc, 1.2) + np.random.default_rng(1).normal(0, 0.02, disc.shape)
    ms = _set()
    v = compute(ms.add("wand", [(110, 110)]), ms.scale, img)
    assert abs(v["eq_diameter"] / PX - 2 * r) <= 1
    assert len(v["outline"]) > 10
    with pytest.raises(ValueError):
        compute(ms.add("wand", [(500, 500)]), ms.scale, img)


# -- line profiles ------------------------------------------------------------------------------


@pytest.mark.parametrize("width,pitch", [(13.3, 31.7), (8.6, 20.4), (21.1, 40.0)])
@pytest.mark.parametrize("method", ["threshold", "gradient"])
def test_profile_line_width_and_pitch(width, pitch, method):
    img = line_space(width=width, pitch=pitch, seed=int(width * 10))
    ms = _set()
    v = compute(ms.add("profile", [(2, 100), (790, 100)], params={"width": 9, "method": method}), ms.scale, img)
    assert abs(v["width_mean"] / PX - width) <= 0.5
    assert abs(v["space_mean"] / PX - (pitch - width)) <= 0.5
    assert abs(v["pitch"] / PX - pitch) <= 0.2 and abs(v["pitch_edges"] / PX - pitch) <= 0.2
    assert v["width_sd"] / PX < 0.3


def test_profile_bright_edge_peaks():
    """SEM shows steps as bright edges: the 'peak' definition measures between them."""
    img = line_space(width=18.4, pitch=40.0, edge_peaks=True, noise=0.02)
    ms = _set()
    v = compute(ms.add("profile", [(2, 100), (790, 100)], params={"width": 9, "method": "peak", "level": 0.5}), ms.scale, img)
    gaps = np.diff(np.asarray(v["edges"]) / PX)
    # alternate gaps: across a line (between its two bright edges), then across a space
    across = gaps[0::2] if gaps[0] < gaps[1] else gaps[1::2]
    assert len(v["edges"]) == 2 * 20 and abs(np.median(across) - 18.4) <= 1.0 and np.ptp(across) < 0.5
    assert abs(v["pitch"] / PX - 40.0) <= 0.2


def test_threshold_level_changes_the_width_as_expected():
    """A higher threshold on blurred lines gives narrower lines: the definition must be recorded."""
    img = line_space(width=12.0, pitch=30.0, blur=2.5, noise=0.0)
    ms = _set()
    w = {lvl: compute(ms.add("profile", [(2, 100), (790, 100)], params={"method": "threshold", "level": lvl}), ms.scale, img)["width_mean"] / PX
         for lvl in (0.3, 0.5, 0.7)}  # fmt: skip
    assert w[0.3] > w[0.5] > w[0.7] and abs(w[0.5] - 12.0) < 0.3


def test_profile_sampling_and_flat_profiles():
    img = np.tile(np.arange(100, dtype=float), (20, 1))
    d, v = sample_profile(img, (10, 10), (50, 10), width=3)
    assert d[-1] == pytest.approx(40) and v[0] == pytest.approx(10) and v[-1] == pytest.approx(50)
    assert profile_edges(np.ones(50)) == []
    with pytest.raises(ValueError):
        profile_edges(np.r_[np.zeros(10), np.ones(10)], method="nope")


# -- records and exports ------------------------------------------------------------------------


@pytest.fixture
def measured(tmp_path):
    from vexel.packs.sem.calibrate import calibrate, load_image

    path = tmp_path / "t.tif"
    write_sem(path, "thermo", pixel_size=PX, rows=400, cols=600)
    cal = calibrate(path, use_ocr=False)
    ms = measure_io.new_set(path, cal)
    ms.add("line", [(100, 100), (300, 180)], name="gate")
    ms.add("polygon", [(400, 100), (500, 120), (480, 220), (390, 200)])
    ms.add("point", [(500, 300), (520, 310)], category="缺陷")
    ms.add("circle", [(300, 250), (340, 290), (300, 330), (260, 290)])
    ms.add("profile", [(20, 380), (300, 380)], params={"width": 5})
    return path, cal, ms, load_image(path)


def test_record_round_trip_gives_identical_values(measured, tmp_path):
    path, cal, ms, img = measured
    before = measure_io.tables(ms, img)["measurements"]
    saved = ms.save(tmp_path / "m.json")
    again = MeasurementSet.load(saved)
    assert again.digest() == ms.digest()
    assert measure_io.tables(again, img)["measurements"].equals(before)
    measure_io.autosave(ms)
    assert measure_io.load_record(cal.sha256).digest() == ms.digest()


def test_a_new_calibration_rescales_every_value(measured):
    _, cal, ms, img = measured
    length = measure_io.tables(ms, img)["measurements"].loc[0, "length (nm)"]
    ms.scale = Scale(pixel_size=2 * PX)
    assert measure_io.tables(ms, img)["measurements"].loc[0, "length (µm)" if "length (µm)" in measure_io.tables(ms, img)["measurements"] else "length (nm)"] == pytest.approx(2 * length)


def test_exports(measured, tmp_path):
    import openpyxl
    from PIL import Image

    path, cal, ms, img = measured
    csvs = measure_io.export_csv(ms, img, tmp_path / "m.csv")
    assert [p.name for p in csvs] == ["m.csv", "m_edges.csv", "m_summary.csv"]
    assert "gate" in csvs[0].read_text(encoding="utf-8-sig")
    wb = openpyxl.load_workbook(measure_io.export_xlsx(ms, img, tmp_path / "m.xlsx"))
    assert wb.sheetnames == ["测量", "汇总", "剖面边缘", "标定", "图像处理"]
    png = measure_io.export_png(img, ms, tmp_path / "m.png", crop_rows=cal.crop_rows)
    assert Image.open(png).size == (600, 400)  # the data bar is cropped off
    svg = measure_io.export_svg(img, ms, tmp_path / "m.svg", crop_rows=cal.crop_rows).read_text(encoding="utf-8")
    assert svg.count("<polyline") >= 2 and "<polygon" in svg and "<ellipse" in svg and "data:image/png;base64," in svg
    summary = measure_io.tables(ms, img)["summary"]
    assert "count (缺陷)" in set(summary["quantity"])


def test_invalid_items_are_refused():
    ms = _set()
    with pytest.raises(ValueError):
        ms.add("line", [(0, 0)])
    with pytest.raises(ValueError):
        ms.add("angle", [(0, 0), (1, 1), (2, 2), (3, 3)])
    with pytest.raises(KeyError):
        ms.get(99)


# -- command line -------------------------------------------------------------------------------


def test_cli_measure_and_profile(measured, tmp_path, capsys):
    from vexel.cli import main

    path, cal, ms, img = measured
    assert main(["sem", "measure", str(path), "--no-ocr"]) == 1  # nothing saved for this image yet
    capsys.readouterr()
    measure_io.autosave(ms)
    out = tmp_path / "out.xlsx"
    assert main(["sem", "measure", str(path), "--no-ocr", "-o", str(out), "-o", str(tmp_path / "out.png")]) == 0
    assert out.is_file() and (tmp_path / "out.png").is_file() and "gate" in capsys.readouterr().out
    assert main(["sem", "measure", str(path), "--no-ocr", "--json", "--tilt", "30"]) == 0
    rows = json.loads(capsys.readouterr().out)
    line = rows[0]
    assert line["dy"] == pytest.approx(80 * PX / math.cos(math.radians(30)))

    grating = tmp_path / "g.tif"
    import tifffile

    tifffile.imwrite(grating, (line_space(width=13.3, pitch=31.7) * 255).astype(np.uint8))
    assert main(["sem", "calibrate", str(grating), "--pixel-size", "2.5nm"]) == 0
    capsys.readouterr()
    assert main(["sem", "profile", str(grating), "--from", "2,100", "--to", "790,100", "--width", "9", "--json", "--no-ocr"]) == 0
    v = json.loads(capsys.readouterr().out)
    assert abs(v["width_mean"] / PX - 13.3) <= 0.5 and abs(v["pitch"] / PX - 31.7) <= 0.2
    assert main(["sem", "profile", str(grating), "--from", "2,100", "--to", "790,100", "--no-ocr"]) == 0
    assert "pitch" in capsys.readouterr().out
