"""SEM module, M1: instrument metadata, data bar and scale bar, OCR, calibration records, CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vexel.packs.sem import calibrate as semcal
from vexel.packs.sem import databar as db
from vexel.packs.sem import ocr
from vexel.packs.sem.metadata import parse_length, read_metadata
from vexel.synthetic.sem import STYLES, write_sem, write_set

needs_ocr = pytest.mark.skipif(not ocr.available()[0], reason="OCR extra (vexel[sem]) not installed")


@pytest.fixture(scope="module")
def sem_set(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sem")
    return {t.style: (p, t) for p, t in write_set(folder)}


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))


# -- metadata ---------------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["thermo", "zeiss", "tescan", "hitachi", "jeol"])
def test_vendor_metadata_gives_the_exact_pixel_size(sem_set, style):
    path, truth = sem_set[style]
    md = read_metadata(path)
    assert md is not None and md.vendor == {"thermo": "thermo_fisher"}.get(style, style)
    # JEOL records a scale bar (rounded to whole pixels), not the pixel size itself
    assert md.pixel_size == pytest.approx(truth.pixel_size, rel=0.005 if style == "jeol" else 1e-6)
    if style in ("thermo", "hitachi", "jeol"):  # these vendors say how tall the micrograph is
        assert md.scan_rows == truth.scan_rows and md.databar_rows == truth.image_shape[0] - truth.scan_rows
    assert md.voltage == pytest.approx(5000)


def test_no_metadata_and_unreasonable_metadata(sem_set, tmp_path):
    assert read_metadata(sem_set["plain"][0]) is None and read_metadata(sem_set["none"][0]) is None
    bad = tmp_path / "bad.tif"
    write_sem(bad, "hitachi", rows=200, cols=300)
    bad.with_suffix(".txt").write_text("[SemImageFile]\nPixelSize=0\nDataSize=300x200\n", encoding="utf-8")
    md = read_metadata(bad)
    assert md.pixel_size is None and any("像素尺寸" in w for w in md.warnings)
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "missing.tif")


def test_resampled_zeiss_image_uses_the_field_width(tmp_path):
    import tifffile

    text = "AP_IMAGE_PIXEL_SIZE\r\nImage Pixel Size = 5.0 nm\r\nAP_WIDTH\r\nWidth = 5.12 \xb5m\r\n".encode("latin-1")
    path = tmp_path / "z.tif"
    tifffile.imwrite(path, np.zeros((384, 512), np.uint8), extratags=[(34118, "B", len(text), text, True)])
    md = read_metadata(path)  # recorded for 1024 columns, saved at 512
    assert md.pixel_size == pytest.approx(10e-9) and "缩放" in md.warnings[0]


def test_parse_lengths_and_magnifications():
    from vexel.packs.sem.metadata import _mag

    assert parse_length("500 nm") == pytest.approx(500e-9) and parse_length("1,5um") == pytest.approx(1.5e-6)
    assert parse_length("2 µm") == pytest.approx(2e-6) and parse_length("WD 4 mm") is None
    assert _mag("50.00 K X") == 50000 and _mag("9,656 X") == 9656 and _mag("x120") == 120
    assert ocr.lengths_in("HFW 1.73 mm") == [pytest.approx(1.73e-3)] and ocr.lengths_in("6.3 pA") == []


# -- data bar and scale bar (pixels only) ------------------------------------------------------


@pytest.mark.parametrize("style", [s for s in STYLES if s != "none"])
def test_databar_and_scale_bar_are_found(sem_set, style):
    path, truth = sem_set[style]
    img = semcal.load_image(path)
    bar = db.find_databar(img)
    assert bar is not None and bar.top == truth.scan_rows
    sb = db.find_scalebar(img, bar)
    assert sb is not None and sb.length_px == pytest.approx(truth.bar_length_px, abs=0.5)


def test_no_databar_in_a_plain_micrograph(sem_set):
    assert db.find_databar(semcal.load_image(sem_set["none"][0])) is None


def test_table_lines_are_not_scale_bars(sem_set):
    """The Thermo Fisher layout has a table line in the scale cell, longer than the bar."""
    path, truth = sem_set["thermo"]
    img = semcal.load_image(path)
    sb = db.find_scalebar(img, db.find_databar(img))
    assert sb.ends == "ticks" and sb.length_px == pytest.approx(truth.bar_length_px, abs=0.5)


# -- calibration ------------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["thermo", "zeiss", "tescan", "hitachi", "jeol"])
def test_metadata_calibration_is_adopted_and_cropped(sem_set, style):
    path, truth = sem_set[style]
    cal = semcal.calibrate(path, use_ocr=False)
    assert cal.ok and cal.source == "metadata" and cal.confidence == "high" and cal.confirmed
    assert cal.crop_rows == truth.scan_rows
    vol = semcal.to_volume(path, cal)
    assert vol.sizes["Y"] == truth.scan_rows and vol.space_unit == "µm"
    assert vol.spacing["x"] == pytest.approx(cal.pixel_size * 1e6)


@needs_ocr
@pytest.mark.parametrize("style", [s for s in STYLES if s != "none"])
def test_databar_calibration_agrees_with_metadata_within_1_percent(sem_set, style):
    """Plan §2.1 acceptance: the pixel size read from the data bar is within 1 % of the truth."""
    path, truth = sem_set[style]
    cal = semcal.calibrate(path, use_metadata=False)
    assert cal.ok and cal.source == "databar" and not cal.confirmed and cal.needs_confirmation
    assert cal.pixel_size == pytest.approx(truth.pixel_size, rel=0.01)
    assert cal.crop_rows == truth.scan_rows


@needs_ocr
def test_cross_checks(sem_set):
    cal = semcal.calibrate(sem_set["thermo"][0])  # metadata, checked against the bar
    assert [c["check"] for c in cal.checks] == ["scale bar"] and cal.checks[0]["agree"]
    cal = semcal.calibrate(sem_set["plain"][0])  # bar, checked against the field width it shows
    assert cal.confidence == "medium" and cal.checks[0]["check"] == "field width" and cal.checks[0]["agree"]


@needs_ocr
def test_wrong_metadata_is_flagged_by_the_data_bar(tmp_path):
    path = tmp_path / "t.tif"
    write_sem(path, "thermo", pixel_size=4e-9)
    import tifffile

    img = tifffile.imread(path)
    tag = f"[Scan]\r\nPixelWidth=8e-09\r\nPixelHeight=8e-09\r\n[Image]\r\nResolutionX={img.shape[1]}\r\nResolutionY=600\r\n"
    tifffile.imwrite(path, img, extratags=[(34682, "s", 0, tag, True)])
    cal = semcal.calibrate(path)
    assert cal.pixel_size == pytest.approx(8e-9) and not cal.checks[0]["agree"] and "请核对" in cal.warnings[-1]


def test_no_calibration_is_never_guessed(sem_set, tmp_path):
    cal = semcal.calibrate(sem_set["none"][0])
    assert not cal.ok and cal.pixel_size is None and "手动" in cal.reason
    cal = semcal.calibrate(sem_set["plain"][0], use_ocr=False)  # bar found, label unread
    assert not cal.ok and "像素" in cal.reason and cal.scale_bar["length_px"] == pytest.approx(sem_set["plain"][1].bar_length_px, abs=0.5)


def test_manual_calibration_and_records(sem_set):
    path, truth = sem_set["plain"]
    cal = semcal.manual(path, bar_length=truth.bar_value)  # the bar's pixels are measured
    assert cal.confirmed and cal.source == "manual"
    assert cal.pixel_size == pytest.approx(truth.bar_value / truth.bar_length_px)
    clicked = semcal.manual(path, bar_length=100e-9, bar_px=50)
    assert clicked.pixel_size == pytest.approx(2e-9) and clicked.scale_bar["ends"] == "clicked"
    with pytest.raises(ValueError):
        semcal.manual(path, pixel_size=5.0)  # 5 m per pixel
    with pytest.raises(ValueError):
        semcal.manual(path)
    semcal.save(clicked)
    again = semcal.for_file(path, use_ocr=False)  # the confirmed record of this exact file
    assert again.source == "manual" and again.pixel_size == pytest.approx(2e-9)
    other = semcal.manual(sem_set["thermo"][0], pixel_size=3e-9)  # disagrees with the metadata: noted
    assert not other.checks[0]["agree"]
    assert semcal.parse_quantity("2.5um") == pytest.approx(2.5e-6)
    with pytest.raises(ValueError):
        semcal.parse_quantity("2.5")


def test_files_route_to_the_sem_module(sem_set):
    from vexel import modules

    for style in ("thermo", "zeiss", "tescan", "hitachi", "jeol"):
        assert modules.for_path(sem_set[style][0])[0].id == "sem"
    assert modules.for_path(sem_set["plain"][0])[0] is None  # a picture without metadata: the person chooses


# -- OCR --------------------------------------------------------------------------------------


def test_reading_values_from_recognised_text():
    T = ocr.Text
    texts = [T("HFW", 1, (175, 443, 199, 454)), T("mag", 1, (126, 444, 161, 455)), T("1.73 mm", 1, (168, 458, 207, 469)),
             T("120 x", 1, (130, 458, 156, 469)), T("500", 1, (397, 444, 417, 456)), T("μm", 1, (415, 445, 433, 456))]  # fmt: skip
    joined = ocr._join_units(texts)
    assert "500 μm" in [t.text for t in joined] and {t.text for t in ocr.scale_labels(joined)} == {"1.73 mm", "500 μm"}
    assert ocr.fields(joined) == {"hfw": pytest.approx(1.73e-3), "mag": 120}
    assert ocr.fields([T("Width = 5.965 µm", 1, (0, 0, 50, 10)), T("EHT = 5.00 kV", 1, (60, 0, 90, 10))]) == {"hfw": pytest.approx(5.965e-6), "hv": 5000}


@needs_ocr
def test_ocr_runs_outside_the_application_process(sem_set):
    """OpenCV (loaded by the OCR engine) and PyAV must not share a process (macOS)."""
    import subprocess

    code = ("import sys; from vexel.packs.sem import calibrate as c; "
            f"cal = c.calibrate({str(sem_set['plain'][0])!r}); print(cal.ok, 'cv2' in sys.modules, 'rapidocr' in sys.modules)")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout.split()
    assert out == ["True", "False", "False"]


def test_ocr_unavailable_is_reported(sem_set, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: (False, "未安装文字识别组件"))
    cal = semcal.calibrate(sem_set["plain"][0])
    assert not cal.ok and "未安装文字识别组件" in cal.warnings


# -- command line -----------------------------------------------------------------------------


def test_cli_sem_calibrate_and_info(sem_set, capsys):
    from vexel.cli import main

    assert main(["sem", "calibrate", str(sem_set["thermo"][0]), "--no-ocr"]) == 0
    assert "2.5 nm/像素" in capsys.readouterr().out
    assert main(["sem", "calibrate", str(sem_set["none"][0]), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)[0]["ok"] is False
    assert main(["sem", "calibrate", str(sem_set["none"][0]), "--bar-length", "100nm", "--bar-pixels", "40"]) == 0
    capsys.readouterr()
    assert main(["sem", "calibrate", str(sem_set["none"][0]), "--json"]) == 0  # the recorded manual calibration
    rec = json.loads(capsys.readouterr().out)[0]
    assert rec["source"] == "manual" and rec["pixel_size"] == pytest.approx(2.5e-9)
    assert main(["sem", "info", str(sem_set["hitachi"][0])]) == 0 and "Hitachi" in capsys.readouterr().out
    assert main(["sem", "info", str(sem_set["plain"][0])]) == 1


@needs_ocr
def test_cli_confirm_records_a_databar_calibration(sem_set, capsys):
    from vexel.cli import main

    path = str(sem_set["plain"][0])
    assert main(["sem", "calibrate", path]) == 0 and "needs confirmation" in capsys.readouterr().out
    assert main(["sem", "calibrate", path, "--confirm"]) == 0
    capsys.readouterr()
    main(["sem", "calibrate", path, "--json"])
    assert json.loads(capsys.readouterr().out)[0]["confirmed"] is True
    assert Path(semcal.store_dir()).is_dir()
