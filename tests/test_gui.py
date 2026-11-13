"""Desktop UI, driven headlessly (Qt offscreen platform)."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from qtpy.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from vexel.engine import Recipe  # noqa: E402


@pytest.fixture(scope="module")
def app():
    QCoreApplication.setOrganizationName("VexelTests")  # keep test settings apart from the user's
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from vexel.ui.main_window import MainWindow

    w = MainWindow()
    w.warnings = []
    w.warn = lambda title, text: w.warnings.append((title, text))  # no modal dialogs in tests
    yield w
    w.close()


def wait_until_idle(window, timeout_s: float = 180) -> None:
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: None if window.is_running() else loop.quit())
    poll.start(20)
    QTimer.singleShot(int(timeout_s * 1000), loop.quit)
    loop.exec()
    poll.stop()
    assert not window.is_running(), "run did not finish in time"


def test_default_settings_reproduce_the_builtin_recipe(window):
    assert window.overrides() == {}
    window.flow_combo.setCurrentIndex(1)
    window.spins["droplets.count_line"][0].setValue(40)
    window.spins["seg.min_area"][0].setValue(150)
    assert window.overrides() == {"droplets.flow_axis": "y", "droplets.count_line": 0.4, "seg.min_area": 150}
    window.reset_advanced()
    window.flow_combo.setCurrentIndex(0)
    assert window.overrides() == {}


def test_open_reads_metadata_and_calibration(window, droplet_dataset):
    assert not window.run_btn.isEnabled()
    assert window.open_path(droplet_dataset / "droplets.ome.tif")
    assert window.run_btn.isEnabled()
    assert "200 帧" in window.info_label.text() and "768×256" in window.info_label.text()
    assert window.pixel_edit.text() == "2" and window.interval_edit.text() == "1"


def test_unreadable_input_is_reported(window, tmp_path):
    (tmp_path / "x.xyz").write_text("no")
    assert not window.open_path(tmp_path / "x.xyz")
    assert window.warnings and window.path is None


def test_invalid_calibration_blocks_the_run(window, droplet_dataset):
    window.open_path(droplet_dataset / "droplets.ome.tif")
    window.pixel_edit.setText("-1")
    assert not window.start()
    assert window.warnings[-1][0] == "标定有误"


def test_full_run_matches_cli_and_is_saved_next_to_data(window, droplet_dataset):
    window.open_path(droplet_dataset / "droplets.ome.tif")
    assert window.start(preview=False)
    assert window.is_running() and not window.run_btn.isEnabled() or window.run_btn.isHidden()
    wait_until_idle(window)
    assert window.warnings == []
    run_dir = window.run_dir
    assert run_dir.parent.parent == droplet_dataset and run_dir.parent.name == "droplets_vexel"
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["recipe"]["sha256"] == Recipe.load("droplet_basic").resolve().digest()
    assert provenance["input"]["space_unit"] == "um" and len(provenance["input"]["sha256"]) == 64
    summary = json.loads((run_dir / "summary.json").read_text())["droplets.summary"]
    assert window.card_n.value.text() == str(summary["n_droplets"])
    assert window.card_d.value.text().endswith("µm")
    assert window.hist.values.size == summary["n_droplets"]
    assert len(window.viewer.paths) == 4
    assert window.stack.currentIndex() == 1 and not window.notes.isVisible()


def test_preview_is_partial_temporary_and_cleaned_up(app, droplet_dataset):
    from vexel.ui.main_window import PREVIEW_FRAMES, MainWindow

    w = MainWindow()
    w.warn = lambda *a: pytest.fail(f"unexpected warning {a}")
    w.open_path(droplet_dataset / "droplets.ome.tif")
    w.interval_edit.clear()  # uncalibrated time: frequency per frame
    assert w.start(preview=True)
    wait_until_idle(w)
    run_dir = w.run_dir
    assert droplet_dataset not in run_dir.parents
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["input"]["frames_used"] == min(200, PREVIEW_FRAMES)
    assert "sha256" not in provenance["input"]
    assert "预览" in w.notes.text() and w.result_title.text() == "预览结果"
    assert "次/帧" in w.card_f.value.text()
    w.close()
    assert not run_dir.exists()


def test_cancel_removes_partial_output(window, droplet_dataset):
    window.open_path(droplet_dataset / "droplets.ome.tif")
    assert window.start()
    out_dir = window._worker.out_dir
    window.cancel()
    wait_until_idle(window)
    assert window.status.text() == "已取消。"
    assert not out_dir.exists()
    assert window.run_btn.isEnabled()


def test_previous_results_can_be_reopened(window, droplet_dataset):
    runs = sorted((droplet_dataset / "droplets_vexel").iterdir())
    assert runs, "depends on test_full_run_matches_cli_and_is_saved_next_to_data"
    assert window.load_results(runs[-1])
    assert window.result_path.text() == str(runs[-1])


def test_cell_counting_switches_recipe_and_shows_encapsulation(window, cells_dataset):
    assert window.recipe_name() == "droplet_basic"
    window.cells_check.setChecked(True)
    assert window.recipe_name() == "droplet_encapsulation" and window.overrides() == {}
    window.polarity_combo.setCurrentIndex(1)
    window.spins["inclusions.min_contrast"][0].setValue(20)
    assert window.overrides() == {"inclusions.polarity": "bright", "inclusions.min_contrast": 20.0}
    window.polarity_combo.setCurrentIndex(0)
    window.reset_advanced()
    assert window.overrides() == {}

    window.open_path(cells_dataset / "droplets.ome.tif")
    assert window.start()
    wait_until_idle(window)
    assert window.warnings == []
    provenance = json.loads((window.run_dir / "provenance.json").read_text())
    assert provenance["recipe"]["sha256"] == Recipe.load("droplet_encapsulation").resolve().digest()
    summary = json.loads((window.run_dir / "summary.json").read_text())["encapsulation.summary"]
    assert window.enc_cards.isVisibleTo(window) and window.count_box.isVisibleTo(window)
    assert window.card_empty.value.text() == f"{100 * summary['frac_empty']:.1f} %"
    assert window.count_chart.k.size >= 3
    assert "黄圈" in window.viewer.caption.text()

    window.cells_check.setChecked(False)
    window.load_results(window.run_dir)  # still shows encapsulation for a run that has it
    assert window.enc_cards.isVisibleTo(window)


def test_fluorescence_option_channels_and_results(window, fluo_dataset):
    window.open_path(fluo_dataset / "droplets.ome.tif")
    assert window.channel_combo.count() == 2 and "fluorescence" in window.fluo_channel_combo.itemText(1)
    window.fluo_check.setChecked(True)
    assert window.recipe_name() == "droplet_fluorescence" and window.overrides() == {}
    window.spins["positivity.threshold"][0].setValue(50)
    assert window.overrides() == {"positivity.threshold": 50.0, "positivity.method": "fixed"}
    window.reset_advanced()
    window.cells_check.setChecked(True)
    window.cells_channel_combo.setCurrentIndex(1)
    assert window.recipe_name() == "droplet_fluorescence_encapsulation"
    assert window.overrides() == {"input.channels.cells": 1}
    window.cells_check.setChecked(False)

    assert window.start()
    wait_until_idle(window)
    assert window.warnings == []
    summary = json.loads((window.run_dir / "summary.json").read_text())["positivity.summary"]
    assert window.fluo_cards.isVisibleTo(window) and window.scatter_box.isVisibleTo(window)
    assert not window.enc_cards.isVisibleTo(window)
    assert window.card_pos.detail.text().startswith(f"{summary['n_positive']} / {summary['n_droplets']}")
    assert window.card_conc.value.text() not in ("", "–")
    assert window.scatter.x.size == summary["n_droplets"] and window.scatter.threshold == pytest.approx(summary["threshold"])


def test_flags_volume_and_report_in_results(window, hard_run, monkeypatch):
    opened = []
    monkeypatch.setattr("vexel.ui.main_window.QDesktopServices.openUrl", lambda url: opened.append(url.toLocalFile()))
    assert window.load_results(hard_run)
    assert window.flag_box.isVisibleTo(window) and window.flag_strip.count() > 0
    window._show_flag(window.flag_strip.item(0))
    assert "轨迹" in window.viewer.caption.text() and window.viewer._extra is not None
    window.viewer.show_index(0)
    assert window.viewer._extra is None and "轨迹" not in window.viewer.caption.text()
    assert "pL" in window.card_d.detail.text()
    window._open_report()
    assert opened and opened[-1].endswith("report.html")


def test_channel_height_override(window, droplet_dataset):
    window.open_path(droplet_dataset / "droplets.ome.tif")
    window.height_edit.setText("40")
    assert window.overrides() == {"droplets.channel_height_um": 40.0}
    window.height_edit.setText("-3")
    assert not window.start() and window.warnings[-1][0] == "标定有误"
    window.height_edit.clear()
