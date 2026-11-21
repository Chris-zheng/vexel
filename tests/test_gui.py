"""Desktop UI, driven headlessly (Qt offscreen platform)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from qtpy.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from vexel.engine import Recipe  # noqa: E402


@pytest.fixture(scope="module")
def app():
    QCoreApplication.setOrganizationName("VexelTests")  # keep test settings apart from the user's
    application = QApplication.instance() or QApplication([])
    yield application
    _flush_deletes()


def _flush_deletes() -> None:
    """Run pending deleteLater calls now, so no widget outlives the test that made it
    (otherwise it is destroyed during interpreter shutdown, in no particular order)."""
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _dispose(window) -> None:
    window.close()
    window.deleteLater()
    _flush_deletes()


@pytest.fixture
def window(app):
    from vexel.ui.main_window import MainWindow

    w = MainWindow()
    w.warnings = []
    w.warn = lambda title, text: w.warnings.append((title, text))  # no modal dialogs in tests
    yield w
    _dispose(w)


def wait_until_idle(window, timeout_s: float = 180) -> None:
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: None if window.is_running() else loop.quit())
    poll.start(20)
    deadline = QTimer()  # an owned timer, stopped below: nothing stays queued after the test
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    deadline.start(int(timeout_s * 1000))
    loop.exec()
    poll.stop()
    deadline.stop()
    _flush_deletes()  # worker and thread are released with deleteLater
    assert not window.is_running(), "run did not finish in time"


def test_default_settings_reproduce_the_builtin_recipe(window):
    assert window.overrides() == {}
    window.flow_combo.setCurrentIndex(1)
    window.spins["droplets.count_line"][0].setValue(40)
    window.spins["seg.min_score"][0].setValue(0.5)
    window.spins["seg.min_diameter_px"][0].setValue(20)
    assert window.overrides() == {
        "droplets.flow_axis": "y", "droplets.count_line": 0.4, "seg.min_score": 0.5, "seg.min_diameter_px": 20.0
    }  # fmt: skip
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
    (tmp_path / "x.xyz").write_text("no", encoding="utf-8")
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
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["recipe"]["sha256"] == Recipe.load("droplet_basic").resolve().digest()
    assert provenance["input"]["space_unit"] == "um" and len(provenance["input"]["sha256"]) == 64
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["droplets.summary"]
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
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["input"]["frames_used"] == min(200, PREVIEW_FRAMES)
    assert "sha256" not in provenance["input"]
    assert "预览" in w.notes.text() and w.result_title.text() == "预览结果"
    assert "次/帧" in w.card_f.value.text()
    _dispose(w)
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
    provenance = json.loads((window.run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["recipe"]["sha256"] == Recipe.load("droplet_encapsulation").resolve().digest()
    summary = json.loads((window.run_dir / "summary.json").read_text(encoding="utf-8"))["encapsulation.summary"]
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
    summary = json.loads((window.run_dir / "summary.json").read_text(encoding="utf-8"))["positivity.summary"]
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


def test_worker_and_thread_are_released_by_qt(window, droplet_dataset):
    from qtpy.QtCore import QThread

    from vexel.ui.worker import RunWorker

    window.open_path(droplet_dataset / "droplets.ome.tif")
    assert window.start(preview=True)
    wait_until_idle(window)
    # nothing left behind: no parked worker or thread objects under the window
    assert not window.findChildren(RunWorker) and not window.findChildren(QThread)
    assert window._worker is None and window._thread is None


# -- live dashboard -------------------------------------------------------------------------


@pytest.fixture
def live(app):
    from vexel.ui.live import LiveWindow

    w = LiveWindow()
    w.warnings = []
    w.warn = lambda title, text: w.warnings.append((title, text))
    yield w
    _dispose(w)


def _speed_index(window, speed) -> int:
    return next(i for i in range(window.speed_combo.count()) if window.speed_combo.itemData(i) == speed)


def test_live_replay_fills_the_dashboard(live, droplet_dataset):
    assert not live.start_btn.isEnabled()
    assert live.set_replay(droplet_dataset / "droplets.ome.tif")
    assert live.pixel_edit.text() == "2" and live.interval_edit.text() == "1"
    live.speed_combo.setCurrentIndex(_speed_index(live, None))
    assert live.start()
    wait_until_idle(live)
    assert live.warnings == []
    assert live.snapshots and live.snapshots[-1]["final"]
    assert live.card_n.value.text() == str(int(live.snapshots[-1]["droplets.n_droplets"]))
    assert live.card_d.value.text().endswith("µm") and "帧/秒" in live.card_fps.value.text()
    assert live.chart_d.x.size == len(live.snapshots) and live.chart_f.x.size > 0
    assert live.frame_view.pixmap() is not None and not live.frame_view.pixmap().isNull()
    p = json.loads((live.run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert p["mode"] == "stream" and live.run_dir.parent.name == "droplets_vexel"
    assert live.report_btn.isVisibleTo(live)


def test_live_watch_runs_a_final_pass(live, tmp_path):
    import tifffile

    from vexel.synthetic.droplets import DropletSynthConfig, generate

    folder = tmp_path / "acq"
    folder.mkdir()
    video, _, _ = generate(DropletSynthConfig(n_frames=150, seed=3))
    for i, f in enumerate(video):
        tifffile.imwrite(folder / f"img_{i}.tif", f)
    (folder / "vexel.json").write_text(json.dumps({"pixel_size": 2.0, "frame_interval": 0.001}), encoding="utf-8")
    assert live.set_watch(folder) and not live.speed_combo.isEnabled()
    live.idle_edit.setText("0.5")
    assert live.start()
    wait_until_idle(live)
    assert live.warnings == []
    assert "全量分析" in live.status.text()
    online = json.loads((live.run_dir / "summary.json").read_text(encoding="utf-8"))["droplets.summary"]["n_droplets"]
    final = json.loads((live.run_dir / "final" / "summary.json").read_text(encoding="utf-8"))["droplets.summary"]["n_droplets"]
    assert online == final > 5
    assert (live.run_dir / "frames.zarr").is_dir()


def test_live_stop_keeps_partial_results(live, droplet_dataset):
    live.set_replay(droplet_dataset / "droplets.ome.tif")
    live.speed_combo.setCurrentIndex(_speed_index(live, 0.02))  # 20 frames/s: 10 s for the whole video
    assert live.start()
    loop = QEventLoop()
    QTimer.singleShot(3500, loop.quit)
    loop.exec()
    live.stop()
    wait_until_idle(live)
    p = json.loads((live.run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert p["status"] == "stopped" and p["stream"]["frames_processed"] < 200
    assert live.status.text().startswith("已停止")


def test_main_window_opens_the_dashboard(window):
    window.open_live()
    assert window.live_window.isVisible()
    window.live_window.close()


# -- review and correction ---------------------------------------------------------------------


def test_review_window_records_edits_and_reruns(app, hard_run):
    from vexel.ui.review import ReviewWindow

    (hard_run / "corrections.json").unlink(missing_ok=True)
    w = ReviewWindow(hard_run)
    w.warn = lambda *a: pytest.fail(f"unexpected warning {a}")
    assert w.queue.count() > 0 and w.filter_combo.currentData() == "flagged"
    w.queue.setCurrentRow(0)
    assert w.selected is not None and w.selected["row"] is not None
    assert w.canvas.view is not None  # zoomed on the selected droplet
    w.field_widgets["n_inclusions"].setValue(7)
    w.set_fields()
    w.verify()
    w.queue.setCurrentRow(1)
    if w.selected is not None:
        w.exclude()
    w.tool_buttons["delete"].setChecked(True)
    y, x = w.selected["y"], w.selected["x"]
    w.selected = None
    w._on_click(y, x)  # deletes the outline in this frame
    kinds = [e.kind for e in w.corrections.edits]
    assert kinds[0] == "table.set" and "labels.delete" in kinds
    _, labels, _ = w.inspector.view(w.frame)
    assert labels[int(round(y)), int(round(x))] == 0
    w.undo()
    assert w.corrections.edits[-1].kind != "labels.delete" or len(w.corrections.edits) < len(kinds)
    saved = json.loads((hard_run / "corrections.json").read_text(encoding="utf-8"))
    assert len(saved["edits"]) == len(w.corrections.edits)
    assert w.apply()
    wait_until_idle(w)
    assert w.status.text().startswith("已按") and w.report_btn.isVisibleTo(w)
    corrected = json.loads((hard_run / "corrected" / "provenance.json").read_text(encoding="utf-8"))
    assert corrected["corrections"]["n_edits"] == len(w.corrections.edits)
    _dispose(w)


def test_review_button_on_results(window, hard_run):
    window.load_results(hard_run)
    assert window.review_btn.isVisibleTo(window)
    window.open_review()
    assert window.review_window.isVisible()
    window.review_window.close()


# -- brush training and the guide ---------------------------------------------------------------


def test_brush_window_trains_previews_and_saves(app, droplet_dataset, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_brush import strokes_from_truth

    from vexel.ops.classify import train
    from vexel.ui.brush import BrushWindow

    w = BrushWindow(droplet_dataset / "droplets.ome.tif", out_dir=tmp_path / "brush")
    assert not w.done_btn.isEnabled()
    _, strokes = strokes_from_truth(droplet_dataset, frames=(20,))
    w.show_frame(20)
    w.strokes[20] = strokes[20].copy()
    w.retrain()
    assert w.model is not None and w.done_btn.isEnabled() and "已在 1 帧上画" in w.frame_label.text()
    prob, labels = w.segment(20)
    assert labels.max() >= 3
    got = []
    w.trained.connect(lambda p, sha, n: got.append((p, sha, n)))
    path, sha = w.finish()
    assert got == [(path, sha, 1)] and Path(path).exists()
    # the preview the person saw is exactly what the saved file gives
    np.testing.assert_array_equal(train(path).probabilities(w.image(20)), w.model.probabilities(w.image(20)))
    _dispose(w)


def test_main_window_uses_brush_training(window, droplet_dataset, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_brush import strokes_from_truth

    from vexel.ops.classify import save_training

    window.open_path(droplet_dataset / "droplets.ome.tif")
    window.seg_combo.blockSignals(True)
    window.seg_combo.setCurrentIndex(window.seg_combo.findData("brush"))
    window.seg_combo.blockSignals(False)
    assert not window.start()  # not trained yet
    assert window.warnings[-1][0] == "还没有训练"
    images, strokes = strokes_from_truth(droplet_dataset)
    path = tmp_path / "b.npz"
    sha = save_training(path, images, strokes, ["液滴", "背景"])
    window._on_brush_trained(str(path), sha, 2)
    recipe = window.build_recipe()
    assert next(s for s in recipe.steps if s.id == "seg").op == "segment.pixel_classifier"
    assert window.start(preview=True)
    wait_until_idle(window)
    assert window.warnings[1:] == []
    p = json.loads((window.run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert next(s for s in p["steps"] if s["id"] == "seg")["op"] == "segment.pixel_classifier"
    window.seg_combo.setCurrentIndex(0)


def test_guide_leads_a_new_user_through_an_analysis(app, tmp_path):
    from vexel.ui.main_window import MainWindow

    w = MainWindow()
    w.warn = lambda *a: pytest.fail(f"unexpected warning {a}")
    w.sample_dir = tmp_path / "sample"
    w.settings.setValue("guide_hidden", "false")
    w.show_guide()
    steps = [w.guide.step]
    w.guide.buttons[0].click()  # 试用示例数据
    steps.append(w.guide.step)
    assert (tmp_path / "sample" / "droplets.ome.tif").exists()
    w.guide.buttons[0].click()  # 快速预览
    wait_until_idle(w)
    steps.append(w.guide.step)
    w.guide.buttons[0].click()  # 开始分析
    steps.append(w.guide.step)
    wait_until_idle(w)
    steps.append(w.guide.step)
    assert steps[:4] == ["data", "preview", "check", "running"] and steps[4] in ("done", "review")
    w.hide_guide()
    assert not w.guide.isVisible() and w.settings.value("guide_hidden") == "true"
    w.show_guide()
    assert w.guide.isVisibleTo(w)
    _dispose(w)


def test_auto_plan_fills_the_settings(window, cells_dataset, monkeypatch):
    monkeypatch.setenv("VEXEL_OFFLINE", "1")
    monkeypatch.setenv("VEXEL_HOME", str(cells_dataset.parent / "vexel-home-gui"))
    window.open_path(cells_dataset / "droplets.ome.tif")
    window.cells_check.setChecked(False)
    window.goal_edit.setText("数每个液滴里的细胞")
    shown = []
    window.confirm_plan = lambda dialog: shown.append(dialog) or True
    assert window.auto_plan()
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: None if window.planning() else loop.quit())
    poll.start(30)
    loop.exec()
    poll.stop()
    assert window.warnings == []
    plan = window.last_plan
    assert plan.status == "ready" and plan.planner == "rules"
    assert shown and shown[0].accept_btn.isEnabled()
    assert window.cells_check.isChecked() and window.recipe_name() == "droplet_encapsulation"
    assert "自动规划" in window.status.text()
    window.cells_check.setChecked(False)


def test_model_settings_dialog_saves_the_choice(app, tmp_path, monkeypatch):
    from vexel.agent import models
    from vexel.ui.model_settings import ModelSettingsDialog

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("VEXEL_OFFLINE", raising=False)
    dialog = ModelSettingsDialog()
    dialog.provider.setCurrentIndex(dialog.provider.findData("gemini"))
    assert "GEMINI_API_KEY" in dialog.key_label.text() and "未设置" in dialog.key_label.text()
    assert dialog.model_edit.text() == models.COMPATIBLE["gemini"]["model"]
    assert dialog.test() is False and "不可用" in dialog.result.text()
    dialog.model_edit.setText("gemini-2.5-pro")
    dialog.save()
    cfg = models.load_config()
    assert cfg["order"] == ["gemini", "rules"] and cfg["gemini"]["model"] == "gemini-2.5-pro"
    dialog.provider.setCurrentIndex(dialog.provider.findData("rules"))
    assert not dialog.model_edit.isEnabled()
    _dispose(dialog)


def test_export_bugreport_from_the_window(window, tmp_path):
    import zipfile

    out = window.export_bugreport(tmp_path / "bug.zip")
    assert out.exists() and "environment.json" in zipfile.ZipFile(out).namelist()
    assert "诊断包已保存" in window.status.text()


def test_batch_window_runs_a_folder(window, tmp_path):
    import pandas as pd

    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    for g, d in (("A", 50.0), ("B", 64.0)):
        for k in range(2):
            write_dataset(tmp_path / "data" / g / f"chip{k}", DropletSynthConfig(n_frames=100, height=128, width=384, diameter_mean_um=d, seed=k))
    window.pixel_edit.setText("2.0")
    assert window.open_batch()
    bw = window.batch_window
    bw.warnings = []
    bw.warn = lambda title, text: bw.warnings.append((title, text))
    assert bw.load_folder(tmp_path / "data") == 4
    assert [bw._cell(i, 1) for i in range(4)] == ["A", "A", "B", "B"]
    bw.table.item(3, 2).setText("chip0")  # B's two files are one chip: averaged into one replicate
    design = bw.save_design(tmp_path / "design.csv")
    saved = pd.read_csv(design, encoding="utf-8-sig")
    assert list(saved["replicate"]) == ["chip0", "chip1", "chip0", "chip0"] and set(saved["pixel_size_um"]) == {2.0}
    bw.out_edit.setText(str(tmp_path / "out"))
    assert bw.start()
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: None if bw.is_running() else loop.quit())
    poll.start(30)
    loop.exec()
    poll.stop()
    assert bw.warnings == [] and bw.result is not None and bw.result.n_ok == 4
    assert [bw._cell(i, 5) for i in range(4)] == ["完成"] * 4
    assert bw.report_btn.isEnabled() and "完成：4 / 4" in bw.status.text()
    units = pd.read_csv(tmp_path / "out" / "replicates.csv")
    assert len(units) == 3  # A: two chips; B: one chip
    window.batch_window = None
    _dispose(bw)


# -- module launcher ------------------------------------------------------------------------------


def test_launcher_routes_files_and_switches_modules(app, droplet_dataset, tmp_path, monkeypatch):
    import tifffile
    from qtpy.QtCore import QSettings

    from vexel.ui import launcher
    from vexel.ui.main_window import MainWindow
    from vexel.ui.sem_window import SemWindow

    monkeypatch.delenv("VEXEL_PREVIEW", raising=False)
    QSettings().setValue("skip_launcher", "false")
    w = launcher.Launcher()
    assert w.buttons["droplet"].isEnabled() and not w.buttons["sem"].isEnabled()
    _dispose(w)

    # a recording opens the droplet module directly, with the module menu added
    win = launcher.start(droplet_dataset / "droplets.ome.tif")
    assert isinstance(win, MainWindow) and win.path is not None and win._module_menu is not None
    titles = [a.text() for a in win._module_menu.actions()]
    assert "微流控液滴" in titles and "SEM 扫描电镜（开发中）" in titles
    assert QSettings().value("last_module") == "droplet"

    # an SEM image while the module is in development: the launcher explains
    sem = tmp_path / "s.tif"
    tifffile.imwrite(sem, np.zeros((64, 80), np.uint8), extratags=[(34682, "s", 0, "[Scan]\r\nPixelWidth=2e-09\r\n", True)])
    shown = launcher.start(sem)
    assert isinstance(shown, launcher.Launcher) and shown.path == sem
    _dispose(shown)
    # asking for it by name, or for a module that does not exist: the launcher explains, no error
    for mid in ("sem", "nothing"):
        shown = launcher.start(module_id=mid)
        assert isinstance(shown, launcher.Launcher)
        _dispose(shown)

    # with previews enabled the SEM window opens and recognises the vendor
    monkeypatch.setenv("VEXEL_PREVIEW", "1")
    sw = launcher.start(sem)
    assert isinstance(sw, SemWindow)
    sw._thread.wait(30000)  # calibration runs off the UI thread
    QCoreApplication.processEvents()
    assert "Thermo Fisher" in sw.meta.text() and "2 nm" in sw.pixel.text()

    # "skip the launcher" opens the last module
    QSettings().setValue("skip_launcher", "true")
    again = launcher.start()
    assert isinstance(again, SemWindow)
    QSettings().setValue("skip_launcher", "false")
    for x in (win, sw, again):
        _dispose(x)
    launcher._open_windows.clear()


def test_sem_window_calibrates_confirms_and_takes_manual_input(app, tmp_path, monkeypatch):
    from vexel.packs.sem import calibrate as semcal
    from vexel.synthetic.sem import write_sem
    from vexel.ui.sem_window import SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    meta = tmp_path / "hitachi.tif"
    write_sem(meta, "hitachi", pixel_size=4.96e-9, rows=300, cols=400)
    w = SemWindow()
    assert w.open_path(meta, wait=True)
    assert "4.96 nm" in w.pixel.text() and "已确认" in w.state.text() and not w.confirm_btn.isVisibleTo(w)
    assert "Hitachi" in w.meta.text() and "信息栏" in w.detail.text()
    assert len(w.view.overlay) >= 1  # the shaded data bar

    plain = tmp_path / "plain.tif"
    truth = write_sem(plain, "plain", pixel_size=2e-9, rows=300, cols=600)
    w.open_path(plain, wait=True)
    if w.cal.ok:  # OCR installed: read from the bar, to be confirmed by the person
        assert w.cal.pixel_size == pytest.approx(truth.pixel_size, rel=0.01) and w.confirm_btn.isVisibleTo(w)
        w._confirm()
        assert w.cal.confirmed and semcal.load(w.cal.sha256).confirmed and not w.confirm_btn.isVisibleTo(w)
    else:
        assert "未标定" in w.pixel.text()

    # manual: two clicks on the scale bar's ends, then its length
    assert w.apply_manual(bar_length="200nm", bar_px=100.0)
    assert w.cal.source == "manual" and "2 nm" in w.pixel.text()
    assert w.apply_manual(pixel_size="3 nm") and semcal.load(w.cal.sha256).pixel_size == pytest.approx(3e-9)
    monkeypatch.setattr("vexel.ui.sem_window.QMessageBox.warning", lambda *a: None)
    assert not w.apply_manual(pixel_size="3")  # no unit: refused
    _dispose(w)


def test_sem_window_measures_edits_undoes_exports_and_restores(app, tmp_path, monkeypatch):
    from qtpy.QtCore import QPointF

    from vexel.synthetic.sem import write_sem
    from vexel.ui.sem_window import SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    path = tmp_path / "t.tif"
    write_sem(path, "thermo", pixel_size=2.5e-9, rows=300, cols=400)
    w = SemWindow()
    w.open_path(path, wait=True)
    assert w.ms is not None and not w.ms.items and all(a.isEnabled() for a in w.tool_actions.values())

    # draw with the canvas the way a person does: a two-click line
    w.set_tool("line")
    w.view._add_point(QPointF(10, 10))
    w.view._add_point(QPointF(110, 10))
    assert [i.kind for i in w.ms.items] == ["line"] and "250 nm" in w.table.item(0, 2).text()
    # a polygon closed with Enter, a count, a profile with the panel's settings
    w.set_tool("polygon")
    for x, y in [(200, 50), (260, 50), (260, 90)]:
        w.view._add_point(QPointF(x, y))
    w.view.finish()
    w.category.setCurrentText("缺陷")
    w.set_tool("point")
    for x, y in [(30, 200), (40, 210)]:
        w.view._add_point(QPointF(x, y))
    w.set_tool("pan")  # leaving the count tool ends the count
    w.edge_method.setCurrentIndex(1)
    w.add_measurement("profile", [[10, 250], [390, 250]])
    assert [i.kind for i in w.ms.items] == ["line", "polygon", "point", "profile"]
    assert w.ms.items[2].category == "缺陷" and w.ms.items[3].params["method"] == "gradient"
    assert w.profile_plot.isVisibleTo(w)  # the new profile is selected and plotted

    # edit: drag the line's end, rename it, delete the polygon; undo and redo
    w._point_moved(1, 1, 210.0, 10.0)
    assert "500 nm" in w.table.item(0, 2).text()
    w.table.item(0, 3).setText("gate")
    assert w.ms.get(1).name == "gate"
    w._select(2)
    w.delete_selected()
    assert [i.id for i in w.ms.items] == [1, 3, 4]
    w.undo()
    assert [i.id for i in w.ms.items] == [1, 2, 3, 4]
    w.redo()
    assert [i.id for i in w.ms.items] == [1, 3, 4]
    w.tilt_on.setChecked(True)
    w.tilt.setValue(60)
    assert w.ms.scale.tilt == 60

    # exports
    for name in ("m.xlsx", "m.csv", "m.png", "m.svg", "m.json"):
        assert w.export(tmp_path / name)[0].is_file()
    _dispose(w)

    # reopening the image restores its measurements (kept by the file's SHA-256)
    w2 = SemWindow()
    w2.open_path(path, wait=True)
    assert [i.id for i in w2.ms.items] == [1, 3, 4] and w2.ms.get(1).name == "gate" and w2.tilt_on.isChecked()
    assert w2.table.rowCount() == 3
    _dispose(w2)


def test_sem_window_processes_compares_averages_and_restores(app, tmp_path, monkeypatch):
    import tifffile
    from qtpy.QtCore import Qt

    from vexel.packs.sem.calibrate import load_image
    from vexel.synthetic.sem import write_sem
    from vexel.ui.sem_window import SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    path = tmp_path / "t.tif"
    write_sem(path, "thermo", pixel_size=2.5e-9, rows=300, cols=400)
    w = SemWindow()
    w.open_path(path, wait=True)
    assert w.result is not None and w.add_step_btn.isEnabled() and not w.export_proc_btn.isEnabled()
    before = w.result.analysis.copy()
    crop = w.cal.crop_rows

    # steps: corrections go before display steps whatever the order of adding
    w.add_step("contrast", gamma=0.8)
    w.add_step("line_level")
    w.add_step("background", method="gaussian")
    assert [s.op for s in w.ms.processing.steps] == ["line_level", "background", "contrast"]
    assert w.steps.count() == 3 and w.steps.currentRow() == 1 and "高斯" in w.params_box.findChildren(type(w.edge_method))[0].currentText()
    assert not np.allclose(w.result.analysis[:crop], before[:crop])
    assert np.array_equal(w.result.analysis[crop:], before[crop:])  # the data bar is untouched
    w.move_step(1, 1)  # a correction cannot move after a display step
    assert [s.op for s in w.ms.processing.steps] == ["line_level", "background", "contrast"]
    w.move_step(1, -1)
    assert [s.op for s in w.ms.processing.steps] == ["background", "line_level", "contrast"]
    w.set_step_params(1, direction="columns")
    assert w.ms.processing.steps[1].params["direction"] == "columns"
    w.steps.item(2).setCheckState(Qt.CheckState.Unchecked)
    assert not w.ms.processing.steps[2].enabled
    w.undo()
    assert w.ms.processing.steps[2].enabled and w.steps.item(2).checkState() == Qt.CheckState.Checked
    w.remove_step(2)
    assert [s.op for s in w.ms.processing.steps] == ["background", "line_level"]
    assert w.export_proc_btn.isEnabled()
    # the parameter form edits the selected step
    w.steps.setCurrentRow(1)
    w._param_widgets["keep"][1].setValue(25.0)
    w._apply_params()
    assert w.ms.processing.steps[1].params["keep"] == 25.0

    # the before/after view and the saturation mask
    w.set_compare(True, 0.3)
    assert w.view._before is not None and w.view._before.pixmap().width() == 120
    w.set_compare(False)
    assert w.view._before is None
    w.set_saturation(enabled=False)
    assert w.result.mask is None and "未标记" in w.sat_info.text()
    w.set_saturation(enabled=True)
    assert w.result.mask is not None

    # measurements use the corrected image; the record keeps the chain
    w.add_measurement("profile", [[10, 150], [390, 150]])
    sheets = __import__("openpyxl").load_workbook(w.export(tmp_path / "m.xlsx")[0]).sheetnames
    assert "图像处理" in sheets

    # the processed image opens calibrated
    out = w.export_processed(tmp_path / "p.tif")
    assert out.is_file() and "vexel" in tifffile.TiffFile(out).pages[0].description
    _dispose(w)
    w3 = SemWindow()
    w3.open_path(out, wait=True)
    assert w3.cal.ok and w3.cal.pixel_size == pytest.approx(2.5e-9) and w3.cal.crop_rows == crop
    _dispose(w3)

    # reopening the original restores the chain
    w2 = SemWindow()
    w2.open_path(path, wait=True)
    assert [s.op for s in w2.ms.processing.steps] == ["background", "line_level"] and w2.steps.count() == 2
    # multi-frame averaging: a second frame shifted by 2 pixels
    img = load_image(path)
    frame2 = tmp_path / "t2.tif"
    shifted = img.copy()
    shifted[:crop] = np.roll(img[:crop], 2, axis=1)
    tifffile.imwrite(frame2, shifted)
    info = w2.average_with([frame2], tmp_path / "avg.tif")
    assert info is not None and info["frames"] == 2
    assert abs(info["shifts"][1][1] + 2) < 0.1 and abs(info["shifts"][1][0]) < 0.1
    assert w2.path.name == "avg.tif" and w2.cal.ok and w2.cal.pixel_size == pytest.approx(2.5e-9)
    _dispose(w2)


def test_sem_window_runs_slow_processing_in_the_background(app, tmp_path, monkeypatch):
    from vexel.synthetic.sem import write_sem
    from vexel.ui.sem_window import SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    path = tmp_path / "t.tif"
    write_sem(path, "thermo", pixel_size=2.5e-9, rows=300, cols=400)
    w = SemWindow()
    w.open_path(path, wait=True)
    w._sync = False  # as when a person opens the image
    before = w.result
    w.add_step("denoise")  # non-local means: slow, runs in the background
    assert w._proc_thread is not None and w.result is before and "处理中" in w.sat_info.text()
    w.add_step("contrast")  # changed meanwhile: the chain then in force runs next
    assert w.wait_processing() and w.wait_processing()
    assert w.result is not before and w._result_key == w.ms.processing.model_dump_json()
    assert not __import__("numpy").array_equal(w.result.analysis, before.analysis)
    # exports wait for the current chain
    w.set_step_params(0, strength=2.0)
    out = w.export(tmp_path / "m.xlsx")
    assert out[0].is_file() and w._result_key == w.ms.processing.model_dump_json()
    w.wait_processing()
    _dispose(w)


def test_sem_window_automatic_measurements(app, tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C
    from vexel.synthetic.sem import rough_lines
    from vexel.ui.sem_window import TAB_AUTO, SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    img, truth = rough_lines(rows=300, cols=400, seed=1)
    path = tmp_path / "lines.tif"
    tifffile.imwrite(path, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
    C.save(C.manual(path, pixel_size=1.5e-9).confirm())
    w = SemWindow()
    w.open_path(path, wait=True)
    assert w.run_btn.isEnabled() and w.auto_list.count() == 0

    # a region drawn the way a person does: two clicks with the region tool
    w.set_tool("region")
    assert w.tabs.currentIndex() == TAB_AUTO
    from qtpy.QtCore import QPointF

    w.view._add_point(QPointF(5, 10))
    w.view._add_point(QPointF(390, 290))
    assert w._pending_region == [5, 10, 390, 290] and w.view._region_item is not None
    an = w.run_analysis("lines")
    assert an is not None and an.error is None and w.ms.analyses[0].region == [5, 10, 390, 290]
    assert an.value("cd_mean") == pytest.approx(truth["width"] * 1.5e-9, rel=0.02)
    assert w.auto_list.count() == 1 and w.auto_list.currentRow() == 0 and w.auto_table.rowCount() == an.value("lines")
    assert "线宽 CD" in w.auto_summary.text() and len(w.view._auto) > 10 and w._pending_region is None
    # a second one on the whole image, with a setting; undo and redo
    w.run_analysis("lines", None, average=3)
    assert [a.params for a in w.ms.analyses] == [{}, {"average": 3}] and w.auto_list.currentRow() == 1
    w.undo()
    assert len(w.ms.analyses) == 1 and w.auto_list.count() == 1
    w.redo()
    assert len(w.ms.analyses) == 2
    w.auto_list.setCurrentRow(0)
    w._delete_analysis()
    assert [a.id for a in w.ms.analyses] == [2]
    # the panel's own settings and the rerun of the selected one
    w.recipe.setCurrentIndex(w.recipe.findData("lines"))
    w._auto_widgets["average"][1].setValue(2)
    w.auto_list.setCurrentRow(0)
    w._rerun_selected()
    assert w.ms.analyses[0].params == {"average": 2}
    # exports and reopening
    sheets = __import__("openpyxl").load_workbook(w.export(tmp_path / "m.xlsx")[0]).sheetnames
    assert "自动测量" in sheets and "自动测量对象" in sheets
    assert w.export(tmp_path / "m.png")[0].is_file()
    _dispose(w)
    w2 = SemWindow()
    w2.open_path(path, wait=True)
    assert [a.id for a in w2.ms.analyses] == [2] and w2.auto_list.count() == 1 and w2.analyses[0].value("lines") >= 8
    _dispose(w2)


def test_sem_window_materials_chart_pooling_and_report(app, tmp_path, monkeypatch):
    import openpyxl
    import tifffile

    from vexel.packs.sem import calibrate as C
    from vexel.synthetic.sem import fibre_mat, particle_field
    from vexel.ui.sem_window import SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    paths = []
    for k in range(3):
        img, _ = particle_field(rows=250, cols=300, n=20, seed=k, touching=0.0)
        p = tmp_path / f"p{k}.tif"
        tifffile.imwrite(p, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(p, pixel_size=2e-9).confirm())
        paths.append(p)
    w = SemWindow()
    w.open_path(paths[0], wait=True)
    an = w.run_analysis("particles")
    assert an.error is None and "class" in an.objects.columns
    assert w.auto_chart.isVisibleTo(w) and w.auto_chart.xq.currentData().startswith("eq_diameter")
    for i in range(w.auto_chart.scale.count()):  # linear, log, log-log draw without errors
        w.auto_chart.scale.setCurrentIndex(i)
    w.auto_chart.weight.setCurrentIndex(1)
    w.auto_chart.yq.setCurrentIndex(w.auto_chart.yq.findData("circularity"))
    w.auto_chart.grab()
    assert w.pool_btn.isEnabled()
    table, per = w.pool_files(paths[1:])
    assert len(per) == 3 and table.iloc[0]["images"] == 3 and table.iloc[0]["objects"] == sum(len(a.objects) for _, a, _ in per)
    assert "合并统计" in openpyxl.load_workbook(w.save_pool(tmp_path / "pool.xlsx", table, per)).sheetnames
    out = w.export(tmp_path / "r.html")[0]
    assert out.suffix == ".html" and "<svg" in out.read_text(encoding="utf-8")
    # fibres: the chart shows the centre-line samples
    img, _ = fibre_mat((300, 300), n=6, seed=1)
    fp = tmp_path / "f.tif"
    tifffile.imwrite(fp, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
    C.save(C.manual(fp, pixel_size=2e-9).confirm())
    w.open_path(fp, wait=True)
    an = w.run_analysis("fibres")
    assert an.error is None and an.value("segments") > 3 and w.auto_chart.weights is not None
    assert "angle (°)" in w.auto_chart.columns
    _dispose(w)


def test_sem_window_auto_plan(app, tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C
    from vexel.synthetic.sem import hole_array
    from vexel.ui.sem_window import TAB_AUTO, SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VEXEL_OFFLINE", "1")
    img, _ = hole_array(seed=1)
    path = tmp_path / "holes.tif"
    tifffile.imwrite(path, (np.clip(img, 0, 1) * 65535).astype(np.uint16))
    C.save(C.manual(path, pixel_size=2e-9).confirm())
    w = SemWindow()
    w.open_path(path, wait=True)
    assert w.plan_btn.isEnabled()
    # declined: nothing enters the record
    plan = w.auto_plan(accept=False)
    assert plan.recipe == "holes" and plan.status == "ready" and not w.ms.analyses and "接触孔" in w.plan_info.text()
    # accepted: the measurement is added, selected and shown; one undo removes it
    w.goal.setText("孔径")
    plan = w.auto_plan(accept=True)
    assert [a.recipe for a in w.ms.analyses] == ["holes"] and w.ms.analyses[0].name == "自动规划"
    assert w.tabs.currentIndex() == TAB_AUTO and w.auto_list.currentRow() == 0 and w.analyses[0].value("holes") == 77
    w.undo()
    assert not w.ms.analyses
    # in the background, as a person uses it
    w._sync = False
    w.auto_plan(accept=True)
    assert w._plan_thread is not None and not w.plan_btn.isEnabled()
    from qtpy.QtCore import QCoreApplication, QElapsedTimer, QThread

    t = QElapsedTimer()
    t.start()
    while w._plan_thread is not None and t.elapsed() < 60000:
        QCoreApplication.processEvents()
        QThread.msleep(10)
    assert w._plan_thread is None and len(w.ms.analyses) == 1 and w.plan_btn.isEnabled()
    _dispose(w)


def test_sem_window_tilt_3d(app, tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C
    from vexel.synthetic import sem as S
    from vexel.ui.sem_window import TAB_3D, SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    z = S.height_map("steps", seed=1)
    tex = S.surface_texture(*z.shape, seed=2)
    paths = []
    for i, t in enumerate((-5, 5)):
        p = tmp_path / f"v{i}.tif"
        tifffile.imwrite(p, (np.clip(S.tilt_view(z, tex, t, seed=i), 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(p, pixel_size=2e-9).confirm())
        paths.append(p)
    w = SemWindow()
    w.open_path(paths[0], wait=True)
    assert not w.tilt_run_btn.isEnabled() and w.tilt_table.rowCount() == 0
    w.add_tilt_views([paths[1]], tilts=[5.0])
    assert w.tabs.currentIndex() == TAB_3D and w.tilt_table.rowCount() == 2 and w.tilt_run_btn.isEnabled()
    assert w.ms.tilt3d.views[0].file == "v1.tif"  # kept relative to the reference image
    w.tilt_table.cellWidget(0, 1).setValue(-5.0)  # the reference's tilt, typed in the table
    assert w.ms.tilt3d.reference_tilt == -5.0
    res = w.run_tilt3d(wait=True)
    assert res is not None and res.hm.valid.mean() > 0.9 and res.hm.pixel_size == pytest.approx(2e-9)
    assert w.tilt_show.currentData() == "height" and "高度范围" in w.tilt_info.text()
    # a step (two terraces 20 px = 40 nm apart) and a height profile
    rows, cols = z.shape
    w.add_step_height([5, 20, 57, rows - 20], [288, 20, cols - 5, rows - 20])
    w.add_height_profile([10, rows / 2], [cols - 10, rows / 2])
    st = w.tilt_result.steps[0]
    assert st["step"] == pytest.approx(40e-9, rel=0.05) and "台阶 1" in w.tilt_info.text() and w.height_plot.isVisibleTo(w)
    assert w.tilt_result.profiles[0]["roughness"]["Ra"] > 0 and "Ra" in w.tilt_info.text()
    w.tilt_cutoff.setValue(100.0)  # λc 100 nm
    assert w.ms.tilt3d.cutoff == pytest.approx(100e-9) and w.tilt_result.profiles[0]["roughness"]["cutoff"] == pytest.approx(100e-9)
    # the drawing tools: a profile by two clicks, a step by two boxes
    from qtpy.QtCore import QPointF

    w.set_tool("hprofile")
    w.view._add_point(QPointF(20, 30))
    w.view._add_point(QPointF(300, 30))
    assert len(w.ms.tilt3d.profiles) == 2
    w._start_step()
    for x, y in [(5, 20), (57, 200), (288, 20), (370, 200)]:
        w.view._add_point(QPointF(x, y))
    assert len(w.ms.tilt3d.steps) == 2 and w.tilt_result.steps[1]["step"] == pytest.approx(40e-9, rel=0.05)
    # display modes and exports
    w.tilt_show.setCurrentIndex(w.tilt_show.findData("confidence"))
    w.tilt_show.setCurrentIndex(w.tilt_show.findData("image"))
    for name in ("h.tif", "h.stl", "h.obj", "h.png", "h.csv"):
        assert w.export_tilt(tmp_path / name).is_file()
    assert "三维" in __import__("openpyxl").load_workbook(w.export(tmp_path / "m.xlsx")[0]).sheetnames
    w._show_3d_preview()
    w._preview_dialog.close()
    # changed settings are flagged until rebuilt; undo restores
    w.tilt_table.cellWidget(1, 1).setValue(6.0)
    assert "已改变" in w.tilt_info.text()
    w.undo()
    assert w.ms.tilt3d.views[0].tilt == 5.0
    _dispose(w)
    # the record keeps the reconstruction's recipe; reopening rebuilds it on request
    w2 = SemWindow()
    w2.open_path(paths[0], wait=True)
    assert w2.ms.tilt3d is not None and len(w2.ms.tilt3d.steps) == 2 and w2.tilt_result is None
    assert w2.run_tilt3d(wait=True).steps[0]["step"] == pytest.approx(40e-9, rel=0.05)
    _dispose(w2)


def test_sem_window_in_english(tmp_path):
    """Every visible text of the SEM window, used on every tab, is English (none left in Chinese)."""
    import subprocess
    import sys

    helper = Path(__file__).parent / "helpers" / "sem_english_ui.py"
    r = subprocess.run([sys.executable, "-W", "ignore", str(helper), str(tmp_path)], capture_output=True, text=True, timeout=900)
    left = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert r.returncode == 0, r.stderr[-2000:]
    assert not left, "\n".join(left[:30])


def test_language_setting(tmp_path, monkeypatch):
    from vexel import i18n

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path))
    monkeypatch.delenv("VEXEL_LANG", raising=False)
    i18n.language.cache_clear()
    assert i18n.language() == "zh" and i18n.tr("标定") == "标定"
    i18n.set_language("en")
    assert i18n.language() == "en"
    assert i18n.tr("标定") == "Calibration"
    assert i18n.tr("找到的完整线条少于 2 条") == "Fewer than 2 complete lines found"
    assert i18n.tr("图像识别为“线条与间隔”（把握 0.97）") == 'Image recognised as "lines and spaces" (confidence 0.97)'
    assert i18n.tr("2.5 nm/像素（仪器元数据，已确认）") == "2.5 nm/px (instrument metadata, confirmed)"
    assert i18n.tr("Plain English") == "Plain English"
    with pytest.raises(ValueError):
        i18n.set_language("fr")
    i18n.set_language("zh")
    assert i18n.tr("标定") == "标定"
