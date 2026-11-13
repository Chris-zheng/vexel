"""Live dashboard: online monitoring of a replayed recording or a watched acquisition folder.

Separate from the analysis wizard (main window): it runs the stream executor in a worker
thread and shows the current frame with outlines, throughput and drops, the rolling
droplet statistics as numbers and charts, and alarms as they are raised.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QCoreApplication, QObject, QPointF, QRectF, Qt, QThread, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vexel.ui.widgets import ACCENT, MUTED, Card, heading, muted

RECIPES = [
    ("液滴", "droplet_basic"),
    ("液滴 + 包裹计数", "droplet_encapsulation"),
    ("液滴 + 荧光阴阳", "droplet_fluorescence"),
    ("液滴 + 荧光 + 包裹计数", "droplet_fluorescence_encapsulation"),
]
SPEEDS = [("实时 1×", 1.0), ("0.5×", 0.5), ("0.2×", 0.2), ("0.1×", 0.1), ("0.05×", 0.05), ("0.02×", 0.02), ("不限速", None)]
FRAME_EVERY_S = 0.25  # the frame view refreshes at most this often


class LineChart(QWidget):
    """A rolling statistic against frame number; dashed lines mark alarms."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.setMinimumHeight(130)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.unit = ""
        self.marks: list[float] = []

    def set_data(self, x: list[float], y: list[float], unit: str = "", marks: list[float] | None = None) -> None:
        x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x_arr) & np.isfinite(y_arr)
        self.x, self.y, self.unit, self.marks = x_arr[ok], y_arr[ok], unit, list(marks or [])
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.palette().windowText().color()
        grey = QColor(MUTED)
        p.setPen(text)
        p.drawText(QRectF(4, 0, self.width() - 8, 18), Qt.AlignmentFlag.AlignLeft, self.title + (f"（{self.unit}）" if self.unit else ""))
        area = QRectF(self.rect()).adjusted(46, 24, -8, -20)
        if self.x.size == 0:
            p.setPen(grey)
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "等待数据")
            return
        xlo, xhi = float(self.x.min()), float(self.x.max())
        if xhi <= xlo:
            xhi = xlo + 1
        ylo, yhi = float(self.y.min()), float(self.y.max())
        pad = (yhi - ylo) * 0.1 or abs(yhi) * 0.05 or 1.0
        ylo, yhi = ylo - pad, yhi + pad

        def at(a: float, b: float) -> QPointF:
            return QPointF(area.left() + (a - xlo) / (xhi - xlo) * area.width(), area.bottom() - (b - ylo) / (yhi - ylo) * area.height())

        p.setPen(QPen(grey, 1))
        p.drawLine(area.bottomLeft(), area.bottomRight())
        warn = QPen(QColor("#d97706"), 1, Qt.PenStyle.DashLine)
        for m in self.marks:
            if xlo <= m <= xhi:
                p.setPen(warn)
                p.drawLine(at(m, ylo), at(m, yhi))
        p.setPen(QPen(QColor(ACCENT), 2))
        pts = [at(a, b) for a, b in zip(self.x, self.y)]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        p.setBrush(QColor(ACCENT))
        for q in pts:
            p.drawEllipse(q, 2.5, 2.5)
        p.setPen(grey)
        p.drawText(QRectF(0, area.top() - 6, 42, 14), Qt.AlignmentFlag.AlignRight, f"{yhi:.4g}")
        p.drawText(QRectF(0, area.bottom() - 8, 42, 14), Qt.AlignmentFlag.AlignRight, f"{ylo:.4g}")
        p.drawText(QRectF(area.left(), area.bottom() + 2, area.width(), 16), Qt.AlignmentFlag.AlignLeft, f"第 {xlo:.0f} 帧")
        p.drawText(QRectF(area.left(), area.bottom() + 2, area.width(), 16), Qt.AlignmentFlag.AlignRight, f"第 {xhi:.0f} 帧")


def overlay_rgb(image: np.ndarray, labels: np.ndarray, marks: np.ndarray | None) -> np.ndarray:
    from vexel.engine.batch_executor import _overlay

    img = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(img, (1, 99))
    rgb = _overlay(img, np.asarray(labels), None if marks is None else np.asarray(marks), lo, hi)
    return np.ascontiguousarray((rgb * 255).round().astype(np.uint8))


class LiveWorker(QObject):
    update = Signal(object)  # rolling snapshot (dict)
    frame = Signal(object, int)  # RGB overlay (H, W, 3) uint8, frame index
    finished = Signal(str, object)  # run directory, final summary of a watched acquisition (or None)
    failed = Signal(str)

    def __init__(self, recipe: Any, source: Any, out_dir: Path, *, final_pass: bool = False) -> None:
        super().__init__()
        self.recipe, self.source, self.out_dir, self.final_pass = recipe, source, Path(out_dir), final_pass
        self._stop = False
        self._last_frame = 0.0

    def stop(self) -> None:
        self._stop = True
        self.source.request_stop()

    def _on_frame(self, index: int, image: np.ndarray, labels: np.ndarray, marks: np.ndarray | None) -> None:
        now = time.monotonic()
        if now - self._last_frame >= FRAME_EVERY_S:
            self._last_frame = now
            self.frame.emit(overlay_rgb(image, labels, marks), index)

    def run(self) -> None:
        from vexel.engine import run_recipe, run_stream

        final = None
        try:
            run_stream(self.recipe, self.source, self.out_dir, on_update=self.update.emit, on_frame=self._on_frame,
                       should_stop=lambda: self._stop)  # fmt: skip
            zarr_path = getattr(self.source, "zarr_path", None)
            if self.final_pass and zarr_path is not None and Path(zarr_path).exists() and not self._stop:
                from vexel.readers import open_volume

                final = run_recipe(self.recipe, open_volume(zarr_path), self.out_dir / "final", hash_input=False).summary
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._done(self.failed, f"{type(e).__name__}: {e}")
            return
        self._done(self.finished, str(self.out_dir), final)

    def _done(self, signal: Signal, *args: Any) -> None:
        app = QCoreApplication.instance()
        if app is not None:  # released in the UI thread, as RunWorker does
            self.moveToThread(app.thread())
        signal.emit(*args)


def _number(placeholder: str) -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    return edit


def _value(edit: QLineEdit) -> float | None:
    text = edit.text().strip()
    if not text:
        return None
    value = float(text)
    if value <= 0:
        raise ValueError("须为正数")
    return value


class LiveWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vexel · 在线监控")
        self.resize(1180, 820)
        self.path: Path | None = None
        self.mode: str | None = None  # "replay" or "watch"
        self.run_dir: Path | None = None
        self._thread: QThread | None = None
        self._worker: LiveWorker | None = None
        self.snapshots: list[dict[str, Any]] = []
        self.alarm_frames: list[float] = []
        self.warn = lambda title, text: QMessageBox.warning(self, title, text)
        self._stopping = False

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)
        outer.addLayout(self._build_controls())
        body = QHBoxLayout()
        body.setSpacing(16)
        self.frame_view = QLabel("开始后在这里显示当前帧")
        self.frame_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_view.setMinimumSize(420, 220)
        self.frame_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_caption = muted("")
        left = QVBoxLayout()
        left.addWidget(self.frame_view, 1)
        left.addWidget(self.frame_caption)
        body.addLayout(left, 3)
        cards = QGridLayout()
        cards.setSpacing(10)
        self.card_fps, self.card_drop = Card("处理速度"), Card("丢帧")
        self.card_n, self.card_d = Card("液滴数（窗口内）"), Card("平均粒径")
        self.card_f, self.card_extra = Card("生成频率"), Card("λ / 阳性比例")
        for i, c in enumerate((self.card_fps, self.card_drop, self.card_n, self.card_d, self.card_f, self.card_extra)):
            cards.addWidget(c, i // 2, i % 2)
        right = QVBoxLayout()
        right.addLayout(cards)
        right.addWidget(heading("报警"))
        self.alarm_list = QListWidget()
        self.alarm_list.setMinimumHeight(120)
        right.addWidget(self.alarm_list, 1)
        body.addLayout(right, 2)
        outer.addLayout(body, 1)
        charts = QHBoxLayout()
        self.chart_d, self.chart_f, self.chart_extra = LineChart("滚动平均粒径"), LineChart("滚动生成频率"), LineChart("滚动 λ")
        for c in (self.chart_d, self.chart_f, self.chart_extra):
            charts.addWidget(c)
        outer.addLayout(charts)
        bottom = QHBoxLayout()
        self.status = muted("选择录像或采集文件夹。")
        self.report_btn = QPushButton("打开报告")
        self.report_btn.setVisible(False)
        self.report_btn.clicked.connect(self._open_report)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.report_btn)
        outer.addLayout(bottom)
        self.setCentralWidget(central)
        self._set_running(False)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        col = QVBoxLayout()
        buttons = QHBoxLayout()
        self.file_btn = QPushButton("回放录像…")
        self.folder_btn = QPushButton("监视采集文件夹…")
        self.file_btn.clicked.connect(self._choose_file)
        self.folder_btn.clicked.connect(self._choose_folder)
        buttons.addWidget(self.file_btn)
        buttons.addWidget(self.folder_btn)
        col.addLayout(buttons)
        self.source_label = QLabel("未选择")
        self.source_label.setObjectName("fileName")
        self.source_label.setWordWrap(True)
        col.addWidget(self.source_label)
        row.addLayout(col, 2)

        form = QFormLayout()
        self.recipe_combo = QComboBox()
        for label, name in RECIPES:
            self.recipe_combo.addItem(label, name)
        self.speed_combo = QComboBox()
        for label, speed in SPEEDS:
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setToolTip("相对录像原始帧率的回放速度；跟不上时自动隔帧处理并记录丢帧")
        form.addRow("分析", self.recipe_combo)
        form.addRow("回放速度", self.speed_combo)
        row.addLayout(form, 2)
        form2 = QFormLayout()
        self.pixel_edit = _number("例如 2.0")
        self.interval_edit = _number("例如 1.0")
        self.idle_edit = _number("10")
        self.idle_edit.setToolTip("监视文件夹时，这么多秒没有新图像就结束（也可以在文件夹里放一个名为 STOP 的文件）")
        form2.addRow("像素尺寸 (µm)", self.pixel_edit)
        form2.addRow("帧间隔 (ms)", self.interval_edit)
        form2.addRow("无新图像结束 (s)", self.idle_edit)
        row.addLayout(form2, 2)
        actions = QVBoxLayout()
        self.start_btn = QPushButton("开始")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("停止")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        actions.addWidget(self.start_btn)
        actions.addWidget(self.stop_btn)
        row.addLayout(actions, 1)
        return row

    # -- source ------------------------------------------------------------------------

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择录像", "", "图像与视频 (*.tif *.tiff *.avi *.mp4 *.mov *.npy);;所有文件 (*)")
        if path:
            self.set_replay(path)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择采集软件写入图像的文件夹")
        if path:
            self.set_watch(path)

    def set_replay(self, path: str | Path) -> bool:
        from vexel.readers import open_volume

        try:
            volume = open_volume(path)
        except Exception as e:  # noqa: BLE001
            self.warn("无法读取", str(e))
            return False
        self.path, self.mode = Path(path), "replay"
        sizes = volume.sizes
        self.source_label.setText(f"回放：{self.path.name}  ·  {sizes['T']} 帧 {sizes['X']}×{sizes['Y']}")
        self.pixel_edit.setText(f"{volume.spacing['x']:g}" if volume.space_unit == "um" else "")
        self.interval_edit.setText(f"{volume.spacing['t'] * 1000:g}" if volume.time_unit == "s" else "")
        self.speed_combo.setEnabled(True)
        self.idle_edit.setEnabled(False)
        self._set_running(False)
        return True

    def set_watch(self, folder: str | Path) -> bool:
        if not Path(folder).is_dir():
            self.warn("不是文件夹", str(folder))
            return False
        self.path, self.mode = Path(folder), "watch"
        self.source_label.setText(f"监视：{self.path}")
        self.speed_combo.setEnabled(False)
        self.idle_edit.setEnabled(True)
        self._set_running(False)
        return True

    # -- run -----------------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None

    def _out_dir(self) -> Path:
        """Next to the data, as the analysis window does: ``<name>_vexel/live-<timestamp>``."""
        stem = self.path.name.split(".")[0] or "live"
        parent = self.path.parent if os.access(self.path.parent, os.W_OK) else Path.home() / "Vexel"
        return parent / f"{stem}_vexel" / f"live-{datetime.now():%Y%m%d-%H%M%S}"

    def start(self) -> bool:
        from vexel.engine import Recipe

        if self.path is None or self.is_running():
            return False
        try:
            pixel = _value(self.pixel_edit)
            interval_ms = _value(self.interval_edit)
            idle = _value(self.idle_edit) or 10.0
        except ValueError as e:
            self.warn("设置有误", f"像素尺寸、帧间隔和等待时间须为正数（{e}）")
            return False
        recipe = Recipe.load(self.recipe_combo.currentData())
        resolved = recipe.resolve()
        out = self._out_dir()
        try:
            if self.mode == "replay":
                from vexel.readers import open_volume
                from vexel.sources import ReplaySource
                from vexel.ui.worker import apply_calibration

                volume = open_volume(self.path)
                apply_calibration(volume, pixel, interval_ms / 1000.0 if interval_ms else None)
                speed = self.speed_combo.currentData()
                if speed is not None and interval_ms is None:
                    self.warn("需要帧间隔", "按帧率回放需要知道帧间隔；不知道时请选“不限速”。")
                    return False
                source = ReplaySource(volume, resolved.input.sources(), resolved.input.z, speed=speed)
                final = False
            else:
                from vexel.sources.watch import WatchSource

                source = WatchSource(self.path, resolved.input.sources(), pixel_size=pixel,
                                     frame_interval_s=interval_ms / 1000.0 if interval_ms else None, idle_timeout_s=idle,
                                     zarr_path=out / "frames.zarr")  # fmt: skip
                final = True
        except Exception as e:  # noqa: BLE001
            self.warn("无法开始", str(e))
            return False

        self.snapshots, self.alarm_frames = [], []
        self.alarm_list.clear()
        for chart in (self.chart_d, self.chart_f, self.chart_extra):
            chart.set_data([], [])
        self.chart_extra.title = "滚动阳性比例" if "fluorescence" in resolved.name else "滚动 λ"
        self.run_dir = out
        self._stopping = False
        self._worker = LiveWorker(recipe, source, out, final_pass=final)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.update.connect(self._on_update)
        self._worker.frame.connect(self._on_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._set_running(True)
        self.status.setText("正在建立背景（缓存开头若干帧）…" if self.mode == "replay" else "等待图像…")
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._worker is not None:
            self._stopping = True
            self._worker.stop()
            self.status.setText("正在停止…")
            self.stop_btn.setEnabled(False)

    def _release(self) -> None:
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        if thread is not None:
            thread.quit()
            thread.wait()
        for obj in (worker, thread):
            if obj is not None and obj.thread() is self.thread():
                obj.setParent(self)
                obj.deleteLater()
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        for w in (self.file_btn, self.folder_btn, self.recipe_combo, self.pixel_edit, self.interval_edit):
            w.setEnabled(not running)
        self.speed_combo.setEnabled(not running and self.mode == "replay")
        self.idle_edit.setEnabled(not running and self.mode == "watch")
        self.start_btn.setEnabled(not running and self.path is not None)
        self.stop_btn.setEnabled(running)
        if running:
            self.report_btn.setVisible(False)

    # -- updates -------------------------------------------------------------------------

    def _on_frame(self, rgb: np.ndarray, index: int) -> None:
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(image).scaled(self.frame_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)  # fmt: skip
        self.frame_view.setPixmap(pix)
        self.frame_caption.setText(f"第 {index} 帧（红线为液滴轮廓）")

    def _on_update(self, snap: dict[str, Any]) -> None:
        self._show_alarms(snap.pop("alarms", []))
        if snap.get("final") and self.snapshots and self.snapshots[-1]["frame"] == snap["frame"]:
            self.snapshots[-1] = snap
        else:
            self.snapshots.append(snap)
        fps, dropped, received = snap.get("fps"), snap.get("dropped", 0), snap.get("received", 0)
        self.card_fps.set(f"{fps:.1f} 帧/秒" if fps else "–", f"已处理 {snap.get('processed', 0)} 帧")
        lat = snap.get("latency_ms")
        share = f"{100 * dropped / received:.0f}%" if received else "0%"
        self.card_drop.set(str(dropped), f"占 {share}" + (f" · 延迟 {lat:.0f} ms" if lat is not None else ""))
        n = snap.get("droplets.n_droplets")
        self.card_n.set("–" if n is None else f"{n:.0f}", f"第 {snap.get('window_start', 0)}–{snap['frame']} 帧")
        unit = "µm" if self.pixel_edit.text().strip() else "px"
        d, cv = snap.get("droplets.diameter_mean"), snap.get("droplets.diameter_cv_percent")
        self.card_d.set("–" if d is None else f"{d:.4g} {unit}", "" if cv is None else f"CV {cv:.2f}%")
        f = snap.get("droplets.frequency")
        f_unit = "Hz" if self.interval_edit.text().strip() else "次/帧"
        self.card_f.set("–" if f is None else f"{f:.4g} {f_unit}")
        lam, pos = snap.get("encapsulation.lambda"), snap.get("positivity.positive_fraction")
        if pos is not None:
            self.card_extra.set(f"{100 * pos:.1f} %", "阳性比例" + (f" · λ {lam:.3g}" if lam is not None else ""))
        elif lam is not None:
            self.card_extra.set(f"{lam:.3g}", "平均每滴细胞数 λ")
        frames = [s["frame"] for s in self.snapshots]
        self.chart_d.set_data(frames, [s.get("droplets.diameter_mean", np.nan) for s in self.snapshots], unit, self.alarm_frames)
        self.chart_f.set_data(frames, [s.get("droplets.frequency", np.nan) for s in self.snapshots], f_unit, self.alarm_frames)
        key = "positivity.positive_fraction" if pos is not None else "encapsulation.lambda"
        self.chart_extra.set_data(frames, [s.get(key, np.nan) for s in self.snapshots], "", self.alarm_frames)
        self.status.setText(f"运行中：已收到 {received} 帧。")

    def _show_alarms(self, rows: list[dict[str, Any]]) -> None:
        for r in rows[self.alarm_list.count() :]:
            kind = {"drift": "漂移", "qc": "质控", "throughput": "处理能力"}.get(r["kind"], r["kind"])
            self.alarm_list.addItem(f"第 {int(r['frame'])} 帧 · {kind} · {r['message']}")
            self.alarm_frames.append(float(r["frame"]))

    def _on_finished(self, out_dir: str, final: Any) -> None:
        self._release()
        text = f"已停止，已处理部分的结果保存在 {out_dir}" if self._stopping else f"完成，结果保存在 {out_dir}"
        if final:
            online = self.snapshots[-1].get("droplets.n_droplets") if self.snapshots else None
            n = final.get("droplets.summary", {}).get("n_droplets")
            text += f"。采集结束后的全量分析：{n} 个液滴（在线 {online:.0f} 个）" if online is not None else ""
        self.status.setText(text + "。")
        self.report_btn.setVisible((Path(out_dir) / "report.html").exists())

    def _on_failed(self, message: str) -> None:
        self._release()
        self.status.setText("未完成。")
        self.warn("在线分析未完成", message)

    def _open_report(self) -> None:
        if self.run_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir / "report.html")))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    import sys

    from qtpy.QtWidgets import QApplication

    from vexel.ui.app import STYLE

    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Vexel")
    app.setStyleSheet(STYLE)
    window = LiveWindow()
    window.show()
    if argv:  # a folder is watched; a file (or an OME-Zarr store) is replayed
        target = Path(argv[0])
        if target.is_dir() and target.suffix != ".zarr":
            window.set_watch(target)
        else:
            window.set_replay(target)
    return int(app.exec())
