"""Vexel main window: choose data, check calibration, run, read the results.

The window is a thin layer over the engine: it builds the same recipe the CLI would
(one of four built-in droplet recipes, chosen by the cell-counting and fluorescence
options, plus parameter and channel overrides) and calls ``run_recipe`` in a worker thread. With default settings a UI run
and ``vexel run -r <that recipe>`` produce the same recipe digest.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QSettings, Qt, QThread, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vexel.engine import Recipe
from vexel.readers import open_volume
from vexel.ui.widgets import Card, CountChart, Histogram, ReviewViewer, ScatterChart, heading, muted
from vexel.ui.worker import RunWorker

RECIPE = "droplet_basic"
RECIPE_CELLS = "droplet_encapsulation"
RECIPE_FLUO = "droplet_fluorescence"
RECIPE_FLUO_CELLS = "droplet_fluorescence_encapsulation"
RECIPES = (RECIPE, RECIPE_CELLS, RECIPE_FLUO, RECIPE_FLUO_CELLS)
PREVIEW_FRAMES = 200
_TO_UM = {"um": 1.0, "nm": 1e-3, "mm": 1e3}
_UNIT_LABEL = {"um": "µm", "px": "px"}

# key, label, min, max, step, decimals, suffix, display scale, tooltip
ADVANCED: list[tuple[str, str, float, float, float, int, str, float, str]] = [
    ("seg.min_threshold", "最低对比度", 0, 255, 1, 1, "", 1, "扣除背景后，低于该灰度差的区域不算液滴。漏检时调低，空白处误检时调高"),
    ("seg.smooth_sigma", "平滑", 0, 5, 0.1, 1, " px", 1, "分割前的高斯平滑。轮廓毛糙时调高，粒径偏大时调低"),
    ("seg.min_area", "最小面积", 0, 100000, 10, 0, " px²", 1, "小于该面积的目标被丢弃"),
    ("seg.min_distance", "粘连分离距离", 1, 1000, 1, 0, " px", 1, "相邻液滴中心的最小距离。一个液滴被切成几块时调高"),
    ("track.max_distance_px", "最大帧间位移", 1, 10000, 1, 1, " px", 1, "同一液滴相邻两帧的最大移动距离。同一液滴被数成多个时调高"),
    ("droplets.count_line", "计数线位置", 5, 95, 5, 0, " %", 100, "沿流动方向的计数线位置；液滴越过此线时计数一次"),
    ("inclusions.cell_diameter_px", "细胞直径", 1, 500, 0.5, 1, " px", 1, "单个细胞的典型直径。两个相邻细胞被数成一个时调小"),
    ("inclusions.min_contrast", "细胞最低对比度", 0, 65535, 1, 1, "", 1, "细胞与液滴内部的最小灰度差。“自动”按每帧估计的噪声设定；漏数时调低，空液滴被数出细胞时调高"),
    ("positivity.threshold", "阳性阈值", 0, 1e9, 1, 1, "", 1, "液滴荧光（扣除背景后的平均值）高于此值判为阳性。“自动”在能分出阴阳两群时自动定阈值；用阴性对照确定阈值后可在此固定"),
]
AUTO_KEYS = {"inclusions.min_contrast", "positivity.threshold"}  # 0 on the spin box means "automatic"


def recipe_defaults(name: str = RECIPE) -> dict[str, Any]:
    """Every overridable setting of a built-in recipe: step parameters and channel numbers."""
    resolved = Recipe.load(name).resolve()
    out: dict[str, Any] = {f"{s.id}.{k}": v for s in resolved.steps for k, v in s.params.items()}
    out["input.channel"] = resolved.input.channel
    out.update({f"input.channels.{k}": v for k, v in resolved.input.channels.items()})
    return out


def _parse_positive(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text:
        return None
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(text)
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float) and not math.isfinite(value):
        return "–"
    return f"{value:.{digits}g}"


def _number_edit(placeholder: str) -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setClearButtonEnabled(True)
    return edit


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vexel · 液滴分析")
        self.resize(1120, 760)
        self.setAcceptDrops(True)
        self.settings = QSettings()
        self.recipe_defaults = {name: recipe_defaults(name) for name in RECIPES}
        self.defaults = self.recipe_defaults[RECIPE_FLUO_CELLS]  # superset: used to initialise the controls
        self.path: Path | None = None
        self.run_dir: Path | None = None
        self._thread: QThread | None = None
        self._worker: RunWorker | None = None
        self._preview = False
        self._preview_dirs: list[Path] = []

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_controls())
        layout.addWidget(self._build_results(), 1)
        self.setCentralWidget(central)
        self._set_ready(False)

    # -- layout -------------------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(340)
        v = QVBoxLayout(panel)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(8)

        v.addWidget(heading("1  数据"))
        row = QHBoxLayout()
        self.open_btn = QPushButton("选择文件…")
        self.folder_btn = QPushButton("图像序列文件夹…")
        self.open_btn.clicked.connect(self._choose_file)
        self.folder_btn.clicked.connect(self._choose_folder)
        row.addWidget(self.open_btn)
        row.addWidget(self.folder_btn)
        v.addLayout(row)
        self.file_label = QLabel("未选择数据")
        self.file_label.setObjectName("fileName")
        self.file_label.setWordWrap(True)
        self.info_label = muted("也可以直接把文件拖进窗口")
        v.addWidget(self.file_label)
        v.addWidget(self.info_label)

        v.addSpacing(10)
        v.addWidget(heading("2  设置"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.pixel_edit = _number_edit("例如 2.0")
        self.interval_edit = _number_edit("例如 1.0")
        self.flow_combo = QComboBox()
        self.flow_combo.addItem("水平（左右流动）", "x")
        self.flow_combo.addItem("竖直（上下流动）", "y")
        form.addRow("像素尺寸 (µm)", self.pixel_edit)
        form.addRow("帧间隔 (ms)", self.interval_edit)
        form.addRow("流动方向", self.flow_combo)
        self.channel_combo = QComboBox()
        self.channel_combo.setToolTip("在哪个通道里分割液滴（液滴轮廓最清楚的通道，通常是明场）")
        form.addRow("液滴通道", self.channel_combo)
        self.cells_check = QCheckBox("统计每个液滴里的细胞数")
        self.cells_check.setToolTip("包裹计数：空滴、单包裹、多包裹比例，并与泊松分布比较")
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItem("比液滴内部暗（明场）", "dark")
        self.polarity_combo.addItem("比液滴内部亮（荧光）", "bright")
        self.cells_channel_combo = QComboBox()
        form.addRow("包裹计数", self.cells_check)
        form.addRow("细胞所在通道", self.cells_channel_combo)
        form.addRow("细胞外观", self.polarity_combo)
        self.fluo_check = QCheckBox("判读阴阳性（荧光）")
        self.fluo_check.setToolTip("测每个液滴的荧光强度，判为阳性或阴性，给出阳性比例与每滴平均拷贝数")
        self.fluo_channel_combo = QComboBox()
        form.addRow("荧光", self.fluo_check)
        form.addRow("荧光通道", self.fluo_channel_combo)
        self.form = form
        self.cells_check.toggled.connect(self._toggle_cells)
        self.fluo_check.toggled.connect(self._toggle_cells)
        v.addLayout(form)
        self.calib_note = muted("")
        v.addWidget(self.calib_note)

        v.addSpacing(6)
        self.adv_toggle = QToolButton()
        self.adv_toggle.setText("高级设置")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_toggle.setAutoRaise(True)
        self.adv_toggle.toggled.connect(self._toggle_advanced)
        v.addWidget(self.adv_toggle)
        self.adv_box = QWidget()
        adv = QFormLayout(self.adv_box)
        adv.setContentsMargins(0, 0, 0, 0)
        self.spins: dict[str, tuple[QDoubleSpinBox, float]] = {}
        self.spin_labels: dict[str, QLabel] = {}
        for key, label, lo, hi, step, decimals, suffix, scale, tip in ADVANCED:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setSuffix(suffix)
            spin.setToolTip(tip)
            if key in AUTO_KEYS:
                spin.setSpecialValueText("自动")
            name = QLabel(label)
            name.setToolTip(tip)
            adv.addRow(name, spin)
            self.spins[key] = (spin, scale)
            self.spin_labels[key] = name
        reset = QPushButton("恢复默认")
        reset.setFlat(True)
        reset.clicked.connect(self.reset_advanced)
        adv.addRow("", reset)
        self.adv_box.setVisible(False)
        v.addWidget(self.adv_box)
        self.reset_advanced()
        self._fill_channels(1, None)
        self._toggle_cells(False)

        v.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setVisible(False)
        self.status = muted("")
        v.addWidget(self.progress)
        v.addWidget(self.status)
        v.addWidget(heading("3  分析"))
        row = QHBoxLayout()
        self.preview_btn = QPushButton("快速预览")
        self.preview_btn.setToolTip(f"只分析前 {PREVIEW_FRAMES} 帧，用来检查设置是否合适；结果不保存")
        self.run_btn = QPushButton("开始分析")
        self.run_btn.setObjectName("primary")
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setVisible(False)
        self.preview_btn.clicked.connect(lambda: self.start(preview=True))
        self.run_btn.clicked.connect(lambda: self.start(preview=False))
        self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.preview_btn)
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.cancel_btn, 1)
        v.addLayout(row)
        return panel

    def _build_results(self) -> QWidget:
        self.stack = QStackedWidget()

        empty = QWidget()
        ev = QVBoxLayout(empty)
        ev.addStretch(1)
        title = self.empty_title = QLabel("把液滴视频或图像序列拖到这里")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = self.empty_sub = muted("支持 TIFF / OME-TIFF、AVI / MP4 / MOV、图像序列文件夹、OME-Zarr")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        old = QPushButton("打开已有结果…")
        old.setFlat(True)
        old.clicked.connect(self._choose_results)
        ev.addWidget(title)
        ev.addWidget(sub)
        ev.addSpacing(8)
        ev.addWidget(old, 0, Qt.AlignmentFlag.AlignHCenter)
        ev.addStretch(1)
        self.stack.addWidget(empty)

        page = QWidget()
        pv = QVBoxLayout(page)
        pv.setContentsMargins(24, 20, 24, 20)
        pv.setSpacing(12)
        top = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.result_title = QLabel("分析结果")
        self.result_title.setObjectName("title")
        self.result_path = muted("")
        self.result_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        titles.addWidget(self.result_title)
        titles.addWidget(self.result_path)
        top.addLayout(titles, 1)
        self.open_dir_btn = QPushButton("打开结果文件夹")
        self.open_dir_btn.clicked.connect(self._open_run_dir)
        top.addWidget(self.open_dir_btn, 0, Qt.AlignmentFlag.AlignTop)
        pv.addLayout(top)

        self.notes = QLabel()
        self.notes.setObjectName("notes")
        self.notes.setWordWrap(True)
        self.notes.setVisible(False)
        pv.addWidget(self.notes)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.card_n = Card("液滴数")
        self.card_d = Card("平均粒径")
        self.card_cv = Card("粒径 CV")
        self.card_f = Card("生成频率")
        for c in (self.card_n, self.card_d, self.card_cv, self.card_f):
            cards.addWidget(c)
        pv.addLayout(cards)

        self.enc_cards = QWidget()
        enc = QHBoxLayout(self.enc_cards)
        enc.setContentsMargins(0, 0, 0, 0)
        enc.setSpacing(12)
        self.card_lambda = Card("平均每滴细胞数 λ")
        self.card_empty = Card("空滴")
        self.card_single = Card("单包裹")
        self.card_multi = Card("多包裹")
        for c in (self.card_lambda, self.card_empty, self.card_single, self.card_multi):
            enc.addWidget(c)
        self.enc_cards.setVisible(False)
        pv.addWidget(self.enc_cards)

        self.fluo_cards = QWidget()
        fl = QHBoxLayout(self.fluo_cards)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(12)
        self.card_pos = Card("阳性液滴")
        self.card_pos_n = Card("阳性 / 总数")
        self.card_copies = Card("每滴平均拷贝数")
        self.card_sep = Card("阴阳分离度 D")
        for c in (self.card_pos, self.card_pos_n, self.card_copies, self.card_sep):
            fl.addWidget(c)
        self.fluo_cards.setVisible(False)
        pv.addWidget(self.fluo_cards)

        pv.addWidget(heading("复核图"))
        self.viewer = ReviewViewer()
        pv.addWidget(self.viewer, 3)
        charts = QHBoxLayout()
        charts.setSpacing(24)
        col = QVBoxLayout()
        col.addWidget(heading("粒径分布"))
        self.hist = Histogram()
        col.addWidget(self.hist, 1)
        charts.addLayout(col, 1)
        self.count_box = QWidget()
        col = QVBoxLayout(self.count_box)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(heading("包裹数分布"))
        self.count_chart = CountChart()
        col.addWidget(self.count_chart, 1)
        self.count_box.setVisible(False)
        charts.addWidget(self.count_box, 1)
        self.scatter_box = QWidget()
        col = QVBoxLayout(self.scatter_box)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(heading("粒径与荧光"))
        self.scatter = ScatterChart()
        col.addWidget(self.scatter, 1)
        self.scatter_box.setVisible(False)
        charts.addWidget(self.scatter_box, 1)
        pv.addLayout(charts, 2)
        self.stack.addWidget(page)
        return self.stack

    # -- data and settings ------------------------------------------------------------------

    def open_path(self, path: str | Path) -> bool:
        if self.is_running():
            return False
        path = Path(path)
        try:
            volume = open_volume(path)
        except Exception as e:
            self.warn("无法打开", f"{path.name}\n\n{e}")
            return False
        self.path = path
        self.settings.setValue("last_dir", str(path.parent))
        sizes = volume.sizes
        channels = f" · {sizes['C']} 通道" if sizes["C"] > 1 else ""
        self.file_label.setText(path.name)
        self.file_label.setToolTip(str(path))
        self.info_label.setText(f"{sizes['T']} 帧 · {sizes['X']}×{sizes['Y']} 像素 · {volume.dtype}{channels}")

        self._fill_channels(sizes["C"], volume.channel_names)
        has_px = volume.space_unit in _TO_UM
        has_dt = volume.time_unit == "s"
        self.pixel_edit.setText(f"{volume.spacing['x'] * _TO_UM[volume.space_unit]:.6g}" if has_px else "")
        self.interval_edit.setText(f"{volume.spacing['t'] * 1000:.6g}" if has_dt else "")
        notes = []
        if has_px or has_dt:
            notes.append("已从文件读取标定，请核对。")
        if volume.metadata.get("reader") == "video":
            notes.append("视频里记录的通常是回放帧率，不是相机采集帧率，请填写实际的帧间隔。")
        if not has_px:
            notes.append("不填像素尺寸时，粒径以像素为单位。")
        self.calib_note.setText("".join(notes))
        self.status.setText("")
        self.empty_title.setText(f"已载入 {path.name}")
        self.empty_sub.setText("核对左侧标定后，点“快速预览”检查设置，或直接“开始分析”。")
        self._set_ready(True)
        return True

    def recipe_name(self) -> str:
        cells, fluo = self.cells_check.isChecked(), self.fluo_check.isChecked()
        return {(False, False): RECIPE, (True, False): RECIPE_CELLS, (False, True): RECIPE_FLUO, (True, True): RECIPE_FLUO_CELLS}[
            (cells, fluo)
        ]

    def _fill_channels(self, n: int, names: list[str] | None) -> None:
        """Offer channels 1..n (shown 1-based; recipes count from 0)."""
        for combo, default in (
            (self.channel_combo, self.defaults["input.channel"]),
            (self.cells_channel_combo, self.defaults["input.channels.cells"]),
            (self.fluo_channel_combo, self.defaults["input.channels.fluo"]),
        ):
            combo.clear()
            for i in range(n):
                name = names[i] if names and i < len(names) and names[i] else ""
                combo.addItem(f"通道 {i + 1}" + (f"（{name}）" if name else ""), i)
            combo.setCurrentIndex(min(default, n - 1))
        self.n_channels = n
        self._toggle_cells()

    def overrides(self) -> dict[str, Any]:
        """Recipe overrides for every setting that differs from the selected built-in recipe."""
        defaults = self.recipe_defaults[self.recipe_name()]
        out: dict[str, Any] = {}
        for key, (spin, scale) in self.spins.items():
            if key not in defaults:
                continue
            default = defaults[key]
            value: Any = spin.value() / scale
            if key in AUTO_KEYS and spin.value() == spin.minimum():
                value = None
            elif isinstance(default, int) and not isinstance(default, bool):
                value = int(round(value))
            if value is None or default is None:
                if value != default:
                    out[key] = value
            elif abs(value - default) > 1e-9:
                out[key] = value
        if out.get("positivity.threshold") is not None:  # a fixed threshold switches the method
            out["positivity.method"] = "fixed"
        for key, combo in (
            ("input.channel", self.channel_combo),
            ("input.channels.cells", self.cells_channel_combo),
            ("input.channels.fluo", self.fluo_channel_combo),
        ):
            if key in defaults and combo.currentData() is not None and combo.currentData() != defaults[key]:
                out[key] = combo.currentData()
        flow = self.flow_combo.currentData()
        if flow != defaults["droplets.flow_axis"]:
            out["droplets.flow_axis"] = flow
        if "inclusions.polarity" in defaults and self.polarity_combo.currentData() != defaults["inclusions.polarity"]:
            out["inclusions.polarity"] = self.polarity_combo.currentData()
        return out

    def reset_advanced(self) -> None:
        for key, (spin, scale) in self.spins.items():
            default = self.defaults[key]
            spin.setValue(spin.minimum() if default is None else default * scale)

    def _toggle_cells(self, _checked: bool = False) -> None:
        cells, fluo = self.cells_check.isChecked(), self.fluo_check.isChecked()
        multi = getattr(self, "n_channels", 1) > 1
        self.form.setRowVisible(self.channel_combo, multi)
        self.form.setRowVisible(self.polarity_combo, cells)
        self.form.setRowVisible(self.cells_channel_combo, cells and multi)
        self.form.setRowVisible(self.fluo_channel_combo, fluo and multi)
        for key, (spin, _) in self.spins.items():
            visible = (cells or not key.startswith("inclusions.")) and (fluo or not key.startswith("positivity."))
            spin.setVisible(visible)
            self.spin_labels[key].setVisible(visible)

    def _toggle_advanced(self, on: bool) -> None:
        self.adv_box.setVisible(on)
        self.adv_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def _default_out_dir(self) -> Path:
        assert self.path is not None
        stem = self.path.name.split(".")[0] or "run"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = self.path.parent if os.access(self.path.parent, os.W_OK) else Path.home() / "Vexel"
        return base / f"{stem}_vexel" / stamp

    # -- running ----------------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None

    def start(self, preview: bool = False) -> bool:
        if self.path is None or self.is_running():
            return False
        try:
            pixel = _parse_positive(self.pixel_edit.text())
            interval = _parse_positive(self.interval_edit.text())
        except ValueError:
            self.warn("标定有误", "像素尺寸和帧间隔必须是正数，或留空。")
            return False
        recipe = Recipe.load(self.recipe_name()).with_overrides(self.overrides())
        try:
            recipe.resolve()
        except Exception as e:
            self.warn("设置有误", str(e))
            return False
        if preview:
            out_dir = Path(tempfile.mkdtemp(prefix="vexel-preview-"))
            self._preview_dirs.append(out_dir)
        else:
            out_dir = self._default_out_dir()
        self._preview = preview
        self._worker = RunWorker(
            recipe,
            self.path,
            pixel,
            interval / 1000.0 if interval else None,
            out_dir,
            max_frames=PREVIEW_FRAMES if preview else None,
            hash_input=not preview,
        )
        self._thread = QThread()  # owned by Python; released in _stop_thread
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._set_running(True)
        self.status.setText("准备中…")
        self.empty_title.setText("正在预览…" if preview else "正在分析…")
        self.empty_sub.setText("可以随时取消。")
        self._thread.start()
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("正在取消…")
            self.cancel_btn.setEnabled(False)

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        # The worker's thread has finished, so both objects are released from here.
        self._thread = None
        self._worker = None
        self._set_running(False)

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        self.status.setText(message)

    def _on_finished(self, out_dir: str) -> None:
        self._stop_thread()
        self.status.setText("预览完成。" if self._preview else "分析完成，结果已保存。")
        self.load_results(Path(out_dir), preview=self._preview)

    def _on_failed(self, message: str, out_dir: str) -> None:
        self._stop_thread()
        self.status.setText("分析未完成。")
        if out_dir:
            self.load_results(Path(out_dir), preview=self._preview)
        self.warn("分析未完成", message)

    def _on_cancelled(self) -> None:
        self._stop_thread()
        self.status.setText("已取消。")
        self.empty_title.setText("已取消")
        self.empty_sub.setText("调整设置后可以重新开始。")

    def _set_running(self, running: bool) -> None:
        for w in (
            self.open_btn, self.folder_btn, self.pixel_edit, self.interval_edit, self.flow_combo, self.channel_combo,
            self.cells_check, self.cells_channel_combo, self.polarity_combo, self.fluo_check, self.fluo_channel_combo,
            self.adv_box,
        ):  # fmt: skip
            w.setEnabled(not running)
        self.preview_btn.setVisible(not running)
        self.run_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.cancel_btn.setEnabled(running)
        self.progress.setVisible(running)
        self.progress.setValue(0)

    def _set_ready(self, ready: bool) -> None:
        self.preview_btn.setEnabled(ready)
        self.run_btn.setEnabled(ready)

    # -- results ------------------------------------------------------------------------------

    def load_results(self, run_dir: Path, preview: bool = False) -> bool:
        run_dir = Path(run_dir)
        try:
            summaries = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            self.warn("无法读取结果", f"{run_dir}\n\n{e}")
            return False
        s = next((v for v in summaries.values() if isinstance(v, dict) and "count_line_px" in v), {})
        enc = next((v for v in summaries.values() if isinstance(v, dict) and "frac_empty" in v), None)
        fluo = next((v for v in summaries.values() if isinstance(v, dict) and "positive_fraction" in v), None)
        unit = _UNIT_LABEL.get(s.get("length_unit", "px"), s.get("length_unit", "px"))
        freq_unit = s.get("frequency_unit", "")
        freq_unit = "次/帧" if freq_unit == "1/frame" else freq_unit

        n = s.get("n_droplets")
        self.card_n.set("–" if n is None else str(n), f"共追踪到 {s['n_tracks_total']} 条轨迹" if "n_tracks_total" in s else "")
        self.card_d.set(f"{_fmt(s.get('diameter_mean'))} {unit}", f"标准差 {_fmt(s.get('diameter_std'), 3)} {unit}")
        cv = s.get("diameter_cv_percent")
        self.card_cv.set("–" if cv is None else f"{cv:.2f} %", f"范围 {_fmt(s.get('diameter_min'))} – {_fmt(s.get('diameter_max'))} {unit}")
        self.card_f.set(f"{_fmt(s.get('frequency'))} {freq_unit}", f"平均间距 {_fmt(s.get('spacing_mean'))} {unit}")

        diameters = np.empty(0)
        table = run_dir / "tables" / "droplets.droplets.csv"
        if table.exists():
            diameters = pd.read_csv(table)["diameter"].to_numpy(dtype=np.float64)
        self.hist.set_data(diameters, unit)
        self.viewer.set_images(sorted((run_dir / "review").glob("*.png")))
        self.viewer.legend = "红线为液滴轮廓，黄圈为检出的细胞" if enc is not None else "红线为分割轮廓"
        self.viewer.show_index(0)
        self._show_encapsulation(run_dir, enc)
        self._show_fluorescence(run_dir, fluo, unit)

        notes = []
        if preview:
            notes.append(f"这是预览：只分析了前 {provenance['input'].get('frames_used')} 帧，结果不代表全部数据，也不会保存。")
        if provenance.get("status") == "stopped_by_qc":
            notes.append("运行被质控规则中止，结果不完整。")
        if s and not s.get("calibrated", True):
            notes.append("未标定：长度单位为像素。")
        if n == 0:
            notes.append("没有检测到液滴。先看复核图，再到“高级设置”里调整最低对比度或最小面积。")
        if enc is not None and enc.get("n_droplets"):
            if enc.get("chi2_p") is not None and enc["chi2_p"] < 0.05:
                d = enc.get("dispersion_index")
                trend = "，D > 1 提示细胞聚集或上样不均" if d and d > 1 else ("，D < 1 提示排布比随机更规则" if d else "")
                notes.append(f"包裹分布与泊松分布差异显著（p = {enc['chi2_p']:.3g}，离散指数 D = {_fmt(d, 3)}{trend}）。")
            if enc.get("n_low_agreement"):
                notes.append(
                    f"有 {enc['n_low_agreement']} 个液滴的细胞数在各帧之间不一致，计数可能不准，"
                    "请在复核图中检查黄圈，必要时调整“细胞直径”或“细胞最低对比度”。"
                )
            if enc.get("n_excluded"):
                notes.append(f"有 {enc['n_excluded']} 个液滴无法计数（内部区域太小），未计入包裹统计。")
        if fluo is not None and fluo.get("n_droplets"):
            if fluo.get("populations") == 1:
                notes.append(
                    f"荧光只发现一群液滴（未见明显的阴阳两群），已按离群值规则判读（阈值 {_fmt(fluo.get('threshold'))}）。"
                    "若样本应含阳性，或全部液滴都是阳性，请用对照样本确定阈值，在“高级设置 › 阳性阈值”中填写。"
                )
            elif fluo.get("saturated"):
                notes.append("所有液滴均为阳性，无法估计每滴拷贝数（已饱和），建议稀释后重测。")
            d = fluo.get("separation_d")
            if fluo.get("threshold_method") == "mixture" and d is not None and d < 4:
                notes.append(f"阴阳两群有重叠（分离度 D = {d:.2f}），靠近阈值的液滴可能判错，请对照散点图确认。")
        for step in provenance.get("steps", []):
            for v in step.get("qc_violations", []):
                bounds = "，".join(f"{k} {v[k]:g}" for k in ("min", "max") if v.get(k) is not None)
                value = "缺失" if v.get("value") is None else f"{v['value']:.4g}"
                notes.append(f"质控提示：{step['id']}.{v['metric']} = {value}（要求 {bounds}），请对照复核图确认。")
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))

        self.result_title.setText("预览结果" if preview else "分析结果")
        self.result_path.setText("临时目录，关闭程序后删除" if preview else str(run_dir))
        self.run_dir = run_dir
        self.stack.setCurrentIndex(1)
        return True

    def _show_encapsulation(self, run_dir: Path, enc: dict | None) -> None:
        self.enc_cards.setVisible(enc is not None)
        self.count_box.setVisible(enc is not None)
        if enc is None:
            return

        def share(key: str) -> str:
            return "–" if enc.get(key) is None else f"{100 * enc[key]:.1f} %"

        lo, hi = enc.get("lambda_ci_low"), enc.get("lambda_ci_high")
        self.card_lambda.set(_fmt(enc.get("lambda"), 3), f"95% 区间 {_fmt(lo, 3)} – {_fmt(hi, 3)}" if lo is not None else "")
        self.card_empty.set(share("frac_empty"), f"泊松预期 {share('poisson_empty')}")
        self.card_single.set(share("frac_single"), f"泊松预期 {share('poisson_single')}")
        self.card_multi.set(share("frac_multi"), f"泊松预期 {share('poisson_multi')}")
        dist = None
        for path in sorted((run_dir / "tables").glob("*.csv")):
            header = path.open(encoding="utf-8").readline().strip().split(",")
            if {"k", "observed_fraction", "poisson_fraction"} <= set(header):
                dist = pd.read_csv(path)
                break
        if dist is None or dist.empty:
            self.count_chart.set_data(np.empty(0), np.empty(0), np.empty(0))
        else:
            self.count_chart.set_data(dist["k"].to_numpy(), dist["observed_fraction"].to_numpy(), dist["poisson_fraction"].to_numpy())

    def _show_fluorescence(self, run_dir: Path, fluo: dict | None, unit: str) -> None:
        self.fluo_cards.setVisible(fluo is not None)
        self.scatter_box.setVisible(fluo is not None)
        if fluo is None:
            return
        p, lo, hi = fluo.get("positive_fraction"), fluo.get("positive_fraction_ci_low"), fluo.get("positive_fraction_ci_high")
        self.card_pos.set("–" if p is None else f"{100 * p:.1f} %", "" if lo is None else f"95% 区间 {100 * lo:.1f} – {100 * hi:.1f} %")
        self.card_pos_n.set(f"{fluo.get('n_positive', 0)} / {fluo.get('n_droplets', 0)}", f"阈值 {_fmt(fluo.get('threshold'))}")
        c, clo, chi = fluo.get("copies_per_droplet"), fluo.get("copies_ci_low"), fluo.get("copies_ci_high")
        self.card_copies.set(_fmt(c, 3), f"95% 区间 {_fmt(clo, 3)} – {_fmt(chi, 3)}" if c is not None else "已饱和或无数据")
        method = {"mixture": "自动分群", "outlier": "离群值规则", "fixed": "固定阈值", "otsu": "大津法"}.get(
            fluo.get("threshold_method"), fluo.get("threshold_method") or ""
        )
        d = fluo.get("separation_d")
        self.card_sep.set("–" if d is None else ("∞" if not math.isfinite(d) else f"{d:.1f}"), method)
        table = None
        for path in sorted((run_dir / "tables").glob("*.csv")):
            header = path.open(encoding="utf-8").readline().strip().split(",")
            if {"positive", "diameter", fluo.get("column", "fluo_mean")} <= set(header):
                table = pd.read_csv(path)
                break
        if table is None or table.empty:
            self.scatter.set_data(np.empty(0), np.empty(0), np.empty(0, dtype=bool), None, unit)
        else:
            positive = table["positive"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
            self.scatter.set_data(table["diameter"].to_numpy(), table[fluo.get("column", "fluo_mean")].to_numpy(), positive, fluo.get("threshold"), unit)

    # -- dialogs and system integration ------------------------------------------------------

    def warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _start_dir(self) -> str:
        return str(self.settings.value("last_dir", str(Path.home())))

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据",
            self._start_dir(),
            "图像与视频 (*.tif *.tiff *.avi *.mp4 *.mov *.mkv *.m4v *.npy);;所有文件 (*)",
        )
        if path:
            self.open_path(path)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图像序列文件夹或 OME-Zarr", self._start_dir())
        if path:
            self.open_path(path)

    def _choose_results(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果文件夹", self._start_dir())
        if path:
            if (Path(path) / "summary.json").exists():
                self.load_results(Path(path))
            else:
                self.warn("不是结果文件夹", "该文件夹里没有 summary.json。")

    def _open_run_dir(self) -> None:
        if self.run_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and not self.is_running():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_path(urls[0].toLocalFile())

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30000)
        for d in self._preview_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(event)
