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
from vexel.report.notes import pick_summaries, run_notes
from vexel.ui.widgets import Card, CountChart, FlagStrip, Histogram, ReviewViewer, ScatterChart, heading, muted
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
    ("seg.min_diameter_px", "最小液滴直径", 0, 10000, 1, 0, " px", 1, "“自动”按每帧液滴的典型大小设定查找范围。小卫星液滴被当成液滴时调高"),
    ("seg.max_diameter_px", "最大液滴直径", 0, 10000, 1, 0, " px", 1, "“自动”按每帧液滴的典型大小设定查找范围。大液滴漏检时调高"),
    ("seg.min_score", "轮廓完整度", 0.05, 1, 0.05, 2, "", 1, "圆周上至少有这一比例落在边缘上才算液滴。漏检时调低，凭空多出圆时调高"),
    ("seg.motion_threshold", "最低对比度", 0, 255, 1, 1, "", 1, "扣除背景后低于该灰度差的地方不找液滴边缘（会按噪声自动抬高）。漏检时调低，空白处误检时调高"),
    ("track.max_distance_px", "最大帧间位移", 1, 10000, 1, 1, " px", 1, "同一液滴相邻两帧的最大移动距离。同一液滴被数成多个时调高"),
    ("droplets.count_line", "计数线位置", 5, 95, 5, 0, " %", 100, "沿流动方向的计数线位置；液滴越过此线时计数一次"),
    ("inclusions.cell_diameter_px", "细胞直径", 1, 500, 0.5, 1, " px", 1, "单个细胞的典型直径。两个相邻细胞被数成一个时调小"),
    ("inclusions.min_contrast", "细胞最低对比度", 0, 65535, 1, 1, "", 1, "细胞与液滴内部的最小灰度差。“自动”按每帧估计的噪声设定；漏数时调低，空液滴被数出细胞时调高"),
    ("positivity.threshold", "阳性阈值", 0, 1e9, 1, 1, "", 1, "液滴荧光（扣除背景后的平均值）高于此值判为阳性。“自动”在能分出阴阳两群时自动定阈值；用阴性对照确定阈值后可在此固定"),
]
FLAG_SHORT = {
    "low_circularity": "形状不圆",
    "size_outlier": "粒径异常",
    "count_unstable": "计数不稳",
    "near_threshold": "近阈值",
    "few_frames": "帧数少",
}
AUTO_KEYS = {"seg.min_diameter_px", "seg.max_diameter_px", "inclusions.min_contrast", "positivity.threshold"}  # 0 on the spin box means "automatic"


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


def _conc(value: float | None) -> str:
    """Concentrations span many orders of magnitude: thousands separators, 3 significant digits below 100."""
    if value is None or not math.isfinite(value):
        return "–"
    return f"{value:,.0f}" if value >= 100 else f"{value:.3g}"


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
        self.live_btn = QPushButton("在线监控（回放录像 / 监视采集文件夹）…")
        self.live_btn.setToolTip("边采集（或边回放）边分析：滚动统计、漂移报警、处理帧率与丢帧")
        self.live_btn.clicked.connect(self.open_live)
        v.addWidget(self.live_btn)
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
        self.height_edit = _number_edit("可不填")
        self.height_edit.setToolTip("流道高度。液滴直径大于它时按被压扁的形状换算体积；不填则按球形换算")
        form.addRow("流道高度 (µm)", self.height_edit)
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
        self.report_btn = QPushButton("打开报告")
        self.report_btn.setToolTip("单文件 HTML 报告，可直接发送或打印")
        self.report_btn.clicked.connect(self._open_report)
        top.addWidget(self.report_btn, 0, Qt.AlignmentFlag.AlignTop)
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
        self.card_copies = Card("每滴平均拷贝数")
        self.card_conc = Card("浓度（拷贝/µL）")
        self.card_sep = Card("阴阳分离度 D")
        for c in (self.card_pos, self.card_copies, self.card_conc, self.card_sep):
            fl.addWidget(c)
        self.fluo_cards.setVisible(False)
        pv.addWidget(self.fluo_cards)

        pv.addWidget(heading("复核图"))
        self.viewer = ReviewViewer()
        pv.addWidget(self.viewer, 3)
        self.flag_box = QWidget()
        fb = QVBoxLayout(self.flag_box)
        fb.setContentsMargins(0, 0, 0, 0)
        fb.setSpacing(4)
        self.flag_heading = heading("可疑液滴")
        fb.addWidget(self.flag_heading)
        self.flag_strip = FlagStrip()
        self.flag_strip.itemClicked.connect(self._show_flag)
        fb.addWidget(self.flag_strip)
        self.flag_box.setVisible(False)
        pv.addWidget(self.flag_box)
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
        try:
            height = _parse_positive(self.height_edit.text())
        except ValueError:
            height = None  # start() reports the bad value
        if height is not None and "droplets.channel_height_um" in defaults:
            out["droplets.channel_height_um"] = height
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
            _parse_positive(self.height_edit.text())
        except ValueError:
            self.warn("标定有误", "像素尺寸、帧间隔和流道高度必须是正数，或留空。")
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
        self._thread = QThread()  # released in _stop_thread
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
        # Called from the worker's own signals: neither object may be destroyed while that
        # signal is still being delivered, so both are handed to Qt (parented to the window,
        # which also ends Python's ownership) and deleted by the event loop afterwards. The
        # worker moved itself back to this thread when it finished.
        thread, worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.quit()
            thread.wait()
        for obj in (worker, thread):
            if obj is not None and obj.thread() is self.thread():
                obj.setParent(self)
                obj.deleteLater()
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
            self.open_btn, self.folder_btn, self.pixel_edit, self.interval_edit, self.height_edit, self.flow_combo, self.channel_combo,
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
        parts = pick_summaries(summaries)
        s, enc, fluo = parts["droplets"] or {}, parts["encapsulation"], parts["fluorescence"]
        unit = _UNIT_LABEL.get(s.get("length_unit", "px"), s.get("length_unit", "px"))
        freq_unit = s.get("frequency_unit", "")
        freq_unit = "次/帧" if freq_unit == "1/frame" else freq_unit

        n = s.get("n_droplets")
        self.card_n.set("–" if n is None else str(n), f"共追踪到 {s['n_tracks_total']} 条轨迹" if "n_tracks_total" in s else "")
        volume = s.get("volume_mean_pl")
        shape = {"sphere": "按球形", "flattened": "按压扁形状"}.get(s.get("volume_model"), "")
        self.card_d.set(
            f"{_fmt(s.get('diameter_mean'))} {unit}",
            f"标准差 {_fmt(s.get('diameter_std'), 3)} {unit}" + (f"\n体积 {_fmt(volume, 3)} pL（{shape}）" if volume is not None else ""),
        )
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
        self._show_flags(run_dir, parts["flags"])
        self._show_encapsulation(run_dir, enc)
        self._show_fluorescence(run_dir, fluo, unit)

        notes = run_notes(summaries, provenance, preview)
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))

        self.result_title.setText("预览结果" if preview else "分析结果")
        self.result_path.setText("临时目录，关闭程序后删除" if preview else str(run_dir))
        self.run_dir = run_dir
        self.stack.setCurrentIndex(1)
        return True

    def _show_flags(self, run_dir: Path, summary: dict | None) -> None:
        crops = run_dir / "review" / "flags"
        table = next(
            (p for p in sorted((run_dir / "tables").glob("*.csv")) if {"reason_text", "track_id"} <= set(p.open(encoding="utf-8").readline().strip().split(","))),
            None,
        )
        items: list[tuple[Path, str, str]] = []
        if table is not None and crops.is_dir():
            for row in pd.read_csv(table).itertuples(index=False):
                path = crops / f"track{int(row.track_id):05d}_frame{int(row.frame):06d}.png"
                if path.exists():
                    short = "/".join(FLAG_SHORT.get(code, code) for code in str(row.reasons).split(";"))
                    items.append((path, short, f"轨迹 {int(row.track_id)} · 第 {int(row.frame)} 帧：{row.reason_text}"))
        self.flag_strip.set_flags(items)
        n = (summary or {}).get("n_flagged", 0) or 0
        total = (summary or {}).get("n_droplets", 0) or 0
        shown = f"，显示前 {len(items)} 个" if len(items) < n else ""
        self.flag_heading.setText(f"可疑液滴（{n} / {total}{shown}）· 点击查看")
        self.flag_box.setVisible(bool(items))

    def _show_flag(self, item) -> None:
        path, reason = item.data(Qt.ItemDataRole.UserRole)
        self.viewer.show_image(Path(path), reason + " · 点“上一张/下一张”回到复核图")

    def _show_encapsulation(self, run_dir: Path, enc: dict | None) -> None:
        self.enc_cards.setVisible(enc is not None)
        self.count_box.setVisible(enc is not None)
        if enc is None:
            return

        def share(key: str) -> str:
            return "–" if enc.get(key) is None else f"{100 * enc[key]:.1f} %"

        lo, hi = enc.get("lambda_ci_low"), enc.get("lambda_ci_high")
        per_ul = enc.get("inclusions_per_ul")
        self.card_lambda.set(
            _fmt(enc.get("lambda"), 3),
            (f"95% 区间 {_fmt(lo, 3)} – {_fmt(hi, 3)}" if lo is not None else "")
            + (f"\n约 {_conc(per_ul)} 个/µL" if per_ul is not None else ""),
        )
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
        counts = f"{fluo.get('n_positive', 0)} / {fluo.get('n_droplets', 0)}"
        self.card_pos.set(
            "–" if p is None else f"{100 * p:.1f} %",
            counts + ("" if lo is None else f" · 95% 区间 {100 * lo:.1f} – {100 * hi:.1f} %"),
        )
        conc, conc_lo, conc_hi = fluo.get("copies_per_ul"), fluo.get("copies_per_ul_ci_low"), fluo.get("copies_per_ul_ci_high")
        self.card_conc.set(
            _conc(conc),
            f"95% 区间 {_conc(conc_lo)} – {_conc(conc_hi)}" if conc is not None else "需要像素标定才能换算体积",
        )
        c, clo, chi = fluo.get("copies_per_droplet"), fluo.get("copies_ci_low"), fluo.get("copies_ci_high")
        self.card_copies.set(_fmt(c, 3), f"95% 区间 {_fmt(clo, 3)} – {_fmt(chi, 3)}" if c is not None else "已饱和或无数据")
        method = {"mixture": "自动分群", "outlier": "离群值规则", "fixed": "固定阈值", "otsu": "大津法"}.get(
            fluo.get("threshold_method"), fluo.get("threshold_method") or ""
        )
        d = fluo.get("separation_d")
        self.card_sep.set("–" if d is None else ("∞" if not math.isfinite(d) else f"{d:.1f}"), f"{method} · 阈值 {_fmt(fluo.get('threshold'))}")
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

    def _open_report(self) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / "report.html"
        if not path.exists():
            from vexel.report import write_report

            try:
                write_report(self.run_dir)
            except Exception as e:  # noqa: BLE001
                self.warn("无法生成报告", str(e))
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

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

    def open_live(self) -> None:
        from vexel.ui.live import LiveWindow

        if getattr(self, "live_window", None) is None:
            self.live_window = LiveWindow()
        self.live_window.show()
        self.live_window.raise_()

    def closeEvent(self, event) -> None:  # noqa: N802
        if getattr(self, "live_window", None) is not None:
            self.live_window.close()
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30000)
        for d in self._preview_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(event)
