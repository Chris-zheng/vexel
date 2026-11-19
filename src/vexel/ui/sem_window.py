"""SEM module window (in development: roadmap 0.0.5).

M1: open an image, calibrate it (instrument metadata, the data bar's scale bar, or by
hand), confirm. M2: measure on it - length, polyline, area (polygon, freehand, magic
wand), angle, counting, circle and ellipse fits, line profiles with edges, widths and
pitch - with a table of results, undo and redo, and exports (CSV, Excel, annotated PNG
and SVG, and the measurement record itself). M3: image processing - line levelling,
Fourier destriping, background levelling, scan-line alignment, denoising, contrast and
sharpening - with a before/after view, the saturated-area mask, and multi-frame
averaging.

Measurements are kept per image (by its SHA-256) and saved after every change, so
reopening the image brings them back. Values are always recomputed from the drawn
geometry and the calibration in force.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from vexel.ui.sem_canvas import TOOL_HINTS, ImageView, ProfilePlot
from vexel.ui.widgets import heading, muted

VENDORS = {"thermo_fisher": "Thermo Fisher（FEI）", "zeiss": "Zeiss", "tescan": "TESCAN", "hitachi": "Hitachi", "jeol": "JEOL"}
SOURCES = {"metadata": "仪器元数据", "databar": "信息栏比例尺识别", "manual": "手动", "none": "无"}
STATE_COLOURS = {"confirmed": "#16a34a", "pending": "#d97706", "none": "#dc2626"}
TOOLS = [("pan", "选择", "V"), ("line", "长度", "L"), ("polyline", "折线", "Shift+L"), ("polygon", "面积", "A"), ("freehand", "自由画", "F"),
         ("wand", "魔棒", "W"), ("angle", "角度", "G"), ("point", "计数", "N"), ("circle", "圆", "C"), ("ellipse", "椭圆", "E"),
         ("profile", "线剖面", "P")]  # fmt: skip
EDGE_METHODS = [("threshold", "阈值（50% 等）"), ("gradient", "最大梯度"), ("peak", "亮边峰值")]
TAB_CALIBRATION, TAB_PROCESSING, TAB_MEASURE = range(3)


class _CalibrateJob(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        from vexel.packs.sem.calibrate import for_file

        try:
            self.done.emit(for_file(self.path))
        except Exception as e:  # noqa: BLE001 - shown to the person
            self.failed.emit(f"{type(e).__name__}: {e}")


class _ProcessJob(QObject):
    """Runs a processing chain away from the window (non-local means on a large image takes seconds)."""

    done = Signal(object, str, object)  # result, chain key, processor
    failed = Signal(str, str, object)

    def __init__(self, processor: Any, chain: Any, key: str) -> None:
        super().__init__()
        self.processor, self.chain, self.key = processor, chain, key

    def run(self) -> None:
        try:
            self.done.emit(self.processor.run(self.chain), self.key, self.processor)
        except Exception as e:  # noqa: BLE001 - shown to the person
            self.failed.emit(f"{type(e).__name__}: {e}", self.key, self.processor)


SLOW = {("denoise", "nlm"), ("denoise", "bilateral"), ("background", "rolling_ball")}


def _slow(chain: Any) -> bool:
    from vexel.packs.sem.process import OPS

    for st in chain.active():
        op = OPS.get(st.op)
        if op is not None and (st.op, op.resolve(st.params).get("method")) in SLOW:
            return True
    return False


class SemWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vexel · SEM（开发中）")
        self.resize(1400, 880)
        self.path: Path | None = None
        self.image: np.ndarray | None = None
        self.cal: Any = None
        self.ms: Any = None  # MeasurementSet
        self.values: dict[int, dict[str, Any]] = {}
        self._undo: list[str] = []
        self._redo: list[str] = []
        self._thread: QThread | None = None
        self._job: _CalibrateJob | None = None
        self.processor: Any = None  # process.Processor of this image
        self.result: Any = None  # process.Result for the chain in force
        self._result_key: str | None = None
        self._proc_thread: QThread | None = None
        self._proc_job: _ProcessJob | None = None
        self._sync = False  # tests and scripts: process on this thread
        self._param_widgets: dict[str, Any] = {}
        self._param_timer = QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.setInterval(350)
        self._param_timer.timeout.connect(self._apply_params)

        self.view = ImageView()
        self.view.shape_done.connect(self._shape_done)
        self.view.point_moved.connect(self._point_moved)
        self.view.item_clicked.connect(self._select)
        self.view.hovered.connect(self._hover)
        self._build_toolbar()

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.view, 1)
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(390)
        for title, build in (("标定", self._build_calibration), ("图像处理", self._build_processing), ("测量", self._build_measurements)):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            panel = QFrame()
            panel.setObjectName("sidebar")
            scroll.setWidget(panel)
            v = QVBoxLayout(panel)
            build(v)
            v.addStretch(1)
            self.tabs.addTab(scroll, title)
        root.addWidget(self.tabs)
        self.setCentralWidget(central)
        self._set_buttons()
        self.statusBar().showMessage("打开一张 SEM 图片开始")

    # -- layout ---------------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = QToolBar("测量工具")
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addAction("打开…", self._choose)
        bar.addSeparator()
        self.tool_actions: dict[str, QAction] = {}
        group = QActionGroup(self)
        for tool, label, key in TOOLS:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setShortcut(QKeySequence(key))
            act.setToolTip(f"{label}（{key}）：{TOOL_HINTS[tool]}")
            act.triggered.connect(lambda _c=False, t=tool: self.set_tool(t))
            group.addAction(act)
            bar.addAction(act)
            self.tool_actions[tool] = act
        self.tool_actions["pan"].setChecked(True)
        bar.addSeparator()
        self.undo_act = bar.addAction("撤销", self.undo)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_act = bar.addAction("重做", self.redo)
        self.redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        bar.addAction("适应窗口", self.view.fit)
        bar.addSeparator()
        self.export_act = bar.addAction("导出…", self._export)
        self.load_act = bar.addAction("载入测量记录…", self._load_record_file)

    def _build_calibration(self, v: QVBoxLayout) -> None:
        self.file_label = muted("")
        v.addWidget(self.file_label)
        v.addWidget(heading("标定"))
        self.pixel = QLabel("–")
        self.pixel.setObjectName("cardValue")
        v.addWidget(self.pixel)
        self.state = QLabel("")
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        self.detail = muted("")
        v.addWidget(self.detail)
        self.notes = QLabel("")
        self.notes.setObjectName("notes")
        self.notes.setWordWrap(True)
        self.notes.hide()
        v.addWidget(self.notes)
        self.confirm_btn = QPushButton("确认标定")
        self.confirm_btn.setObjectName("primary")
        self.confirm_btn.clicked.connect(self._confirm)
        v.addWidget(self.confirm_btn)
        row = QHBoxLayout()
        self.pick_btn = QPushButton("点比例尺两端…")
        self.pick_btn.setToolTip("在图上依次点击比例尺的两端，再输入它代表的长度")
        self.pick_btn.clicked.connect(lambda: self.set_tool("calibrate"))
        self.size_btn = QPushButton("输入像素尺寸…")
        self.size_btn.clicked.connect(self._enter_size)
        row.addWidget(self.pick_btn)
        row.addWidget(self.size_btn)
        v.addLayout(row)
        self.meta = muted("")
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.meta)

    def _build_measurements(self, v: QVBoxLayout) -> None:
        v.addWidget(heading("测量"))
        form = QFormLayout()
        self.category = QComboBox()
        self.category.setEditable(True)
        self.category.addItems(["", "颗粒", "缺陷", "孔"])
        self.category.setToolTip("计数工具：这一组标记属于哪一类")
        form.addRow("计数类别", self.category)
        self.profile_width = QSpinBox()
        self.profile_width.setRange(1, 101)
        self.profile_width.setValue(5)
        self.profile_width.setSuffix(" 像素")
        self.profile_width.setToolTip("线剖面在垂直方向上平均的宽度：越宽越抗噪，但会把倾斜的边缘抹宽")
        form.addRow("剖面宽度", self.profile_width)
        self.edge_method = QComboBox()
        for key, label in EDGE_METHODS:
            self.edge_method.addItem(label, key)
        self.edge_method.setToolTip("边缘的定义不同，测得的线宽也不同；结果中总会写明所用定义")
        form.addRow("边缘定义", self.edge_method)
        self.edge_level = QSpinBox()
        self.edge_level.setRange(5, 95)
        self.edge_level.setValue(50)
        self.edge_level.setSuffix(" %")
        form.addRow("阈值", self.edge_level)
        self.tilt_on = QCheckBox("倾斜校正")
        self.tilt_on.setToolTip("样品台倾斜时，垂直方向的距离被缩短；勾选后按 1/cos(倾角) 校正（倾转轴为图像水平方向）")
        self.tilt = QDoubleSpinBox()
        self.tilt.setRange(0, 89)
        self.tilt.setDecimals(1)
        self.tilt.setSuffix("°")
        self.tilt.setEnabled(False)
        self.tilt_on.toggled.connect(lambda on: (self.tilt.setEnabled(on), self._apply_scale()))
        self.tilt.valueChanged.connect(lambda _v: self._apply_scale())
        trow = QHBoxLayout()
        trow.addWidget(self.tilt_on)
        trow.addWidget(self.tilt)
        form.addRow(trow)
        for w in (self.edge_method, self.edge_level, self.profile_width):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(lambda _x: self._profile_settings_changed())
        v.addLayout(form)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "工具", "结果", "名称"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.itemChanged.connect(self._name_edited)
        v.addWidget(self.table, 1)
        brow = QHBoxLayout()
        self.delete_btn = QPushButton("删除所选")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setShortcut(QKeySequence.StandardKey.Delete)
        brow.addWidget(self.delete_btn)
        brow.addStretch(1)
        v.addLayout(brow)
        self.summary = muted("")
        v.addWidget(self.summary)
        self.profile_plot = ProfilePlot()
        self.profile_plot.hide()
        v.addWidget(self.profile_plot)
        self.profile_info = muted("")
        v.addWidget(self.profile_info)

    def _build_processing(self, v: QVBoxLayout) -> None:
        from vexel.packs.sem.process import OPS

        v.addWidget(heading("图像处理"))
        v.addWidget(muted("校正步骤的结果用于测量；“仅显示”的步骤只改变显示和导出的图。只处理信息栏以上的部分。"))
        row = QHBoxLayout()
        self.op_choice = QComboBox()
        for name, op in OPS.items():
            self.op_choice.addItem(op.label + ("（仅显示）" if op.role == "display" else ""), name)
            self.op_choice.setItemData(self.op_choice.count() - 1, op.help, Qt.ItemDataRole.ToolTipRole)
        self.add_step_btn = QPushButton("添加")
        self.add_step_btn.clicked.connect(lambda: self.add_step(self.op_choice.currentData()))
        row.addWidget(self.op_choice, 1)
        row.addWidget(self.add_step_btn)
        v.addLayout(row)
        self.steps = QListWidget()
        self.steps.setMaximumHeight(150)
        self.steps.currentRowChanged.connect(lambda _r: self._show_params())
        self.steps.itemChanged.connect(self._step_toggled)
        v.addWidget(self.steps)
        brow = QHBoxLayout()
        self.up_btn = QPushButton("上移")
        self.up_btn.clicked.connect(lambda: self.move_step(self.steps.currentRow(), -1))
        self.down_btn = QPushButton("下移")
        self.down_btn.clicked.connect(lambda: self.move_step(self.steps.currentRow(), 1))
        self.remove_step_btn = QPushButton("删除")
        self.remove_step_btn.clicked.connect(lambda: self.remove_step(self.steps.currentRow()))
        for b in (self.up_btn, self.down_btn, self.remove_step_btn):
            brow.addWidget(b)
        v.addLayout(brow)
        self.params_box = QWidget()
        self.params_form = QFormLayout(self.params_box)
        self.params_form.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.params_box)
        self.step_help = muted("")
        v.addWidget(self.step_help)

        v.addWidget(heading("饱和区"))
        self.sat_on = QCheckBox("标记饱和区（魔棒和线剖面不使用）")
        self.sat_on.setToolTip("严重荷电或探测器饱和处的信息已经丢失，只能标记，不能“修复”")
        self.sat_on.toggled.connect(lambda on: self.set_saturation(enabled=on))
        v.addWidget(self.sat_on)
        self.sat_show = QCheckBox("在图上显示（红色）")
        self.sat_show.setChecked(True)
        self.sat_show.toggled.connect(lambda _on: self._show_mask())
        v.addWidget(self.sat_show)
        self.sat_info = muted("")
        v.addWidget(self.sat_info)
        self.proc_notes = QLabel("")
        self.proc_notes.setObjectName("notes")
        self.proc_notes.setWordWrap(True)
        self.proc_notes.hide()
        v.addWidget(self.proc_notes)

        v.addWidget(heading("前后对比"))
        crow = QHBoxLayout()
        self.compare_on = QCheckBox("左侧显示原图")
        self.compare_on.toggled.connect(lambda _on: self._apply_split())
        self.compare_pos = QSlider(Qt.Orientation.Horizontal)
        self.compare_pos.setRange(0, 100)
        self.compare_pos.setValue(50)
        self.compare_pos.valueChanged.connect(lambda _v: self._apply_split())
        crow.addWidget(self.compare_on)
        crow.addWidget(self.compare_pos, 1)
        v.addLayout(crow)

        v.addWidget(heading("多帧与导出"))
        orow = QHBoxLayout()
        self.average_btn = QPushButton("多帧平均…")
        self.average_btn.setToolTip("选择同一位置的其他帧：对齐（亚像素）后平均，结果另存为新图并沿用本图的标定")
        self.average_btn.clicked.connect(self._choose_frames)
        self.export_proc_btn = QPushButton("导出处理后的图…")
        self.export_proc_btn.setToolTip("校正后的图（32 位浮点 TIFF，信息栏保留），在 Vexel 中打开时沿用本图的标定")
        self.export_proc_btn.clicked.connect(self._export_processed)
        orow.addWidget(self.average_btn)
        orow.addWidget(self.export_proc_btn)
        v.addLayout(orow)
        v.addWidget(muted("荷电最有效的办法在采集端（降低电压、缩短驻留时间、交错扫描、镀膜）；软件处理只是补救。"))

    # -- opening and calibrating --------------------------------------------------------

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 SEM 图片", "", "图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path, *, wait: bool = False) -> bool:
        from vexel.packs.sem.calibrate import load_image

        path = Path(path)
        try:
            img = load_image(path)
        except Exception as e:  # noqa: BLE001 - shown to the person
            self.file_label.setText(f"无法读取：{type(e).__name__}: {e}")
            return False
        self.path, self.image, self.cal, self.ms, self.values = path, img, None, None, {}
        self.processor, self.result, self._result_key = None, None, None
        self._undo, self._redo = [], []
        self.view.set_image(img)
        self.file_label.setText(f"{path.name}：{img.shape[1]}×{img.shape[0]} 像素")
        self.pixel.setText("…")
        self.state.setText("正在读取元数据与信息栏…")
        self.detail.setText("")
        self.notes.hide()
        self.meta.setText("")
        self._refresh()
        self._sync = wait
        if wait:  # tests and scripts: calibrate on this thread
            from vexel.packs.sem.calibrate import for_file

            self._show(for_file(path))
            return True
        self._stop_thread()
        self._thread = QThread()
        self._job = _CalibrateJob(path)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.done.connect(self._show)
        self._job.failed.connect(lambda msg: self.state.setText(f"标定失败：{msg}"))
        self._job.done.connect(self._thread.quit)
        self._job.failed.connect(self._thread.quit)
        self._thread.start()
        return True

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        self._stop_thread()
        if self._proc_thread is not None:
            self._proc_thread.quit()
            self._proc_thread.wait(60000)
        super().closeEvent(event)

    # -- calibration display -----------------------------------------------------------------

    def _show(self, cal: Any) -> None:
        from vexel.packs.sem import measure_io
        from vexel.packs.sem.calibrate import format_length

        if self.path is None or cal.file != self.path.name:
            return
        self.cal = cal
        self.view.show_calibration(cal)
        if cal.ok:
            self.pixel.setText(f"{format_length(cal.pixel_size)} / 像素")
            key = "confirmed" if cal.confirmed else "pending"
            text = f"来源：{SOURCES.get(cal.source, cal.source)}，" + ("已确认" if cal.confirmed else "待确认：请核对后点击“确认标定”")
            if cal.confirmed and cal.confirmed_by == "person":
                text += f"（{cal.confirmed_at[:10]}）"
        else:
            self.pixel.setText("未标定")
            key, text = "none", cal.reason + "。在未标定前，测量以像素为单位。"
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {STATE_COLOURS[key]}; font-weight: 600;")
        lines = []
        if cal.scale_bar:
            lines.append(f"比例尺：{cal.scale_bar['length_px']:.1f} 像素" + (f" = {cal.label}" if cal.label else ""))
        for chk in cal.checks:
            name = {"scale bar": "信息栏比例尺", "field width": "视场宽度", "metadata": "仪器元数据"}.get(chk["check"], chk["check"])
            lines.append(f"核对（{name}）：{format_length(chk['pixel_size'])}/像素，相差 {chk['difference'] * 100:+.2f}%" + ("" if chk["agree"] else "，不一致"))
        if cal.crop_rows:
            how = {"metadata": "元数据给出", "detected": "自动识别", "manual": "手动"}.get(cal.crop_source, cal.crop_source)
            lines.append(f"分析区域：第 0–{cal.crop_rows - 1} 行；下方 {cal.image_shape[0] - cal.crop_rows} 行是信息栏（{how}）")
        self.detail.setText("\n".join(lines))
        if cal.warnings:
            self.notes.setText("\n".join(f"• {w}" for w in cal.warnings))
            self.notes.show()
        else:
            self.notes.hide()
        md = cal.metadata or {}
        info = []
        if md:
            info.append(f"厂商：{VENDORS.get(md.get('vendor'), md.get('vendor'))}" + (f"（{md['instrument']}）" if md.get("instrument") else ""))
            for k, label, fmt in (("voltage", "加速电压", lambda x: f"{x / 1e3:.3g} kV"), ("magnification", "放大倍数", lambda x: f"{x:,.0f}×"),
                                  ("hfw", "视场宽度", format_length), ("working_distance", "工作距离", format_length),
                                  ("tilt", "样品台倾角", lambda x: f"{x:.2f}°"), ("detector", "探测器", str), ("date", "时间", str)):  # fmt: skip
                if md.get(k) not in (None, ""):
                    info.append(f"{label}：{fmt(md[k])}")
        else:
            info.append("文件中没有仪器元数据")
        self.meta.setText("\n".join(info))
        if md.get("tilt") and abs(md["tilt"]) >= 0.5 and not self.tilt_on.isChecked():
            self.tilt.setValue(abs(float(md["tilt"])))
        # measurements: the record of this image, or a new one, under the calibration in force
        if self.processor is None or self.processor.crop_rows != (cal.crop_rows or None):
            from vexel.packs.sem.process import Processor

            self.processor, self.result, self._result_key = Processor(self.image, cal.crop_rows), None, None
        if self.ms is None or self.ms.image.get("sha256") != cal.sha256:
            self.ms = measure_io.load_record(cal.sha256) or measure_io.new_set(self.path, cal)
            if self.ms.scale.tilt:
                self.tilt_on.setChecked(True)
                self.tilt.setValue(self.ms.scale.tilt)
        self._apply_scale(record=False)
        self.statusBar().showMessage(cal.describe())

    def _apply_scale(self, record: bool = True) -> None:
        from vexel.packs.sem.measure import scale_from

        if self.ms is None:
            return
        tilt = float(self.tilt.value()) if self.tilt_on.isChecked() else 0.0
        new = scale_from(self.cal, tilt)
        if new != self.ms.scale:
            if record:
                self._push_undo()
            self.ms.scale = new
            self._changed()
        else:
            self._refresh()

    def _set_buttons(self) -> None:
        has = self.path is not None
        self.confirm_btn.setVisible(bool(self.cal is not None and self.cal.needs_confirmation))
        self.pick_btn.setEnabled(has)
        self.size_btn.setEnabled(has)
        for act in self.tool_actions.values():
            act.setEnabled(has)
        self.undo_act.setEnabled(bool(self._undo))
        self.redo_act.setEnabled(bool(self._redo))
        self.export_act.setEnabled(bool(self.ms and self.ms.items))
        self.load_act.setEnabled(self.ms is not None)
        self.delete_btn.setEnabled(self.view.selected is not None)
        ready = self.ms is not None and self.processor is not None
        row = self.steps.currentRow()
        n = len(self.ms.processing.steps) if self.ms is not None else 0
        for w in (self.add_step_btn, self.op_choice, self.sat_on, self.compare_on, self.average_btn):
            w.setEnabled(ready)
        self.export_proc_btn.setEnabled(ready and bool(self.ms.processing.active()))
        self.up_btn.setEnabled(ready and row > 0)
        self.down_btn.setEnabled(ready and 0 <= row < n - 1)
        self.remove_step_btn.setEnabled(ready and 0 <= row < n)

    def _hover(self, x: float, y: float) -> None:
        if self.image is None or not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            return
        text = f"x {x:.0f}, y {y:.0f} 像素"
        if self.cal is not None and self.cal.ok:
            from vexel.packs.sem.calibrate import format_length

            text += f"  ({format_length(x * self.cal.pixel_size)}, {format_length(y * (self.cal.pixel_size_y or self.cal.pixel_size))})"
        hint = TOOL_HINTS.get(self.view.tool, "")
        self.statusBar().showMessage(f"{hint}  ·  {text}" if self.view.tool != "pan" else text)

    # -- confirming and manual calibration ------------------------------------------------

    def _confirm(self) -> None:
        from vexel.packs.sem.calibrate import save

        if self.cal is None or not self.cal.ok:
            return
        self.cal.confirm("person")
        save(self.cal)
        self._show(self.cal)

    def _enter_size(self) -> None:
        text, ok = QInputDialog.getText(self, "像素尺寸", "每个像素的尺寸（例如 2.5 nm、0.82 µm）：")
        if ok and text.strip():
            self.apply_manual(pixel_size=text)

    def apply_manual(self, *, pixel_size: str | None = None, bar_length: str | None = None, bar_px: float | None = None) -> bool:
        from vexel.packs.sem.calibrate import manual, parse_quantity, save

        if self.path is None:
            return False
        try:
            cal = manual(self.path, pixel_size=parse_quantity(pixel_size) if pixel_size else None,
                         bar_length=parse_quantity(bar_length) if bar_length else None, bar_px=bar_px)  # fmt: skip
        except ValueError as e:
            QMessageBox.warning(self, "无法标定", str(e))
            return False
        save(cal)
        self._show(cal)
        return True

    # -- measuring ------------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        if tool == "calibrate":
            self.tabs.setCurrentIndex(TAB_CALIBRATION)
        elif tool != "pan":
            self.tabs.setCurrentIndex(TAB_MEASURE)  # the results appear there
        self.view.set_tool(tool)
        if tool in self.tool_actions:
            self.tool_actions[tool].setChecked(True)
        else:
            for act in self.tool_actions.values():
                act.setChecked(False)
        self.view.setFocus()
        self.statusBar().showMessage(TOOL_HINTS.get(tool, ""))

    def _profile_params(self) -> dict[str, Any]:
        return {"width": int(self.profile_width.value()), "method": self.edge_method.currentData(), "level": self.edge_level.value() / 100}

    def _shape_done(self, tool: str, points: list[list[float]]) -> None:
        if tool == "calibrate":
            self.set_tool("pan")
            length_px = float(np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]))
            if length_px < 2:
                return
            text, ok = QInputDialog.getText(self, "比例尺长度", f"两点相距 {length_px:.1f} 像素。它代表的长度（例如 500 nm、2 µm）：")
            if ok and text.strip():
                self.apply_manual(bar_length=text, bar_px=length_px)
            return
        self.add_measurement(tool, points)

    def add_measurement(self, kind: str, points: list[list[float]], **kw: Any) -> Any:
        """Add one measurement (also used by tests and scripts); returns the item, or None if refused."""
        if self.ms is None:
            return None
        params: dict[str, Any] = dict(kw.pop("params", {}))
        if kind == "profile":
            params = {**self._profile_params(), **params}
        if kind == "point":
            kw.setdefault("category", self.category.currentText().strip())
        self._push_undo()
        try:
            item = self.ms.add(kind, points, params=params, **kw)
            from vexel.packs.sem.measure import compute

            res = self._processed()
            compute(item, self.ms.scale, res.analysis if res is not None else self.image, res.mask if res is not None else None)
        except ValueError as e:
            self._pop_undo()
            self.statusBar().showMessage(f"无法测量：{e}")
            return None
        self.tabs.setCurrentIndex(TAB_MEASURE)
        self._changed(select=item.id)
        return item

    def _point_moved(self, item_id: int, index: int, x: float, y: float) -> None:
        if self.ms is None:
            return
        self._push_undo()
        item = self.ms.get(item_id)
        pts = list(item.points)
        pts[index] = (float(x), float(y))
        item.points = pts
        self._changed(select=item_id)

    def _profile_settings_changed(self) -> None:
        """New profile settings apply to the selected profile (and to the next ones drawn)."""
        if self.ms is None or self.view.selected is None:
            return
        item = self.ms.get(self.view.selected)
        if item.kind != "profile":
            return
        self._push_undo()
        item.params = {**item.params, **self._profile_params()}
        self._changed(select=item.id)

    def delete_selected(self) -> None:
        if self.ms is None or self.view.selected is None:
            return
        self._push_undo()
        self.ms.remove(self.view.selected)
        self._changed(select=None)

    def _name_edited(self, cell: QTableWidgetItem) -> None:
        if self.ms is None or cell.column() != 3 or self._filling:
            return
        item_id = int(self.table.item(cell.row(), 0).text())
        item = self.ms.get(item_id)
        if item.name != cell.text():
            self._push_undo()
            item.name = cell.text()
            self._changed(select=item_id)

    # -- undo, redo, autosave -------------------------------------------------------------------

    def _push_undo(self) -> None:
        if self.ms is not None:
            self._undo.append(self.ms.model_dump_json())
            self._redo.clear()

    def _pop_undo(self) -> None:
        if self._undo:
            self._undo.pop()

    def undo(self) -> None:
        self._swap(self._undo, self._redo)

    def redo(self) -> None:
        self._swap(self._redo, self._undo)

    def _swap(self, take: list[str], put: list[str]) -> None:
        from vexel.packs.sem.measure import MeasurementSet

        if not take or self.ms is None:
            return
        put.append(self.ms.model_dump_json())
        self.ms = MeasurementSet.model_validate_json(take.pop())
        self.tilt_on.blockSignals(True)
        self.tilt.blockSignals(True)
        self.tilt_on.setChecked(bool(self.ms.scale.tilt))
        self.tilt.setEnabled(bool(self.ms.scale.tilt))
        if self.ms.scale.tilt:
            self.tilt.setValue(self.ms.scale.tilt)
        self.tilt_on.blockSignals(False)
        self.tilt.blockSignals(False)
        self._changed(select=None, record=False)

    def _changed(self, select: int | None = -1, record: bool = True) -> None:
        from vexel.packs.sem import measure_io

        if self.ms is not None:
            measure_io.autosave(self.ms)
        self._refresh(select)

    # -- display ---------------------------------------------------------------------------

    _filling = False

    def _refresh(self, select: int | None = -1) -> None:
        from vexel.packs.sem import measure_io
        from vexel.packs.sem.measure import KIND_NAMES, compute_all

        if select == -1:
            select = self.view.selected
        if self.ms is None:
            self.table.setRowCount(0)
            self.view.draw([], None)
            self._set_buttons()
            return
        if select is not None and select not in [i.id for i in self.ms.items]:
            select = None
        res = self._processed()
        values = compute_all(self.ms, res.analysis if res is not None else self.image, res.mask if res is not None else None)
        self.values = {v["id"]: v for v in values}
        shapes = measure_io._shapes(self.ms, values)
        self.view.draw([(item.id, *shape) for item, shape in zip(self.ms.items, shapes)], select)
        sel_item = self.ms.get(select) if select is not None else None
        self.view.show_handles(select, list(sel_item.points) if sel_item is not None else None)
        unit, f = measure_io.display_unit(self.ms)
        self._filling = True
        self.table.setRowCount(len(self.ms.items))
        for row, item in enumerate(self.ms.items):
            v = self.values[item.id]
            label = measure_io.label_of(v, unit, f).split(" ", 1)
            cells = [str(item.id), KIND_NAMES.get(item.kind, item.kind), (v.get("error") or (label[1] if len(label) > 1 else "")), item.name]
            for col, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if col != 3:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, cell)
            if item.id == select:
                self.table.selectRow(row)
        if select is None:
            self.table.clearSelection()
        self._filling = False
        n = len(self.ms.items)
        unit_note = "" if self.ms.scale.pixel_size else "（未标定：以像素为单位）"
        tilt_note = f"；倾斜校正 {self.ms.scale.tilt:.1f}°" if self.ms.scale.tilt else ""
        self.summary.setText(f"{n} 项测量{unit_note}{tilt_note}。自动保存，重新打开这张图时恢复。" if n else f"选择上方工具在图上测量{unit_note}。")
        self._show_profile(self.values.get(select) if select is not None else None, unit, f)
        self._fill_steps()
        self._set_buttons()

    def _show_profile(self, v: dict[str, Any] | None, unit: str, f: float) -> None:
        if not v or v.get("kind") != "profile" or "error" in v:
            self.profile_plot.hide()
            self.profile_info.setText("")
            return
        self.profile_plot.show()
        self.profile_plot.show_profile(v)
        num = __import__("vexel.packs.sem.measure_io", fromlist=["num"]).num
        lines = [f"边缘定义：{dict(EDGE_METHODS)[v['method']]}" + (f"，阈值 {v['level']:.0%}" if v["method"] == "threshold" else "") + f"；找到 {len(v['edges'])} 个边缘"]
        if v.get("width_mean") is not None:
            lines.append(f"线宽 {num(v['width_mean'] / f)} ± {num((v['width_sd'] or 0) / f)} {unit}（{len(v['widths'])} 条）")
        if v.get("space_mean") is not None:
            lines.append(f"间隔 {num(v['space_mean'] / f)} {unit}")
        if v.get("pitch") is not None:
            lines.append(f"周期 {num(v['pitch'] / f)} {unit}（自相关）" + (f"；按边缘 {num(v['pitch_edges'] / f)} {unit}" if v.get("pitch_edges") else ""))
        if v.get("peak_spacing") is not None:
            lines.append(f"峰间距 {num(v['peak_spacing'] / f)} {unit}")
        self.profile_info.setText("\n".join(lines))

    def _select(self, item_id: int) -> None:
        if item_id >= 0:
            self.tabs.setCurrentIndex(TAB_MEASURE)
        self._refresh(None if item_id < 0 else item_id)

    def _table_selected(self) -> None:
        if self._filling:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            item_id = int(self.table.item(rows[0].row(), 0).text())
            if item_id != self.view.selected:
                self._refresh(item_id)

    # -- exports --------------------------------------------------------------------------

    def _export(self) -> None:
        if self.ms is None or self.path is None:
            return
        stem = self.path.with_suffix("").name + "_measurements"
        path, chosen = QFileDialog.getSaveFileName(self, "导出测量", str(self.path.parent / stem),
                                                   "Excel (*.xlsx);;CSV (*.csv);;标注图 PNG (*.png);;标注图 SVG (*.svg);;测量记录 (*.json)")  # fmt: skip
        if path:
            self.export(path)

    def export(self, path: str | Path) -> list[Path]:
        from vexel.packs.sem import measure_io

        path = Path(path)
        crop = self.cal.crop_rows if self.cal is not None else None
        suffix = path.suffix.lower()
        res = self._processed(block=True)
        if suffix == ".xlsx":
            out = [measure_io.export_xlsx(self.ms, self.image, path, result=res)]
        elif suffix == ".csv":
            out = measure_io.export_csv(self.ms, self.image, path, result=res)
        elif suffix == ".png":
            out = [measure_io.export_png(self.image, self.ms, path, crop_rows=crop, result=res)]
        elif suffix == ".svg":
            out = [measure_io.export_svg(self.image, self.ms, path, crop_rows=crop, result=res)]
        elif suffix == ".json":
            out = [self.ms.save(path)]
        else:
            raise ValueError(f"unknown export type: {suffix}")
        self.statusBar().showMessage("已导出：" + "、".join(p.name for p in out))
        return out

    def _load_record_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入测量记录", str(self.path.parent if self.path else ""), "测量记录 (*.json)")
        if path:
            self.load_record(path)

    def load_record(self, path: str | Path) -> bool:
        from vexel.packs.sem.measure import MeasurementSet

        try:
            ms = MeasurementSet.load(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "无法载入", str(e))
            return False
        if self.cal is not None and ms.image.get("sha256") and ms.image["sha256"] != self.cal.sha256:
            answer = QMessageBox.question(self, "不是同一张图", f"这份记录属于 {ms.image.get('file', '另一张图')}，不是当前图片。仍要载入它的测量吗？")
            if answer != QMessageBox.StandardButton.Yes:
                return False
            ms.image = self.ms.image if self.ms is not None else ms.image
        self._push_undo()
        keep_scale = self.ms.scale if self.ms is not None else ms.scale
        self.ms = ms
        self.ms.scale = keep_scale  # values follow the calibration in force for this image
        self._changed(select=None)
        return True


    # -- image processing ---------------------------------------------------------------------

    def _processed(self, block: bool = False) -> Any:
        """The processing result for the chain in force (recomputed only when the chain changed).

        A chain with a slow step runs in the background: until it is done, the previous
        result stands in (values and picture update when it arrives). ``block`` (exports)
        waits for the current result instead."""
        if self.ms is None or self.processor is None:
            return None
        key = self.ms.processing.model_dump_json()
        if key != self._result_key and not (block or self._sync) and _slow(self.ms.processing):
            self._start_processing(key)
            return self.result
        if key != self._result_key:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.result = self.processor.run(self.ms.processing)
            except (ValueError, MemoryError) as e:
                self.statusBar().showMessage(f"图像处理失败：{e}")
                self.result = None
            finally:
                QGuiApplication.restoreOverrideCursor()
            self._result_key = key
            self._show_result()
        return self.result

    def _start_processing(self, key: str) -> None:
        if self._proc_thread is not None:
            return  # one at a time: when it is done, the chain then in force is run
        self._proc_thread = QThread()
        self._proc_job = _ProcessJob(self.processor, self.ms.processing.model_copy(deep=True), key)
        self._proc_job.moveToThread(self._proc_thread)
        self._proc_thread.started.connect(self._proc_job.run)
        self._proc_job.done.connect(self._processing_done)
        self._proc_job.failed.connect(self._processing_failed)
        self._proc_thread.start()
        self.sat_info.setText("图像处理中…")
        self.statusBar().showMessage("图像处理中…（完成后数值和图像自动更新）")

    def _end_processing(self) -> None:
        if self._proc_thread is not None:
            self._proc_thread.quit()
            self._proc_thread.wait()
            self._proc_thread, self._proc_job = None, None

    def _processing_done(self, result: Any, key: str, processor: Any) -> None:
        self._end_processing()
        if processor is not self.processor or self.ms is None:
            return
        if key == self.ms.processing.model_dump_json():
            self.result, self._result_key = result, key
            self._show_result()
            self.statusBar().showMessage("图像处理完成")
        self._refresh()  # also starts the chain in force if it changed meanwhile

    def _processing_failed(self, msg: str, key: str, processor: Any) -> None:
        self._end_processing()
        if processor is self.processor:
            self._result_key = key  # do not retry the same chain
            self.statusBar().showMessage(f"图像处理失败：{msg}")

    def wait_processing(self, timeout_ms: int = 60000) -> bool:
        """Wait for background processing to finish (tests and scripts)."""
        from qtpy.QtCore import QCoreApplication, QElapsedTimer

        t = QElapsedTimer()
        t.start()
        while self._proc_thread is not None and t.elapsed() < timeout_ms:
            QCoreApplication.processEvents()
            QThread.msleep(10)
        return self._proc_thread is None

    def _show_result(self) -> None:
        res = self.result
        if res is None:
            return
        self.view.set_picture(res.display, res.original)
        self._show_mask()
        sat = res.saturation
        if not self.ms.processing.saturation.enabled:
            self.sat_info.setText("未标记饱和区")
        elif sat.get("pixels"):
            self.sat_info.setText(f"饱和像素 {sat['pixels']:,}（{sat['fraction']:.2%}），{sat['regions']} 处；阈值 {sat['high']:g}"
                                  + (f"；暗端 {sat['low']:g}" if sat.get("low") is not None else ""))  # fmt: skip
        else:
            self.sat_info.setText("没有发现饱和区")
        notes = [n for n in res.notes if not n.startswith("饱和区")]
        self.proc_notes.setText("\n".join(f"• {n}" for n in notes))
        self.proc_notes.setVisible(bool(notes))

    def _show_mask(self) -> None:
        res = self.result
        self.view.set_mask(res.mask if res is not None else None, self.sat_show.isChecked())

    def _apply_split(self) -> None:
        self.view.set_split(self.compare_pos.value() / 100 if self.compare_on.isChecked() else None)

    def set_compare(self, on: bool, position: float = 0.5) -> None:
        self.compare_pos.setValue(int(round(position * 100)))
        self.compare_on.setChecked(on)

    _filling_steps = False

    _steps_sig: str | None = None
    _params_key: Any = None

    def _fill_steps(self) -> None:
        from vexel.packs.sem.process import OPS

        if self.ms is None:
            return
        sig = json.dumps([[st.op, st.enabled] for st in self.ms.processing.steps] + [self.ms.processing.saturation.enabled])
        if sig == self._steps_sig and self.steps.count() == len(self.ms.processing.steps):
            self._show_params()
            return
        self._steps_sig = sig
        self._filling_steps = True
        row = self.steps.currentRow()
        self.steps.clear()
        for i, st in enumerate(self.ms.processing.steps, start=1):
            op = OPS.get(st.op)
            label = f"{i}. {op.label if op else st.op}" + ("（仅显示）" if op and op.role == "display" else "")
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if st.enabled else Qt.CheckState.Unchecked)
            if op:
                it.setToolTip(op.help)
            self.steps.addItem(it)
        self.steps.setCurrentRow(min(row, self.steps.count() - 1) if row >= 0 else -1)
        self.sat_on.blockSignals(True)
        self.sat_on.setChecked(self.ms.processing.saturation.enabled)
        self.sat_on.blockSignals(False)
        self._filling_steps = False
        self._show_params()

    def _edit_processing(self, change: Any, select: int | None = None) -> None:
        if self.ms is None:
            return
        self._push_undo()
        try:
            change(self.ms.processing)
        except ValueError as e:
            self._pop_undo()
            QMessageBox.warning(self, "无法处理", str(e))
            return
        if select is not None:
            self._filling_steps = True
            self._fill_steps()
            self.steps.setCurrentRow(select)
            self._filling_steps = False
        self._changed()

    def add_step(self, op: str, **params: Any) -> None:
        """Add a processing step (corrections go before the display steps)."""
        from vexel.packs.sem.process import OPS, step

        self.tabs.setCurrentIndex(TAB_PROCESSING)
        new = step(op, **params)
        steps = self.ms.processing.steps if self.ms is not None else []
        index = len(steps)
        if OPS[op].role == "correct":
            index = next((i for i, s in enumerate(steps) if OPS.get(s.op) and OPS[s.op].role == "display"), len(steps))
        self._edit_processing(lambda p: p.steps.insert(index, new), select=index)
        if OPS[op].caution:
            self.statusBar().showMessage(f"{OPS[op].label}：{OPS[op].caution}")

    def remove_step(self, index: int) -> None:
        if self.ms is None or not 0 <= index < len(self.ms.processing.steps):
            return
        self._edit_processing(lambda p: p.steps.pop(index), select=min(index, len(self.ms.processing.steps) - 2))

    def move_step(self, index: int, delta: int) -> None:
        from vexel.packs.sem.process import OPS

        if self.ms is None:
            return
        steps = self.ms.processing.steps
        j = index + delta
        if not (0 <= index < len(steps) and 0 <= j < len(steps)):
            return
        if OPS[steps[index].op].role != OPS[steps[j].op].role:
            self.statusBar().showMessage("“仅显示”的步骤总在校正步骤之后运行")
            return

        def swap(p: Any) -> None:
            p.steps[index], p.steps[j] = p.steps[j], p.steps[index]

        self._edit_processing(swap, select=j)

    def set_step_params(self, index: int, **params: Any) -> None:
        from vexel.packs.sem.process import OPS

        if self.ms is None or not 0 <= index < len(self.ms.processing.steps):
            return
        st = self.ms.processing.steps[index]
        merged = {**st.params, **params}
        OPS[st.op].resolve(merged)
        if merged != st.params:
            self._edit_processing(lambda p: setattr(p.steps[index], "params", merged))

    def set_saturation(self, **kw: Any) -> None:
        if self.ms is None or self._filling_steps:
            return
        sat = self.ms.processing.saturation
        if any(getattr(sat, k) != v for k, v in kw.items()):
            self._edit_processing(lambda p: [setattr(p.saturation, k, v) for k, v in kw.items()])

    def _step_toggled(self, it: QListWidgetItem) -> None:
        if self._filling_steps or self.ms is None:
            return
        i = self.steps.row(it)
        on = it.checkState() == Qt.CheckState.Checked
        if 0 <= i < len(self.ms.processing.steps) and self.ms.processing.steps[i].enabled != on:
            self._edit_processing(lambda p: setattr(p.steps[i], "enabled", on))

    def _show_params(self) -> None:
        from vexel.packs.sem.process import OPS

        if self._filling_steps:
            return
        row = self.steps.currentRow()
        steps = self.ms.processing.steps if self.ms is not None else []
        key = (row, steps[row].op, json.dumps(steps[row].params, sort_keys=True)) if 0 <= row < len(steps) else (row,)
        if key == self._params_key and (self._param_widgets or len(key) == 1):
            return  # the form already shows this step (and is perhaps being edited)
        self._params_key = key
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets = {}
        if self.ms is None or not 0 <= row < len(self.ms.processing.steps):
            self.step_help.setText("")
            self._set_buttons()
            return
        st = self.ms.processing.steps[row]
        op = OPS.get(st.op)
        if op is None:
            self.step_help.setText(f"未知的处理步骤：{st.op}")
            return
        vals = op.resolve(st.params)
        for prm in op.params:
            if prm.kind == "choice":
                w: Any = QComboBox()
                for value, label in prm.choices:
                    w.addItem(label, value)
                w.setCurrentIndex(max(0, [c[0] for c in prm.choices].index(vals[prm.name])))
                w.currentIndexChanged.connect(lambda _i: self._param_timer.start())
            elif prm.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(vals[prm.name]))
                w.toggled.connect(lambda _b: self._param_timer.start())
            elif prm.kind == "int":
                w = QSpinBox()
                w.setRange(int(prm.lo), int(min(prm.hi, 2**31 - 1)))
                w.setValue(int(vals[prm.name]))
                w.valueChanged.connect(lambda _x: self._param_timer.start())
            else:
                w = QDoubleSpinBox()
                w.setDecimals(3 if prm.step < 0.1 else 2 if prm.step < 1 else 1)
                w.setRange(prm.lo, prm.hi)
                w.setSingleStep(prm.step)
                w.setValue(float(vals[prm.name]))
                w.valueChanged.connect(lambda _x: self._param_timer.start())
            if prm.help:
                w.setToolTip(prm.help)
            self._param_widgets[prm.name] = (prm, w)
            self.params_form.addRow(prm.label, w)
        self.step_help.setText(op.help + (f"。注意：{op.caution}" if op.caution else ""))
        self._set_buttons()

    def _apply_params(self) -> None:
        row = self.steps.currentRow()
        if self.ms is None or not 0 <= row < len(self.ms.processing.steps) or not self._param_widgets:
            return
        params: dict[str, Any] = {}
        for name, (prm, w) in self._param_widgets.items():
            if prm.kind == "choice":
                params[name] = w.currentData()
            elif prm.kind == "bool":
                params[name] = w.isChecked()
            else:
                params[name] = w.value()
        st = self.ms.processing.steps[row]
        self._params_key = (row, st.op, json.dumps({**st.params, **params}, sort_keys=True))  # keep the form being edited
        self.set_step_params(row, **params)

    def _choose_frames(self) -> None:
        if self.path is None:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "选择同一位置的其他帧（本图为第 1 帧）", str(self.path.parent), "图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        paths = [p for p in paths if Path(p) != self.path]
        if not paths:
            return
        out, _ = QFileDialog.getSaveFileName(self, "保存平均后的图", str(self.path.with_name(f"{self.path.stem}_avg{len(paths) + 1}.tif")), "TIFF (*.tif)")
        if out:
            self.average_with(paths, out)

    def average_with(self, others: list[str | Path], out: str | Path) -> dict[str, Any] | None:
        """Average this image with ``others`` (same place) into ``out`` and open the result."""
        from vexel.packs.sem.process import average_files

        if self.path is None:
            return None
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            info = average_files([self.path, *others], out, use_ocr=False)
        except (ValueError, OSError) as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "无法平均", str(e))
            return None
        QGuiApplication.restoreOverrideCursor()
        drift = max(float(np.hypot(*s)) for s in info["shifts"])
        self.open_path(out, wait=True)
        self.statusBar().showMessage(f"{info['frames']} 帧平均：噪声 {info['noise_single']:.3g} → {info['noise_mean']:.3g}；最大漂移 {drift:.2f} 像素")
        return info

    def _export_processed(self) -> None:
        if self.path is None:
            return
        out, _ = QFileDialog.getSaveFileName(self, "导出处理后的图", str(self.path.with_name(self.path.stem + "_processed.tif")), "TIFF (*.tif)")
        if out:
            self.export_processed(out)

    def export_processed(self, path: str | Path) -> Path | None:
        from vexel.packs.sem.process import describe, write_derived

        res = self._processed(block=True)
        if res is None or self.path is None:
            return None
        out = write_derived(path, res.analysis, self.cal, {"operation": "图像处理", "source": self.path.name, "steps": describe(self.ms.processing)})
        self.statusBar().showMessage(f"已导出：{Path(out).name}")
        return Path(out)


__all__ = ["SemWindow", "json"]
