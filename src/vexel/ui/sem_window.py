"""SEM module window (in development: roadmap 0.0.5).

M1: open an image, calibrate it (instrument metadata, the data bar's scale bar, or by
hand), confirm. M2: measure on it - length, polyline, area (polygon, freehand, magic
wand), angle, counting, circle and ellipse fits, line profiles with edges, widths and
pitch - with a table of results, undo and redo, and exports (CSV, Excel, annotated PNG
and SVG, and the measurement record itself).

Measurements are kept per image (by its SHA-256) and saved after every change, so
reopening the image brings them back. Values are always recomputed from the drawn
geometry and the calibration in force.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtGui import QAction, QActionGroup, QKeySequence
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
    QScrollArea,
    QSpinBox,
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        panel = QFrame()
        panel.setObjectName("sidebar")
        scroll.setWidget(panel)
        v = QVBoxLayout(panel)
        self._build_calibration(v)
        self._build_measurements(v)
        root.addWidget(scroll)
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
        self._undo, self._redo = [], []
        self.view.set_image(img)
        self.file_label.setText(f"{path.name}：{img.shape[1]}×{img.shape[0]} 像素")
        self.pixel.setText("…")
        self.state.setText("正在读取元数据与信息栏…")
        self.detail.setText("")
        self.notes.hide()
        self.meta.setText("")
        self._refresh()
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

            compute(item, self.ms.scale, self.image)
        except ValueError as e:
            self._pop_undo()
            self.statusBar().showMessage(f"无法测量：{e}")
            return None
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
        values = compute_all(self.ms, self.image)
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
        if suffix == ".xlsx":
            out = [measure_io.export_xlsx(self.ms, self.image, path)]
        elif suffix == ".csv":
            out = measure_io.export_csv(self.ms, self.image, path)
        elif suffix == ".png":
            out = [measure_io.export_png(self.image, self.ms, path, crop_rows=crop)]
        elif suffix == ".svg":
            out = [measure_io.export_svg(self.image, self.ms, path, crop_rows=crop)]
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


__all__ = ["SemWindow", "json"]
