"""SEM module window (in development: roadmap 0.0.5).

M1: open an image, calibrate it (instrument metadata, the data bar's scale bar, or by
hand), confirm. M2: measure on it - length, polyline, area (polygon, freehand, magic
wand), angle, counting, circle and ellipse fits, line profiles with edges, widths and
pitch - with a table of results, undo and redo, and exports (CSV, Excel, annotated PNG
and SVG, and the measurement record itself). M3: image processing - line levelling,
Fourier destriping, background levelling, scan-line alignment, denoising, contrast and
sharpening - with a before/after view, the saturated-area mask, and multi-frame
averaging. M4: automatic measurements on a region - line/space CD and roughness, contact
holes, particles, cross-section layers and profiles, tips.

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
    QLineEdit,
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

from vexel.i18n import tr
from vexel.ui.sem_canvas import TOOL_HINTS, ImageView, ProfilePlot
from vexel.ui.widgets import heading, muted

VENDORS = {"thermo_fisher": tr("Thermo Fisher（FEI）"), "zeiss": "Zeiss", "tescan": "TESCAN", "hitachi": "Hitachi", "jeol": "JEOL", "imagej": tr("ImageJ 标定")}
SOURCES = {"metadata": tr("仪器元数据"), "databar": tr("信息栏比例尺识别"), "manual": tr("手动"), "none": tr("无")}
STATE_COLOURS = {"confirmed": "#16a34a", "pending": "#d97706", "none": "#dc2626"}
TOOLS = [("pan", tr("选择"), "V"), ("line", tr("长度"), "L"), ("polyline", tr("折线"), "Shift+L"), ("polygon", tr("面积"), "A"), ("freehand", tr("自由画"), "F"),
         ("wand", tr("魔棒"), "W"), ("angle", tr("角度"), "G"), ("point", tr("计数"), "N"), ("circle", tr("圆"), "C"), ("ellipse", tr("椭圆"), "E"),
         ("profile", tr("线剖面"), "P")]  # fmt: skip
PREVIEW_RECIPES = ("nested", "traces")  # listed only with VEXEL_PREVIEW=1 (0.8 feature audit)
SYNTHETIC_ONLY = ("lines", "holes", "layers", "profile", "tip", "pores", "fibres")  # accepted on synthetic images only so far
CORE_TOOLS = ("pan", "line", "polygon", "point")  # on the toolbar; the others under "更多工具"
EDGE_METHODS = [("threshold", tr("阈值（50% 等）")), ("gradient", tr("最大梯度")), ("peak", tr("亮边峰值"))]
TAB_CALIBRATION, TAB_PROCESSING, TAB_MEASURE, TAB_AUTO, TAB_3D = range(5)


def _hide_when_empty(table: Any) -> None:
    """A results table shows only when it has rows (no empty boxes on the first screen)."""
    import shiboken6

    model = table.model()

    def update(*_a: Any) -> None:
        if shiboken6.isValid(table):  # the model still signals while the window is being destroyed
            table.setVisible(table.rowCount() > 0)

    for sig in (model.rowsInserted, model.rowsRemoved, model.modelReset):
        sig.connect(update)
    update()


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


def _param_widget(prm: Any, value: Any) -> Any:
    """An input for one recipe or processing setting."""
    if prm.kind == "choice":
        w: Any = QComboBox()
        w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(6)
        for v, label in prm.choices:
            w.addItem(tr(label), v)
        w.setCurrentIndex(max(0, [c[0] for c in prm.choices].index(value)))
    elif prm.kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(value))
    elif prm.kind == "int":
        w = QSpinBox()
        w.setRange(int(prm.lo), int(min(prm.hi, 2**31 - 1)))
        w.setValue(int(value))
    else:
        w = QDoubleSpinBox()
        w.setDecimals(3 if prm.step < 0.1 else 2 if prm.step < 1 else 1)
        w.setRange(prm.lo, prm.hi)
        w.setSingleStep(prm.step)
        w.setValue(float(value))
    if tr(prm.help):
        w.setToolTip(tr(prm.help))
    return w


def _param_value(prm: Any, w: Any) -> Any:
    if prm.kind == "choice":
        return w.currentData()
    if prm.kind == "bool":
        return w.isChecked()
    return w.value()


def plan_preview(plan: Any, ms: Any, image: Any) -> Any:
    """The plan's measurement as it would be on this image (on a copy of the record), for the
    card and the picture before the person decides; None when the plan has no measurement."""
    from vexel.packs.sem import measure_io

    if plan is None or plan.recipe is None or ms is None:
        return None
    m = ms.model_copy(deep=True)
    try:
        spec = plan.apply(m)
        return next((a for a in measure_io.analyses_of(m, image) if a.spec.id == spec.id), None)
    except (ValueError, OSError):
        return None


class _PlanJob(QObject):
    """Plans an SEM measurement away from the window (a language-model tier may take a while)
    and computes its preview."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, kwargs: dict[str, Any], ms: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.ms = ms

    def run(self) -> None:
        from vexel.packs.sem.plan import make_plan

        try:
            plan = make_plan(**self.kwargs)
            self.done.emit((plan, plan_preview(plan, self.ms, self.kwargs["image"])))
        except Exception as e:  # noqa: BLE001 - shown to the person
            self.failed.emit(f"{type(e).__name__}: {e}")


class _TiltJob(QObject):
    """Reconstructs a height map away from the window (several seconds per pair)."""

    done = Signal(object, str)
    failed = Signal(str, str)

    def __init__(self, spec: Any, path: Path, key: str) -> None:
        super().__init__()
        self.spec, self.path, self.key = spec, path, key

    def run(self) -> None:
        from vexel.packs.sem.tilt3d import run_spec

        try:
            self.done.emit(run_spec(self.spec, self.path), self.key)
        except Exception as e:  # noqa: BLE001 - shown to the person
            self.failed.emit(f"{type(e).__name__}: {e}", self.key)


class _AutoJob(QObject):
    """Automatic measurements of a large image, off the window's thread."""

    done = Signal(object)  # {cache key: Analysis}

    def __init__(self, todo: list[tuple[str, Any]], image: Any, scale: Any, mask: Any, crop: Any) -> None:
        super().__init__()
        self.todo, self.image, self.scale, self.mask, self.crop = todo, image, scale, mask, crop

    def run(self) -> None:
        from vexel.packs.sem.analysis import Analysis, run

        out = {}
        for k, spec in self.todo:
            try:
                out[k] = run(spec, self.image, self.scale, self.mask, self.crop)
            except Exception as e:  # noqa: BLE001 - shown to the person in the list
                out[k] = Analysis(spec=spec, error=f"{type(e).__name__}: {e}")
        self.done.emit(out)


BACKGROUND_PIXELS = 4_000_000  # automatic measurements of larger images run off the window's thread


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


def _config() -> dict[str, Any]:
    from vexel.i18n import _config_path

    try:
        return json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _set_config(key: str, value: Any) -> None:
    from vexel.i18n import _config_path

    cfg = _config()
    cfg[key] = value
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


class SemWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("Vexel · SEM（开发中）"))
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
        self.analyses: list[Any] = []  # analysis.Analysis for each spec of the record
        self._auto_key: str | None = None
        self._pending_region: list[float] | None = None
        self._auto_widgets: dict[str, Any] = {}
        self.tilt_result: Any = None  # tilt3d.Tilt3DResult of the record's spec
        self._tilt_key: str | None = None
        self._tilt_thread: QThread | None = None
        self._tilt_job: _TiltJob | None = None
        self._box_first: list[float] | None = None
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
        from vexel.i18n import language
        from vexel.ui.sem_cards import CalibrationBanner

        side = QWidget()
        side.setFixedWidth(390 if language() == "zh" else 450)  # English tab names are longer
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        self.banner = CalibrationBanner()
        self.banner.confirm.connect(self._confirm)
        self.banner.open_calibration.connect(lambda: self.tabs.setCurrentIndex(TAB_CALIBRATION))
        sv.addWidget(self.banner)
        self.sat_badge = QLabel("")
        self.sat_badge.setObjectName("notes")
        self.sat_badge.setWordWrap(True)
        self.sat_badge.linkActivated.connect(lambda _l: self.sat_show.setChecked(True))
        self.sat_badge.hide()
        sv.addWidget(self.sat_badge)
        self.start_panel = QFrame()
        spv = QVBoxLayout(self.start_panel)
        spv.setContentsMargins(16, 24, 16, 16)
        spv.addWidget(heading(tr("开始")))
        spv.addWidget(muted(tr("打开一张 SEM 图片：标定会自动完成（仪器元数据或信息栏的比例尺），然后自动规划测量。")))
        self.start_btn = QPushButton(tr("打开 SEM 图片…"))
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._choose)
        spv.addWidget(self.start_btn)
        spv.addStretch(1)
        sv.addWidget(self.start_panel, 1)
        sv.addWidget(self.tabs, 1)
        for title, build in ((tr("标定"), self._build_calibration), (tr("图像处理"), self._build_processing), (tr("测量"), self._build_measurements),
                             (tr("自动测量"), self._build_auto), (tr("三维"), self._build_3d)):  # fmt: skip
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # the panel wraps instead
            panel = QFrame()
            panel.setObjectName("sidebar")
            scroll.setWidget(panel)
            v = QVBoxLayout(panel)
            build(v)
            v.addStretch(1)
            self.tabs.addTab(scroll, title)
        self.view_menu.addSeparator()
        self.autoplan_act = self.view_menu.addAction(tr("打开图片时自动规划"))
        self.autoplan_act.setCheckable(True)
        self.autoplan_act.setChecked(self.autoplan_on.isChecked())
        self.autoplan_act.toggled.connect(self.autoplan_on.setChecked)
        self.autoplan_on.toggled.connect(self.autoplan_act.setChecked)
        self._update_tabs()
        for box in self.tabs.findChildren(QComboBox):  # long choices must not widen the panel
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(6)
        root.addWidget(side)
        self.setCentralWidget(central)
        self._set_buttons()
        self.say(tr("打开一张 SEM 图片开始"))

    # -- layout ---------------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """A short toolbar - open, the four everyday tools, the others under "更多工具", undo,
        export - and menus for everything used now and then (0.8: simple first screen)."""
        from qtpy.QtWidgets import QMenu, QToolButton

        bar = QToolBar(tr("测量工具"))
        bar.setMovable(False)
        self.addToolBar(bar)
        self.open_act = bar.addAction(tr("打开…"), self._choose)
        self.open_act.setShortcut(QKeySequence.StandardKey.Open)
        bar.addSeparator()
        self.tool_actions: dict[str, QAction] = {}
        group = QActionGroup(self)
        for tool, label, key in TOOLS:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setShortcut(QKeySequence(key))
            act.setToolTip(tr(f"{label}（{key}）：{TOOL_HINTS[tool]}"))
            act.triggered.connect(lambda _c=False, t=tool: self.set_tool(t))
            group.addAction(act)
            self.tool_actions[tool] = act
            if tool in CORE_TOOLS:
                bar.addAction(act)
        self.more_menu = QMenu(tr("更多工具"), self)
        for tool, _label, _key in TOOLS:
            if tool not in CORE_TOOLS:
                self.more_menu.addAction(self.tool_actions[tool])
                self.addAction(self.tool_actions[tool])  # shortcuts work while the menu is closed
        self.more_btn = QToolButton()
        self.more_btn.setText(tr("更多工具"))
        self.more_btn.setMenu(self.more_menu)
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_btn.setToolTip(tr("折线、自由画、魔棒、角度、圆、椭圆、线剖面"))
        bar.addWidget(self.more_btn)
        group.triggered.connect(self._tool_chosen)
        self.tool_actions["pan"].setChecked(True)
        bar.addSeparator()
        self.undo_act = bar.addAction(tr("撤销"), self.undo)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self.export_act = bar.addAction(tr("导出…"), self._export)
        self.redo_act = QAction(tr("重做"), self)
        self.redo_act.triggered.connect(self.redo)
        self.redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self.fit_act = QAction(tr("适应窗口"), self)
        self.fit_act.triggered.connect(self.view.fit)
        self.fit_act.setShortcut(QKeySequence("Ctrl+0"))
        self.load_act = QAction(tr("载入测量记录…"), self)
        self.load_act.triggered.connect(self._load_record_file)
        self.bug_act = QAction(tr("诊断包…"), self)
        self.bug_act.triggered.connect(self._bugreport)
        self.bug_act.setToolTip(tr("报告问题用：版本、系统、这张图的标定与测量记录；不含图像，也不含信息栏上的文字"))
        from vexel.i18n import language

        other = "en" if language() == "zh" else "zh"
        self.lang_act = QAction("English" if other == "en" else "中文", self)
        self.lang_act.triggered.connect(lambda: self.switch_language(other))
        self.lang_act.setToolTip("Show the SEM window in English (when it is next opened)" if other == "en" else "下次打开 SEM 窗口时使用中文界面")
        self.processing_act = QAction(tr("图像处理"), self)
        self.processing_act.setCheckable(True)
        self.processing_act.setToolTip(tr("校平、去条纹、背景、去噪、对比度、饱和区与多帧平均"))
        self.processing_act.toggled.connect(lambda _on: self._update_tabs(show_processing=_on))
        menus = self.menuBar()
        m = menus.addMenu(tr("文件"))
        m.addAction(self.open_act)
        m.addAction(self.load_act)
        m.addAction(self.export_act)
        m = menus.addMenu(tr("编辑"))
        m.addAction(self.undo_act)
        m.addAction(self.redo_act)
        m = menus.addMenu(tr("视图"))
        self.view_menu = m
        m.addAction(self.fit_act)
        m.addAction(self.processing_act)
        m = menus.addMenu(tr("帮助"))
        m.addAction(self.bug_act)
        m.addAction(self.lang_act)
        self.addAction(self.redo_act)
        self.addAction(self.fit_act)

    def _tool_chosen(self, act: QAction) -> None:
        """The "更多工具" button names the tool in use when it is one of its tools."""
        tool = next((t for t, a in self.tool_actions.items() if a is act), None)
        self.more_btn.setText(tr("更多工具") if tool in CORE_TOOLS or tool is None else f"{act.text()} ▾")
        self.more_btn.setDown(False)

    def _build_calibration(self, v: QVBoxLayout) -> None:
        self.file_label = muted("")
        v.addWidget(self.file_label)
        v.addWidget(heading(tr("标定")))
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
        self.confirm_btn = QPushButton(tr("确认标定"))
        self.confirm_btn.setObjectName("primary")
        self.confirm_btn.clicked.connect(self._confirm)
        v.addWidget(self.confirm_btn)
        row = QHBoxLayout()
        self.pick_btn = QPushButton(tr("点比例尺两端…"))
        self.pick_btn.setToolTip(tr("在图上依次点击比例尺的两端，再输入它代表的长度"))
        self.pick_btn.clicked.connect(lambda: self.set_tool("calibrate"))
        self.size_btn = QPushButton(tr("输入像素尺寸…"))
        self.size_btn.clicked.connect(self._enter_size)
        row.addWidget(self.pick_btn)
        row.addWidget(self.size_btn)
        v.addLayout(row)
        self.cal_change_btn = QPushButton(tr("更改标定…"))
        self.cal_change_btn.setFlat(True)
        self.cal_change_btn.setStyleSheet("text-align: left; padding: 2px 0;")
        self.cal_change_btn.clicked.connect(lambda: (setattr(self, "_cal_edit", True), self._set_buttons()))
        v.addWidget(self.cal_change_btn)
        self._cal_edit = False
        self.tilt_on = QCheckBox(tr("样品台倾斜校正"))
        self.tilt_on.setToolTip(tr("样品台倾斜时，垂直方向的距离被缩短；勾选后按 1/cos(倾角) 校正（倾转轴为图像水平方向）"))
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
        trow.addStretch(1)
        self.tilt_row = QWidget()
        self.tilt_row.setLayout(trow)
        v.addWidget(self.tilt_row)
        self.meta = muted("")
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.meta)

    def _build_measurements(self, v: QVBoxLayout) -> None:
        v.addWidget(heading(tr("测量")))
        form = QFormLayout()
        self.category = QComboBox()
        self.category.setEditable(True)
        self.category.addItems(["", tr("颗粒"), tr("缺陷"), tr("孔")])
        self.category.setToolTip(tr("计数工具：这一组标记属于哪一类"))
        form.addRow(tr("计数类别"), self.category)
        self.profile_width = QSpinBox()
        self.profile_width.setRange(1, 101)
        self.profile_width.setValue(5)
        self.profile_width.setSuffix(tr(" 像素"))
        self.profile_width.setToolTip(tr("线剖面在垂直方向上平均的宽度：越宽越抗噪，但会把倾斜的边缘抹宽"))
        form.addRow(tr("剖面宽度"), self.profile_width)
        self.edge_method = QComboBox()
        for key, label in EDGE_METHODS:
            self.edge_method.addItem(label, key)
        self.edge_method.setToolTip(tr("边缘的定义不同，测得的线宽也不同；结果中总会写明所用定义"))
        form.addRow(tr("边缘定义"), self.edge_method)
        self.edge_level = QSpinBox()
        self.edge_level.setRange(5, 95)
        self.edge_level.setValue(50)
        self.edge_level.setSuffix(" %")
        form.addRow(tr("阈值"), self.edge_level)
        for w in (self.edge_method, self.edge_level, self.profile_width):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(lambda _x: self._profile_settings_changed())
        v.addLayout(form)
        self.meas_form = form
        self.tool_hint = muted(tr("用上方工具栏在图上测量；结果列在下面。"))
        v.addWidget(self.tool_hint)
        self._update_tool_settings("pan")

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", tr("工具"), tr("结果"), tr("名称")])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        hh.resizeSection(3, 90)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.itemChanged.connect(self._name_edited)
        v.addWidget(self.table, 1)
        _hide_when_empty(self.table)
        brow = QHBoxLayout()
        self.delete_btn = QPushButton(tr("删除所选"))
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

        v.addWidget(heading(tr("图像处理")))
        v.addWidget(muted(tr("校正步骤的结果用于测量；“仅显示”的步骤只改变显示和导出的图。只处理信息栏以上的部分。")))
        row = QHBoxLayout()
        self.op_choice = QComboBox()
        for name, op in OPS.items():
            self.op_choice.addItem(tr(op.label) + (tr("（仅显示）") if op.role == "display" else ""), name)
            self.op_choice.setItemData(self.op_choice.count() - 1, tr(op.help), Qt.ItemDataRole.ToolTipRole)
        self.add_step_btn = QPushButton(tr("添加"))
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
        self.up_btn = QPushButton(tr("上移"))
        self.up_btn.clicked.connect(lambda: self.move_step(self.steps.currentRow(), -1))
        self.down_btn = QPushButton(tr("下移"))
        self.down_btn.clicked.connect(lambda: self.move_step(self.steps.currentRow(), 1))
        self.remove_step_btn = QPushButton(tr("删除"))
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

        v.addWidget(heading(tr("饱和区")))
        self.sat_on = QCheckBox(tr("标记饱和区（魔棒和线剖面不使用）"))
        self.sat_on.setToolTip(tr("严重荷电或探测器饱和处的信息已经丢失，只能标记，不能“修复”"))
        self.sat_on.toggled.connect(lambda on: self.set_saturation(enabled=on))
        v.addWidget(self.sat_on)
        self.sat_show = QCheckBox(tr("在图上显示（红色）"))
        self.sat_show.setChecked(False)  # a warning above the panel offers to show it
        self.sat_show.toggled.connect(lambda _on: (self._show_mask(), self._show_sat_badge()))
        v.addWidget(self.sat_show)
        self.sat_info = muted("")
        v.addWidget(self.sat_info)
        self.proc_notes = QLabel("")
        self.proc_notes.setObjectName("notes")
        self.proc_notes.setWordWrap(True)
        self.proc_notes.hide()
        v.addWidget(self.proc_notes)

        v.addWidget(heading(tr("前后对比")))
        crow = QHBoxLayout()
        self.compare_on = QCheckBox(tr("左侧显示原图"))
        self.compare_on.toggled.connect(lambda _on: self._apply_split())
        self.compare_pos = QSlider(Qt.Orientation.Horizontal)
        self.compare_pos.setRange(0, 100)
        self.compare_pos.setValue(50)
        self.compare_pos.valueChanged.connect(lambda _v: self._apply_split())
        crow.addWidget(self.compare_on)
        crow.addWidget(self.compare_pos, 1)
        v.addLayout(crow)

        v.addWidget(heading(tr("多帧与导出")))
        orow = QHBoxLayout()
        self.average_btn = QPushButton(tr("多帧平均…"))
        self.average_btn.setToolTip(tr("选择同一位置的其他帧：对齐（亚像素）后平均，结果另存为新图并沿用本图的标定"))
        self.average_btn.clicked.connect(self._choose_frames)
        self.export_proc_btn = QPushButton(tr("导出处理后的图…"))
        self.export_proc_btn.setToolTip(tr("校正后的图（32 位浮点 TIFF，信息栏保留），在 Vexel 中打开时沿用本图的标定"))
        self.export_proc_btn.clicked.connect(self._export_processed)
        orow.addWidget(self.average_btn)
        orow.addWidget(self.export_proc_btn)
        v.addLayout(orow)
        v.addWidget(muted(tr("荷电最有效的办法在采集端（降低电压、缩短驻留时间、交错扫描、镀膜）；软件处理只是补救。")))

    def _build_auto(self, v: QVBoxLayout) -> None:
        from vexel.packs.sem.analysis import RECIPES

        v.addWidget(heading(tr("自动测量")))
        prow = QHBoxLayout()
        self.goal = QLineEdit()
        self.goal.setPlaceholderText(tr("目标（可不填）：例如 线宽和粗糙度、孔径、膜厚"))
        self.plan_btn = QPushButton(tr("自动规划"))
        self.plan_btn.setObjectName("primary")
        self.plan_btn.setToolTip(tr("识别图像类型，选择测量方法、设置、区域和图像处理，先试运行；你确认后才加入记录"))
        self.plan_btn.clicked.connect(lambda: self.auto_plan())
        prow.addWidget(self.goal, 1)
        prow.addWidget(self.plan_btn)
        v.addLayout(prow)
        self.goal.returnPressed.connect(lambda: self.auto_plan())
        self.autoplan_on = QCheckBox(tr("打开图片时自动规划"))
        self.autoplan_on.setToolTip(tr("图片还没有自动测量时，打开后立即识别、规划并试运行；结果先给你看，采用后才加入记录"))
        self.autoplan_on.setChecked(bool(_config().get("sem_autoplan", True)))
        self.autoplan_on.toggled.connect(lambda on: _set_config("sem_autoplan", bool(on)))
        self.autoplan_on.hide()  # a preference: 视图 → 打开图片时自动规划
        v.addWidget(self.autoplan_on)
        from vexel.ui.sem_cards import PlanCard

        self.plan_card = PlanCard()
        self.plan_card.accepted.connect(lambda: self._decide_plan("accept"))
        self.plan_card.adjust.connect(lambda: self._decide_plan("adjust"))
        self.plan_card.dismissed.connect(lambda: self._decide_plan("dismiss"))
        v.addWidget(self.plan_card)
        v.addWidget(muted(tr("或手动选择：")))
        self.recipe = QComboBox()
        from vexel.ui.launcher import preview_enabled

        for name, rec in RECIPES.items():
            if name in PREVIEW_RECIPES and not preview_enabled():
                continue  # 0.8: new, not yet checked on real images (VEXEL_PREVIEW=1 shows them)
            self.recipe.addItem(tr(rec.label), name)
        self.recipe.currentIndexChanged.connect(lambda _i: self._show_recipe())
        v.addWidget(self.recipe)
        self.recipe_help = muted("")
        v.addWidget(self.recipe_help)
        rrow = QHBoxLayout()
        self.region_btn = QPushButton(tr("框选区域"))
        self.region_btn.setToolTip(tr("在图上点击区域的两个对角"))
        self.region_btn.clicked.connect(lambda: self.set_tool("region"))
        self.whole_btn = QPushButton(tr("整张图"))
        self.whole_btn.clicked.connect(lambda: self.set_region(None))
        rrow.addWidget(self.region_btn)
        rrow.addWidget(self.whole_btn)
        v.addLayout(rrow)
        self.region_label = muted(tr("区域：整张图（信息栏以上）"))
        v.addWidget(self.region_label)
        self.advanced_btn = QPushButton("")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setFlat(True)
        self.advanced_btn.setStyleSheet("text-align: left; padding: 2px 0;")
        self.advanced_btn.setToolTip(tr("每种方法的全部设置；一般不用改。选中一项自动测量时，改动会立即重算它"))
        self.advanced_btn.toggled.connect(lambda _on: self._mark_params())
        v.addWidget(self.advanced_btn)
        self.auto_box = QWidget()
        box_v = QVBoxLayout(self.auto_box)
        box_v.setContentsMargins(0, 0, 0, 0)
        self.auto_form = QFormLayout()
        self.auto_form.setContentsMargins(0, 0, 0, 0)
        self.auto_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        box_v.addLayout(self.auto_form)
        self.reset_params_btn = QPushButton(tr("恢复默认设置"))
        self.reset_params_btn.clicked.connect(self.reset_params)
        box_v.addWidget(self.reset_params_btn)
        self.auto_box.hide()
        v.addWidget(self.auto_box)
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(400)
        self._live_timer.timeout.connect(self._live_apply)
        arow = QHBoxLayout()
        self.run_btn = QPushButton(tr("运行"))
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(lambda: self.run_analysis())
        self.rerun_btn = QPushButton(tr("用当前设置重算所选"))
        self.rerun_btn.clicked.connect(self._rerun_selected)
        arow.addWidget(self.run_btn)
        arow.addWidget(self.rerun_btn)
        v.addLayout(arow)
        self.auto_list = QListWidget()
        self.auto_list.setMaximumHeight(110)
        self.auto_list.currentRowChanged.connect(lambda _r: self._show_auto_detail())
        v.addWidget(self.auto_list)
        drow = QHBoxLayout()
        self.auto_delete = QPushButton(tr("删除所选自动测量"))
        self.auto_delete.clicked.connect(self._delete_analysis)
        self.auto_show = QCheckBox(tr("在图上显示"))
        self.auto_show.setChecked(True)
        self.auto_show.setToolTip(tr("自动测量的轮廓与边缘画在图上；手动测量时可以先隐藏"))
        self.auto_show.toggled.connect(self.view.set_show_auto)
        drow.addWidget(self.auto_delete, 1)
        drow.addWidget(self.auto_show)
        v.addLayout(drow)
        # correct by example
        xrow = QHBoxLayout()
        self.example_btn = QPushButton(tr("点选纠错"))
        self.example_btn.setCheckable(True)
        self.example_btn.setToolTip(tr("在图上点例子：左键点应该测到的对象，右键（或 Shift+左键）点不该测到的；Vexel 据此换设置重测，给出几个候选"))
        self.example_btn.toggled.connect(lambda on: self.set_tool("example" if on else "pan"))
        self.tune_btn = QPushButton(tr("按例子调整"))
        self.tune_btn.setObjectName("primary")
        self.tune_btn.clicked.connect(lambda: self.tune_examples())
        self.clear_examples_btn = QPushButton(tr("清除例子"))
        self.clear_examples_btn.clicked.connect(self.clear_examples)
        xrow.addWidget(self.example_btn)
        xrow.addWidget(self.tune_btn)
        xrow.addWidget(self.clear_examples_btn)
        v.addLayout(xrow)
        self.example_info = muted("")
        v.addWidget(self.example_info)
        self._examples: dict[str, list[list[float]]] = {"yes": [], "no": []}
        # saved methods
        mrow = QHBoxLayout()
        self.method_box = QComboBox()
        self.method_box.setToolTip(tr("保存过的方案：方法、设置、区域与图像处理"))
        self.method_box.currentIndexChanged.connect(lambda _i: self._method_selected())
        self.method_apply_btn = QPushButton(tr("应用方案"))
        self.method_apply_btn.clicked.connect(lambda: self.apply_method())
        self.method_save_btn = QPushButton(tr("存为方案…"))
        self.method_save_btn.setToolTip(tr("把所选自动测量（连同本图的图像处理）存为方案，以后可以一键用在别的图上"))
        self.method_save_btn.clicked.connect(self._save_method_dialog)
        mrow.addWidget(self.method_box, 1)
        mrow.addWidget(self.method_apply_btn)
        mrow.addWidget(self.method_save_btn)
        v.addLayout(mrow)
        self.method_auto = QCheckBox(tr("打开图片时自动套用这个方案"))
        self.method_auto.setToolTip(tr("还没有测量的图片打开后直接按这个方案测量（不再自动规划）"))
        self.method_auto.toggled.connect(self._method_auto_toggled)
        v.addWidget(self.method_auto)
        self.batch_btn = QPushButton(tr("批量测量…"))
        self.batch_btn.setToolTip(tr("用所选方案测量一批图片：每张用自己的标定；只把需要看的（出错、未确认标定、有提示、与其余相差很大）排在前面"))
        self.batch_btn.clicked.connect(self._batch_dialog)
        v.addWidget(self.batch_btn)
        self._fill_methods()
        self.auto_summary = QLabel("")
        self.auto_summary.setWordWrap(True)
        self.auto_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.auto_summary)
        self.auto_notes = QLabel("")
        self.auto_notes.setObjectName("notes")
        self.auto_notes.setWordWrap(True)
        self.auto_notes.hide()
        v.addWidget(self.auto_notes)
        self.auto_table = QTableWidget(0, 0)
        self.auto_table.verticalHeader().setVisible(False)
        self.auto_table.verticalHeader().setDefaultSectionSize(20)
        self.auto_table.setMinimumHeight(180)
        self.auto_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        v.addWidget(self.auto_table)
        _hide_when_empty(self.auto_table)
        from vexel.ui.sem_charts import DistributionChart

        self.auto_chart = DistributionChart()
        self.auto_chart.hide()
        v.addWidget(self.auto_chart)
        self.pool_btn = QPushButton(tr("多张图合并统计…"))
        self.pool_btn.setToolTip(tr("用所选自动测量的方法和设置（及本图的图像处理）测量其他图片，把所有对象合在一起统计；每张图用自己的标定"))
        self.pool_btn.clicked.connect(self._pool_dialog)
        v.addWidget(self.pool_btn)
        self._show_recipe()

    def _build_3d(self, v: QVBoxLayout) -> None:
        v.addWidget(heading(tr("倾转三维（2.5D 高度图）")))
        v.addWidget(muted(tr("同一位置在不同样品台倾角下拍的图：本图为参考图（高度图画在它上面），再添加其他倾角的图。")))
        self.tilt_table = QTableWidget(0, 2)
        self.tilt_table.setHorizontalHeaderLabels([tr("图像"), tr("倾角 (°)")])
        self.tilt_table.verticalHeader().setVisible(False)
        self.tilt_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tilt_table.setMaximumHeight(130)
        v.addWidget(self.tilt_table)
        row = QHBoxLayout()
        self.tilt_add_btn = QPushButton(tr("添加倾转图…"))
        self.tilt_add_btn.clicked.connect(self._choose_tilt_views)
        self.tilt_remove_btn = QPushButton(tr("移除所选"))
        self.tilt_remove_btn.clicked.connect(self._remove_tilt_view)
        row.addWidget(self.tilt_add_btn)
        row.addWidget(self.tilt_remove_btn)
        v.addLayout(row)
        form = QFormLayout()
        self.tilt_axis = QDoubleSpinBox()
        self.tilt_axis.setRange(-180, 180)
        self.tilt_axis.setDecimals(1)
        self.tilt_axis.setSuffix("°")
        self.tilt_axis.setToolTip(tr("倾转轴在图像中的方向（与水平方向的夹角）：通常为 0°（水平）；扫描旋转过时按实际方向填写"))
        form.addRow(tr("倾转轴方向"), self.tilt_axis)
        self.tilt_level = QComboBox()
        for k, label in ((1, tr("扣除平面")), (2, tr("扣除二次曲面")), (0, tr("不校平"))):
            self.tilt_level.addItem(label, k)
        form.addRow(tr("校平"), self.tilt_level)
        self.tilt_smooth = QDoubleSpinBox()
        self.tilt_smooth.setRange(0, 10)
        self.tilt_smooth.setValue(1.5)
        self.tilt_smooth.setSuffix(tr(" 像素"))
        self.tilt_smooth.setToolTip(tr("对高度做高斯平滑：降低噪声，细节变模糊"))
        form.addRow(tr("平滑"), self.tilt_smooth)
        self.tilt_conf = QDoubleSpinBox()
        self.tilt_conf.setRange(0, 1)
        self.tilt_conf.setSingleStep(0.05)
        self.tilt_conf.setValue(0.3)
        self.tilt_conf.setToolTip(tr("匹配置信度低于此值的像素标为不可靠（缺少纹理的平坦区域），不参与粗糙度和台阶高度"))
        form.addRow(tr("最低置信度"), self.tilt_conf)
        self.tilt_cutoff = QDoubleSpinBox()
        self.tilt_cutoff.setRange(0, 1e7)
        self.tilt_cutoff.setDecimals(1)
        self.tilt_cutoff.setSpecialValueText(tr("自动（剖面长度的 1/5）"))
        self.tilt_cutoff.setToolTip(tr("剖面粗糙度（ISO 21920）的截止波长 λc：比它长的起伏算作波纹度，从粗糙度中扣除"))
        self.tilt_cutoff.valueChanged.connect(self._tilt_cutoff_edited)
        form.addRow(tr("剖面 λc"), self.tilt_cutoff)
        v.addLayout(form)
        self.tilt_run_btn = QPushButton(tr("重建高度图"))
        self.tilt_run_btn.setObjectName("primary")
        self.tilt_run_btn.clicked.connect(lambda: self.run_tilt3d())
        v.addWidget(self.tilt_run_btn)
        srow = QHBoxLayout()
        srow.addWidget(QLabel(tr("显示")))
        self.tilt_show = QComboBox()
        for k, label in (("image", tr("原图")), ("height", tr("高度图（彩色）")), ("confidence", tr("置信度"))):
            self.tilt_show.addItem(label, k)
        self.tilt_show.currentIndexChanged.connect(lambda _i: self._show_tilt_picture())
        srow.addWidget(self.tilt_show, 1)
        self.view3d_btn = QPushButton(tr("三维预览…"))
        self.view3d_btn.clicked.connect(self._show_3d_preview)
        srow.addWidget(self.view3d_btn)
        v.addLayout(srow)
        mrow = QHBoxLayout()
        self.hprofile_btn = QPushButton(tr("画高度剖面"))
        self.hprofile_btn.clicked.connect(lambda: self.set_tool("hprofile"))
        self.step_btn = QPushButton(tr("测台阶高度"))
        self.step_btn.setToolTip(tr("先框选下平面，再框选上平面；按 ISO 5436-1 用两个平行平面拟合，不受整体倾斜影响"))
        self.step_btn.clicked.connect(lambda: self._start_step())
        mrow.addWidget(self.hprofile_btn)
        mrow.addWidget(self.step_btn)
        v.addLayout(mrow)
        self.tilt_info = QLabel("")
        self.tilt_info.setWordWrap(True)
        self.tilt_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.tilt_info)
        self.tilt_notes = QLabel("")
        self.tilt_notes.setObjectName("notes")
        self.tilt_notes.setWordWrap(True)
        self.tilt_notes.hide()
        v.addWidget(self.tilt_notes)
        self.height_plot = ProfilePlot()
        self.height_plot.hide()
        v.addWidget(self.height_plot)
        erow = QHBoxLayout()
        self.tilt_export_btn = QPushButton(tr("导出…"))
        self.tilt_export_btn.setToolTip(tr("高度图 TIFF（高度、置信度、不确定度）、STL / OBJ 网格、彩色高度图 PNG、三维预览 PNG、剖面 CSV"))
        self.tilt_export_btn.clicked.connect(self._export_tilt)
        self.tilt_clear_btn = QPushButton(tr("清除三维"))
        self.tilt_clear_btn.clicked.connect(self._clear_tilt)
        erow.addWidget(self.tilt_export_btn)
        erow.addWidget(self.tilt_clear_btn)
        v.addLayout(erow)

    # -- opening and calibrating --------------------------------------------------------

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("打开 SEM 图片"), "", tr("图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"))
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path, *, wait: bool = False, plan: bool | None = None) -> bool:
        """Open an image: calibrate it and load its record. ``plan``: recognise and plan it
        right away when the record has no automatic measurement (None: the panel's switch;
        with ``wait``, only when asked)."""
        self._cal_edit = False
        from vexel.packs.sem.calibrate import load_image

        self._plan_on_open = plan
        self.plan_card.hide()
        self._examples = {"yes": [], "no": []}
        self.example_info.setText("")
        if self.example_btn.isChecked():
            self.example_btn.setChecked(False)

        path = Path(path)
        try:
            img = load_image(path)
        except Exception as e:  # noqa: BLE001 - shown to the person
            why = str(e) if isinstance(e, ValueError) and str(e) else f"{type(e).__name__}: {e}"  # our refusals say it all
            self.file_label.setText(tr(f"无法读取：{why}"))
            self.say(tr(f"无法打开 {path.name}"))
            QMessageBox.warning(self, tr("无法打开"), tr(f"无法读取这个文件：\n{why}"))
            return False
        self.path, self.image, self.cal, self.ms, self.values = path, img, None, None, {}
        self.processor, self.result, self._result_key = None, None, None
        self.analyses, self._auto_key, self._pending_region = [], None, None
        self.tilt_result, self._tilt_key, self._box_first = None, None, None
        self._undo, self._redo = [], []
        self.view.set_image(img)
        self.file_label.setText(tr(f"{path.name}：{img.shape[1]}×{img.shape[0]} 像素"))
        self.pixel.setText("…")
        self.state.setText(tr("正在读取元数据与信息栏…"))
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
        self._job.failed.connect(self._calibration_failed)  # a method: runs on the window's thread, not the worker's
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
        self._end_plan()
        self._end_tilt()
        if self._proc_thread is not None:
            self._proc_thread.quit()
            self._proc_thread.wait(60000)
        self._end_auto()
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
            self.pixel.setText(tr(f"{format_length(cal.pixel_size)} / 像素"))
            key = "confirmed" if cal.confirmed else "pending"
            text = tr(f"来源：{SOURCES.get(cal.source, cal.source)}，") + (tr("已确认") if cal.confirmed else tr("待确认：请核对后点击“确认标定”"))
            if cal.confirmed and cal.confirmed_by == "person":
                text += tr(f"（{cal.confirmed_at[:10]}）")
        else:
            self.pixel.setText(tr("未标定"))
            key, text = "none", tr(cal.reason) + tr("。在未标定前，测量以像素为单位。")
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {STATE_COLOURS[key]}; font-weight: 600;")
        lines = []
        if cal.scale_bar:
            lines.append(tr(f"比例尺：{cal.scale_bar['length_px']:.1f} 像素") + (f" = {cal.label}" if cal.label else ""))
        for chk in cal.checks:
            name = {"scale bar": tr("信息栏比例尺"), "field width": tr("视场宽度"), "metadata": tr("仪器元数据")}.get(chk["check"], chk["check"])
            lines.append(tr(f"核对（{name}）：{format_length(chk['pixel_size'])}/像素，相差 {chk['difference'] * 100:+.2f}%") + ("" if chk["agree"] else tr("，不一致")))
        if cal.crop_rows:
            how = {"metadata": tr("元数据给出"), "detected": tr("自动识别"), "manual": tr("手动")}.get(cal.crop_source, cal.crop_source)
            lines.append(tr(f"分析区域：第 0–{cal.crop_rows - 1} 行；下方 {cal.image_shape[0] - cal.crop_rows} 行是信息栏（{how}）"))
        self.detail.setText("\n".join(lines))
        if cal.warnings:
            self.notes.setText("\n".join(f"• {tr(w)}" for w in cal.warnings))
            self.notes.show()
        else:
            self.notes.hide()
        md = cal.metadata or {}
        info = []
        if md:
            info.append(tr(f"厂商：{VENDORS.get(md.get('vendor'), md.get('vendor'))}") + (tr(f"（{md['instrument']}）") if md.get("instrument") else ""))
            for k, label, fmt in (("voltage", tr("加速电压"), lambda x: f"{x / 1e3:.3g} kV"), ("magnification", tr("放大倍数"), lambda x: f"{x:,.0f}×"),
                                  ("hfw", tr("视场宽度"), format_length), ("working_distance", tr("工作距离"), format_length),
                                  ("tilt", tr("样品台倾角"), lambda x: f"{x:.2f}°"), ("detector", tr("探测器"), str), ("date", tr("时间"), str)):  # fmt: skip
                if md.get(k) not in (None, ""):
                    info.append(tr(f"{label}：{fmt(md[k])}"))
        else:
            info.append(tr("文件中没有仪器元数据"))
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
        self.say(tr(cal.describe()))
        self._show_banner()
        want = self._plan_on_open
        if want is None:
            want = self.autoplan_on.isChecked() and not self._sync
        self._plan_on_open = False
        empty = not self.ms.analyses and not self.ms.items and self.ms.tilt3d is None
        from vexel.packs.sem import methods

        m = methods.auto() if empty else None
        if m is not None:
            self.apply_method(m.name, record=False)
            self.say(tr(f"已按方案“{m.name}”测量"))
        elif want and empty:
            self.auto_plan()

    _plan_on_open: bool | None = None

    def _show_banner(self) -> None:
        from vexel.packs.sem.calibrate import format_length

        cal = self.cal
        if cal is None:
            self.banner.hide()
            return
        if not cal.ok:
            self.banner.set_state("none", tr("未标定：长度以像素为单位"))
            return
        src = tr(SOURCES.get(cal.source, cal.source))
        size = tr(f"{format_length(cal.pixel_size)} / 像素")
        if cal.needs_confirmation:
            self.banner.set_state("pending", f"<b>{size}</b> · {src} · " + tr("待确认"))
        else:
            self.banner.set_state("confirmed", f"<b>{size}</b> · {src}")

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
        self.start_panel.setVisible(not has)
        self.tabs.setVisible(has)
        self.confirm_btn.setVisible(bool(self.cal is not None and self.cal.needs_confirmation))
        self.pick_btn.setEnabled(has)
        self.size_btn.setEnabled(has)
        settled = self.cal is not None and self.cal.ok and not self.cal.needs_confirmation
        editing = not settled or self._cal_edit
        self.pick_btn.setVisible(editing)
        self.size_btn.setVisible(editing)
        self.tilt_row.setVisible(editing or self.tilt_on.isChecked())
        self.cal_change_btn.setVisible(not editing)
        for act in self.tool_actions.values():
            act.setEnabled(has)
        self.more_btn.setEnabled(has)
        self.undo_act.setEnabled(bool(self._undo))
        self.redo_act.setEnabled(bool(self._redo))
        self.export_act.setEnabled(bool(self.ms and self.ms.items))
        self.load_act.setEnabled(self.ms is not None)
        self.bug_act.setEnabled(self.path is not None)
        self.delete_btn.setEnabled(self.view.selected is not None)
        self.delete_btn.setVisible(self.table.rowCount() > 0)
        ready = self.ms is not None and self.processor is not None
        row = self.steps.currentRow()
        n = len(self.ms.processing.steps) if self.ms is not None else 0
        for w in (self.add_step_btn, self.op_choice, self.sat_on, self.compare_on, self.average_btn):
            w.setEnabled(ready)
        self.export_proc_btn.setEnabled(ready and bool(self.ms.processing.active()))
        self.up_btn.setEnabled(ready and row > 0)
        self.down_btn.setEnabled(ready and 0 <= row < n - 1)
        self.remove_step_btn.setEnabled(ready and 0 <= row < n)
        arow = self.auto_list.currentRow()
        for w in (self.run_btn, self.region_btn, self.whole_btn, self.recipe):
            w.setEnabled(ready)
        self.plan_btn.setEnabled(ready and getattr(self, "_plan_thread", None) is None)
        self.plan_card.setEnabled(ready)
        has_tilt = ready and self.ms.tilt3d is not None and bool(self.ms.tilt3d.views)
        self._update_tabs()
        done = self.tilt_result is not None
        for w in (self.tilt_add_btn, self.tilt_axis, self.tilt_level, self.tilt_smooth, self.tilt_conf):
            w.setEnabled(ready)
        self.tilt_cutoff.setEnabled(ready and self.ms.tilt3d is not None)
        self.tilt_remove_btn.setEnabled(has_tilt and self.tilt_table.currentRow() > 0)
        self.tilt_run_btn.setEnabled(has_tilt and self._tilt_thread is None)
        for w in (self.tilt_show, self.view3d_btn, self.hprofile_btn, self.step_btn, self.tilt_export_btn):
            w.setEnabled(done)
        self.tilt_clear_btn.setEnabled(ready and self.ms.tilt3d is not None)
        self.rerun_btn.setEnabled(ready and arow >= 0)
        self.pool_btn.setEnabled(ready and 0 <= arow < len(self.analyses) and self.analyses[arow].spec.recipe in ("particles", "pores", "holes", "fibres"))
        self.auto_delete.setEnabled(ready and arow >= 0)
        from vexel.packs.sem.examples import OBJECT_RECIPES

        objects_ok = ready and 0 <= arow < len(self.analyses) and self.analyses[arow].spec.recipe in OBJECT_RECIPES
        self.example_btn.setEnabled(objects_ok or self.example_btn.isChecked())
        n_ex = len(self._examples["yes"]) + len(self._examples["no"])
        self.tune_btn.setEnabled(objects_ok and n_ex > 0)
        self.clear_examples_btn.setEnabled(n_ex > 0)
        has_method = self.method_box.currentData() is not None
        self.method_apply_btn.setEnabled(ready and has_method)
        self.method_save_btn.setEnabled(ready and arow >= 0)
        self.method_auto.setEnabled(has_method)
        self.batch_btn.setEnabled(has_method)
        # what shows: only what can be used now (0.8: controls appear when they have something to act on)
        any_auto = bool(self.analyses)
        saved = any(self.method_box.itemData(i) is not None for i in range(self.method_box.count()))
        self.rerun_btn.setVisible(arow >= 0)
        for w in (self.auto_list, self.auto_delete, self.auto_show):
            w.setVisible(any_auto)
        examples = objects_ok or n_ex > 0 or self.example_btn.isChecked()
        for w in (self.example_btn, self.tune_btn, self.clear_examples_btn):
            w.setVisible(examples)
        self.example_info.setVisible(examples and bool(self.example_info.text()))
        for w in (self.method_box, self.method_apply_btn, self.method_save_btn):
            w.setVisible(any_auto or saved)
        self.method_save_btn.setVisible(any_auto)
        self.method_apply_btn.setVisible(saved)
        self.method_auto.setVisible(has_method)
        self.batch_btn.setVisible(saved)
        self.pool_btn.setVisible(self.pool_btn.isEnabled())

    def switch_language(self, lang: str) -> None:
        """Remember the interface language; it applies when the window is opened again."""
        from vexel.i18n import set_language

        set_language(lang)
        QMessageBox.information(self, "Vexel", "The SEM window will be in English the next time it is opened." if lang == "en"
                                else "下次打开 SEM 窗口时将使用中文界面。")  # fmt: skip

    def say(self, text: str, seconds: int = 10) -> None:
        """A message in the status bar that goes away by itself."""
        self.statusBar().showMessage(text, seconds * 1000)

    def _hover(self, x: float, y: float) -> None:
        if self.image is None or not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            return
        text = tr(f"x {x:.0f}, y {y:.0f} 像素")
        if self.cal is not None and self.cal.ok:
            from vexel.packs.sem.calibrate import format_length

            text += f"  ({format_length(x * self.cal.pixel_size)}, {format_length(y * (self.cal.pixel_size_y or self.cal.pixel_size))})"
        hint = TOOL_HINTS.get(self.view.tool, "")
        self.statusBar().showMessage(f"{hint}  ·  {text}" if self.view.tool != "pan" else text)  # follows the pointer

    # -- confirming and manual calibration ------------------------------------------------

    def _confirm(self) -> None:
        from vexel.packs.sem.calibrate import save

        if self.cal is None or not self.cal.ok:
            return
        self.cal.confirm("person")
        save(self.cal)
        self._show(self.cal)

    def _enter_size(self) -> None:
        text, ok = QInputDialog.getText(self, tr("像素尺寸"), tr("每个像素的尺寸（例如 2.5 nm、0.82 µm）："))
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
            QMessageBox.warning(self, tr("无法标定"), str(e))
            return False
        save(cal)
        self._show(cal)
        return True

    def _update_3d_tab(self) -> None:
        """The 3D tab (tilt 2.5D) is a preview in 0.8: shown with VEXEL_PREVIEW=1, or when the
        open record already has tilt views (made with an earlier version), so they stay reachable."""
        from vexel.ui.launcher import preview_enabled

        ms = getattr(self, "ms", None)
        show = preview_enabled() or (ms is not None and ms.tilt3d is not None)
        self.tabs.setTabVisible(TAB_3D, show)
        if not show and self.tabs.currentIndex() == TAB_3D:
            self.tabs.setCurrentIndex(TAB_MEASURE)

    # -- measuring ------------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        if tool == "calibrate":
            self.tabs.setCurrentIndex(TAB_CALIBRATION)
        elif tool in ("region", "example"):
            self.tabs.setCurrentIndex(TAB_AUTO)
        elif tool in ("hprofile", "box_lower", "box_upper"):
            self._update_3d_tab()
            self.tabs.setCurrentIndex(TAB_3D)
        elif tool != "pan":
            self.tabs.setCurrentIndex(TAB_MEASURE)  # the results appear there
        self.view.set_tool(tool)
        if tool in self.tool_actions:
            self.tool_actions[tool].setChecked(True)
            self._tool_chosen(self.tool_actions[tool])
        else:
            for act in self.tool_actions.values():
                act.setChecked(False)
        self._update_tool_settings(tool)
        self.view.setFocus()
        self.say(TOOL_HINTS.get(tool, ""))

    def _update_tool_settings(self, tool: str) -> None:
        """Only the settings of the tool in use: the count category for counting, width and edge
        definition for line profiles."""
        form = self.meas_form
        form.setRowVisible(self.category, tool == "point")
        for w in (self.profile_width, self.edge_method, self.edge_level):
            form.setRowVisible(w, tool == "profile")

    def _update_tabs(self, show_processing: bool | None = None) -> None:
        """Image processing shows when it is asked for (视图 → 图像处理) or the record has
        processing steps; 3D as a preview (``_update_3d_tab``)."""
        ms = getattr(self, "ms", None)
        has_steps = ms is not None and bool(ms.processing.steps)
        want = self.processing_act.isChecked() if show_processing is None else show_processing
        if has_steps and not self.processing_act.isChecked():
            self.processing_act.blockSignals(True)
            self.processing_act.setChecked(True)
            self.processing_act.blockSignals(False)
            want = True
        self.tabs.setTabVisible(TAB_PROCESSING, bool(want or has_steps))
        if not (want or has_steps) and self.tabs.currentIndex() == TAB_PROCESSING:
            self.tabs.setCurrentIndex(TAB_MEASURE)
        if show_processing:
            self.tabs.setCurrentIndex(TAB_PROCESSING)
        self._update_3d_tab()

    def _profile_params(self) -> dict[str, Any]:
        return {"width": int(self.profile_width.value()), "method": self.edge_method.currentData(), "level": self.edge_level.value() / 100}

    def _shape_done(self, tool: str, points: list[list[float]]) -> None:
        if tool in ("hprofile", "box_lower", "box_upper"):
            self._tilt_shape(tool, points)
            return
        if tool in ("example_yes", "example_no"):
            self.add_example(points[0], tool == "example_yes")
            return
        if tool == "region":
            self.set_tool("pan")
            (x0, y0), (x1, y1) = points
            self.set_region([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
            return
        if tool == "calibrate":
            self.set_tool("pan")
            length_px = float(np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]))
            if length_px < 2:
                return
            text, ok = QInputDialog.getText(self, tr("比例尺长度"), tr(f"两点相距 {length_px:.1f} 像素。它代表的长度（例如 500 nm、2 µm）："))
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
            self.say(tr(f"无法测量：{e}"))
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
            cells = [str(item.id), tr(KIND_NAMES.get(item.kind, item.kind)), (v.get("error") or (label[1] if len(label) > 1 else "")), item.name]
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
        unit_note = "" if self.ms.scale.pixel_size else tr("（未标定：以像素为单位）")
        tilt_note = tr(f"；倾斜校正 {self.ms.scale.tilt:.1f}°") if self.ms.scale.tilt else ""
        self.summary.setText(tr(f"{n} 项测量{unit_note}{tilt_note}。自动保存，重新打开这张图时恢复。") if n else tr(f"选择上方工具在图上测量{unit_note}。"))
        self._show_profile(self.values.get(select) if select is not None else None, unit, f)
        self._refresh_auto(res, unit, f)
        self._refresh_tilt()
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
        definition = tr(f"阈值 {v['level']:.0%}") if v["method"] == "threshold" else dict(EDGE_METHODS)[v["method"]]
        lines = [tr(f"边缘定义：{definition}；找到 {len(v['edges'])} 个边缘")]
        if v.get("width_mean") is not None:
            lines.append(tr(f"线宽 {num(v['width_mean'] / f)} ± {num((v['width_sd'] or 0) / f)} {unit}（{len(v['widths'])} 条）"))
        if v.get("space_mean") is not None:
            lines.append(tr(f"间隔 {num(v['space_mean'] / f)} {unit}"))
        if v.get("pitch") is not None:
            lines.append(tr(f"周期 {num(v['pitch'] / f)} {unit}（自相关）") + (tr(f"；按边缘 {num(v['pitch_edges'] / f)} {unit}") if v.get("pitch_edges") else ""))
        if v.get("peak_spacing") is not None:
            lines.append(tr(f"峰间距 {num(v['peak_spacing'] / f)} {unit}"))
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
        path, chosen = QFileDialog.getSaveFileName(self, tr("导出测量"), str(self.path.parent / stem),
                                                   tr("Excel (*.xlsx);;CSV (*.csv);;报告 HTML (*.html);;标注图 PNG (*.png);;标注图 SVG (*.svg);;测量记录 (*.json)"))  # fmt: skip
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
            if self.tilt_result is not None:
                self._tilt_sheet(path)
        elif suffix == ".csv":
            out = measure_io.export_csv(self.ms, self.image, path, result=res)
        elif suffix == ".png":
            out = [measure_io.export_png(self.image, self.ms, path, crop_rows=crop, result=res)]
        elif suffix == ".svg":
            out = [measure_io.export_svg(self.image, self.ms, path, crop_rows=crop, result=res)]
        elif suffix == ".json":
            out = [self.ms.save(path)]
        elif suffix in (".html", ".htm"):
            from vexel.packs.sem.report import write_report

            out = [write_report(self.ms, self.image, path, tilt=self.tilt_result, result=res)]
        else:
            raise ValueError(f"unknown export type: {suffix}")
        self.say(tr("已导出：") + tr("、").join(p.name for p in out))
        return out

    def _bugreport(self) -> None:
        from qtpy.QtWidgets import QInputDialog

        if self.path is None:
            return
        note, ok = QInputDialog.getMultiLineText(self, tr("诊断包"), tr("发生了什么？（可以不填）"))
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("保存诊断包"), str(self.path.with_name(self.path.stem + tr("_诊断包.zip"))), "Zip (*.zip)")
        if path:
            out = self.bugreport(path, note)
            self.say(tr(f"已保存诊断包：{out.name}（发送前可以解压查看）"))

    def bugreport(self, path: str | Path, note: str = "") -> Path:
        from vexel.bugreport import write_bundle

        if self.ms is not None:
            from vexel.packs.sem import measure_io

            measure_io.autosave(self.ms)
        return write_bundle(path, note=note, sem_images=[self.path] if self.path else [])

    def _load_record_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("载入测量记录"), str(self.path.parent if self.path else ""), tr("测量记录 (*.json)"))
        if path:
            self.load_record(path)

    def load_record(self, path: str | Path) -> bool:
        from vexel.packs.sem.measure import MeasurementSet

        try:
            ms = MeasurementSet.load(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, tr("无法载入"), str(e))
            return False
        if self.cal is not None and ms.image.get("sha256") and ms.image["sha256"] != self.cal.sha256:
            answer = QMessageBox.question(self, tr("不是同一张图"), tr(f"这份记录属于 {ms.image.get('file', '另一张图')}，不是当前图片。仍要载入它的测量吗？"))
            if answer != QMessageBox.StandardButton.Yes:
                return False
            ms.image = self.ms.image if self.ms is not None else ms.image
        self._push_undo()
        keep_scale = self.ms.scale if self.ms is not None else ms.scale
        self.ms = ms
        self.ms.scale = keep_scale  # values follow the calibration in force for this image
        self._changed(select=None)
        return True


    # -- automatic measurements ----------------------------------------------------------------

    def _show_recipe(self) -> None:
        from vexel.packs.sem.analysis import RECIPES

        rec = RECIPES[self.recipe.currentData()]
        self.recipe_help.setText(tr(rec.help))
        while self.auto_form.rowCount():
            self.auto_form.removeRow(0)
        self._auto_widgets = {}
        self._auto_labels: dict[str, QLabel] = {}
        for prm in rec.params:
            w = _param_widget(prm, prm.default)
            self._auto_widgets[prm.name] = (prm, w)
            lab = QLabel(tr(prm.label))
            lab.setWordWrap(True)
            self._auto_labels[prm.name] = lab
            self.auto_form.addRow(lab, w)
            for sig in ("currentIndexChanged", "toggled", "valueChanged"):
                if hasattr(w, sig):
                    getattr(w, sig).connect(lambda *_a: self._param_edited())
                    break
        self._mark_params()

    def _default_text(self, prm: Any) -> str:
        if prm.kind == "choice":
            return tr(dict(prm.choices).get(prm.default, str(prm.default)))
        if prm.kind == "bool":
            return tr("是") if prm.default else tr("否")
        return f"{prm.default:g}"

    def _mark_params(self) -> None:
        """Changed settings in bold with their default in the tooltip; the drawer's button
        says how many are changed."""
        changed = self._recipe_params() if self._auto_widgets else {}
        for name, (prm, _w) in self._auto_widgets.items():
            lab = self._auto_labels[name]
            on = name in changed
            lab.setText(("● " if on else "") + tr(prm.label))
            lab.setStyleSheet("font-weight: 600; color: #2563eb;" if on else "")
            lab.setToolTip(tr(f"默认：{self._default_text(prm)}"))
        n = len(changed)
        opened = self.advanced_btn.isChecked()
        self.auto_box.setVisible(opened)
        self.advanced_btn.setText(("▾ " if opened else "▸ ") + tr("高级设置") + (tr(f"（{n} 项已改）") if n else ""))
        self.reset_params_btn.setEnabled(bool(n))

    def _param_edited(self) -> None:
        self._mark_params()
        if not getattr(self, "_loading_params", False):
            self._live_timer.start()

    def _live_apply(self, refresh: bool = True) -> bool:
        """Changed settings apply at once to the automatic measurement shown in the panel (the
        selected one) when it uses the same method."""
        if self.ms is None or self._shown_spec is None:
            return False
        spec = next((a for a in self.ms.analyses if a.id == self._shown_spec), None)
        new = self._recipe_params()
        if spec is None or spec.recipe != self.recipe.currentData() or spec.params == new:
            return False
        self._push_undo()
        spec.params = new
        if refresh:
            self._auto_select = spec.id
            self._changed()
        return True

    def reset_params(self) -> None:
        self._load_params({})
        self._live_apply()

    def _load_params(self, params: dict[str, Any], recipe: str | None = None) -> None:
        """Show a method and its settings in the panel (without re-running anything)."""
        self._loading_params = True
        try:
            if recipe and recipe != self.recipe.currentData():
                if self.recipe.findData(recipe) < 0 and recipe in RECIPES:  # a preview method in a record made with it
                    self.recipe.addItem(tr(RECIPES[recipe].label) + tr("（预览）"), recipe)
                self.recipe.setCurrentIndex(self.recipe.findData(recipe))
            for name, (prm, w) in self._auto_widgets.items():
                v = params.get(name, prm.default)
                if prm.kind == "choice":
                    w.setCurrentIndex(max(0, w.findData(v)))
                elif prm.kind == "bool":
                    w.setChecked(bool(v))
                else:
                    w.setValue(v)
        finally:
            self._loading_params = False
        self._mark_params()

    def _recipe_params(self) -> dict[str, Any]:
        out = {}
        for name, (prm, w) in self._auto_widgets.items():
            v = _param_value(prm, w)
            if v != prm.default:
                out[name] = v
        return out

    def set_region(self, region: list[float] | None) -> None:
        self._pending_region = region
        self.view.show_region(region)
        if region:
            x0, y0, x1, y1 = region
            self.region_label.setText(tr(f"区域：x {x0:.0f}–{x1:.0f}，y {y0:.0f}–{y1:.0f}（{x1 - x0:.0f}×{y1 - y0:.0f} 像素）"))
        else:
            self.region_label.setText(tr("区域：整张图（信息栏以上）"))

    def run_analysis(self, recipe: str | None = None, region: list[float] | None | str = "pending", **params: Any) -> Any:
        """Add an automatic measurement (the panel's recipe, region and settings unless given) and run it."""
        if self.ms is None:
            return None
        recipe = recipe or self.recipe.currentData()
        region = self._pending_region if region == "pending" else region
        params = params or (self._recipe_params() if recipe == self.recipe.currentData() else {})
        self._push_undo()
        try:
            spec = self.ms.add_analysis(recipe, region, **params)  # type: ignore[arg-type]
        except ValueError as e:
            self._pop_undo()
            QMessageBox.warning(self, tr("无法测量"), str(e))
            return None
        self.tabs.setCurrentIndex(TAB_AUTO)
        self._auto_select = spec.id
        self.set_region(None)
        self._changed()
        return next((a for a in self.analyses if a.spec.id == spec.id), None)

    _auto_select: int | None = None
    _plan_thread: QThread | None = None
    _plan_job: _PlanJob | None = None
    last_plan: Any = None

    def auto_plan(self, goal: str | None = None, *, accept: bool | None = None, wait: bool | None = None) -> Any:
        """Recognise the image and plan its measurement (tried first). The plan is shown with its
        reasons; it enters the record only when accepted (``accept``: decide without asking)."""
        from vexel.packs.sem.plan import memory_store

        if self.ms is None or self.image is None:
            return None
        kwargs = {"image": self.image, "cal": self.cal, "goal": self.goal.text().strip() if goal is None else goal, "scale": self.ms.scale,
                  "crop_rows": self.ms.image.get("crop_rows"), "base_steps": list(self.ms.processing.steps), "memory": memory_store()}  # fmt: skip
        self._plan_accept = accept
        if wait if wait is not None else self._sync:
            from vexel.packs.sem.plan import make_plan

            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                plan = make_plan(**kwargs)
                preview = plan_preview(plan, self.ms, self.image) if accept is None else None
            finally:
                QGuiApplication.restoreOverrideCursor()
            return self._plan_ready((plan, preview))
        if self._plan_thread is not None:
            return None
        self.plan_card.busy(tr("正在识别图像并试运行…"))
        self._plan_thread = QThread()
        self._plan_job = _PlanJob(kwargs, self.ms.model_copy(deep=True))
        self._plan_job.moveToThread(self._plan_thread)
        self._plan_thread.started.connect(self._plan_job.run)
        self._plan_job.done.connect(self._plan_ready)
        self._plan_job.failed.connect(self._plan_failed)  # a method: queued to the window's thread
        self._plan_thread.start()
        self._set_buttons()
        return None

    def _plan_failed(self, msg: str) -> None:
        self._end_plan()
        self.plan_card.set_plan(f"<b>{tr('自动规划')}</b>", "needs_user", [], [("✗", tr(f"规划失败：{msg}"))], [], False)

    def _calibration_failed(self, msg: str) -> None:
        self.state.setText(tr(f"标定失败：{msg}"))

    def _tilt_failed(self, msg: str, _key: str) -> None:
        self._end_tilt()
        self.tilt_info.setText(tr(f"无法重建：{msg}"))

    def _end_plan(self) -> None:
        if self._plan_thread is not None:
            self._plan_thread.quit()
            self._plan_thread.wait()
            self._plan_thread, self._plan_job = None, None
        self._set_buttons()

    def _plan_ready(self, result: Any) -> Any:
        """Show the plan on its card and its trial on the image; it enters the record only
        when accepted (``auto_plan(accept=...)`` decides without asking)."""
        from vexel.packs.sem import measure_io
        from vexel.packs.sem.analysis import RECIPES
        from vexel.packs.sem.recognize import KIND_NAMES

        plan, preview = result if isinstance(result, tuple) else (result, None)
        self._end_plan()
        self.last_plan = plan
        self.plan_preview = preview
        fp = plan.fingerprint
        usable = plan.recipe is not None and plan.status != "needs_user"
        who = tr("离线规则") if plan.planner == "rules" else (plan.model or plan.planner)
        what = tr(RECIPES[plan.recipe].label) if plan.recipe else tr("无")
        title = f"<b>{tr(KIND_NAMES.get(fp.kind, fp.kind))}</b> → <b>{what}</b><br><span style='color:#6b7280'>" + tr(
            f"识别置信度 {fp.confidence:.2f} · 规划：{who}") + "</span>"
        numbers = []
        if preview is not None and not preview.error and self.ms is not None:
            unit, f = measure_io.display_unit(self.ms)
            # the main count first, then the first two lengths or angles
            vals = [(lab, v, kind) for k, lab, v, kind in preview.summary if v is not None and k != "at_border" and not k.startswith("class_")]
            picked = [x for x in vals if x[2] == "count"][:1] + [x for x in vals if x[2] in ("length", "angle")][:2]
            numbers = [f"{tr(lab)} <b>{self._fmt(v, kind, unit, f)}</b>" for lab, v, kind in picked]
        best = plan.best
        checks: list[tuple[str, str]] = []
        if best is not None:
            checks += [("✗", tr(t)) for t in ([best.error] if best.error else []) + best.problems]
            checks += [("!", tr(t)) for t in best.warnings[:3]]
            if not checks:
                checks.append(("✓", tr("试运行通过全部检查")))
        reasons = [tr(r) for r in plan.reasons if not r.startswith("试运行通过")]
        self.plan_card.set_plan(title, plan.status, numbers, checks, reasons, usable)
        if preview is not None and not preview.error:
            self.view.show_analysis([sh for an in self.analyses if not an.error for sh in an.shapes] + preview.shapes)
        self.tabs.setCurrentIndex(TAB_AUTO)
        accept = self._plan_accept
        self._plan_accept = None
        if accept is not None:
            return self._decide_plan("accept" if accept else "dismiss")
        return plan

    _plan_accept: bool | None = None
    plan_preview: Any = None

    def _decide_plan(self, choice: str) -> Any:
        """accept: into the record; adjust: method, settings and region into the panel;
        dismiss: neither (remembered as not accepted)."""
        from vexel.packs.sem.plan import memory_store, remember

        plan = self.last_plan
        self.plan_card.hide()
        if plan is None or self.ms is None:
            return plan
        usable = plan.recipe is not None and plan.status != "needs_user"
        if choice == "accept" and usable:
            self._push_undo()
            spec = plan.apply(self.ms)
            remember(memory_store(), plan, True, str(self.path) if self.path else None)
            self.tabs.setCurrentIndex(TAB_AUTO)
            self._auto_select = spec.id
            self._changed()
            return plan
        if choice == "adjust" and plan.recipe:
            self.auto_list.setCurrentRow(-1)
            self._load_params(plan.params, plan.recipe)
            self.set_region(plan.region)
            self.advanced_btn.setChecked(True)
        elif usable:
            remember(memory_store(), plan, False)
        self.view.show_analysis([sh for an in self.analyses if not an.error for sh in an.shapes])
        return plan

    # -- correct by example ----------------------------------------------------------------

    def add_example(self, point: list[float], positive: bool) -> None:
        self._examples["yes" if positive else "no"].append([float(point[0]), float(point[1])])
        self.view.show_examples(self._examples["yes"], self._examples["no"])
        self.example_info.setText(tr(f"例子：这个 {len(self._examples['yes'])} 个，不是这个 {len(self._examples['no'])} 个"))
        self._set_buttons()

    def clear_examples(self) -> None:
        self._examples = {"yes": [], "no": []}
        self.view.show_examples([], [])
        self.example_info.setText("")
        self._set_buttons()

    def tune_examples(self, choose: int | None = None) -> list[Any]:
        """Search the selected measurement's settings for the clicked examples; show the best
        candidates side by side (``choose``: take that one without asking)."""
        from vexel.packs.sem.examples import tune

        row = self.auto_list.currentRow()
        if self.ms is None or not 0 <= row < len(self.ms.analyses):
            return []
        spec = self.ms.analyses[row]
        res = self._processed(block=True)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cands = tune(res.analysis, spec, self.ms.scale, [tuple(p) for p in self._examples["yes"]], [tuple(p) for p in self._examples["no"]],
                         mask=res.mask, crop_rows=self.ms.image.get("crop_rows"))  # fmt: skip
        except ValueError as e:
            QMessageBox.warning(self, tr("无法调整"), str(e))
            return []
        finally:
            QGuiApplication.restoreOverrideCursor()
        if choose is not None:
            if 0 <= choose < len(cands):
                self.choose_candidate(cands[choose])
            return cands
        self._candidates_dialog(cands, res)
        return cands

    def choose_candidate(self, cand: Any) -> None:
        from vexel.packs.sem.analysis import RECIPES

        row = self.auto_list.currentRow()
        spec = self.ms.analyses[row]
        defaults = {p.name: p.default for p in RECIPES[spec.recipe].params}
        self._push_undo()
        spec.params = {k: v for k, v in cand.params.items() if defaults.get(k) != v}
        self._shown_spec = None  # the panel shows the new settings
        self._auto_select = spec.id
        self.clear_examples()
        if self.example_btn.isChecked():
            self.example_btn.setChecked(False)
        self._changed()

    def _thumbnail(self, cand: Any, res: Any, width: int = 300) -> Any:
        from PIL import Image, ImageDraw
        from qtpy.QtGui import QImage, QPixmap

        img = Image.fromarray(res.display).convert("RGB")
        x0, y0, x1, y1 = cand.analysis.extra.get("region") or [0, 0, img.width, img.height]
        dr = ImageDraw.Draw(img)
        lw = max(1, img.width // 400)
        for kind, geom, colour, _a, label in cand.analysis.shapes:
            if kind in ("polygon", "polyline") and len(geom) >= 2 and not str(label).startswith("A"):
                dr.line([tuple(p) for p in geom] + ([tuple(geom[0])] if kind == "polygon" else []), fill=colour, width=lw)
        for x, y in self._examples["yes"]:
            dr.ellipse([x - 8, y - 8, x + 8, y + 8], outline="#16a34a", width=lw + 1)
        for x, y in self._examples["no"]:
            dr.line([x - 7, y - 7, x + 7, y + 7], fill="#dc2626", width=lw + 1)
            dr.line([x - 7, y + 7, x + 7, y - 7], fill="#dc2626", width=lw + 1)
        img = img.crop((int(x0), int(y0), int(x1), int(y1)))
        img.thumbnail((width, width))
        data = img.tobytes()
        q = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q)

    def _candidates_dialog(self, cands: list[Any], res: Any) -> None:
        from qtpy.QtWidgets import QDialog, QDialogButtonBox

        from vexel.packs.sem import measure_io
        from vexel.packs.sem.analysis import RECIPES

        if not cands:
            QMessageBox.information(self, tr("按例子调整"), tr("没有找到符合例子的设置：再多点几个例子，或换一种方法"))
            return
        unit, f = measure_io.display_unit(self.ms)
        n_ex = len(self._examples["yes"]) + len(self._examples["no"])
        labels = {p.name: tr(p.label) for p in RECIPES[cands[0].analysis.spec.recipe].params}
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("按例子调整：选一个"))
        lay = QHBoxLayout(dlg)
        for i, c in enumerate(cands):
            col = QVBoxLayout()
            pic = QLabel()
            pic.setPixmap(self._thumbnail(c, res))
            col.addWidget(pic)
            main = next(((lab, v, kind) for _k, lab, v, kind in c.analysis.summary if kind in ("length", "ratio") and v is not None), None)
            lines = [tr(f"<b>符合 {c.satisfied}/{n_ex} 个例子</b>"), tr(f"对象 {len(c.analysis.objects)} 个")]
            if main:
                lines.append(f"{tr(main[0])}：{self._fmt(main[1], main[2], unit, f)}")
            changes = [f"{labels.get(k, k)} = {tr(str(c.params[k]))}" for k in c.changes]
            lines.append(tr("改动：") + ("，".join(changes) if changes else tr("无")))
            text = QLabel("<br>".join(lines))
            text.setWordWrap(True)
            col.addWidget(text)
            pick = QPushButton(tr("用这个"))
            if i == 0:
                pick.setObjectName("primary")
            pick.clicked.connect(lambda _c=False, cand=c: (self.choose_candidate(cand), dlg.accept()))
            col.addWidget(pick)
            lay.addLayout(col)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        close.rejected.connect(dlg.reject)
        lay.addWidget(close)
        self._candidates = dlg
        dlg.exec()

    # -- saved methods and batches -----------------------------------------------------------

    def _fill_methods(self, select: str | None = None) -> None:
        from vexel.packs.sem import methods

        auto = methods.auto()
        keep = select or self.method_box.currentData()
        self.method_box.blockSignals(True)
        self.method_box.clear()
        self.method_box.addItem(tr("（方案）"), None)
        for m in methods.list_methods():
            self.method_box.addItem(m.name + (tr("（自动套用）") if auto and auto.name == m.name else ""), m.name)
        i = self.method_box.findData(keep)
        self.method_box.setCurrentIndex(max(i, 0))
        self.method_box.blockSignals(False)
        self._method_selected()

    def _method_selected(self) -> None:
        from vexel.packs.sem import methods

        name = self.method_box.currentData()
        auto = methods.auto()
        self.method_auto.blockSignals(True)
        self.method_auto.setChecked(bool(name and auto and auto.name == name))
        self.method_auto.blockSignals(False)
        if hasattr(self, "tilt_cutoff"):  # the window is built (the 3D tab comes last)
            self._set_buttons()

    def _method_auto_toggled(self, on: bool) -> None:
        from vexel.packs.sem import methods

        name = self.method_box.currentData()
        if name is None:
            return
        methods.set_auto(name if on else None)
        self._fill_methods(name)

    def save_method(self, name: str, row: int | None = None) -> Any:
        from vexel.packs.sem import methods

        row = self.auto_list.currentRow() if row is None else row
        if self.ms is None or not 0 <= row < len(self.ms.analyses):
            return None
        m = methods.save(methods.from_record(self.ms, self.ms.analyses[row], name))
        self._fill_methods(m.name)
        self.say(tr(f"已存为方案“{m.name}”"))
        return m

    def _save_method_dialog(self) -> None:
        from vexel.packs.sem.analysis import RECIPES

        row = self.auto_list.currentRow()
        if self.ms is None or not 0 <= row < len(self.ms.analyses):
            return
        spec = self.ms.analyses[row]
        default = spec.name if spec.name and spec.name != "自动规划" else tr(RECIPES[spec.recipe].label)
        name, ok = QInputDialog.getText(self, tr("存为方案"), tr("方案名称（例如“M1 线宽”）："), text=default)
        if ok and name.strip():
            self.save_method(name)

    def apply_method(self, name: str | None = None, record: bool = True) -> Any:
        from vexel.packs.sem import methods

        name = name or self.method_box.currentData()
        m = methods.get(name) if name else None
        if self.ms is None or m is None:
            return None
        if record:
            self._push_undo()
        spec = methods.apply(m, self.ms)
        self.tabs.setCurrentIndex(TAB_AUTO)
        self._auto_select = spec.id
        self._changed()
        return spec

    def batch_files(self, paths: list[str | Path], name: str | None = None) -> list[Any]:
        from vexel.packs.sem import methods

        name = name or self.method_box.currentData()
        m = methods.get(name) if name else None
        if m is None:
            raise ValueError("choose a saved method first")
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return methods.run_files(m, paths)
        finally:
            QGuiApplication.restoreOverrideCursor()

    def _batch_dialog(self) -> None:
        from qtpy.QtWidgets import QDialog, QDialogButtonBox

        from vexel.packs.sem import measure_io, methods

        files, _ = QFileDialog.getOpenFileNames(self, tr("选择要批量测量的图片"), str(self.path.parent if self.path else ""),
                                                tr("图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"))  # fmt: skip
        if not files:
            return
        try:
            results = self.batch_files(files)
        except ValueError as e:
            QMessageBox.warning(self, tr("无法批量测量"), str(e))
            return
        unit, f = measure_io.display_unit(self.ms) if self.ms is not None else ("px", 1.0)
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(f"批量测量：{self.method_box.currentData()}"))
        lay = QVBoxLayout(dlg)
        n_check = sum(r.status != "ok" for r in results)
        lay.addWidget(QLabel(tr(f"共 {len(results)} 张：需要看 {n_check} 张（出错 {sum(r.status == 'error' for r in results)}），正常 {len(results) - n_check} 张。双击一行打开那张图。")))
        only = QCheckBox(tr("只显示需要看的"))
        only.setChecked(n_check > 0)
        lay.addWidget(only)
        tbl = QTableWidget(0, 4)
        tbl.setHorizontalHeaderLabels([tr("状态"), tr("文件"), tr("结果"), tr("原因")])
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        names = {"ok": tr("正常"), "check": tr("需要看"), "error": tr("出错")}

        def fill() -> None:
            shown = [r for r in results if r.status != "ok" or not only.isChecked()]
            tbl.setRowCount(len(shown))
            for i, r in enumerate(shown):
                label, v, kind = r.main
                val = f"{tr(label)} {self._fmt(v, kind, unit, f)}" if v is not None else "–"
                for j, text in enumerate([names[r.status], r.path.name, val, "；".join(tr(x) for x in r.reasons)]):
                    cell = QTableWidgetItem(text)
                    cell.setData(Qt.ItemDataRole.UserRole, str(r.path))
                    tbl.setItem(i, j, cell)
            tbl.resizeColumnsToContents()

        only.toggled.connect(lambda _on: fill())
        tbl.cellDoubleClicked.connect(lambda row, _col: (self.open_path(tbl.item(row, 0).data(Qt.ItemDataRole.UserRole)), dlg.accept()))
        fill()
        lay.addWidget(tbl)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        save = buttons.addButton(tr("导出 Excel…"), QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(dlg.reject)

        def _save() -> None:
            path, _ = QFileDialog.getSaveFileName(dlg, tr("导出批量结果"), str(Path(files[0]).with_name("batch.xlsx")), "Excel (*.xlsx)")
            if path:
                methods.table(results, unit, f).to_excel(path, index=False)
                self.say(tr(f"已导出：{Path(path).name}"))

        save.clicked.connect(_save)
        lay.addWidget(buttons)
        dlg.resize(820, 520)
        self._batch = dlg
        dlg.exec()

    def _rerun_selected(self) -> None:
        row = self.auto_list.currentRow()
        if self.ms is None or not 0 <= row < len(self.ms.analyses):
            return
        spec = self.ms.analyses[row]
        if spec.recipe != self.recipe.currentData():
            self.say(tr("所选自动测量用的是另一种方法：先在上方选同一方法"))
            return
        self._push_undo()
        spec.params = self._recipe_params()
        if self._pending_region:
            spec.region = self._pending_region
            self.set_region(None)
        self._auto_select = spec.id
        self._changed()

    def _delete_analysis(self) -> None:
        row = self.auto_list.currentRow()
        if self.ms is None or not 0 <= row < len(self.ms.analyses):
            return
        self._push_undo()
        self.ms.analyses.pop(row)
        self._changed()

    def _refresh_auto(self, res: Any, unit: str, f: float) -> None:
        from vexel.packs.sem.analysis import RECIPES, run

        if self.ms is None:
            return
        key = json.dumps([[a.model_dump() for a in self.ms.analyses], self._result_key, self.ms.scale.model_dump()], sort_keys=True, default=str)
        if key != self._auto_key:
            img = res.analysis if res is not None else self.image
            mask = res.mask if res is not None else None
            # each measurement is recomputed only when its own settings, the processed image or the
            # scale changed: adding a measurement does not redo the others (on a 50 MP image each
            # takes seconds)
            crop = self.ms.image.get("crop_rows")
            base = json.dumps([self._result_key, self.ms.scale.model_dump(), crop], sort_keys=True, default=str)
            cache = getattr(self, "_an_cache", {})
            keys = [(base + json.dumps(spec.model_dump(), sort_keys=True, default=str), spec) for spec in self.ms.analyses]
            todo = [(k, spec) for k, spec in keys if k not in cache]
            pending = False
            if todo and not self._sync and img is not None and img.size > BACKGROUND_PIXELS:
                self._start_auto(todo, img, mask, crop)  # the list says "计算中…" until it is done
                pending = True
            else:
                QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    for k, spec in todo:
                        cache[k] = run(spec, img, self.ms.scale, mask, crop)
                finally:
                    QGuiApplication.restoreOverrideCursor()
            from vexel.packs.sem.analysis import Analysis

            self.analyses = [cache.get(k) or Analysis(spec=spec, extra={"pending": True}) for k, spec in keys]
            self._an_cache = {k: cache[k] for k, _ in keys if k in cache}
            if pending:
                key = None  # computed again when the job is done
            self._auto_key = key
            self.view.show_analysis([sh for an in self.analyses if not an.error for sh in an.shapes])
        want = self._auto_select
        row = self.auto_list.currentRow()
        self.auto_list.blockSignals(True)
        self.auto_list.clear()
        for an in self.analyses:
            text = f"A{an.spec.id} {tr(RECIPES[an.spec.recipe].label) if an.spec.recipe in RECIPES else an.spec.recipe}"
            main = next(((lab, v, kind) for k, lab, v, kind in an.summary if kind in ("length", "angle") and v is not None), None)
            if an.error:
                text += tr(f"：{an.error}")
            elif an.extra.get("pending"):
                text += tr("：计算中…")
            elif main:
                text += tr(f"：{tr(main[0])} {self._fmt(main[1], main[2], unit, f)}")
            self.auto_list.addItem(text)
        if want is not None:
            row = next((i for i, a in enumerate(self.analyses) if a.spec.id == want), row)
            self._auto_select = None
        self.auto_list.setCurrentRow(min(row, self.auto_list.count() - 1) if row >= 0 or want is not None else -1)
        self.auto_list.blockSignals(False)
        self._show_auto_detail(unit, f)

    @staticmethod
    def _fmt(v: Any, kind: str, unit: str, f: float) -> str:
        from vexel.packs.sem.measure_io import num

        if v is None:
            return "–"
        if kind == "length":
            return f"{num(v / f)} {unit}"
        if kind == "area":
            return f"{num(v / f**2)} {unit}²"
        if kind == "angle":
            return f"{v:.2f}°"
        if kind == "ratio":
            return f"{v:.3f}"
        return tr(str(v))

    _shown_spec: int | None = None

    def _show_auto_detail(self, unit: str | None = None, f: float | None = None) -> None:
        from vexel.packs.sem import measure_io
        from vexel.packs.sem.analysis import objects_table

        row = self.auto_list.currentRow()
        if self.ms is None or not 0 <= row < len(self.analyses):
            self._shown_spec = None
            self.auto_summary.setText(tr("选择方法、框选区域（或用整张图），点“运行”。") if self.ms is not None else "")
            self.auto_notes.hide()
            self.auto_table.setRowCount(0)
            self.auto_table.setColumnCount(0)
            self.auto_chart.hide()
            self._set_buttons()
            return
        if unit is None or f is None:
            unit, f = measure_io.display_unit(self.ms)
        an = self.analyses[row]
        if self._shown_spec != an.spec.id and self._live_timer.isActive():  # a setting was just changed: keep it
            self._live_timer.stop()
            if self._live_apply(refresh=False):
                QTimer.singleShot(0, lambda: self._changed())
        if self._shown_spec != an.spec.id:  # the panel shows the selected measurement's method and settings
            self._shown_spec = an.spec.id
            self._load_params(an.spec.params, an.spec.recipe)
        if an.error:
            self.auto_summary.setText(tr(f"无法测量：{an.error}"))
        else:
            lines = [f"<b>A{an.spec.id}</b>" + (tr(f"（区域 {an.extra.get('region')}）") if an.spec.region else tr("（整张图）"))]
            lines += [tr(f"{tr(lab)}：<b>{self._fmt(v, kind, unit, f)}</b>") for k, lab, v, kind in an.summary if v is not None]
            if an.spec.recipe in SYNTHETIC_ONLY:
                lines.append(f"<span style='color:#b45309'>{tr('这种测量只在合成图像上验收过：请与你们现有的测量结果对照后再使用')}</span>")
            elif an.spec.recipe in PREVIEW_RECIPES:
                lines.append(f"<span style='color:#b45309'>{tr('预览：新的测量，还没有在真实图像上验收')}</span>")
            self.auto_summary.setText("<br>".join(lines))
        self.auto_notes.setText("\n".join(f"• {tr(w)}" for w in an.warnings))
        self.auto_notes.setVisible(bool(an.warnings))
        df = objects_table([an], unit, f).drop(columns=["analysis", "recipe"], errors="ignore").head(500)
        self.auto_table.setRowCount(len(df))
        self.auto_table.setColumnCount(len(df.columns))
        self.auto_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, r in enumerate(df.itertuples(index=False)):
            for j, val in enumerate(r):
                text = f"{val:.4g}" if isinstance(val, float) else str(val)
                self.auto_table.setItem(i, j, QTableWidgetItem(text))
        self._fill_chart(an, unit, f)
        self._set_buttons()

    def _fill_chart(self, an: Any, unit: str, f: float) -> None:
        import pandas as pd

        from vexel.packs.sem.analysis import objects_table
        from vexel.packs.sem.population import SIZE

        if an.error:
            self.auto_chart.hide()
            return
        if an.spec.recipe == "fibres" and an.extra.get("samples"):
            s = an.extra["samples"]
            d = f"diameter ({unit})"
            self.auto_chart.set_columns({d: np.asarray(s["diameter"]) / f, "angle (°)": np.asarray(s["angle"])}, d, np.asarray(s["length"]))
            return
        df = objects_table([an], unit, f)
        skip = {"analysis", "recipe", "particle", "pore", "hole", "line", "layer", "segment", "touching", "x (" + unit + ")", "y (" + unit + ")"}
        cols = {c: df[c].to_numpy(float) for c in df.columns if c not in skip and c.split(" (")[0] not in ("x", "y") and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])}
        size = SIZE.get(an.spec.recipe, "")
        prefer = next((c for c in cols if c.split(" (")[0] == size), next(iter(cols), ""))
        if len(df) < 2 or not cols:
            self.auto_chart.hide()
            return
        self.auto_chart.set_columns(cols, prefer)

    def pool_files(self, paths: list[str | Path], row: int | None = None) -> tuple[Any, list[Any]]:
        """Measure ``paths`` with the selected automatic measurement's recipe, settings and
        region (and this image's processing), each with its own calibration; with this
        image's result, pooled. Returns (pooled table, [(path, Analysis, calibrated)])."""
        from vexel.packs.sem import calibrate as semcal
        from vexel.packs.sem import measure_io
        from vexel.packs.sem.analysis import AnalysisSpec, run
        from vexel.packs.sem.measure import scale_from
        from vexel.packs.sem.population import pooled

        row = self.auto_list.currentRow() if row is None else row
        if self.ms is None or not 0 <= row < len(self.analyses):
            raise ValueError("select an automatic measurement first")
        base = self.analyses[row]
        unit, f = measure_io.display_unit(self.ms)
        out = [(self.path, base, bool(self.ms.scale.pixel_size))]
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for p in paths:
                p = Path(p)
                if self.path is not None and p.resolve() == self.path.resolve():
                    continue
                try:
                    cal = semcal.for_file(p)
                    img = semcal.load_image(p)
                    ms = measure_io.load_record(cal.sha256) or measure_io.new_set(p, cal)
                    ms.scale = scale_from(cal, ms.scale.tilt)
                    ms.processing = self.ms.processing.model_copy(deep=True)
                    res = measure_io.processed(ms, img)
                    spec = AnalysisSpec(id=1, recipe=base.spec.recipe, region=base.spec.region, params=dict(base.spec.params))
                    an = run(spec, res.analysis, ms.scale, res.mask, cal.crop_rows)
                except (ValueError, OSError) as e:
                    an = type(base)(spec=base.spec, error=f"{type(e).__name__}: {e}")
                    cal = None
                out.append((p, an, bool(cal is not None and cal.ok)))
        finally:
            QGuiApplication.restoreOverrideCursor()
        table = pooled([(str(Path(p).parent.name), an, ok) for p, an, ok in out], unit, f)
        return table, out

    def _pool_dialog(self) -> None:
        from qtpy.QtWidgets import QDialog, QDialogButtonBox

        from vexel.packs.sem import measure_io
        from vexel.packs.sem.population import sizes
        from vexel.ui.sem_charts import DistributionChart

        files, _ = QFileDialog.getOpenFileNames(self, tr("选择要合并统计的其他图片"), str(self.path.parent if self.path else ""),
                                                tr("图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"))  # fmt: skip
        if not files:
            return
        try:
            table, per = self.pool_files(files)
        except ValueError as e:
            QMessageBox.warning(self, tr("无法合并"), str(e))
            return
        unit, f = measure_io.display_unit(self.ms)
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("多张图合并统计"))
        lay = QVBoxLayout(dlg)
        lines = []
        for p, an, ok in per:
            n = len(sizes(an)[0])
            state = tr(f"错误：{an.error}") if an.error else tr(f"{n} 个对象") + ("" if ok else tr("（未标定，不计入）"))
            lines.append(f"{Path(p).name}：{state}")
        info = QLabel("<br>".join(lines))
        info.setWordWrap(True)
        lay.addWidget(info)
        if len(table):
            r = table.iloc[0]
            count = r.get("objects", r.get("samples"))
            lay.addWidget(QLabel(tr(f"合并：{int(r['images'])} 张图，{int(count)} 个；平均 {r['mean']:.4g} {r['unit']}，标准差 {r['sd']:.4g}，"
                                    f"d10 / d50 / d90 = {r['d10']:.4g} / {r['d50']:.4g} / {r['d90']:.4g} {r['unit']}")))  # fmt: skip
            vals, wts = [], []
            for _, an, ok in per:
                if ok or not any(o for _, _, o in per):
                    v, w = sizes(an)
                    vals.append(v)
                    wts.append(w)
            chart = DistributionChart()
            k = f if any(o for _, _, o in per) else 1.0
            name = f"size ({r['unit']})"
            chart.set_columns({name: np.concatenate(vals) / k}, name, np.concatenate(wts))
            lay.addWidget(chart)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        save = buttons.addButton(tr("保存 Excel…"), QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(dlg.reject)

        def _save() -> None:
            path, _ = QFileDialog.getSaveFileName(dlg, tr("保存合并统计"), str(self.path.with_name(self.path.stem + "_pooled.xlsx")), "Excel (*.xlsx)")
            if path:
                self.save_pool(path, table, per)
                self.say(tr(f"已导出：{Path(path).name}"))

        save.clicked.connect(_save)
        lay.addWidget(buttons)
        dlg.resize(560, 480)
        dlg.exec()

    def save_pool(self, path: str | Path, table: Any, per: list[Any]) -> Path:
        import pandas as pd

        from vexel.packs.sem import measure_io
        from vexel.packs.sem.analysis import objects_table

        unit, f = measure_io.display_unit(self.ms)
        objs = []
        for p, an, _ok in per:
            df = objects_table([an], unit, f)
            if len(df):
                df.insert(0, "file", Path(p).name)
                objs.append(df)
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            (table if len(table) else pd.DataFrame({"": []})).to_excel(xw, sheet_name="合并统计", index=False)
            pd.DataFrame([{"file": Path(p).name, "error": an.error, "calibrated": ok} for p, an, ok in per]).to_excel(xw, sheet_name="图片", index=False)
            (pd.concat(objs, ignore_index=True) if objs else pd.DataFrame({"": []})).to_excel(xw, sheet_name="对象", index=False)
        return Path(path)

    # -- tilt 3D ---------------------------------------------------------------------------------

    @staticmethod
    def _recon_key(spec: Any) -> str:
        """What the reconstruction itself depends on (profiles, steps and levelling do not)."""
        return json.dumps(spec.model_dump(include={"reference_tilt", "views", "axis", "attachment", "smooth", "min_confidence"}), sort_keys=True)

    def _tilt_spec(self) -> Any:
        from vexel.packs.sem.tilt3d import Tilt3DSpec

        if self.ms.tilt3d is None:
            from vexel.packs.sem.tilt3d import stage_tilt

            t = stage_tilt(self.path) if self.path else None
            self.ms.tilt3d = Tilt3DSpec(reference_tilt=t or 0.0)
        return self.ms.tilt3d

    def add_tilt_views(self, paths: list[str | Path], tilts: list[float] | None = None) -> None:
        """Add views of this place at other tilts (angles from the metadata unless given)."""
        from vexel.packs.sem.calibrate import file_sha256
        from vexel.packs.sem.tilt3d import TiltView, stage_tilt

        if self.ms is None or self.path is None:
            return
        self._push_undo()
        spec = self._tilt_spec()
        for i, p in enumerate(paths):
            p = Path(p)
            t = tilts[i] if tilts else stage_tilt(p)
            try:
                rel = str(p.resolve().relative_to(self.path.resolve().parent))
            except ValueError:
                rel = str(p.resolve())
            spec.views.append(TiltView(file=rel, sha256=file_sha256(p), tilt=float(t) if t is not None else 0.0))
            if t is None:
                self.say(tr(f"{p.name} 的元数据中没有倾角：请在表中填写"))
        self._update_3d_tab()
        self.tabs.setCurrentIndex(TAB_3D)
        self._changed()

    def _choose_tilt_views(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, tr("选择同一位置其他倾角的图"), str(self.path.parent if self.path else ""),
                                                tr("图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"))  # fmt: skip
        paths = [p for p in paths if Path(p) != self.path]
        if paths:
            self.add_tilt_views(paths)

    def _remove_tilt_view(self) -> None:
        row = self.tilt_table.currentRow()
        if self.ms is None or self.ms.tilt3d is None or row < 1:
            return
        self._push_undo()
        self.ms.tilt3d.views.pop(row - 1)
        self._changed()

    def _clear_tilt(self) -> None:
        if self.ms is None or self.ms.tilt3d is None:
            return
        self._push_undo()
        self.ms.tilt3d = None
        self.tilt_result, self._tilt_key = None, None
        self._changed()

    _filling_tilt = False

    def _fill_tilt_table(self) -> None:
        spec = self.ms.tilt3d if self.ms is not None else None
        self._filling_tilt = True
        rows = [] if spec is None else [(self.path.name if self.path else tr("本图"), spec.reference_tilt, True)] + [
            (Path(v.file).name, v.tilt, False) for v in spec.views]  # fmt: skip
        self.tilt_table.setRowCount(len(rows))
        for i, (name, tilt, ref) in enumerate(rows):
            cell = QTableWidgetItem(name + (tr("（参考）") if ref else ""))
            cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tilt_table.setItem(i, 0, cell)
            spin = self.tilt_table.cellWidget(i, 1)
            if not isinstance(spin, QDoubleSpinBox):
                spin = QDoubleSpinBox()
                spin.setRange(-90, 90)
                spin.setDecimals(2)
                spin.valueChanged.connect(lambda value, r=i: self._tilt_angle_edited(r, value))
                self.tilt_table.setCellWidget(i, 1, spin)
            spin.blockSignals(True)
            spin.setValue(tilt)
            spin.blockSignals(False)
        if spec is not None:
            for w, value in ((self.tilt_axis, spec.axis), (self.tilt_smooth, spec.smooth), (self.tilt_conf, spec.min_confidence)):
                w.blockSignals(True)
                w.setValue(value)
                w.blockSignals(False)
            self.tilt_level.blockSignals(True)
            self.tilt_level.setCurrentIndex(self.tilt_level.findData(spec.level))
            self.tilt_level.blockSignals(False)
        k = 1e-9 if self.ms is not None and self.ms.scale.pixel_size else 1.0
        self.tilt_cutoff.blockSignals(True)
        self.tilt_cutoff.setSuffix(" nm" if k != 1.0 else tr(" 像素"))
        self.tilt_cutoff.setValue((spec.cutoff / k) if spec is not None else 0.0)
        self.tilt_cutoff.blockSignals(False)
        self._filling_tilt = False

    def _tilt_cutoff_edited(self, value: float) -> None:
        if self._filling_tilt or self.ms is None or self.ms.tilt3d is None:
            return
        k = 1e-9 if self.ms.scale.pixel_size else 1.0
        self._push_undo()
        self.ms.tilt3d.cutoff = float(value) * k
        self._changed()

    def _tilt_angle_edited(self, row: int, value: float) -> None:
        if self._filling_tilt or self.ms is None or self.ms.tilt3d is None:
            return
        self._push_undo()
        if row == 0:
            self.ms.tilt3d.reference_tilt = float(value)
        elif row - 1 < len(self.ms.tilt3d.views):
            self.ms.tilt3d.views[row - 1].tilt = float(value)
        self._changed()

    def run_tilt3d(self, wait: bool | None = None) -> Any:
        """Reconstruct the height map with the panel's settings (kept in the record)."""
        if self.ms is None or self.path is None or self.ms.tilt3d is None or not self.ms.tilt3d.views:
            return None
        spec = self.ms.tilt3d
        new = {"axis": float(self.tilt_axis.value()), "level": int(self.tilt_level.currentData()), "smooth": float(self.tilt_smooth.value()),
               "min_confidence": float(self.tilt_conf.value())}  # fmt: skip
        if any(getattr(spec, k) != v for k, v in new.items()):
            from vexel.packs.sem import measure_io

            self._push_undo()
            for k, v in new.items():
                setattr(spec, k, v)
            measure_io.autosave(self.ms)
        key = self._recon_key(spec)
        if wait if wait is not None else self._sync:
            from vexel.packs.sem.tilt3d import run_spec

            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                res = run_spec(spec, self.path)
            except (ValueError, OSError) as e:
                QGuiApplication.restoreOverrideCursor()
                self.tilt_info.setText(tr(f"无法重建：{e}"))
                return None
            QGuiApplication.restoreOverrideCursor()
            self._tilt_done(res, key)
            return res
        if self._tilt_thread is not None:
            return None
        self.tilt_info.setText(tr("正在重建高度图…（每对图几秒钟）"))
        self._tilt_thread = QThread()
        self._tilt_job = _TiltJob(spec.model_copy(deep=True), self.path, key)
        self._tilt_job.moveToThread(self._tilt_thread)
        self._tilt_thread.started.connect(self._tilt_job.run)
        self._tilt_job.done.connect(self._tilt_done)
        self._tilt_job.failed.connect(self._tilt_failed)  # a method: queued to the window's thread
        self._tilt_thread.start()
        self._set_buttons()
        return None

    def _end_tilt(self) -> None:
        if self._tilt_thread is not None:
            self._tilt_thread.quit()
            self._tilt_thread.wait()
            self._tilt_thread, self._tilt_job = None, None
        self._set_buttons()

    def _tilt_done(self, res: Any, key: str) -> None:
        self._end_tilt()
        self.tilt_result, self._tilt_key = res, key
        if self.tilt_show.currentData() == "image":
            self.tilt_show.setCurrentIndex(self.tilt_show.findData("height"))
        self._update_3d_tab()
        self.tabs.setCurrentIndex(TAB_3D)
        self._refresh()

    def _tilt_shape(self, tool: str, points: list[list[float]]) -> None:
        if self.ms is None or self.ms.tilt3d is None:
            return
        (x0, y0), (x1, y1) = points
        if tool == "hprofile":
            self.set_tool("pan")
            self._push_undo()
            self.ms.tilt3d.profiles.append([[x0, y0], [x1, y1]])
            self._changed()
            return
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        if tool == "box_lower":
            self._box_first = box
            self.view.show_region(box)
            self.set_tool("box_upper")
            self.say(tr("再框选上平面"))
            return
        self.set_tool("pan")
        self.view.show_region(None)
        if self._box_first is not None:
            self._push_undo()
            self.ms.tilt3d.steps.append({"lower": self._box_first, "upper": box})
            self._box_first = None
            self._changed()

    def _start_step(self) -> None:
        self._box_first = None
        self.set_tool("box_lower")
        self.say(tr("先框选下平面（点击两个对角）"))

    def add_step_height(self, lower: list[float], upper: list[float]) -> None:  # tests and scripts
        self._push_undo()
        self.ms.tilt3d.steps.append({"lower": lower, "upper": upper})
        self._changed()

    def add_height_profile(self, a: list[float], b: list[float]) -> None:  # tests and scripts
        self._push_undo()
        self.ms.tilt3d.profiles.append([a, b])
        self._changed()

    def _refresh_tilt(self) -> None:
        from vexel.packs.sem.tilt3d import analyse

        if self.ms is None:
            return
        self._fill_tilt_table()
        spec = self.ms.tilt3d
        res = self.tilt_result
        if spec is None or res is None:
            self.tilt_info.setText("" if spec is None else tr("添加倾转图后点“重建高度图”"))
            self.tilt_notes.hide()
            self.height_plot.hide()
            self._show_tilt_picture()
            return
        # cheap: the measurements on the height map follow the record (profiles, steps, levelling)
        stale = self._tilt_key != self._recon_key(spec)
        res = analyse(res.hm, spec, res.warnings)
        self.tilt_result = res
        hm = res.hm
        unit, f = ("nm", 1e-9) if hm.pixel_size else (tr("像素"), 1.0)
        valid = hm.valid
        lines = []
        if valid.any():
            zr = float(np.percentile(res.z[valid], 99) - np.percentile(res.z[valid], 1))
            lines.append(tr(f"高度范围（1–99%）：<b>{zr / f:.4g} {unit}</b>；可靠像素 {valid.mean():.0%}；不确定度中位数 {np.median(hm.uncertainty[valid]) / f:.3g} {unit}"))
        r = res.roughness
        if r:
            lines.append(tr("粗糙度（ISO 25178，校平后）：") + tr("，").join(f"{k} <b>{r[k] / f:.4g}</b> {unit}" for k in ("Sa", "Sq", "Sz")) + tr(f"，Ssk {r['Ssk']:.3f}，Sku {r['Sku']:.3f}"))
        for i, st in enumerate(res.steps, start=1):
            lines.append(tr(f"台阶 {i}：") + (st["error"] if "error" in st else f"<b>{st['step'] / f:.4g} ± {st['uncertainty'] / f:.2g} {unit}</b>"))
        for i, p in enumerate(res.profiles, start=1):
            lines.append(tr(f"剖面 {i}：高差 {np.ptp(p['height']) / f:.4g} {unit}"))
            pr = p.get("roughness") or {}
            if pr:
                lines.append("&nbsp;&nbsp;" + tr("，").join(f"{q} <b>{pr[q] / f:.4g}</b> {unit}" for q in ("Ra", "Rq", "Rz")) + tr(f"（ISO 21920，λc {pr['cutoff'] / f:.4g} {unit}）"))
        if hm.pairs:
            lines.append(tr("倾角对：") + tr("；").join(tr(f"{p['tilt_reference']:g}° / {p['tilt']:g}°（匹配 {p['match']:.2f}）") for p in hm.pairs))
        if stale:
            lines.insert(0, tr("<b>倾角、图像或设置已改变：点“重建高度图”更新</b>"))
        self.tilt_info.setText("<br>".join(lines))
        self.tilt_notes.setText("\n".join(f"• {tr(w)}" for w in res.warnings))
        self.tilt_notes.setVisible(bool(res.warnings))
        if res.profiles:
            p = res.profiles[-1]
            self.height_plot.show()
            self.height_plot.show_profile({"profile": p["height"], "distance": p["distance"], "edges": [], "polarity": []})
        else:
            self.height_plot.hide()
        self._show_tilt_picture()
        shapes = [("polyline", [list(a), list(b)], "#f472b6", tuple(b), f"H{i}") for i, (a, b) in enumerate(spec.profiles, start=1)]
        for i, st in enumerate(spec.steps, start=1):
            for key, colour in (("lower", "#22d3ee"), ("upper", "#facc15")):
                x0, y0, x1, y1 = st[key]
                shapes.append(("polygon", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], colour, (x0, y0), f"S{i}{'↓' if key == 'lower' else '↑'}"))
        self.view.show_analysis([sh for an in self.analyses if not an.error for sh in an.shapes] + shapes)

    def _show_tilt_picture(self) -> None:
        from vexel.packs.sem.tilt3d import colour

        res, mode = self.tilt_result, self.tilt_show.currentData() if hasattr(self, "tilt_show") else "image"
        if res is None or mode == "image" or self.image is None:
            if self.result is not None:
                self.view.set_picture(self.result.display, self.result.original)
            return
        hm = res.hm
        if mode == "height":
            pic = colour(res.z, hm.valid)
        else:
            g = (np.clip(hm.confidence, 0, 1) * 255).astype(np.uint8)
            pic = np.stack([g, g, g], -1)
        full = np.zeros((*self.image.shape, 3), np.uint8)
        full[: pic.shape[0], : pic.shape[1]] = pic
        if self.result is not None and pic.shape[0] < self.image.shape[0]:  # the data bar stays as it is
            bar = self.result.display[pic.shape[0] :]
            full[pic.shape[0] :] = np.stack([bar] * 3, -1)
        self.view.set_picture(full, self.result.original if self.result is not None else None)

    def _show_3d_preview(self) -> None:
        from qtpy.QtWidgets import QDialog

        from vexel.packs.sem.tilt3d import render_3d
        from vexel.ui.sem_canvas import to_qimage

        res = self.tilt_result
        if res is None:
            return
        from qtpy.QtGui import QPixmap

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("三维预览（示意，不用于测量）"))
        lay = QVBoxLayout(dlg)
        pic = QLabel()
        lay.addWidget(pic, 1)
        form = QFormLayout()
        sliders = {}
        for key, label, lo, hi, val in (("azimuth", tr("方位角"), 0, 360, 225), ("elevation", tr("仰角"), 10, 80, 35), ("exaggeration", tr("高度放大"), 1, 20, 1)):
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            form.addRow(label, sl)
            sliders[key] = sl
        lay.addLayout(form)

        def draw() -> None:
            img = render_3d(res.z, azimuth=sliders["azimuth"].value(), elevation=sliders["elevation"].value(),
                            exaggeration=sliders["exaggeration"].value(), mask=res.hm.valid, sx=res.hm.sx, sy=res.hm.sy)  # fmt: skip
            pic.setPixmap(QPixmap.fromImage(to_qimage(img)))

        for sl in sliders.values():
            sl.valueChanged.connect(lambda _v: draw())
        draw()
        dlg.resize(700, 620)
        self._preview_dialog = dlg
        dlg.show()

    def _tilt_sheet(self, path: Path) -> None:
        """The height map's results as one more sheet of an Excel export."""
        import pandas as pd

        res = self.tilt_result
        hm = res.hm
        unit, f = ("nm", 1e-9) if hm.pixel_size else ("px", 1.0)
        rows = [{"quantity": f"{k} ({unit})" if k in ("Sa", "Sq", "Sp", "Sv", "Sz") else k, "value": v / f if k in ("Sa", "Sq", "Sp", "Sv", "Sz") else v}
                for k, v in res.roughness.items()]  # fmt: skip
        rows += [{"quantity": f"step {i} ({unit})", "value": st.get("step", float("nan")) / f, "uncertainty": st.get("uncertainty", float("nan")) / f,
                  "lower box": str(st["lower"]), "upper box": str(st["upper"])} for i, st in enumerate(res.steps, start=1)]  # fmt: skip
        rows += [{"quantity": "reliable fraction", "value": float(hm.valid.mean())},
                 {"quantity": "tilts (°)", "value": ", ".join(f"{p['tilt_reference']:g}/{p['tilt']:g}" for p in hm.pairs)}]  # fmt: skip
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as xw:
            pd.DataFrame(rows).to_excel(xw, sheet_name=tr("三维"), index=False)

    def _export_tilt(self) -> None:
        if self.tilt_result is None or self.path is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("导出三维"), str(self.path.with_name(self.path.stem + "_height")),
                                              tr("高度图 TIFF (*.tif);;STL 网格 (*.stl);;OBJ 网格 (*.obj);;彩色高度图 PNG (*.png);;剖面 CSV (*.csv)"))  # fmt: skip
        if path:
            self.export_tilt(path)

    def export_tilt(self, path: str | Path) -> Path | None:
        from PIL import Image

        from vexel.packs.sem import tilt3d as T

        res = self.tilt_result
        if res is None:
            return None
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            out = T.save_tiff(res.hm, path, res.z, {"reference": self.path.name if self.path else None})
        elif suffix in (".stl", ".obj"):
            out = T.save_mesh(res.hm, path, res.z)
        elif suffix == ".png":
            Image.fromarray(T.colour(res.z, res.hm.valid)).save(path)
            out = path
        elif suffix == ".csv":
            import pandas as pd

            rows = [{"profile": i, "distance": d, "height": h, "confidence": c} for i, p in enumerate(res.profiles, start=1)
                    for d, h, c in zip(p["distance"], p["height"], p["confidence"])]  # fmt: skip
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            out = path
        else:
            raise ValueError(f"unknown export type: {suffix}")
        self.say(tr(f"已导出：{Path(out).name}"))
        return Path(out)

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
                self.say(tr(f"图像处理失败：{e}"))
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
        self.sat_info.setText(tr("图像处理中…"))
        self.say(tr("图像处理中…（完成后数值和图像自动更新）"))

    _auto_thread: QThread | None = None
    _auto_job: _AutoJob | None = None

    def _start_auto(self, todo: list[tuple[str, Any]], img: Any, mask: Any, crop: Any) -> None:
        if self._auto_thread is not None:
            return  # one at a time: when it is done, what is still missing runs next
        self._auto_thread = QThread()
        self._auto_job = _AutoJob(todo, img, self.ms.scale.model_copy(deep=True), mask, crop)
        self._auto_job.moveToThread(self._auto_thread)
        self._auto_thread.started.connect(self._auto_job.run)
        self._auto_job.done.connect(self._auto_done)
        self._auto_thread.start()
        self.say(tr("自动测量计算中…（大图需要一些时间，窗口可以继续使用）"))

    def _end_auto(self) -> None:
        if self._auto_thread is not None:
            self._auto_thread.quit()
            self._auto_thread.wait()
            self._auto_thread, self._auto_job = None, None

    def _auto_done(self, results: dict[str, Any]) -> None:
        self._end_auto()
        if self.ms is None:
            return
        cache = getattr(self, "_an_cache", {})
        cache.update(results)
        self._an_cache = cache
        self._auto_key = None
        self._refresh()

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
            self.say(tr("图像处理完成"))
        self._refresh()  # also starts the chain in force if it changed meanwhile

    def _processing_failed(self, msg: str, key: str, processor: Any) -> None:
        self._end_processing()
        if processor is self.processor:
            self._result_key = key  # do not retry the same chain
            self.say(tr(f"图像处理失败：{msg}"))

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
            self.sat_info.setText(tr("未标记饱和区"))
        elif sat.get("pixels"):
            self.sat_info.setText(tr(f"饱和像素 {sat['pixels']:,}（{sat['fraction']:.2%}），{sat['regions']} 处；阈值 {sat['high']:g}")
                                  + (tr(f"；暗端 {sat['low']:g}") if sat.get("low") is not None else ""))  # fmt: skip
        else:
            self.sat_info.setText(tr("没有发现饱和区"))
        self._show_sat_badge()
        notes = [n for n in res.notes if not n.startswith("饱和区")]  # pack notes, still untranslated here
        self.proc_notes.setText("\n".join(f"• {tr(n)}" for n in notes))
        self.proc_notes.setVisible(bool(notes))

    def _show_sat_badge(self) -> None:
        """A warning above the panel when part of the image is saturated (0.05 % or more) and
        the red marking is not shown."""
        res = self.result
        sat = res.saturation if res is not None else {}
        frac = float(sat.get("fraction") or 0.0) if self.ms is not None and self.ms.processing.saturation.enabled else 0.0
        show = frac >= 5e-4 and not self.sat_show.isChecked()
        if show:
            self.sat_badge.setText(tr(f"⚠ 图中有饱和区（{frac:.2%}）：那里的信息已经丢失，魔棒和线剖面不使用这些像素。") + f" <a href='show'>{tr('在图上显示')}</a>")
        self.sat_badge.setVisible(show)

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
            label = f"{i}. {tr(op.label) if op else st.op}" + (tr("（仅显示）") if op and op.role == "display" else "")
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if st.enabled else Qt.CheckState.Unchecked)
            if op:
                it.setToolTip(tr(op.help))
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
            QMessageBox.warning(self, tr("无法处理"), str(e))
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

        self.processing_act.setChecked(True)
        self.tabs.setCurrentIndex(TAB_PROCESSING)
        new = step(op, **params)
        steps = self.ms.processing.steps if self.ms is not None else []
        index = len(steps)
        if OPS[op].role == "correct":
            index = next((i for i, s in enumerate(steps) if OPS.get(s.op) and OPS[s.op].role == "display"), len(steps))
        self._edit_processing(lambda p: p.steps.insert(index, new), select=index)
        if tr((OPS[op].caution)):
            self.say(tr(f"{tr((OPS[op].label))}：{tr((OPS[op].caution))}"))

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
            self.say(tr("“仅显示”的步骤总在校正步骤之后运行"))
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
            self.step_help.setText(tr(f"未知的处理步骤：{st.op}"))
            return
        vals = op.resolve(st.params)
        for prm in op.params:
            if prm.kind == "choice":
                w: Any = QComboBox()
                for value, label in prm.choices:
                    w.addItem(tr(label), value)
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
            if tr(prm.help):
                w.setToolTip(tr(prm.help))
            self._param_widgets[prm.name] = (prm, w)
            self.params_form.addRow(tr(prm.label), w)
        self.step_help.setText(tr(op.help))  # its caution is in the notes below
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
        paths, _ = QFileDialog.getOpenFileNames(self, tr("选择同一位置的其他帧（本图为第 1 帧）"), str(self.path.parent), tr("图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"))
        paths = [p for p in paths if Path(p) != self.path]
        if not paths:
            return
        out, _ = QFileDialog.getSaveFileName(self, tr("保存平均后的图"), str(self.path.with_name(f"{self.path.stem}_avg{len(paths) + 1}.tif")), "TIFF (*.tif)")
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
            QMessageBox.warning(self, tr("无法平均"), str(e))
            return None
        QGuiApplication.restoreOverrideCursor()
        drift = max(float(np.hypot(*s)) for s in info["shifts"])
        self.open_path(out, wait=True)
        self.say(tr(f"{info['frames']} 帧平均：噪声 {info['noise_single']:.3g} → {info['noise_mean']:.3g}；最大漂移 {drift:.2f} 像素"))
        return info

    def _export_processed(self) -> None:
        if self.path is None:
            return
        out, _ = QFileDialog.getSaveFileName(self, tr("导出处理后的图"), str(self.path.with_name(self.path.stem + "_processed.tif")), "TIFF (*.tif)")
        if out:
            self.export_processed(out)

    def export_processed(self, path: str | Path) -> Path | None:
        from vexel.packs.sem.process import describe, write_derived

        res = self._processed(block=True)
        if res is None or self.path is None:
            return None
        out = write_derived(path, res.analysis, self.cal, {"operation": tr("图像处理"), "source": self.path.name, "steps": describe(self.ms.processing)})
        self.say(tr(f"已导出：{Path(out).name}"))
        return Path(out)


__all__ = ["SemWindow", "json"]
