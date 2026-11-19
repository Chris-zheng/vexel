"""SEM module window (in development: roadmap 0.0.5). M1: open an image, calibrate it, confirm.

The image view shows the micrograph, the data bar (shaded: it is excluded from analysis)
and the scale bar Vexel measured (outlined). The panel on the right says how many
nanometres a pixel is, where that came from, how it was checked, and whether a person
has confirmed it. A calibration read from the data bar counts only after "确认".

Manual calibration: click both ends of the scale bar and type its length, or type the
pixel size. Every confirmed calibration is recorded against the file's SHA-256.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from qtpy.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vexel.ui.widgets import heading, muted

VENDORS = {"thermo_fisher": "Thermo Fisher（FEI）", "zeiss": "Zeiss", "tescan": "TESCAN", "hitachi": "Hitachi", "jeol": "JEOL"}
SOURCES = {"metadata": "仪器元数据", "databar": "信息栏比例尺识别", "manual": "手动", "none": "无"}
STATE_COLOURS = {"confirmed": "#16a34a", "pending": "#d97706", "none": "#dc2626"}


def to_qimage(img: np.ndarray) -> QImage:
    lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.5))
    gray = np.clip((img.astype(np.float64) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    gray = np.ascontiguousarray(gray)
    return QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format.Format_Grayscale8).copy()


class ImageView(QGraphicsView):
    """Zoomable image with overlays; in "pick" mode two clicks give a length in pixels."""

    picked = Signal(float, float, float, float)  # x0, y0, x1, y1 (image pixels)
    hovered = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.picking = False
        self._points: list[QPointF] = []
        self._pick_items: list[Any] = []
        self.overlay: list[Any] = []

    def set_image(self, img: np.ndarray) -> None:
        self.scene().clear()
        self.overlay, self._pick_items, self._points = [], [], []
        self.scene().addPixmap(QPixmap.fromImage(to_qimage(img)))
        self.scene().setSceneRect(QRectF(0, 0, img.shape[1], img.shape[0]))
        self.fit()

    def fit(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def show_calibration(self, cal: Any) -> None:
        for item in self.overlay:
            self.scene().removeItem(item)
        self.overlay = []
        rect = self.scene().sceneRect()
        if cal.crop_rows:
            shade = self.scene().addRect(QRectF(0, cal.crop_rows, rect.width(), rect.height() - cal.crop_rows),
                                         QPen(QColor(37, 99, 235), 0), QBrush(QColor(37, 99, 235, 45)))  # fmt: skip
            self.overlay.append(shade)
        if cal.scale_bar and "x0" in cal.scale_bar:
            sb = cal.scale_bar
            pen = QPen(QColor("#f59e0b"), 2)
            pen.setCosmetic(True)
            box = self.scene().addRect(QRectF(sb["x0"] - 3, sb["y"] - 6, sb["x1"] - sb["x0"] + 6, 12), pen)
            self.overlay.append(box)

    def start_pick(self) -> None:
        self.picking = True
        self._points = []
        for item in self._pick_items:
            self.scene().removeItem(item)
        self._pick_items = []
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def stop_pick(self) -> None:
        self.picking = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().unsetCursor()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        if not self.picking or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        p = self.mapToScene(event.position().toPoint())
        self._points.append(p)
        pen = QPen(QColor("#f59e0b"), 2)
        pen.setCosmetic(True)
        self._pick_items.append(self.scene().addEllipse(QRectF(p.x() - 2, p.y() - 2, 4, 4), pen))
        if len(self._points) == 2:
            a, b = self._points
            self._pick_items.append(self.scene().addLine(a.x(), a.y(), b.x(), b.y(), pen))
            self.stop_pick()
            self.picked.emit(a.x(), a.y(), b.x(), b.y())

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        p = self.mapToScene(event.position().toPoint())
        self.hovered.emit(p.x(), p.y())
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)


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
        self.resize(1280, 820)
        self.path: Path | None = None
        self.image: np.ndarray | None = None
        self.cal: Any = None
        self._thread: QThread | None = None
        self._job: _CalibrateJob | None = None

        central = QWidget()
        root = QHBoxLayout(central)
        self.view = ImageView()
        self.view.picked.connect(self._picked)
        self.view.hovered.connect(self._hover)
        root.addWidget(self.view, 1)

        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(340)
        v = QVBoxLayout(panel)
        self.open_btn = QPushButton("打开 SEM 图片…")
        self.open_btn.clicked.connect(self._choose)
        v.addWidget(self.open_btn)
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
        self.pick_btn.clicked.connect(self._start_pick)
        self.size_btn = QPushButton("输入像素尺寸…")
        self.size_btn.clicked.connect(self._enter_size)
        row.addWidget(self.pick_btn)
        row.addWidget(self.size_btn)
        v.addLayout(row)
        v.addWidget(heading("仪器信息"))
        self.meta = muted("")
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.meta)
        v.addStretch(1)
        v.addWidget(muted("下一阶段（M2）：长度、面积、角度、周长、计数与线剖面测量。"))
        root.addWidget(panel)
        self.setCentralWidget(central)
        self._set_buttons()
        self.statusBar().showMessage("打开一张 SEM 图片开始")

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
        self.path, self.image, self.cal = path, img, None
        self.view.set_image(img)
        self.file_label.setText(f"{path.name}：{img.shape[1]}×{img.shape[0]} 像素")
        self.pixel.setText("…")
        self.state.setText("正在读取元数据与信息栏…")
        self.detail.setText("")
        self.notes.hide()
        self.meta.setText("")
        self._set_buttons()
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

    # -- showing ------------------------------------------------------------------------

    def _show(self, cal: Any) -> None:
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
            key, text = "none", cal.reason
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
            lines.append(f"分析区域：第 0–{cal.crop_rows - 1} 行；下方 {cal.image_shape[0] - cal.crop_rows} 行是信息栏（{how}），不参与分析")
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
            for key, label, fmt in (("voltage", "加速电压", lambda v: f"{v / 1e3:.3g} kV"), ("magnification", "放大倍数", lambda v: f"{v:,.0f}×"),
                                    ("hfw", "视场宽度", format_length), ("working_distance", "工作距离", format_length),
                                    ("tilt", "样品台倾角", lambda v: f"{v:.2f}°"), ("detector", "探测器", str), ("date", "时间", str)):  # fmt: skip
                if md.get(key) not in (None, ""):
                    info.append(f"{label}：{fmt(md[key])}")
        else:
            info.append("文件中没有仪器元数据")
        self.meta.setText("\n".join(info))
        self._set_buttons()
        self.statusBar().showMessage(cal.describe())

    def _set_buttons(self) -> None:
        has = self.path is not None
        self.confirm_btn.setVisible(bool(self.cal is not None and self.cal.needs_confirmation))
        self.pick_btn.setEnabled(has)
        self.size_btn.setEnabled(has)

    def _hover(self, x: float, y: float) -> None:
        if self.image is None or not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            return
        text = f"x {x:.0f}, y {y:.0f} 像素"
        if self.cal is not None and self.cal.ok:
            from vexel.packs.sem.calibrate import format_length

            text += f"  ({format_length(x * self.cal.pixel_size)}, {format_length(y * (self.cal.pixel_size_y or self.cal.pixel_size))})"
        if self.view.picking:
            text = "点击比例尺的" + ("右端" if self.view._points else "左端") + "  ·  " + text
        self.statusBar().showMessage(text)

    # -- confirming and manual calibration ------------------------------------------------

    def _confirm(self) -> None:
        from vexel.packs.sem.calibrate import save

        if self.cal is None or not self.cal.ok:
            return
        self.cal.confirm("person")
        save(self.cal)
        self._show(self.cal)

    def _start_pick(self) -> None:
        self.view.start_pick()
        self.statusBar().showMessage("点击比例尺的左端")

    def _picked(self, x0: float, y0: float, x1: float, y1: float) -> None:
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        if length_px < 2:
            return
        text, ok = QInputDialog.getText(self, "比例尺长度", f"两点相距 {length_px:.1f} 像素。它代表的长度（例如 500 nm、2 µm）：")
        if ok and text.strip():
            self.apply_manual(bar_length=text, bar_px=length_px)

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
