"""SEM measurement canvas: a zoomable image on which the person draws measurements.

Tools (``ImageView.set_tool``) and how to use them:

==========  ===========================================================================
tool        clicks
==========  ===========================================================================
pan         drag to move, wheel to zoom; click a measurement to select it
line        two clicks
profile     two clicks (a line profile across lines, edges or a step)
angle       three clicks: an arm, the vertex, the other arm
polyline    clicks; double-click or Enter to finish
polygon     clicks; double-click or Enter to close
freehand    press, draw, release
wand        one click inside a region
circle      three or more clicks on the edge; double-click or Enter to fit
ellipse     five or more clicks on the edge; double-click or Enter to fit
point       each click marks one object; Enter (or another tool) ends the count
==========  ===========================================================================

Esc cancels the shape being drawn; Backspace removes its last point. A selected
measurement shows handles: drag one to move that point.

The picture can be replaced without touching the drawings (``set_picture``: after image
processing), shown split into before (left) and after (right) at ``set_split``, with the
saturated pixels tinted red (``set_mask``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

AUTO_FINISH = {"line": 2, "profile": 2, "angle": 3, "calibrate": 2}
FINISH_ON_ENTER = {"polyline": 2, "polygon": 3, "circle": 3, "ellipse": 5, "point": 1}
TOOL_HINTS = {
    "pan": "拖动平移，滚轮缩放；点击测量可选中，拖动小圆点修改",
    "line": "点击起点和终点",
    "profile": "点击剖面线的起点和终点（横穿线条或台阶）",
    "angle": "依次点击：一条边上的点、顶点、另一条边上的点",
    "polyline": "逐点点击；双击或按回车结束",
    "polygon": "逐点点击轮廓；双击或按回车闭合",
    "freehand": "按住鼠标沿轮廓画，松开结束",
    "wand": "点击区域内部，自动找出连通的相似区域",
    "circle": "在圆周上点击三个或更多点；双击或回车拟合",
    "ellipse": "在椭圆周上点击五个或更多点；双击或回车拟合",
    "point": "点击每个要计数的目标；回车结束这一组",
    "calibrate": "依次点击比例尺的两端",
}


def to_qimage(img: np.ndarray, stretch: bool = True) -> QImage:
    """Grey image to QImage: stretched 0.5-99.5 % to black-white, or (``stretch`` False,
    uint8 already prepared for display) as it is."""
    if stretch or img.dtype != np.uint8:
        lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.5))
        gray = np.clip((img.astype(np.float64) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    else:
        gray = img
    gray = np.ascontiguousarray(gray)
    return QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format.Format_Grayscale8).copy()


def _pen(colour: str, width: float = 2.0, dash: bool = False) -> QPen:
    pen = QPen(QColor(colour), width)
    pen.setCosmetic(True)  # constant width on screen at any zoom
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


class Handle(QGraphicsEllipseItem):
    """A draggable point of the selected measurement."""

    def __init__(self, view: "ImageView", item_id: int, index: int, x: float, y: float) -> None:
        super().__init__(-5, -5, 10, 10)
        self.view, self.item_id, self.index = view, item_id, index
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("white")))
        self.setPen(_pen("#111827", 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)  # same size at any zoom
        self.setZValue(10)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        super().mouseReleaseEvent(event)
        self.view.point_moved.emit(self.item_id, self.index, self.pos().x(), self.pos().y())


class ImageView(QGraphicsView):
    shape_done = Signal(str, list)  # tool, [[x, y], ...]
    point_moved = Signal(int, int, float, float)  # item id, point index, x, y
    item_clicked = Signal(int)  # item id (or -1: empty space)
    hovered = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.tool = "pan"
        self.points: list[QPointF] = []
        self._preview: list[Any] = []
        self._drawn: dict[int, list[Any]] = {}
        self._handles: list[Handle] = []
        self.overlay: list[Any] = []  # calibration overlay (data bar, scale bar)
        self._pix: Any = None  # the picture
        self._before: Any = None  # the unprocessed picture, left of the split
        self._divider: Any = None
        self._mask_item: Any = None
        self._original: QImage | None = None
        self.split: float | None = None  # 0..1 of the width, or None: no comparison
        self.shapes: list[tuple[int, str, Any, str, tuple[float, float], str]] = []
        self._freehand = False
        self.selected: int | None = None
        self.set_tool("pan")

    # -- image and calibration overlay ---------------------------------------------------------

    def set_image(self, img: np.ndarray) -> None:
        self.scene().clear()
        self.overlay, self._preview, self._handles, self._drawn, self.points = [], [], [], {}, []
        self._before = self._divider = self._mask_item = self._original = None
        self.split = None
        self._pix = self.scene().addPixmap(QPixmap.fromImage(to_qimage(img)))
        self._pix.setZValue(-10)
        self.scene().setSceneRect(QRectF(0, 0, img.shape[1], img.shape[0]))
        self.fit()

    def set_picture(self, display: np.ndarray, original: np.ndarray | None = None) -> None:
        """Show ``display`` (uint8, prepared) in place of the picture, keeping drawings and zoom;
        ``original`` is what the before/after view shows on the left."""
        if self._pix is None:
            return
        self._pix.setPixmap(QPixmap.fromImage(to_qimage(display, stretch=False)))
        self._original = to_qimage(original, stretch=False) if original is not None else None
        self.set_split(self.split)

    def set_split(self, split: float | None) -> None:
        """Before/after view: the unprocessed picture left of ``split`` (0..1 of the width)."""
        self.split = split
        for it in (self._before, self._divider):
            if it is not None:
                self.scene().removeItem(it)
        self._before = self._divider = None
        if split is None or self._original is None:
            return
        w, h = self._original.width(), self._original.height()
        x = int(round(min(max(split, 0.0), 1.0) * w))
        if x > 0:
            self._before = self.scene().addPixmap(QPixmap.fromImage(self._original.copy(0, 0, x, h)))
            self._before.setZValue(-9)
        pen = _pen("#facc15", 2)
        pen.setCosmetic(True)
        self._divider = self.scene().addLine(x, 0, x, h, pen)
        self._divider.setZValue(-7)

    def set_mask(self, mask: np.ndarray | None, visible: bool = True) -> None:
        """Tint ``mask`` pixels (saturated) red."""
        if self._mask_item is not None:
            self.scene().removeItem(self._mask_item)
            self._mask_item = None
        if mask is None or not visible or not mask.any():
            return
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask] = (239, 68, 68, 120)
        rgba = np.ascontiguousarray(rgba)
        img = QImage(rgba.data, mask.shape[1], mask.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        self._mask_item = self.scene().addPixmap(QPixmap.fromImage(img))
        self._mask_item.setZValue(-8)

    def fit(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def show_calibration(self, cal: Any) -> None:
        for item in self.overlay:
            self.scene().removeItem(item)
        self.overlay = []
        rect = self.scene().sceneRect()
        if cal.crop_rows:
            shade = self.scene().addRect(QRectF(0, cal.crop_rows, rect.width(), rect.height() - cal.crop_rows),
                                         _pen("#2563eb", 1), QBrush(QColor(37, 99, 235, 45)))  # fmt: skip
            self.overlay.append(shade)
        if cal.scale_bar and "x0" in cal.scale_bar:
            sb = cal.scale_bar
            self.overlay.append(self.scene().addRect(QRectF(sb["x0"] - 3, sb["y"] - 6, sb["x1"] - sb["x0"] + 6, 12), _pen("#f59e0b")))

    # -- tools --------------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        if self.tool == "point" and tool != "point" and self.points:
            self.finish()
        self.cancel()
        self.tool = tool
        drag = QGraphicsView.DragMode.ScrollHandDrag if tool == "pan" else QGraphicsView.DragMode.NoDrag
        self.setDragMode(drag)
        if tool == "pan":
            self.viewport().unsetCursor()
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def cancel(self) -> None:
        self.points = []
        self._freehand = False
        self._update_preview(None)

    def finish(self) -> None:
        need = FINISH_ON_ENTER.get(self.tool)
        if need is not None and len(self.points) >= need:
            pts = [[p.x(), p.y()] for p in self.points]
            tool = self.tool
            self.cancel()
            self.shape_done.emit(tool, pts)

    def _add_point(self, p: QPointF) -> None:
        self.points.append(p)
        n = AUTO_FINISH.get(self.tool)
        if n is not None and len(self.points) == n:
            pts = [[q.x(), q.y()] for q in self.points]
            tool = self.tool
            self.cancel()
            self.shape_done.emit(tool, pts)
        else:
            self._update_preview(None)

    def _update_preview(self, cursor: QPointF | None) -> None:
        for item in self._preview:
            self.scene().removeItem(item)
        self._preview = []
        pts = list(self.points) + ([cursor] if cursor is not None and self.points and not self._freehand and self.tool != "point" else [])
        if not pts:
            return
        pen = _pen("#f59e0b", 2, dash=True)
        if self.tool in ("point", "circle", "ellipse"):
            for p in self.points:
                self._preview.append(self.scene().addEllipse(QRectF(p.x() - 4, p.y() - 4, 8, 8), _pen("#f59e0b", 2)))
            return
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        if self.tool == "polygon" and len(pts) > 2:
            path.closeSubpath()
        self._preview.append(self.scene().addPath(path, pen))

    # -- drawing the measurements --------------------------------------------------------

    def draw(self, shapes: list[tuple[int, str, Any, str, tuple[float, float], str]], selected: int | None = None) -> None:
        """Redraw every measurement: (id, kind of drawing, geometry, colour, label anchor, label)."""
        for items in self._drawn.values():
            for it in items:
                self.scene().removeItem(it)
        self._drawn = {}
        self.shapes = shapes
        self.selected = selected
        font = QFont()
        font.setPointSize(11)
        for item_id, kind, geom, colour, anchor, label in shapes:
            width = 3.0 if item_id == selected else 2.0
            pen = _pen(colour, width)
            drawn: list[Any] = []
            if kind == "points":
                for x, y in geom:
                    drawn.append(self.scene().addEllipse(QRectF(x - 4, y - 4, 8, 8), pen))
            elif kind in ("polygon", "polyline"):
                path = QPainterPath(QPointF(*geom[0]))
                for x, y in geom[1:]:
                    path.lineTo(x, y)
                if kind == "polygon":
                    path.closeSubpath()
                drawn.append(self.scene().addPath(path, pen))
            elif kind == "ellipse":
                cx, cy, a, b, deg = geom
                path = QPainterPath()
                t = np.linspace(0, 2 * np.pi, 121)
                th = math.radians(deg)
                xs = cx + a * np.cos(t) * math.cos(th) + b * np.sin(t) * math.sin(th)
                ys = cy - a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)
                path.moveTo(xs[0], ys[0])
                for x, y in zip(xs[1:], ys[1:]):
                    path.lineTo(x, y)
                drawn.append(self.scene().addPath(path, pen))
            text = QGraphicsSimpleTextItem(label)
            text.setFont(font)
            text.setBrush(QBrush(QColor(colour)))
            text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            text.setPos(anchor[0] + 4, anchor[1] - 4)
            back = self.scene().addRect(QRectF(text.boundingRect()).adjusted(-2, -1, 2, 1), QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0, 170)))
            back.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            back.setPos(text.pos())
            self.scene().addItem(text)
            drawn += [back, text]
            for d in drawn:
                d.setData(0, item_id)
            self._drawn[item_id] = drawn

    def show_handles(self, item_id: int | None, points: list[tuple[float, float]] | None) -> None:
        for h in self._handles:
            self.scene().removeItem(h)
        self._handles = []
        if item_id is None or not points:
            return
        for i, (x, y) in enumerate(points):
            h = Handle(self, item_id, i, x, y)
            self.scene().addItem(h)
            self._handles.append(h)

    # -- events ---------------------------------------------------------------------------

    def _scene_pos(self, event: Any) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        p = self._scene_pos(event)
        if self.tool == "pan":
            hit = [it for it in self.items(event.position().toPoint()) if isinstance(it, Handle)]
            if hit:
                return super().mousePressEvent(event)
            ids = [it.data(0) for it in self.items(event.position().toPoint()) if it.data(0) is not None]
            self.item_clicked.emit(int(ids[0]) if ids else -1)
            return super().mousePressEvent(event)
        if self.tool == "freehand":
            self.points = [p]
            self._freehand = True
            return
        if self.tool == "wand":
            self.shape_done.emit("wand", [[p.x(), p.y()]])
            return
        self._add_point(p)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        p = self._scene_pos(event)
        self.hovered.emit(p.x(), p.y())
        if self._freehand:
            last = self.points[-1]
            if math.hypot(p.x() - last.x(), p.y() - last.y()) >= 2.0:
                self.points.append(p)
                self._update_preview(None)
            return
        if self.points:
            self._update_preview(p)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        if self._freehand:
            pts = [[q.x(), q.y()] for q in self.points]
            self.cancel()
            if len(pts) >= 3:
                self.shape_done.emit("polygon", pts)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        if self.tool in FINISH_ON_ENTER and self.tool != "point":
            # the double-click's first press already added this point
            self.finish()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish()
        elif key == Qt.Key.Key_Escape:
            self.cancel()
        elif key == Qt.Key.Key_Backspace and self.points:
            self.points.pop()
            self._update_preview(None)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)


class ProfilePlot(QGraphicsView):
    """Grey level along a profile, with the edges found and the edge level."""

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMinimumHeight(140)
        self.setMaximumHeight(170)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def show_profile(self, values: dict[str, Any] | None) -> None:
        sc = self.scene()
        sc.clear()
        if not values or "profile" not in values:
            return
        y = np.asarray(values["profile"], float)
        d = np.asarray(values["distance"], float)
        w, h = 300.0, 120.0
        if len(y) < 2 or d[-1] <= 0:
            return
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, 1e-12)
        xs = d / d[-1] * w
        ys = h - (y - lo) / span * h
        path = QPainterPath(QPointF(xs[0], ys[0]))
        for a, b in zip(xs[1:], ys[1:]):
            path.lineTo(a, b)
        sc.addRect(QRectF(0, 0, w, h), _pen("#8a8f98", 1))
        sc.addPath(path, _pen("#e5e7eb", 1.5))
        for pos, pol in zip(values["edges"], values["polarity"]):
            x = pos / d[-1] * w
            sc.addLine(x, 0, x, h, _pen("#22c55e" if pol > 0 else "#ef4444", 1))
        self.fitInView(QRectF(-4, -4, w + 8, h + 8), Qt.AspectRatioMode.IgnoreAspectRatio)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        super().resizeEvent(event)
        self.fitInView(QRectF(-4, -4, 308, 128), Qt.AspectRatioMode.IgnoreAspectRatio)
