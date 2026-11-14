"""Review and correction window for a finished run.

Generic over analyses: the run's review adapter (from its pack) says which table lists
the objects, which labels the outline tools edit, how an object is excluded and which
values can be set. Edits are recorded in ``corrections.json`` next to the run and
applied by re-running the recipe (``corrected/``), so corrected results stay
reproducible. The original run is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QCoreApplication, QObject, QPointF, QRectF, Qt, QThread, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vexel.engine.corrections import Corrections, Edit
from vexel.engine.review import CORRECTED_DIR, FrameInspector, rerun_with_corrections, review_adapter
from vexel.ui.widgets import heading, muted

TOOLS = [
    ("select", "选择", "点一个目标，在右侧查看并修改"),
    ("delete", "删除误检", "点一个目标，删除它的轮廓（不是液滴的杂质、气泡等）"),
    ("paint", "补画", "按下拖动画一个圆，作为漏检的新目标"),
    ("erase", "擦除", "按下拖动画一个圆，擦掉轮廓的这一部分"),
    ("merge", "合并", "依次点两个目标，合成一个（一个液滴被切成两块时）"),
]
KIND_TEXT = {
    "labels.delete": "删除轮廓", "labels.merge": "合并", "labels.paint": "补画", "labels.erase": "擦除",
    "table.set": "修改", "table.exclude": "排除",
}  # fmt: skip


def _read(run_dir: Path, ref: str) -> pd.DataFrame | None:
    path = run_dir / "tables" / f"{ref}.csv"
    return pd.read_csv(path) if path.exists() else None


class Canvas(QWidget):
    """One frame with outlines; reports clicks and dragged circles in image pixels."""

    clicked = Signal(float, float)  # y, x
    circled = Signal(float, float, float)  # y, x, r

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 220)
        self.setMouseTracking(True)
        self.rgb: np.ndarray | None = None
        self.highlight: tuple[float, float, float] | None = None  # y, x, r
        self.drag_circles = False
        self.view: tuple[float, float, float, float] | None = None  # y0, x0, y1, x1 shown (None: whole frame)
        self._press: tuple[float, float] | None = None
        self._drag: tuple[float, float, float] | None = None

    def set_image(self, rgb: np.ndarray) -> None:
        self.rgb = np.ascontiguousarray(rgb)
        self.update()

    def _region(self) -> tuple[int, int, int, int]:
        h, w = self.rgb.shape[:2]
        if self.view is None:
            return 0, 0, h, w
        y0, x0, y1, x1 = self.view
        y0, x0 = max(0, int(y0)), max(0, int(x0))
        return y0, x0, min(h, int(np.ceil(y1))), min(w, int(np.ceil(x1)))

    def _geometry(self) -> tuple[float, float, float, int, int]:
        y0, x0, y1, x1 = self._region()
        scale = min(self.width() / (x1 - x0), self.height() / (y1 - y0))
        return scale, (self.width() - (x1 - x0) * scale) / 2, (self.height() - (y1 - y0) * scale) / 2, y0, x0

    def to_image(self, px: float, py: float) -> tuple[float, float]:
        scale, ox, oy, y0, x0 = self._geometry()
        return y0 + (py - oy) / scale, x0 + (px - ox) / scale

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        if self.rgb is None:
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, "正在载入…")
            return
        y0, x0, y1, x1 = self._region()
        crop = np.ascontiguousarray(self.rgb[y0:y1, x0:x1])
        scale, ox, oy, _, _ = self._geometry()
        h, w = crop.shape[:2]
        image = QImage(crop.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        p.drawImage(QRectF(ox, oy, w * scale, h * scale), image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for circle, color in ((self.highlight, "#facc15"), (self._drag, "#22d3ee")):
            if circle is not None:
                y, x, r = circle
                p.setPen(QPen(QColor(color), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(ox + (x - x0) * scale, oy + (y - y0) * scale), r * scale, r * scale)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.rgb is None:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        self._press = self.to_image(pos.x(), pos.y())
        if not self.drag_circles:
            self.clicked.emit(*self._press)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is not None and self.drag_circles:
            pos = event.position() if hasattr(event, "position") else event.pos()
            y, x = self.to_image(pos.x(), pos.y())
            self._drag = (*self._press, float(np.hypot(y - self._press[0], x - self._press[1])))
            self.update()

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._drag is not None and self._drag[2] >= 2:
            self.circled.emit(*self._drag)
        self._press, self._drag = None, None
        self.update()


class RerunWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, run_dir: Path, corrections: Corrections) -> None:
        super().__init__()
        self.run_dir, self.corrections = run_dir, corrections

    def run(self) -> None:
        try:
            result = rerun_with_corrections(self.run_dir, self.corrections, progress=self.progress.emit)
        except Exception as e:  # noqa: BLE001
            self._done(self.failed, f"{type(e).__name__}: {e}")
            return
        self._done(self.finished, str(result.out_dir))

    def _done(self, signal: Signal, *args: Any) -> None:
        app = QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        signal.emit(*args)


class ReviewWindow(QMainWindow):
    def __init__(self, run_dir: str | Path) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.warn = lambda title, text: QMessageBox.warning(self, title, text)
        self.inspector = FrameInspector(self.run_dir)
        self.corrections: Corrections = self.inspector.corrections
        self.adapter = review_adapter(self.inspector.recipe)
        self.objects = _read(self.run_dir, self.adapter.objects)
        self.tracks = _read(self.run_dir, self.adapter.exclude)
        self.flags = _read(self.run_dir, self.adapter.flags) if self.adapter.flags else None
        self.frame = 0
        self.selected: dict[str, Any] | None = None  # {"y", "x", "row" (object row or None)}
        self._merge_first: tuple[float, float] | None = None
        self._thread: QThread | None = None
        self._worker: RerunWorker | None = None
        self.setWindowTitle(f"Vexel · 复核与纠错 · {self.run_dir.name}")
        self.resize(1280, 820)
        self._build()
        self._fill_queue()
        self._fill_edits()
        first = self.queue.item(0)
        if first is not None:
            self.queue.setCurrentRow(0)
        else:
            self.show_frame(0)

    # -- layout --------------------------------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(14)

        left = QVBoxLayout()
        left.addWidget(heading(f"{self.adapter.object_name}列表"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("需要复核", "flagged")
        self.filter_combo.addItem("全部", "all")
        self.filter_combo.addItem("已修改", "edited")
        self.filter_combo.currentIndexChanged.connect(self._fill_queue)
        left.addWidget(self.filter_combo)
        self.queue = QListWidget()
        self.queue.setMinimumWidth(250)
        self.queue.setWordWrap(True)
        self.queue.currentItemChanged.connect(lambda item, _prev: self._goto_item(item))
        left.addWidget(self.queue, 1)
        outer.addLayout(left, 0)

        mid = QVBoxLayout()
        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons: dict[str, QPushButton] = {}
        for key, label, tip in TOOLS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            self.tool_group.addButton(b)
            self.tool_buttons[key] = b
            tools.addWidget(b)
            b.toggled.connect(lambda on, k=key: on and self.set_tool(k))
        tools.addStretch(1)
        self.zoom_btn = QPushButton("放大选中")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setChecked(True)
        self.zoom_btn.setToolTip("选中目标后放大到它周围；关掉则显示整帧")
        self.zoom_btn.toggled.connect(lambda _on: self._apply_zoom())
        tools.addWidget(self.zoom_btn)
        self.prev_btn, self.next_btn = QPushButton("◀ 上一帧"), QPushButton("下一帧 ▶")
        self.prev_btn.clicked.connect(lambda: self.show_frame(self.frame - 1))
        self.next_btn.clicked.connect(lambda: self.show_frame(self.frame + 1))
        tools.addWidget(self.prev_btn)
        tools.addWidget(self.next_btn)
        mid.addLayout(tools)
        self.canvas = Canvas()
        self.canvas.clicked.connect(self._on_click)
        self.canvas.circled.connect(self._on_circle)
        mid.addWidget(self.canvas, 1)
        self.caption = muted("")
        mid.addWidget(self.caption)
        outer.addLayout(mid, 1)

        right = QVBoxLayout()
        right.addWidget(heading("选中的" + self.adapter.object_name))
        self.sel_label = muted("在图上点一个目标，或从左侧列表选择。")
        right.addWidget(self.sel_label)
        form = QFormLayout()
        self.field_widgets: dict[str, Any] = {}
        for f in self.adapter.fields:
            if f.kind == "int":
                w = QSpinBox()
                w.setRange(0, 999)
            else:
                w = QComboBox()
                w.addItem("阳性", True)
                w.addItem("阴性", False)
            w.setToolTip(f.help)
            form.addRow(f.label, w)
            self.field_widgets[f.key] = w
        right.addLayout(form)
        row = QHBoxLayout()
        self.set_btn = QPushButton("保存修改")
        self.set_btn.setToolTip("把上面的值记为人工确认的结果")
        self.set_btn.clicked.connect(self.set_fields)
        self.verify_btn = QPushButton("确认无误")
        self.verify_btn.setToolTip("自动结果正确：标为人工确认，不再列为需要复核")
        self.verify_btn.clicked.connect(self.verify)
        row.addWidget(self.set_btn)
        row.addWidget(self.verify_btn)
        right.addLayout(row)
        self.exclude_btn = QPushButton(f"排除此{self.adapter.object_name}")
        self.exclude_btn.setToolTip("从统计中去掉整个目标（杂质、气泡、碎片等）")
        self.exclude_btn.clicked.connect(self.exclude)
        right.addWidget(self.exclude_btn)
        right.addSpacing(10)
        right.addWidget(heading("修改记录"))
        self.edit_list = QListWidget()
        right.addWidget(self.edit_list, 1)
        self.undo_btn = QPushButton("撤销最后一步")
        self.undo_btn.clicked.connect(self.undo)
        right.addWidget(self.undo_btn)
        self.apply_btn = QPushButton("应用并重算")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.setToolTip("按全部修改重新运行配方，结果写入 corrected/，原结果不变")
        self.apply_btn.clicked.connect(self.apply)
        right.addWidget(self.apply_btn)
        self.report_btn = QPushButton("打开修正后的报告")
        self.report_btn.clicked.connect(self._open_report)
        self.report_btn.setVisible((self.run_dir / CORRECTED_DIR / "report.html").exists())
        right.addWidget(self.report_btn)
        self.status = muted("")
        right.addWidget(self.status)
        holder = QWidget()
        holder.setLayout(right)
        holder.setFixedWidth(300)
        outer.addWidget(holder)
        self.setCentralWidget(central)
        self.tool_buttons["select"].setChecked(True)
        self._update_selection_ui()

    # -- queue -----------------------------------------------------------------------------

    def _object_rows(self) -> list[dict[str, Any]]:
        if self.objects is None:
            return []
        rows = self.objects.to_dict("records")
        reasons = {}
        if self.flags is not None and "track_id" in self.flags.columns:
            reasons = {int(r["track_id"]): r for r in self.flags.to_dict("records")}
        out = []
        for r in rows:
            tid = int(r["track_id"]) if "track_id" in r else None
            flag = reasons.get(tid)
            frame, y, x = self._anchor(r, flag)
            out.append({**r, "_flag": flag, "_frame": frame, "_y": y, "_x": x})
        return out

    def _anchor(self, row: dict[str, Any], flag: dict[str, Any] | None) -> tuple[int, float, float]:
        """Frame and position to show an object at: the flagged frame, else its detection
        nearest the counting line crossing."""
        if flag is not None:
            return int(flag["frame"]), float(flag["centroid_y_px"]), float(flag["centroid_x_px"])
        if self.tracks is not None and "track_id" in row:
            dets = self.tracks[self.tracks["track_id"] == row["track_id"]]
            if len(dets):
                target = row.get("crossing_frame", dets["frame"].median())
                d = dets.iloc[int(np.argmin(np.abs(dets["frame"].to_numpy() - float(target))))]
                return int(d["frame"]), float(d["centroid_y_px"]), float(d["centroid_x_px"])
        return int(row.get("frame", 0)), float(row.get("centroid_y_px", 0)), float(row.get("centroid_x_px", 0))

    def _edited_tracks(self) -> set[int]:
        out = set()
        if self.tracks is None:
            return out
        for e in self.corrections.edits:
            if e.at is None:
                continue
            dets = self.tracks[self.tracks["frame"] == e.frame]
            if len(dets):
                d = np.hypot(dets["centroid_y_px"] - e.at[0], dets["centroid_x_px"] - e.at[1]).to_numpy()
                i = int(np.argmin(d))
                reach = 0.6 * float(dets["diameter_px"].iloc[i]) if "diameter_px" in dets else 10.0
                if d[i] <= max(5.0, reach):
                    out.add(int(dets["track_id"].iloc[i]))
        return out

    def _fill_queue(self) -> None:
        mode = self.filter_combo.currentData()
        edited = self._edited_tracks()
        self.queue.blockSignals(True)
        self.queue.clear()
        rows = self._object_rows()
        if mode == "flagged":
            rows = [r for r in rows if r["_flag"] is not None]
        elif mode == "edited":
            rows = [r for r in rows if int(r.get("track_id", -1)) in edited]
        for r in rows:
            tid = r.get("track_id")
            mark = "［已修改］" if tid is not None and int(tid) in edited else ""
            why = "、".join(self.adapter.reasons.get(c, c) for c in str(r["_flag"]["reasons"]).split(";")) if r["_flag"] is not None else ""
            text = f"{mark}{self.adapter.object_name} {int(tid) if tid is not None else ''} · 第 {r['_frame']} 帧"
            detail = self.adapter.describe(r)
            item = QListWidgetItem(text + (f"\n{why}" if why else "") + (f"\n{detail}" if detail else ""))
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.queue.addItem(item)
        self.queue.blockSignals(False)
        if mode == "flagged" and not rows:
            self.queue.addItem(QListWidgetItem("没有需要复核的目标。选“全部”查看所有目标。"))

    def _goto_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        r = item.data(Qt.ItemDataRole.UserRole)
        if not r:
            return
        self.show_frame(int(r["_frame"]))
        self.select(float(r["_y"]), float(r["_x"]))

    # -- frames and selection --------------------------------------------------------------

    def show_frame(self, t: int) -> None:
        from vexel.ui.live import overlay_rgb

        t = int(np.clip(t, 0, self.inspector.n_frames - 1))
        self.frame = t
        image, labels, marks = self.inspector.view(t)
        self._labels = labels
        self.canvas.set_image(overlay_rgb(image, labels, marks))
        self.caption.setText(f"第 {t} 帧 / 共 {self.inspector.n_frames} 帧 · 红线为轮廓" + ("，黄圈为检出的细胞" if marks is not None else ""))
        self.prev_btn.setEnabled(t > 0)
        self.next_btn.setEnabled(t < self.inspector.n_frames - 1)
        follow = self.selected["track_id"] if self.selected is not None else None
        self.canvas.highlight, self.selected = None, None
        if follow is not None and self.tracks is not None:  # keep following the same object
            det = self.tracks[(self.tracks["frame"] == t) & (self.tracks["track_id"] == follow)]
            if len(det):
                self.select(float(det["centroid_y_px"].iloc[0]), float(det["centroid_x_px"].iloc[0]))
                return
        self._update_selection_ui()
        self._apply_zoom()

    def _object_at(self, y: float, x: float) -> dict[str, Any] | None:
        """The object row whose track has a detection under the point in this frame."""
        if self.tracks is None or self.objects is None:
            return None
        dets = self.tracks[self.tracks["frame"] == self.frame]
        if not len(dets):
            return None
        d = np.hypot(dets["centroid_y_px"].to_numpy() - y, dets["centroid_x_px"].to_numpy() - x)
        i = int(np.argmin(d))
        reach = 0.6 * float(dets["diameter_px"].iloc[i]) if "diameter_px" in dets else 10.0
        if d[i] > max(5.0, reach):
            return None
        tid = dets["track_id"].iloc[i]
        rows = self.objects[self.objects["track_id"] == tid] if "track_id" in self.objects else self.objects.iloc[:0]
        det = dets.iloc[i]
        return {"track_id": int(tid), "y": float(det["centroid_y_px"]), "x": float(det["centroid_x_px"]),
                "r": float(det.get("diameter_px", 20)) / 2, "row": rows.iloc[0].to_dict() if len(rows) else None}  # fmt: skip

    def select(self, y: float, x: float) -> None:
        obj = self._object_at(y, x)
        self.selected = obj
        self.canvas.highlight = (obj["y"], obj["x"], obj["r"] + 3) if obj else None
        self._update_selection_ui()
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        obj = self.selected
        if self.zoom_btn.isChecked() and obj is not None and self.canvas.rgb is not None:
            h, w = self.canvas.rgb.shape[:2]
            half = max(4.0 * obj["r"], 60.0)
            aspect = self.canvas.width() / max(self.canvas.height(), 1)
            hy, hx = half, half * max(aspect, 1.0)
            cy, cx = np.clip(obj["y"], hy, max(hy, h - hy)), np.clip(obj["x"], hx, max(hx, w - hx))
            self.canvas.view = (cy - hy, cx - hx, cy + hy, cx + hx)
        elif self.canvas.view is not None and (obj is None or not self.zoom_btn.isChecked()):
            if not self.zoom_btn.isChecked():
                self.canvas.view = None
        self.canvas.update()

    def _update_selection_ui(self) -> None:
        obj = self.selected
        has_row = obj is not None and obj["row"] is not None
        for w in (*self.field_widgets.values(), self.set_btn, self.verify_btn):
            w.setEnabled(has_row)
        self.exclude_btn.setEnabled(obj is not None)
        if obj is None:
            self.sel_label.setText("在图上点一个目标，或从左侧列表选择。")
            return
        row = obj["row"] or {}
        self.sel_label.setText(f"{self.adapter.object_name} {obj['track_id']} · 第 {self.frame} 帧" + (
            f"\n{self.adapter.describe(row)}" if row else "\n（未计入统计：没有越过计数线或轨迹太短）"))  # fmt: skip
        for f in self.adapter.fields:
            w = self.field_widgets[f.key]
            value = row.get(f.shown or f.key)
            if f.kind == "int" and value is not None and value == value:
                w.setValue(int(value))
            elif f.kind == "bool" and value is not None:
                w.setCurrentIndex(0 if str(value).lower() in ("true", "1") else 1)

    # -- tools ------------------------------------------------------------------------------

    def set_tool(self, key: str) -> None:
        self.tool = key
        self.canvas.drag_circles = key in ("paint", "erase")
        self._merge_first = None
        hint = dict((k, tip) for k, _, tip in TOOLS)[key]
        self.status.setText(hint)

    def _record(self, edit: Edit) -> None:
        self.corrections.add(edit)
        self.inspector.corrections = self.corrections
        self.inspector.invalidate(edit.frame)
        self._fill_edits()

    def _on_click(self, y: float, x: float) -> None:
        tool = getattr(self, "tool", "select")
        labels_ref = self.adapter.labels
        if tool == "select":
            self.select(y, x)
            return
        if tool == "delete":
            if not self._labels[int(round(np.clip(y, 0, self._labels.shape[0] - 1))), int(round(np.clip(x, 0, self._labels.shape[1] - 1)))]:
                self.status.setText("这里没有轮廓。")
                return
            self._record(Edit(kind="labels.delete", target=labels_ref, frame=self.frame, at=(y, x)))
        elif tool == "merge":
            if self._merge_first is None:
                self._merge_first = (y, x)
                self.status.setText("再点要合并的另一个目标。")
                return
            self._record(Edit(kind="labels.merge", target=labels_ref, frame=self.frame, points=[self._merge_first, (y, x)]))
            self._merge_first = None
        self.show_frame(self.frame)

    def _on_circle(self, y: float, x: float, r: float) -> None:
        kind = "labels.paint" if getattr(self, "tool", "") == "paint" else "labels.erase"
        self._record(Edit(kind=kind, target=self.adapter.labels, frame=self.frame, shape={"circle": [y, x, r]}))
        self.show_frame(self.frame)

    def set_fields(self) -> None:
        obj = self.selected
        if obj is None or obj["row"] is None:
            return
        for f in self.adapter.fields:
            w = self.field_widgets[f.key]
            value = int(w.value()) if f.kind == "int" else bool(w.currentData())
            current = obj["row"].get(f.shown or f.key)
            if f.kind == "bool":
                current = None if current is None else str(current).lower() in ("true", "1")
            if current is None or (current != current) or value != current:
                self._record(Edit(kind="table.set", target=f.target, frame=self.frame, at=(obj["y"], obj["x"]), values={f.key: value}))
        self._fill_queue()

    def verify(self) -> None:
        obj = self.selected
        if obj is None or obj["row"] is None:
            return
        target = self.adapter.fields[0].target if self.adapter.fields else self.adapter.objects
        self._record(Edit(kind="table.set", target=target, frame=self.frame, at=(obj["y"], obj["x"]), values={"verified": True}))
        self._fill_queue()

    def exclude(self) -> None:
        obj = self.selected
        if obj is None:
            return
        self._record(Edit(kind="table.exclude", target=self.adapter.exclude, frame=self.frame, at=(obj["y"], obj["x"])))
        self._fill_queue()

    def undo(self) -> None:
        if not self.corrections.edits:
            return
        edit = self.corrections.edits.pop()
        self.inspector.corrections = self.corrections
        self.inspector.invalidate(edit.frame)
        self._fill_edits()
        self._fill_queue()
        if edit.frame == self.frame:
            self.show_frame(self.frame)

    def _fill_edits(self) -> None:
        self.edit_list.clear()
        for e in self.corrections.edits:
            what = KIND_TEXT[e.kind]
            if e.values:
                what += " " + "，".join(f"{k} = {v}" for k, v in e.values.items())
            self.edit_list.addItem(f"第 {e.frame} 帧 · {what}")
        self.undo_btn.setEnabled(bool(self.corrections.edits))
        self.apply_btn.setEnabled(bool(self.corrections.edits) and self._thread is None)
        self.corrections.save(self.run_dir / "corrections.json")

    # -- re-run -------------------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None

    def apply(self) -> bool:
        if not self.corrections.edits or self.is_running():
            return False
        self._worker = RerunWorker(self.run_dir, self.corrections)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.setText(f"重算中：{m}"))
        self._worker.finished.connect(self._on_applied)
        self._worker.failed.connect(self._on_failed)
        self.apply_btn.setEnabled(False)
        self.status.setText("重算中…")
        self._thread.start()
        return True

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
        self.apply_btn.setEnabled(bool(self.corrections.edits))

    def _on_applied(self, out_dir: str) -> None:
        self._release()
        prov = json.loads((Path(out_dir) / "provenance.json").read_text())
        c = prov.get("corrections", {})
        missed = c.get("n_edits", 0) - c.get("n_applied", 0)
        self.status.setText(f"已按 {c.get('n_applied', 0)} 处修改重算，结果在 {out_dir}" + (f"（{missed} 处没有找到目标）" if missed else "") + "。")
        self.report_btn.setVisible((Path(out_dir) / "report.html").exists())

    def _on_failed(self, message: str) -> None:
        self._release()
        self.status.setText("重算未完成。")
        self.warn("重算未完成", message)

    def _open_report(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir / CORRECTED_DIR / "report.html")))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(60000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    import sys

    from qtpy.QtWidgets import QApplication

    from vexel.ui.app import STYLE

    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Vexel")
    app.setStyleSheet(STYLE)
    window = ReviewWindow(argv[0])
    window.show()
    return int(app.exec())
