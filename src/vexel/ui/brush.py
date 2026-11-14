"""Brush training: paint a few strokes, see the segmentation they give, use it.

Two brushes (objects, background) and an eraser paint on any frame; after every stroke
the pixel classifier is re-trained from all strokes and the current frame shows the
object probability (blue) with the outlines it produces (red). "完成" saves the training
file (patches around the strokes) next to the data and hands it to the caller.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vexel.ui.review import Canvas
from vexel.ui.widgets import heading, muted

CLASSES = ["液滴", "背景"]
COLORS = {1: (37, 99, 235), 2: (234, 88, 12)}  # stroke colours per class


class BrushCanvas(Canvas):
    """Canvas that reports drags as stroke segments (image pixels)."""

    stroke = Signal(float, float, float, float)  # y0, x0, y1, x1
    released = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.rgb is None:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        self._press = self.to_image(pos.x(), pos.y())
        self.stroke.emit(*self._press, *self._press)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is None:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        here = self.to_image(pos.x(), pos.y())
        self.stroke.emit(*self._press, *here)
        self._press = here

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._press is not None:
            self._press = None
            self.released.emit()


class BrushWindow(QMainWindow):
    trained = Signal(str, str, int)  # training file, SHA-256, number of painted frames

    def __init__(self, path: str | Path, channel: int = 0, out_dir: str | Path | None = None) -> None:
        super().__init__()
        from vexel.readers import open_volume

        self.path = Path(path)
        self.volume = open_volume(self.path)
        self.frames_lazy = self.volume.frames(channel, 0)
        self.n_frames = int(self.frames_lazy.shape[0])
        self.out_dir = Path(out_dir) if out_dir else self.path.parent / f"{self.path.name.split('.')[0]}_vexel" / "brush"
        self.strokes: dict[int, np.ndarray] = {}
        self.images: dict[int, np.ndarray] = {}
        self.model: Any = None
        self.frame = 0
        self.warn = lambda title, text: QMessageBox.warning(self, title, text)
        self.setWindowTitle(f"Vexel · 画笔训练 · {self.path.name}")
        self.resize(1180, 760)
        self._build()
        self.show_frame(0)

    def _build(self) -> None:
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(16, 12, 16, 12)
        v.addWidget(heading("用画笔告诉 Vexel 哪里是液滴、哪里是背景"))
        v.addWidget(muted("在液滴内部画几笔（蓝），在背景、流道壁、液滴之间的空隙画几笔（橙）。每画一笔，当前帧马上显示识别效果："
                          "蓝色深浅是“是液滴”的把握，红线是由此得到的轮廓。哪里错了就在哪里补一笔；可以在几帧上画。"))  # fmt: skip
        tools = QHBoxLayout()
        self.group = QButtonGroup(self)
        self.brush_buttons = {}
        for cls, name in ((1, "液滴画笔"), (2, "背景画笔"), (0, "橡皮")):
            b = QPushButton(name)
            b.setCheckable(True)
            self.group.addButton(b)
            self.brush_buttons[cls] = b
            tools.addWidget(b)
        self.brush_buttons[1].setChecked(True)
        tools.addWidget(QLabel("笔刷"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 40)
        self.size_spin.setValue(4)
        self.size_spin.setSuffix(" px")
        tools.addWidget(self.size_spin)
        self.preview_check = QCheckBox("显示识别效果")
        self.preview_check.setChecked(True)
        self.preview_check.toggled.connect(lambda _on: self.render())
        tools.addWidget(self.preview_check)
        clear = QPushButton("清除本帧笔画")
        clear.clicked.connect(self.clear_frame)
        tools.addWidget(clear)
        tools.addStretch(1)
        v.addLayout(tools)
        self.canvas = BrushCanvas()
        self.canvas.stroke.connect(self._paint)
        self.canvas.released.connect(self.retrain)
        v.addWidget(self.canvas, 1)
        nav = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.valueChanged.connect(self.show_frame)
        self.frame_label = muted("")
        nav.addWidget(QLabel("帧"))
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.frame_label)
        v.addLayout(nav)
        bottom = QHBoxLayout()
        self.status = muted("先在一个液滴里画一笔，再在背景上画一笔。")
        bottom.addWidget(self.status, 1)
        self.done_btn = QPushButton("完成，用于分析")
        self.done_btn.setObjectName("primary")
        self.done_btn.clicked.connect(self.finish)
        bottom.addWidget(self.done_btn)
        v.addLayout(bottom)
        self.setCentralWidget(central)
        self._update_buttons()

    # -- painting -------------------------------------------------------------------------------

    def image(self, t: int) -> np.ndarray:
        if t not in self.images:
            self.images[t] = np.asarray(self.frames_lazy[t]).astype(np.float32)
        return self.images[t]

    def _paint(self, y0: float, x0: float, y1: float, x1: float) -> None:
        from skimage.draw import disk

        cls = next(c for c, b in self.brush_buttons.items() if b.isChecked())
        img = self.image(self.frame)
        s = self.strokes.setdefault(self.frame, np.zeros(img.shape, np.uint8))
        r = self.size_spin.value() / 2
        n = max(1, int(np.hypot(y1 - y0, x1 - x0)))
        for k in range(n + 1):
            rr, cc = disk((y0 + (y1 - y0) * k / n, x0 + (x1 - x0) * k / n), max(r, 0.6), shape=s.shape)
            s[rr, cc] = cls
        self.render(quick=True)

    def clear_frame(self) -> None:
        self.strokes.pop(self.frame, None)
        self.retrain()

    def paint_stroke(self, cls: int, points: list[tuple[float, float]], size: int = 4) -> None:
        """Programmatic stroke (tests, scripted demos): a polyline in image pixels."""
        self.brush_buttons[cls].setChecked(True)
        self.size_spin.setValue(size)
        for (y0, x0), (y1, x1) in zip(points, points[1:] or points):
            self._paint(y0, x0, y1, x1)
        self.retrain()

    def counts(self) -> dict[int, int]:
        out = {1: 0, 2: 0}
        for s in self.strokes.values():
            for c in (1, 2):
                out[c] += int((s == c).sum())
        return out

    # -- training and preview ---------------------------------------------------------------------

    def _arrays(self) -> dict[str, Any] | None:
        from vexel.ops.classify import PATCH_MARGIN

        painted = [t for t, s in self.strokes.items() if (s > 0).any()]
        if not painted:
            return None
        all_px = np.concatenate([self.image(t).ravel() for t in painted])
        lo, hi = np.percentile(all_px, (0.5, 99.5))
        arrays: dict[str, Any] = {"classes": np.array(CLASSES), "range": np.array([lo, max(hi, lo + 1)]), "frames": np.array(painted)}
        for t in painted:
            s = self.strokes[t]
            ys, xs = np.nonzero(s)
            y0, y1 = max(0, ys.min() - PATCH_MARGIN), min(s.shape[0], ys.max() + PATCH_MARGIN + 1)
            x0, x1 = max(0, xs.min() - PATCH_MARGIN), min(s.shape[1], xs.max() + PATCH_MARGIN + 1)
            arrays[f"image_{t}"] = self.image(t)[y0:y1, x0:x1]
            arrays[f"strokes_{t}"] = s[y0:y1, x0:x1]
        return arrays

    def retrain(self) -> None:
        from vexel.ops.classify import train

        arrays = self._arrays()
        c = self.counts()
        self.model = None
        if arrays is not None and c[1] and c[2]:
            try:
                self.model = train(arrays=arrays)
            except ValueError as e:
                self.status.setText(str(e))
        if self.model is None:
            self.status.setText("液滴和背景都至少画一笔。" if not (c[1] and c[2]) else self.status.text())
        else:
            self.status.setText(f"已用 {len(arrays['frames'])} 帧上的 {c[1] + c[2]} 个像素训练。检查红线是否套准每个液滴；不对的地方补一笔。")
        self._update_buttons()
        self._frame_text()
        self.render()

    def segment(self, t: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Object probability and labels of frame ``t`` with the current strokes (the same
        post-processing as the segment.pixel_classifier operator's defaults)."""
        from vexel.ops.classify import labels_from_probability

        if self.model is None:
            return None
        prob = self.model.probabilities(self.image(t))[..., 0]
        return prob, labels_from_probability(prob)

    def render(self, quick: bool = False) -> None:
        from skimage.segmentation import find_boundaries

        img = self.image(self.frame)
        lo, hi = np.percentile(img, (1, 99))
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.dstack([gray] * 3)
        if self.preview_check.isChecked() and self.model is not None and not quick:
            seg = self.segment(self.frame)
            if seg is not None:
                prob, labels = seg
                rgb = rgb * (1 - 0.45 * prob[..., None]) + 0.45 * prob[..., None] * np.array([0.15, 0.4, 0.95])
                rgb[find_boundaries(labels, mode="outer")] = (1.0, 0.15, 0.15)
                self._last_labels = labels
        s = self.strokes.get(self.frame)
        if s is not None:
            for cls, color in COLORS.items():
                rgb[s == cls] = np.array(color) / 255
        self.canvas.set_image((rgb * 255).round().astype(np.uint8))

    def show_frame(self, t: int) -> None:
        self.frame = int(np.clip(t, 0, self.n_frames - 1))
        if self.slider.value() != self.frame:
            self.slider.setValue(self.frame)
        self._frame_text()
        self.render()

    def _frame_text(self) -> None:
        painted = sum(1 for s in self.strokes.values() if (s > 0).any())
        self.frame_label.setText(f"第 {self.frame} 帧 / {self.n_frames}（已在 {painted} 帧上画）")

    def _update_buttons(self) -> None:
        self.done_btn.setEnabled(self.model is not None)

    # -- finish ---------------------------------------------------------------------------------

    def finish(self) -> tuple[str, str] | None:
        from vexel.ops.classify import save_training

        if self.model is None:
            return None
        frames = {t: self.image(t) for t, s in self.strokes.items() if (s > 0).any()}
        path = self.out_dir / f"brush-{datetime.now():%Y%m%d-%H%M%S}.npz"
        sha = save_training(path, frames, {t: self.strokes[t] for t in frames}, CLASSES)
        self.training = (str(path), sha)
        self.trained.emit(str(path), sha, len(frames))
        self.status.setText(f"已保存训练文件 {path}")
        self.close()
        return self.training

