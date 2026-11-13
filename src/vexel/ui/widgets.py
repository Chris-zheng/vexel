"""Small display widgets for the results page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

ACCENT = "#2563eb"
MUTED = "#8a8f98"


def muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("heading")
    return label


class Card(QFrame):
    """One headline number with a title and a one-line detail."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(2)
        self.title = muted(title)
        self.value = QLabel("–")
        self.value.setObjectName("cardValue")
        self.detail = muted("")
        for w in (self.title, self.value, self.detail):
            layout.addWidget(w)

    def set(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class Histogram(QWidget):
    """Bar histogram with a dashed mean line."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(140)
        self.values = np.empty(0)
        self.unit = ""

    def set_data(self, values: np.ndarray, unit: str) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.values = values[np.isfinite(values)]
        self.unit = unit
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text_color = self.palette().windowText().color()
        muted_color = QColor(MUTED)
        area = QRectF(self.rect()).adjusted(4, 22, -4, -22)
        v = self.values
        if v.size == 0:
            p.setPen(muted_color)
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, "没有可显示的粒径")
            return
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        bins = int(np.clip(np.sqrt(v.size) * 1.5, 5, 30))
        counts, _ = np.histogram(v, bins=bins, range=(lo, hi))
        width = area.width() / bins
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(ACCENT))
        for i, c in enumerate(counts):
            h = area.height() * c / counts.max()
            p.drawRoundedRect(QRectF(area.left() + i * width + 1, area.bottom() - h, max(1.0, width - 2), h), 2, 2)

        mean = float(v.mean())
        x = area.left() + (mean - lo) / (hi - lo) * area.width()
        pen = QPen(text_color)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
        label = f"均值 {mean:.4g} {self.unit}"
        if x + 150 < area.right():
            p.drawText(QRectF(x + 6, 0, 150, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        else:
            p.drawText(QRectF(x - 156, 0, 150, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

        p.setPen(muted_color)
        bottom = QRectF(area.left(), area.bottom() + 4, area.width(), 18)
        p.drawText(bottom, Qt.AlignmentFlag.AlignLeft, f"{lo:.4g}")
        p.drawText(bottom, Qt.AlignmentFlag.AlignRight, f"{hi:.4g} {self.unit}")
        p.drawText(bottom, Qt.AlignmentFlag.AlignHCenter, f"n = {v.size}")


class _ClickableLabel(QLabel):
    def __init__(self, viewer: "ReviewViewer") -> None:
        super().__init__()
        self._viewer = viewer

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        self._viewer.open_current()


class ReviewViewer(QWidget):
    """Shows the review overlays one at a time."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.image = _ClickableLabel(self)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(140)
        # Ignored: the label must not grow to fit its pixmap, or rescaling would loop.
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image.setToolTip("点击用系统看图软件打开原图")
        layout.addWidget(self.image, 1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("‹ 上一张")
        self.next_btn = QPushButton("下一张 ›")
        self.caption = muted("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_btn.clicked.connect(lambda: self.show_index(self.index - 1))
        self.next_btn.clicked.connect(lambda: self.show_index(self.index + 1))
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.caption, 1)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

        self.paths: list[Path] = []
        self.index = 0
        self._pixmap: QPixmap | None = None

    def set_images(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self.show_index(0)

    def show_index(self, index: int) -> None:
        n = len(self.paths)
        self.prev_btn.setEnabled(n > 1 and index > 0)
        self.next_btn.setEnabled(n > 1 and index < n - 1)
        if n == 0:
            self._pixmap = None
            self.image.clear()
            self.image.setText("没有复核图")
            self.caption.setText("")
            return
        self.index = max(0, min(index, n - 1))
        path = self.paths[self.index]
        self._pixmap = QPixmap(str(path))
        match = re.search(r"(\d+)", path.stem)
        frame = f" · 第 {int(match.group(1))} 帧" if match else ""
        self.caption.setText(f"{self.index + 1} / {n}{frame} · 红线为分割轮廓")
        self._rescale()

    def open_current(self) -> None:
        if self.paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths[self.index])))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image.setPixmap(scaled)
