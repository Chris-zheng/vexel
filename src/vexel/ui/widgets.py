"""Small display widgets for the results page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
        self._extra: Path | None = None
        self.legend = "红线为分割轮廓"
        self.index = 0
        self._pixmap: QPixmap | None = None

    def set_images(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self.show_index(0)

    def show_index(self, index: int) -> None:
        if self._extra is not None:  # leaving a single image: resume the series where it was
            self._extra = None
            index = self.index
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
        self.caption.setText(f"{self.index + 1} / {n}{frame} · {self.legend}")
        self._rescale()

    def show_image(self, path: Path, caption: str) -> None:
        """Show one image outside the review series (e.g. a flagged droplet); the
        previous/next buttons return to the series."""
        self._extra = Path(path)
        self._pixmap = QPixmap(str(path))
        self.caption.setText(caption)
        self.prev_btn.setEnabled(bool(self.paths))
        self.next_btn.setEnabled(bool(self.paths))
        self._rescale()

    def open_current(self) -> None:
        target = self._extra if self._extra is not None else (self.paths[self.index] if self.paths else None)
        if target is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

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


class CountChart(QWidget):
    """Share of droplets with k cells: measured (filled) next to the Poisson expectation."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(140)
        self.k = np.empty(0, dtype=int)
        self.observed = np.empty(0)
        self.expected = np.empty(0)
        self._folded = False

    def set_data(self, k: np.ndarray, observed: np.ndarray, expected: np.ndarray, max_classes: int = 8) -> None:
        k = np.asarray(k, dtype=int)
        observed = np.asarray(observed, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        self._folded = k.size > max_classes
        if self._folded:  # fold the tail into the last class
            observed = np.append(observed[: max_classes - 1], observed[max_classes - 1 :].sum())
            expected = np.append(expected[: max_classes - 1], expected[max_classes - 1 :].sum())
            k = k[:max_classes]
        self.k, self.observed, self.expected = k, observed, expected
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted_color = QColor(MUTED)
        text_color = self.palette().windowText().color()
        if self.k.size == 0:
            p.setPen(muted_color)
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, "没有包裹数据")
            return
        area = QRectF(self.rect()).adjusted(4, 40, -4, -22)  # room for the legend and bar labels
        top = max(float(self.observed.max()), float(self.expected.max()), 1e-9)
        slot = area.width() / self.k.size
        bar = min(28.0, slot * 0.36)
        poisson_color = QColor(MUTED)
        poisson_color.setAlpha(110)
        for i, k in enumerate(self.k):
            cx = area.left() + (i + 0.5) * slot
            for value, color, dx in ((self.observed[i], QColor(ACCENT), -bar - 1), (self.expected[i], poisson_color, 1)):
                h = area.height() * value / top
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                p.drawRoundedRect(QRectF(cx + dx, area.bottom() - h, bar, h), 2, 2)
            p.setPen(text_color)
            p.drawText(
                QRectF(cx - bar - 20, area.bottom() - area.height() * self.observed[i] / top - 16, bar + 40, 14),
                Qt.AlignmentFlag.AlignHCenter,
                f"{100 * self.observed[i]:.0f}%",
            )
            p.setPen(muted_color)
            label = f"≥{k}" if (self._folded and i == self.k.size - 1) else str(k)
            p.drawText(QRectF(cx - 30, area.bottom() + 4, 60, 16), Qt.AlignmentFlag.AlignHCenter, label)
        # legend
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(ACCENT))
        p.drawRoundedRect(QRectF(area.left(), 4, 10, 10), 2, 2)
        p.setBrush(poisson_color)
        p.drawRoundedRect(QRectF(area.left() + 60, 4, 10, 10), 2, 2)
        p.setPen(text_color)
        p.drawText(QRectF(area.left() + 14, 0, 44, 18), Qt.AlignmentFlag.AlignVCenter, "实测")
        p.drawText(QRectF(area.left() + 74, 0, 80, 18), Qt.AlignmentFlag.AlignVCenter, "泊松预期")
        p.setPen(muted_color)
        p.drawText(QRectF(area.right() - 160, 0, 160, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "横轴：每滴细胞数")


class ScatterChart(QWidget):
    """Droplet diameter against fluorescence; positives in the accent colour, the
    threshold as a dashed line."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(140)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.positive = np.empty(0, dtype=bool)
        self.threshold: float | None = None
        self.x_unit = ""

    def set_data(self, x: np.ndarray, y: np.ndarray, positive: np.ndarray, threshold: float | None, x_unit: str) -> None:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(y)
        self.x, self.y = x[keep], y[keep]
        self.positive = np.asarray(positive, dtype=bool)[keep]
        self.threshold = threshold
        self.x_unit = x_unit
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted_color = QColor(MUTED)
        text_color = self.palette().windowText().color()
        if self.x.size == 0:
            p.setPen(muted_color)
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, "没有荧光数据")
            return
        area = QRectF(self.rect()).adjusted(40, 22, -8, -22)
        ys = self.y if self.threshold is None else np.append(self.y, self.threshold)
        x0, x1 = float(self.x.min()), float(self.x.max())
        y0, y1 = float(ys.min()), float(ys.max())
        padx, pady = max((x1 - x0) * 0.05, 1e-6), max((y1 - y0) * 0.05, 1e-6)
        x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady

        def to_px(xv: float, yv: float) -> QPointF:
            return QPointF(
                area.left() + (xv - x0) / (x1 - x0) * area.width(),
                area.bottom() - (yv - y0) / (y1 - y0) * area.height(),
            )

        neg = QColor(MUTED)
        neg.setAlpha(150)
        pos = QColor(ACCENT)
        pos.setAlpha(200)
        p.setPen(Qt.PenStyle.NoPen)
        for xv, yv, is_pos in zip(self.x, self.y, self.positive):
            p.setBrush(pos if is_pos else neg)
            p.drawEllipse(to_px(xv, yv), 3.0, 3.0)
        if self.threshold is not None:
            pen = QPen(text_color)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            a, b = to_px(x0, self.threshold), to_px(x1, self.threshold)
            p.drawLine(a, b)
            p.drawText(QRectF(b.x() - 150, b.y() - 18, 150, 16), Qt.AlignmentFlag.AlignRight, f"阈值 {self.threshold:.4g}")
        p.setPen(muted_color)
        p.drawText(QRectF(0, 0, 200, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "纵轴：荧光强度")
        p.drawText(QRectF(0, area.top() - 2, 36, 16), Qt.AlignmentFlag.AlignRight, f"{y1:.3g}")
        p.drawText(QRectF(0, area.bottom() - 14, 36, 16), Qt.AlignmentFlag.AlignRight, f"{y0:.3g}")
        bottom = QRectF(area.left(), area.bottom() + 4, area.width(), 18)
        p.drawText(bottom, Qt.AlignmentFlag.AlignLeft, f"{x0:.4g}")
        p.drawText(bottom, Qt.AlignmentFlag.AlignHCenter, "横轴：粒径")
        p.drawText(bottom, Qt.AlignmentFlag.AlignRight, f"{x1:.4g} {self.x_unit}")


class FlagStrip(QListWidget):
    """Thumbnails of flagged droplets; the reason is under each one and in the tooltip."""

    def __init__(self) -> None:
        super().__init__()
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(QSize(72, 72))
        self.setFixedHeight(118)
        self.setSpacing(4)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_flags(self, items: list[tuple[Path, str, str]]) -> None:
        """items: (crop image, short label, full reason)."""
        self.clear()
        for path, label, reason in items:
            item = QListWidgetItem(QIcon(str(path)), label)
            item.setToolTip(reason)
            item.setData(Qt.ItemDataRole.UserRole, (str(path), reason))
            item.setSizeHint(QSize(96, 104))
            self.addItem(item)
