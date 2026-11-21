"""Distribution chart for SEM objects: a histogram of one quantity (by number or by volume,
linear or logarithmic) or a scatter plot of two."""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from vexel.i18n import tr
from vexel.ui.widgets import ACCENT, MUTED


class _Plot(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(170)
        self.x = np.empty(0)
        self.y: np.ndarray | None = None
        self.w = np.empty(0)
        self.xlabel = self.ylabel = ""
        self.logx = self.logy = False

    def set(self, x: np.ndarray, y: np.ndarray | None, w: np.ndarray, xlabel: str, ylabel: str, logx: bool, logy: bool) -> None:
        self.x, self.y, self.w, self.xlabel, self.ylabel, self.logx, self.logy = x, y, w, xlabel, ylabel, logx, logy
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.palette().windowText().color()
        muted = QColor(MUTED)
        area = QRectF(self.rect()).adjusted(44, 18, -8, -30)
        x = self.x
        ok = np.isfinite(x) & ((x > 0) if self.logx else True)
        if self.y is not None:
            ok &= np.isfinite(self.y) & ((self.y > 0) if self.logy else True)
        if ok.sum() == 0:
            p.setPen(muted)
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, tr("没有可显示的数据"))
            return
        fx = np.log10 if self.logx else (lambda v: v)
        X = fx(x[ok])
        lo, hi = float(X.min()), float(X.max())
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5

        def px(v: float) -> float:
            return area.left() + (v - lo) / (hi - lo) * area.width()

        def lab(v: float, log: bool) -> str:
            return f"{10**v:.3g}" if log else f"{v:.4g}"

        p.setPen(QPen(muted))
        p.drawLine(QPointF(area.left(), area.bottom()), QPointF(area.right(), area.bottom()))
        if self.y is None:
            w = self.w[ok]
            bins = int(np.clip(math.sqrt(len(X)) * 1.5, 5, 40))
            counts, _ = np.histogram(X, bins=bins, range=(lo, hi), weights=w)
            top = counts.max() or 1.0
            ys = np.log10(np.maximum(counts, top * 1e-3) / (top * 1e-3)) / 3 if self.logy else counts / top
            bw = area.width() / bins
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT))
            for i, c in enumerate(ys):
                if counts[i] <= 0:
                    continue
                hgt = area.height() * float(c)
                p.drawRect(QRectF(area.left() + i * bw + 0.5, area.bottom() - hgt, max(bw - 1, 0.5), hgt))
            p.setPen(muted)
            p.drawText(QRectF(0, area.top() - 2, area.left() - 4, 14), Qt.AlignmentFlag.AlignRight, self.ylabel)
        else:
            fy = np.log10 if self.logy else (lambda v: v)
            Y = fy(self.y[ok])
            ylo, yhi = float(Y.min()), float(Y.max())
            if yhi - ylo < 1e-12:
                ylo, yhi = ylo - 0.5, yhi + 0.5
            p.setPen(Qt.PenStyle.NoPen)
            c = QColor(ACCENT)
            c.setAlpha(170)
            p.setBrush(c)
            for a, b in zip(X, Y):
                p.drawEllipse(QPointF(px(a), area.bottom() - (b - ylo) / (yhi - ylo) * area.height()), 2.5, 2.5)
            p.setPen(muted)
            p.drawLine(QPointF(area.left(), area.top()), QPointF(area.left(), area.bottom()))
            p.drawText(QRectF(0, area.top() - 6, area.left() - 4, 14), Qt.AlignmentFlag.AlignRight, lab(yhi, self.logy))
            p.drawText(QRectF(0, area.bottom() - 8, area.left() - 4, 14), Qt.AlignmentFlag.AlignRight, lab(ylo, self.logy))
            p.drawText(QRectF(area.left() + 4, 0, area.width(), 16), Qt.AlignmentFlag.AlignLeft, self.ylabel)
        p.setPen(text)
        bottom = QRectF(area.left(), area.bottom() + 4, area.width(), 16)
        p.drawText(bottom, Qt.AlignmentFlag.AlignLeft, lab(lo, self.logx))
        p.drawText(bottom, Qt.AlignmentFlag.AlignRight, lab(hi, self.logx))
        p.setPen(muted)
        p.drawText(bottom, Qt.AlignmentFlag.AlignHCenter, f"{self.xlabel}  (n = {int(ok.sum())})")


class DistributionChart(QWidget):
    """Columns of an object table (name -> values) with a histogram or scatter plot."""

    def __init__(self) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.xq, self.yq, self.scale, self.weight = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        for text, data in ((tr("线性"), "lin"), (tr("对数"), "log"), (tr("双对数"), "loglog")):
            self.scale.addItem(text, data)
        for text, data in ((tr("按数量"), "n"), (tr("按体积"), "v")):
            self.weight.addItem(text, data)
        self.weight.setToolTip(tr("按体积：每个对象按直径³加权（颗粒）"))
        for w in (self.xq, self.yq, self.scale, self.weight):
            w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(4)
            row.addWidget(w, 2 if w in (self.xq, self.yq) else 1)
            w.currentIndexChanged.connect(lambda _i: self._draw())
        v.addLayout(row)
        self.plot = _Plot()
        v.addWidget(self.plot)
        self.columns: dict[str, np.ndarray] = {}
        self.size_key = ""
        self.weights: np.ndarray | None = None

    def set_columns(self, columns: dict[str, np.ndarray], prefer: str = "", weights: np.ndarray | None = None) -> None:
        """``prefer``: the column to show first (the object size); ``weights``: number weights
        (fibre samples stand for their length)."""
        keep_x, keep_y = self.xq.currentData(), self.yq.currentData()
        self.columns = {k: np.asarray(v, float) for k, v in columns.items()}
        self.size_key = prefer
        self.weights = weights
        for box in (self.xq, self.yq):
            box.blockSignals(True)
            box.clear()
        self.yq.addItem(tr("（直方图）"), "")
        for k in self.columns:
            self.xq.addItem(k, k)
            self.yq.addItem(k, k)
        i = self.xq.findData(keep_x if keep_x in self.columns else prefer)
        self.xq.setCurrentIndex(max(i, 0))
        self.yq.setCurrentIndex(max(self.yq.findData(keep_y if keep_y in self.columns else ""), 0))
        for box in (self.xq, self.yq):
            box.blockSignals(False)
        self.setVisible(bool(self.columns))
        self._draw()

    def _draw(self) -> None:
        xk, yk = self.xq.currentData(), self.yq.currentData()
        if not xk or xk not in self.columns:
            self.plot.set(np.empty(0), None, np.empty(0), "", "", False, False)
            return
        x = self.columns[xk]
        mode = self.scale.currentData()
        logx, logy = mode in ("log", "loglog"), mode == "loglog"
        w = self.weights if self.weights is not None and len(self.weights) == len(x) else np.ones_like(x)
        by_volume = self.weight.currentData() == "v" and not yk
        self.weight.setEnabled(not yk)
        if by_volume:
            size = self.columns.get(self.size_key, x)
            w = w * np.where(np.isfinite(size), size, 0) ** 3
        ylabel = (tr("体积") if by_volume else tr("数量")) if not yk else yk
        self.plot.set(x, self.columns[yk] if yk else None, w, xk, ylabel, logx, logy)


__all__ = ["DistributionChart"]
