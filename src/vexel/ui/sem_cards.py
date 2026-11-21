"""Cards for the SEM window: the automatic plan (result first, with its checks and three
choices: accept, adjust, do it myself) and the calibration banner above the panel."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from vexel.i18n import tr

STATUS = {  # plan status -> (label, colour)
    "ready": ("试运行通过", "#16a34a"),
    "check": ("通过，有提示", "#d97706"),
    "needs_user": ("需要你来决定", "#dc2626"),
    "busy": ("正在规划…", "#6b7280"),
}


def _label(text: str = "", *, wrap: bool = True, rich: bool = False) -> QLabel:
    w = QLabel(text)
    w.setWordWrap(wrap)
    if rich:
        w.setTextFormat(Qt.TextFormat.RichText)
    w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return w


class PlanCard(QFrame):
    """What the automatic planner found and tried: the image type and measurement, the trial's
    key numbers, its checks, the first reasons (all on request) and three choices."""

    accepted = Signal()
    adjust = Signal()
    dismissed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("planCard")
        self.setStyleSheet("#planCard { border: 1px solid palette(mid); border-radius: 8px; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)
        top = QHBoxLayout()
        self.title = _label(rich=True)
        self.chip = QLabel("")
        self.chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        top.addWidget(self.title, 1)
        top.addWidget(self.chip)
        v.addLayout(top)
        self.numbers = _label(rich=True)
        v.addWidget(self.numbers)
        self.checks = _label(rich=True)
        v.addWidget(self.checks)
        self.reasons = _label(rich=True)
        self.reasons.setObjectName("muted")
        v.addWidget(self.reasons)
        self.more = QPushButton("")
        self.more.setFlat(True)
        self.more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.more.setStyleSheet("text-align: left; padding: 0; color: palette(link);")
        self.more.clicked.connect(self._toggle)
        v.addWidget(self.more)
        row = QHBoxLayout()
        self.accept_btn = QPushButton(tr("采用"))
        self.accept_btn.setObjectName("primary")
        self.accept_btn.setToolTip(tr("加入这张图的测量记录（可撤销）"))
        self.adjust_btn = QPushButton(tr("调整"))
        self.adjust_btn.setToolTip(tr("把方法、设置和区域填到下方，改好后点“运行”"))
        self.self_btn = QPushButton(tr("我自己来"))
        self.self_btn.setToolTip(tr("不用这个规划，在下方手动选择方法"))
        self.accept_btn.clicked.connect(self.accepted.emit)
        self.adjust_btn.clicked.connect(self.adjust.emit)
        self.self_btn.clicked.connect(self.dismissed.emit)
        for b in (self.accept_btn, self.adjust_btn, self.self_btn):
            row.addWidget(b)
        v.addLayout(row)
        self._all: list[str] = []
        self._open = False
        self.hide()

    def _chip(self, key: str) -> None:
        text, colour = STATUS[key]
        self.chip.setText(f'<span style="color:{colour}; font-weight:600">● {tr(text)}</span>')

    def busy(self, text: str) -> None:
        self.title.setText(f"<b>{tr('自动规划')}</b>")
        self._chip("busy")
        self.numbers.setText(text)
        for w in (self.checks, self.reasons, self.more, self.accept_btn, self.adjust_btn, self.self_btn):
            w.hide()
        self.show()

    def set_plan(self, title: str, status: str, numbers: list[str], checks: list[tuple[str, str]], reasons: list[str], usable: bool) -> None:
        """``checks``: (mark ✓ ! ✗, text); ``reasons`` already translated."""
        self.title.setText(title)
        self._chip(status)
        self.numbers.setText(" · ".join(numbers))
        self.numbers.setVisible(bool(numbers))
        colour = {"✓": "#16a34a", "!": "#d97706", "✗": "#dc2626"}
        self.checks.setText("<br>".join(f'<span style="color:{colour.get(m, "inherit")}">{m}</span> {t}' for m, t in checks))
        self.checks.setVisible(bool(checks))
        self._all = reasons
        self._open = False
        self._fill_reasons()
        self.accept_btn.setVisible(usable)
        self.adjust_btn.setVisible(usable)
        self.self_btn.setVisible(True)
        self.self_btn.setText(tr("我自己来") if usable else tr("关闭"))
        self.show()

    def _fill_reasons(self) -> None:
        shown = self._all if self._open else self._all[:3]
        self.reasons.setText("<br>".join(f"· {r}" for r in shown))
        self.reasons.setVisible(bool(shown))
        extra = len(self._all) - 3
        self.more.setVisible(extra > 0)
        self.more.setText(tr("收起理由") if self._open else tr(f"全部理由（还有 {extra} 条）") if extra > 0 else "")

    def _toggle(self) -> None:
        self._open = not self._open
        self._fill_reasons()


class CalibrationBanner(QFrame):
    """One line above the panel: the pixel size and where it came from; a button to confirm
    when it waits for a person, or to calibrate when there is none."""

    confirm = Signal()
    open_calibration = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("calBanner")
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 6, 10, 6)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(self.text, 1)
        self.button = QPushButton("")
        self.button.clicked.connect(self._clicked)
        h.addWidget(self.button)
        self._mode = ""
        self.hide()

    def set_state(self, mode: str, text: str) -> None:
        """``mode``: confirmed | pending | none."""
        self._mode = mode
        colour = {"confirmed": "#16a34a", "pending": "#d97706", "none": "#dc2626"}[mode]
        bg = {"confirmed": "transparent", "pending": "rgba(217,119,6,0.10)", "none": "rgba(220,38,38,0.08)"}[mode]
        self.setStyleSheet(f"#calBanner {{ background: {bg}; border-bottom: 1px solid palette(mid); }}")
        self.text.setText(f'<span style="color:{colour}">●</span> {text}')
        self.button.setVisible(mode != "confirmed")
        self.button.setText(tr("确认") if mode == "pending" else tr("标定…"))
        self.button.setObjectName("primary" if mode == "pending" else "")
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)
        self.show()

    def _clicked(self) -> None:
        (self.confirm if self._mode == "pending" else self.open_calibration).emit()


__all__ = ["CalibrationBanner", "PlanCard"]
