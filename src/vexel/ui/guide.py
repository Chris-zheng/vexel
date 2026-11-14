"""The "next step" bar: tells a first-time user what to do now, with the button for it."""

from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class GuideBar(QFrame):
    def __init__(self, on_close: Callable[[], None]) -> None:
        super().__init__()
        self.setObjectName("guide")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 10, 10, 10)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = QLabel()
        self.title.setObjectName("guideTitle")
        self.body = QLabel()
        self.body.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.body)
        outer.addLayout(text, 1)
        self.buttons: list[QPushButton] = []
        self._actions = QHBoxLayout()
        outer.addLayout(self._actions)
        close = QPushButton("×")
        close.setObjectName("guideClose")
        close.setToolTip("不再显示引导（可在“帮助”中重新打开）")
        close.setFixedWidth(28)
        close.clicked.connect(on_close)
        outer.addWidget(close)
        self.step = ""

    def set_step(self, step: str, title: str, body: str, actions: list[tuple[str, Callable[[], object]]] = ()) -> None:
        self.step = step
        self.title.setText(title)
        self.body.setText(body)
        for b in self.buttons:
            self._actions.removeWidget(b)
            b.deleteLater()
        self.buttons = []
        for i, (label, fn) in enumerate(actions):
            b = QPushButton(label)
            if i == 0:
                b.setObjectName("primary")
            b.clicked.connect(lambda _c=False, f=fn: f())
            self._actions.addWidget(b)
            self.buttons.append(b)
