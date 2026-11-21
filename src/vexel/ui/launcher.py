"""Application shell: choose a module, open its window, switch between modules.

Module windows know nothing about each other or about this shell; the shell adds a
"模块" menu to each window it opens. Several module windows can be open at once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from vexel.modules import Module, for_path, get, modules, resolve
from vexel.ui.widgets import heading, muted

_open_windows: list[Any] = []  # keep module windows alive (Qt owns nothing parentless)


def preview_enabled() -> bool:
    return os.environ.get("VEXEL_PREVIEW", "").strip() not in ("", "0", "false")


def can_open(m: Module) -> bool:
    return m.ready or preview_enabled()


def add_module_menu(window: Any) -> None:
    """A "模块" menu on a module window: switch to (open) another module, or show the launcher."""
    if not isinstance(window, QMainWindow) or getattr(window, "_module_menu", None) is not None:
        return
    menu = window.menuBar().addMenu("模块")
    for m in modules():
        action = menu.addAction(m.name + ("（开发中）" if not m.ready else ""), lambda _c=False, mid=m.id: open_module(mid))
        action.setEnabled(can_open(m))
    menu.addSeparator()
    menu.addAction("选择模块…", show_launcher)
    window._module_menu = menu


def open_module(module_id: str, path: str | Path | None = None) -> Any:
    """Open (another window of) a module, optionally with a file."""
    m = get(module_id)
    if not can_open(m):
        raise ValueError(f"module {m.name} is still in development")
    window = resolve(m.window)()
    add_module_menu(window)
    window.setProperty("vexel_module", m.id)
    window.show()
    if path is not None and hasattr(window, "open_path"):
        window.open_path(str(path))
    _open_windows.append(window)
    QSettings().setValue("last_module", m.id)
    return window


class Launcher(QWidget):
    def __init__(self, path: str | Path | None = None, note: str = "") -> None:
        super().__init__()
        self.setWindowTitle("Vexel")
        self.resize(720, 360)
        self.path = Path(path) if path else None
        v = QVBoxLayout(self)
        title = QLabel("Vexel")
        title.setObjectName("title")
        v.addWidget(title)
        v.addWidget(muted("科学图像分析平台。选择要使用的模块；各模块独立运行，可以同时打开。"))
        if note:
            v.addWidget(muted(note))
        if self.path is not None:
            v.addWidget(muted(f"要打开的文件：{self.path.name}"))
        self.buttons: dict[str, QPushButton] = {}
        for m in modules():
            card = QFrame()
            card.setObjectName("card")
            row = QHBoxLayout(card)
            text = QVBoxLayout()
            text.addWidget(heading(m.name + ("（开发中）" if not m.ready else "")))
            text.addWidget(muted(m.description))
            row.addLayout(text, 1)
            button = QPushButton("打开" if can_open(m) else "开发中")
            button.setEnabled(can_open(m))
            button.clicked.connect(lambda _c=False, mid=m.id: self.choose(mid))
            row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            self.buttons[m.id] = button
            v.addWidget(card)
        v.addStretch(1)
        self.skip = QCheckBox("下次启动时直接打开上次使用的模块")
        self.skip.setChecked(QSettings().value("skip_launcher", "false") == "true")
        self.skip.toggled.connect(lambda on: QSettings().setValue("skip_launcher", "true" if on else "false"))
        v.addWidget(self.skip)
        self.opened: Any = None

    def choose(self, module_id: str) -> Any:
        self.opened = open_module(module_id, self.path)
        self.close()
        return self.opened


_launcher: Launcher | None = None


def show_launcher(path: str | Path | None = None, note: str = "") -> Launcher:
    global _launcher
    _launcher = Launcher(path, note)
    _launcher.show()
    _launcher.raise_()
    return _launcher


def start(path: str | Path | None = None, module_id: str | None = None) -> Any:
    """What ``vexel gui`` shows first: the named module; else the module the file clearly
    belongs to; else the last module (when the person asked to skip the launcher); else
    the launcher."""
    if module_id:
        try:
            m = get(module_id)
        except KeyError:
            return show_launcher(path, f"没有名为“{module_id}”的模块，请选择。")
        if not can_open(m):
            return show_launcher(path, f"“{m.name}”模块还在开发中：设置环境变量 VEXEL_PREVIEW=1 后再启动即可打开预览版。")
        return open_module(module_id, path)
    if path is not None:
        m, _scores = for_path(path)
        if m is not None and can_open(m):
            return open_module(m.id, path)
        if m is not None:
            return show_launcher(path, f"这个文件看起来属于“{m.name}”模块，该模块还在开发中。可以选择其他模块打开。")
        return show_launcher(path, "无法从文件判断用哪个模块，请选择。")
    last = QSettings().value("last_module", "")
    if QSettings().value("skip_launcher", "false") == "true" and last:
        try:
            if can_open(get(last)):
                return open_module(last)
        except KeyError:
            pass
    return show_launcher()
