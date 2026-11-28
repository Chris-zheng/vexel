"""The desktop app's safety net: an error that no code caught does not end the program, nor
pass in silence.

``install(app)`` replaces ``sys.excepthook`` and ``threading.excepthook``. An uncaught error
(in a button's handler, a timer, a thread) is

- written to ``$VEXEL_HOME/logs/error-<time>.log``: the traceback, Vexel and Python versions
  and the operating system - no image data, and file paths shortened to file names;
- shown to the person in one dialog (at most one every 10 s: a repeated error does not pile
  up dialogs), saying the program keeps running, where the log is, and how to report it
  (帮助 → 诊断包: the bundle includes the latest logs).

Errors in background jobs are already turned into messages by each job (``failed`` signals);
this catches the rest.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

_last_dialog = 0.0
_installed = False
MAX_LOGS = 50


def logs_dir() -> Path:
    return Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel")) / "logs"


def _shorten(text: str) -> str:
    """Absolute paths to file names (user names, project and customer folder names stay out)."""
    for q in ('"', "'"):  # quoted paths (tracebacks quote them; they may contain spaces)
        text = re.sub(q + r"[^" + q + r"\n]*[\\/]([^" + q + r"\\/\n]+)" + q, q + r"…/\1" + q, text)
    return re.sub(r"(?:[A-Za-z]:)?[\\/](?:[^\\/\s\"':]+[\\/])+([^\\/\s\"':]+)", r"…/\1", text)


def write_log(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None, where: str = "") -> Path | None:
    """One error log; the oldest are removed beyond ``MAX_LOGS``. None when it cannot be written."""
    from vexel import __version__

    try:
        d = logs_dir()
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = d / f"error-{stamp}.log"
        body = "".join(traceback.format_exception(exc_type, exc, tb))
        head = [f"vexel {__version__}", f"python {platform.python_version()}", f"{platform.system()} {platform.release()} {platform.machine()}",
                f"time {datetime.now().isoformat(timespec='seconds')}", f"where {where or 'main thread'}", ""]  # fmt: skip
        path.write_text("\n".join(head) + _shorten(body), encoding="utf-8")
        for old in sorted(d.glob("error-*.log"))[:-MAX_LOGS]:
            old.unlink(missing_ok=True)
        return path
    except OSError:
        return None


def recent_logs(n: int = 5) -> list[Path]:
    d = logs_dir()
    return sorted(d.glob("error-*.log"))[-n:] if d.is_dir() else []


def _tell(exc: BaseException, log: Path | None) -> None:
    """One dialog, not more than one every 10 seconds; nothing when there is no window system."""
    global _last_dialog
    from qtpy.QtCore import Qt, QThread, QTimer
    from qtpy.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance()
    if app is None or time.monotonic() - _last_dialog < 10:
        return
    _last_dialog = time.monotonic()
    where = tr_text("错误已记录在 {}。", "The error was logged in {}.").format(log.name if log else "–")
    text = tr_text("出了一个没有预料到的错误：{}\n\n程序继续运行，但刚才的操作可能没有完成。", "An unexpected error occurred: {}\n\nVexel keeps running, but the last action may not have finished.").format(
        f"{type(exc).__name__}: {str(exc)[:300]}")  # fmt: skip
    hint = tr_text("报告问题时请用 帮助 → 诊断包…（会附上最近的错误记录，不含图像）。", "To report it: Help → Diagnostics… (it includes the latest error logs, no images).")

    def show() -> None:
        box = QMessageBox(QMessageBox.Icon.Warning, "Vexel", f"{text}\n\n{where}\n{hint}")
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _open_boxes.append(box)
        box.finished.connect(lambda _r: _open_boxes.remove(box) if box in _open_boxes else None)
        box.show()

    if QThread.currentThread() is app.thread():
        show()
    else:
        QTimer.singleShot(0, app, show)  # from another thread: show it on the main thread


_open_boxes: list[Any] = []  # keep the non-modal dialogs alive until closed


def tr_text(zh: str, en: str) -> str:
    from vexel.i18n import language

    return en if language() == "en" else zh


def handle(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None, where: str = "") -> Path | None:
    """Log an uncaught error and tell the person. Ctrl+C and exit still end the program."""
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc, tb)
        return None
    log = write_log(exc_type, exc, tb, where)
    traceback.print_exception(exc_type, exc, tb)  # still on the console for developers
    try:
        _tell(exc, log)
    except Exception:  # noqa: BLE001 - the safety net must not fail itself
        pass
    return log


def install() -> None:
    """Replace the hooks once (idempotent)."""
    global _installed
    if _installed:
        return
    sys.excepthook = handle
    threading.excepthook = lambda args: handle(args.exc_type, args.exc_value, args.exc_traceback, f"thread {args.thread.name if args.thread else '?'}")
    _installed = True


__all__ = ["handle", "install", "logs_dir", "recent_logs", "write_log"]
