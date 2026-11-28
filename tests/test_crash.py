"""The desktop safety net (``vexel.ui.crash``): an uncaught error is logged (paths shortened,
no image data), shown once, and included in the diagnostics bundle."""

from __future__ import annotations

import os
import sys
import zipfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _raise_in_customer_folder() -> None:
    raise ValueError('cannot read "/Users/someone/客户 A/lot 42/wafer.tif"')


def test_uncaught_error_is_logged_shown_and_bundled(tmp_path, monkeypatch):
    from qtpy.QtWidgets import QApplication

    from vexel import bugreport
    from vexel.ui import crash

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(crash, "_last_dialog", 0.0)
    try:
        _raise_in_customer_folder()
    except ValueError:
        log = crash.handle(*sys.exc_info())
    assert log is not None and log.parent == crash.logs_dir()
    text = log.read_text(encoding="utf-8")
    assert "ValueError" in text and "wafer.tif" in text and "客户 A" not in text and "someone" not in text
    app.processEvents()
    assert len(crash._open_boxes) == 1 and "ValueError" in crash._open_boxes[0].text()
    # the same error again within 10 s: logged, but no second dialog
    try:
        _raise_in_customer_folder()
    except ValueError:
        crash.handle(*sys.exc_info())
    assert len(crash._open_boxes) == 1 and len(crash.recent_logs(10)) == 2
    for box in list(crash._open_boxes):
        box.close()
    app.processEvents()
    out = bugreport.write_bundle(tmp_path / "bug.zip")
    names = zipfile.ZipFile(out).namelist()
    assert sum(n.startswith("logs/error-") for n in names) == 2


def test_keyboard_interrupt_is_not_swallowed(monkeypatch):
    from vexel.ui import crash

    seen = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: seen.append(a[0]))
    assert crash.handle(KeyboardInterrupt, KeyboardInterrupt(), None) is None and seen == [KeyboardInterrupt]


@pytest.mark.parametrize("n", [3])
def test_old_logs_are_pruned(tmp_path, monkeypatch, n):
    from vexel.ui import crash

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(crash, "MAX_LOGS", n)
    for _ in range(n + 4):
        crash.write_log(RuntimeError, RuntimeError("x"), None)
    assert len(crash.recent_logs(100)) == n
