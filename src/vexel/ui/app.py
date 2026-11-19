"""Desktop entry point: ``vexel gui`` or ``vexel-gui``."""

from __future__ import annotations

import sys

STYLE = """
QFrame#sidebar { border-right: 1px solid rgba(128, 128, 128, 0.28); }
QLabel#heading { font-weight: 600; font-size: 13px; margin-top: 2px; }
QLabel#muted { color: #8a8f98; font-size: 12px; }
QLabel#fileName { font-weight: 600; }
QLabel#title { font-size: 18px; font-weight: 600; }
QLabel#emptyTitle { font-size: 20px; font-weight: 600; }
QLabel#cardValue { font-size: 22px; font-weight: 600; }
QFrame#card { border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 8px; }
QToolButton { border: none; background: transparent; padding: 2px 0; font-weight: 600; }
QLabel#notes { background: rgba(245, 158, 11, 0.16); border-radius: 6px; padding: 8px 10px; }
QPushButton#primary {
    background: #2563eb; color: white; border: none; border-radius: 6px;
    padding: 7px 16px; font-weight: 600;
}
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: palette(mid); color: palette(window); }
QFrame#guide { background: rgba(37, 99, 235, 0.10); border: 1px solid rgba(37, 99, 235, 0.35); border-radius: 8px; }
QLabel#guideTitle { font-weight: 600; font-size: 14px; }
QPushButton#guideClose { border: none; background: transparent; font-size: 16px; }
QProgressBar { border: none; border-radius: 3px; background: palette(mid); }
QProgressBar::chunk { border-radius: 3px; background: #2563eb; }
"""


def main(argv: list[str] | None = None) -> int:
    """``vexel gui [file] [--module ID]``: the module named, else the one the file belongs
    to, else the module launcher (see :mod:`vexel.ui.launcher`)."""
    import argparse

    from qtpy.QtWidgets import QApplication

    from vexel.ui.launcher import start

    parser = argparse.ArgumentParser(prog="vexel-gui")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--module", help="open this module directly (see `vexel modules`)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Vexel")
    app.setOrganizationName("Vexel")
    app.setStyleSheet(STYLE)
    start(args.input, args.module)
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
