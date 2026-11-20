"""Open the SEM window in English, use every tab, and print every visible text that still
contains Chinese (one per line). Run in its own process: the language is chosen at import."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["VEXEL_LANG"] = "en"
os.environ["VEXEL_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QListWidget, QTableWidget, QTabWidget,  # noqa: E402
                            QToolBar, QWidget)

from vexel.i18n import CJK, missing  # noqa: E402


def texts(root: QWidget) -> list[str]:
    out = [root.windowTitle()]
    for w in root.findChildren(QWidget):
        out.append(w.toolTip())
        if isinstance(w, QLabel):
            out.append(w.text())
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())] + [str(w.itemData(i, 3) or "") for i in range(w.count())]
        elif isinstance(w, QLineEdit):
            out += [w.placeholderText(), w.text()]
        elif isinstance(w, QListWidget):
            out += [w.item(i).text() for i in range(w.count())] + [w.item(i).toolTip() for i in range(w.count())]
        elif isinstance(w, QTableWidget):
            out += [w.horizontalHeaderItem(c).text() for c in range(w.columnCount()) if w.horizontalHeaderItem(c)]
            out += [w.item(r, c).text() for r in range(w.rowCount()) for c in range(w.columnCount()) if w.item(r, c)]
        elif isinstance(w, QTabWidget):
            out += [w.tabText(i) for i in range(w.count())]
        elif isinstance(w, QToolBar):
            out += [a.text() for a in w.actions()] + [a.toolTip() for a in w.actions()]
        if hasattr(w, "suffix"):
            out.append(w.suffix())
    return out


def main(folder: str) -> int:
    from vexel.packs.sem import calibrate as C
    from vexel.synthetic import sem as S
    from vexel.ui.sem_window import SemWindow

    d = Path(folder)
    os.environ["VEXEL_HOME"] = str(d / "home")
    p = d / "thermo.tif"
    S.write_sem(p, "thermo", pixel_size=2.5e-9, rows=400, cols=500)
    app = QApplication.instance() or QApplication([])
    seen: list[str] = []
    w = SemWindow()
    w.show()
    w.open_path(p, wait=True)
    seen += texts(w)
    w.add_measurement("line", [[50, 50], [300, 80]])
    w.add_measurement("profile", [[20, 200], [480, 200]])
    w.add_measurement("wand", [[100, 100]])
    w.add_step("line_level")
    w.add_step("contrast")
    w.set_saturation(enabled=True)
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        app.processEvents()
        seen += texts(w)
    w.auto_plan(accept=True, wait=True)
    seen += texts(w)
    for name in ("lines", "holes", "particles", "layers", "profile", "tip"):
        w.recipe.setCurrentIndex(w.recipe.findData(name))
        w.run_analysis()
        app.processEvents()
        seen += texts(w)
    # tilt 3D
    z = S.height_map("steps", 300, 360, seed=1)
    tex = S.surface_texture(*z.shape, seed=2)
    views = []
    for i, t in enumerate((-5, 5)):
        q = d / f"v{i}.tif"
        tifffile.imwrite(q, (np.clip(S.tilt_view(z, tex, t, seed=i), 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(q, pixel_size=2e-9).confirm())
        views.append(q)
    w2 = SemWindow()
    w2.show()
    w2.open_path(views[0], wait=True)
    w2.add_tilt_views([views[1]], tilts=[5.0])
    w2.run_tilt3d(wait=True)
    w2.add_step_height([5, 20, 50, 280], [280, 20, 355, 280])
    w2.add_height_profile([10, 150], [350, 150])
    w2.tilt_table.cellWidget(1, 1).setValue(6.0)
    app.processEvents()
    seen += texts(w2) + [w2.statusBar().currentMessage(), w.statusBar().currentMessage()]
    switch = {w.lang_act.text(), w.lang_act.toolTip()}  # the way back to Chinese is meant to read in Chinese
    bad = sorted({t for t in seen if t and CJK.search(t) and t not in switch})
    for t in bad:
        print(repr(t))
    for t in sorted(missing()):
        print("MISSING", repr(t))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
