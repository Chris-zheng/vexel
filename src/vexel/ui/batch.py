"""Batch window: many recordings, one recipe (the main window's current settings), group
statistics. Same engine as ``vexel batch run``; the batch folder can be resumed from either."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from qtpy.QtCore import QCoreApplication, QObject, Qt, QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vexel.batch.design import Design, DesignError, format_settings, scan_folder
from vexel.engine import Recipe
from vexel.ui.widgets import heading, muted

COLUMNS = ["文件", "组", "重复", "像素尺寸 (µm)", "帧间隔 (ms)", "状态"]
STATUS = {"completed": "完成", "resumed": "完成（沿用）", "failed": "失败", "missing": "找不到文件", "stopped_by_qc": "质控中止"}


class BatchWorker(QObject):
    progress = Signal(int, int, int, str, str)  # done, total, table row, status, error
    message = Signal(str)
    finished = Signal(object)  # BatchResult
    failed = Signal(str)

    def __init__(self, design: Design, recipe: Recipe, out_dir: Path) -> None:
        super().__init__()
        self.design, self.recipe, self.out_dir = design, recipe, Path(out_dir)
        self._stop = False
        self._row_of = {r.index: i for i, r in enumerate(design.rows)}

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:
        from vexel.batch import run_batch

        try:
            result = run_batch(self.design, self.recipe, self.out_dir, progress=self.message.emit, should_stop=lambda: self._stop,
                               on_file=lambda i, n, o: self.progress.emit(i, n, self._row_of[o.row], o.status, o.error))  # fmt: skip
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._done(self.failed, f"{type(e).__name__}: {e}")
            return
        self._done(self.finished, result)

    def _done(self, signal: Signal, *args: Any) -> None:
        app = QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        signal.emit(*args)


class BatchWindow(QWidget):
    def __init__(self, recipe: Recipe, recipe_label: str, pixel_um: float | None = None, interval_s: float | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Vexel 批量分析")
        self.resize(900, 640)
        self.recipe, self.pixel_um, self.interval_s = recipe, pixel_um, interval_s
        self.base: Path | None = None  # folder that relative file names refer to
        self.design_path: Path | None = None
        self.result: Any = None
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None

        v = QVBoxLayout(self)
        v.addWidget(heading("批量分析与分组统计"))
        v.addWidget(muted(f"所有文件使用同一套设置：{recipe_label}。在表里填写每个文件属于哪个组（实验条件）和哪个重复"
                          "（独立的芯片、批次或日期；同组同一重复的多个文件先取平均）。统计以重复为单位，不以液滴为单位。"))  # fmt: skip
        row = QHBoxLayout()
        self.folder_btn = QPushButton("选择文件夹…")
        self.folder_btn.setToolTip("找出文件夹（含子文件夹）里所有录像；第一层子文件夹名作为组，第二层作为重复")
        self.folder_btn.clicked.connect(self._choose_folder)
        self.load_btn = QPushButton("打开设计表…")
        self.load_btn.clicked.connect(self._choose_design)
        self.save_btn = QPushButton("保存设计表…")
        self.save_btn.clicked.connect(self._save_design)
        for b in (self.folder_btn, self.load_btn, self.save_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        v.addWidget(self.table, 1)
        calib = []
        if pixel_um:
            calib.append(f"像素尺寸 {pixel_um:g} µm")
        if interval_s:
            calib.append(f"帧间隔 {interval_s * 1000:g} ms")
        v.addWidget(muted("表里空着的标定用主窗口的值（" + "，".join(calib) + "）。" if calib else
                          "表里空着的标定按文件自带的元数据；没有时结果以像素和帧为单位。"))  # fmt: skip
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出文件夹"))
        self.out_edit = QLineEdit()
        out_row.addWidget(self.out_edit, 1)
        browse = QPushButton("…")
        browse.clicked.connect(self._choose_out)
        out_row.addWidget(browse)
        v.addLayout(out_row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        self.status = muted("选择一个文件夹，或打开设计表。")
        v.addWidget(self.status)
        actions = QHBoxLayout()
        self.run_btn = QPushButton("开始批量分析")
        self.run_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("停止")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.report_btn = QPushButton("打开报告")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.open_report)
        actions.addStretch(1)
        for b in (self.cancel_btn, self.report_btn, self.run_btn):
            actions.addWidget(b)
        v.addLayout(actions)
        self._update_buttons()

    # -- the design table ----------------------------------------------------------------

    def warn(self, title: str, text: str) -> None:  # tests replace this
        QMessageBox.warning(self, title, text)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含录像的文件夹")
        if folder:
            self.load_folder(folder)

    def _choose_design(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开设计表", "", "CSV (*.csv *.txt)")
        if path:
            self.load_design(path)

    def load_folder(self, folder: str | Path) -> int:
        folder = Path(folder)
        table = scan_folder(folder)
        if table.empty:
            self.warn("没有找到录像", f"{folder} 里没有找到视频、TIFF 或图像序列。")
            return 0
        self.base, self.design_path = folder, None
        self._fill([(f, g, r, "", "") for f, g, r in table[["file", "group", "replicate"]].itertuples(index=False)])
        self._suggest_out(folder)
        self.status.setText(f"找到 {len(table)} 个录像。请检查“组”和“重复”两列，然后开始。")
        return len(table)

    def load_design(self, path: str | Path) -> int:
        try:
            d = Design.load(path)
        except DesignError as e:
            self.warn("设计表有误", str(e))
            return 0
        self.base, self.design_path = Path(path).parent, Path(path)
        self._extra = {r.index: (r.settings, r.meta) for r in d.rows}
        rows = []
        for r in d.rows:
            try:
                name = str(r.file.relative_to(self.base))
            except ValueError:
                name = str(r.file)
            rows.append((name, r.group, r.replicate, f"{r.pixel_size_um:g}" if r.pixel_size_um else "",
                         f"{r.frame_interval_s * 1000:g}" if r.frame_interval_s else ""))  # fmt: skip
        self._fill(rows)
        self._suggest_out(self.base)
        self.status.setText(f"设计表：{len(rows)} 个文件，{len(d.groups)} 组。")
        return len(rows)

    def _fill(self, rows: list[tuple[str, str, str, str, str]]) -> None:
        self.table.setRowCount(len(rows))
        for i, values in enumerate(rows):
            for j, value in enumerate([*values, ""]):
                item = QTableWidgetItem(str(value))
                if j in (0, 5):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)
        self.report_btn.setEnabled(False)
        self._update_buttons()

    def _cell(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item is not None else ""

    def design(self) -> Design:
        """The table as a design (raises DesignError on a bad entry)."""
        import pandas as pd

        extra = getattr(self, "_extra", {}) if self.design_path is not None else {}
        recs = []
        for i in range(self.table.rowCount()):
            interval = self._cell(i, 4)
            settings, meta = extra.get(i + 1, ({}, {}))
            pixel = self._cell(i, 3) or (f"{self.pixel_um:g}" if self.pixel_um else "")
            try:
                frame = f"{float(interval) / 1000:g}" if interval else (f"{self.interval_s:g}" if self.interval_s else "")
            except ValueError:
                raise DesignError(f"row {i + 1}: frame interval must be a number, got {interval!r}") from None
            recs.append({"file": self._cell(i, 0), "group": self._cell(i, 1), "replicate": self._cell(i, 2), "pixel_size_um": pixel,
                         "frame_interval_s": frame, "settings": format_settings(settings), **meta})  # fmt: skip
        d = Design.from_frame(pd.DataFrame(recs, dtype=str).fillna(""), base=self.base)
        return d

    def _save_design(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存设计表", str((self.base or Path.home()) / "design.csv"), "CSV (*.csv)")
        if path:
            self.save_design(path)

    def save_design(self, path: str | Path) -> Path:
        d = self.design()
        frame = d.to_frame()
        if self.base is not None:
            frame["file"] = [os.path.relpath(f, Path(path).parent) for f in frame["file"]]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        self.status.setText(f"设计表已保存：{path}（可以用 Excel 编辑，或用命令 vexel batch run 运行）")
        return Path(path)

    def _suggest_out(self, folder: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = folder if os.access(folder, os.W_OK) else Path.home() / "Vexel"
        self.out_edit.setText(str(base / f"vexel_batch_{stamp}"))

    def _choose_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹（新的空文件夹，或要继续的批量结果）")
        if folder:
            self.out_edit.setText(folder)

    # -- running -------------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None

    def _update_buttons(self) -> None:
        running = self.is_running()
        self.run_btn.setEnabled(not running and self.table.rowCount() > 0)
        self.cancel_btn.setEnabled(running)
        for b in (self.folder_btn, self.load_btn, self.save_btn):
            b.setEnabled(not running)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers if running else QAbstractItemView.EditTrigger.AllEditTriggers)

    def start(self) -> bool:
        if self.is_running() or self.table.rowCount() == 0:
            return False
        try:
            design = self.design()
        except DesignError as e:
            self.warn("表格有误", str(e))
            return False
        if not self.out_edit.text().strip():
            self.warn("没有输出文件夹", "请选择保存结果的文件夹。")
            return False
        for i in range(self.table.rowCount()):
            self.table.item(i, 5).setText("等待")
        self.progress.setRange(0, len(design.rows))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._worker = BatchWorker(design, self.recipe, Path(self.out_edit.text().strip()))
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.message.connect(self._on_message)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._update_buttons()
        self._thread.start()
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("当前文件分析完后停止；之后用同一个输出文件夹再点“开始”会从停下的地方继续。")
            self.cancel_btn.setEnabled(False)

    def _on_message(self, text: str) -> None:
        if text.startswith("["):
            self.status.setText("正在分析 " + text)

    def _on_progress(self, done: int, total: int, row: int, status: str, error: str) -> None:
        self.progress.setValue(done)
        item = self.table.item(row, 5)
        if item is not None:
            item.setText(STATUS.get(status, status) + (f"：{error[:80]}" if error else ""))
            item.setToolTip(error)

    def _release(self) -> None:
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        if thread is not None:
            thread.quit()
            thread.wait()
        for obj in (worker, thread):
            if obj is not None and obj.thread() is self.thread():
                obj.setParent(self)
                obj.deleteLater()
        self._update_buttons()

    def _on_finished(self, result: Any) -> None:
        self._release()
        self.result = result
        bad = len(result.failed)
        if result.stats is None:
            self.status.setText(f"已停止：完成 {result.n_ok} 个文件。用同一个输出文件夹再点“开始”可以继续。")
            return
        self.report_btn.setEnabled((result.out_dir / "report.html").exists())
        self.status.setText(f"完成：{result.n_ok} / {len(result.outcomes)} 个文件" + (f"，{bad} 个没有完成（见状态列）" if bad else "") +
                            f"。结果在 {result.out_dir}。")  # fmt: skip

    def _on_failed(self, message: str) -> None:
        self._release()
        self.status.setText("批量分析没有完成。")
        self.warn("批量分析出错", message)

    def open_report(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.out_dir / "report.html")))

    def shutdown(self) -> None:
        """Stop after the current file and wait (the app is quitting); the batch can be resumed."""
        if self._worker is not None:
            self._worker.cancel()
        self._release()  # waits for the thread
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.is_running():  # the batch keeps its place; stop it first, it can be resumed later
            self.status.setText("批量分析正在进行：先点“停止”，当前文件分析完后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)
