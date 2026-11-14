"""Automatic planning in the desktop UI: run the planner off the UI thread, show the plan,
apply it to the window's settings when the user accepts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy.QtCore import QCoreApplication, QObject, Signal
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QListWidget, QVBoxLayout

from vexel.ui.widgets import heading, muted

STATUS_TEXT = {
    "ready": "可以运行：试运行通过全部质控规则",
    "check": "可以运行，但试运行有质控提示（见下）",
    "needs_user": "需要你处理：没有得到可用的识别结果",
    "draft": "草案（未试运行）",
}


class PlanWorker(QObject):
    finished = Signal(object)  # Plan
    failed = Signal(str)

    def __init__(self, path: Path, goal: str, calibration: dict[str, Any]) -> None:
        super().__init__()
        self.path, self.goal, self.calibration = Path(path), goal, calibration

    def run(self) -> None:
        from vexel.agent.memory import RecipeMemory
        from vexel.agent.planner import make_plan
        from vexel.readers import open_volume
        from vexel.ui.worker import apply_calibration

        try:
            volume = open_volume(self.path)
            apply_calibration(volume, self.calibration.get("pixel"), self.calibration.get("interval"))
            plan = make_plan(volume, self.goal, memory=RecipeMemory())
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._done(self.failed, f"{type(e).__name__}: {e}")
            return
        self._done(self.finished, plan)

    def _done(self, signal: Signal, *args: Any) -> None:
        app = QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        signal.emit(*args)


class PlanDialog(QDialog):
    def __init__(self, plan: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("自动规划")
        self.resize(640, 460)
        v = QVBoxLayout(self)
        who = "规则引擎（离线）" if plan.planner == "rules" else f"{plan.model or plan.planner}"
        v.addWidget(heading(STATUS_TEXT.get(plan.status, plan.status)))
        v.addWidget(muted(f"方案由 {who} 起草，在前几百帧上试运行过。采用后会填入左侧设置，仍可修改。"))
        reasons = QListWidget()
        reasons.setWordWrap(True)
        for r in plan.reasons:
            reasons.addItem(r)
        v.addWidget(reasons, 1)
        if plan.skipped:
            v.addWidget(muted("未使用：" + "；".join(plan.skipped)))
        self.buttons = QDialogButtonBox()
        self.accept_btn = self.buttons.addButton("采用", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        self.accept_btn.setEnabled(plan.status != "needs_user")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        self.note = QLabel("")
        v.addWidget(self.note)
