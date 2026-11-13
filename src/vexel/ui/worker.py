"""Runs the engine off the UI thread and reports progress through Qt signals."""

from __future__ import annotations

import shutil
from pathlib import Path

from qtpy.QtCore import QObject, Signal

from vexel.core.volume import SpatialVolume
from vexel.engine import QCGateError, Recipe, run_recipe
from vexel.readers import open_volume

STEP_LABELS = {
    "preprocess.static_background": "扣除背景",
    "preprocess.background_correction": "背景校正",
    "preprocess.gaussian": "平滑",
    "preprocess.median": "去噪",
    "segment.threshold_watershed": "分割液滴",
    "segment.circles": "识别液滴",
    "measure.regionprops": "测量",
    "track.link": "追踪",
    "droplet.summary": "统计",
    "stats.describe": "统计",
}


def apply_calibration(volume: SpatialVolume, pixel_size_um: float | None, frame_interval_s: float | None) -> None:
    """The form is the single source of truth: an empty field means uncalibrated, even
    when the file carries metadata (the form was pre-filled from it, so the user saw it)."""
    if pixel_size_um is None:
        volume.spacing["y"] = volume.spacing["x"] = 1.0
        volume.space_unit = "px"
    else:
        volume.spacing["y"] = volume.spacing["x"] = float(pixel_size_um)
        volume.space_unit = "um"
    if frame_interval_s is None:
        volume.spacing["t"] = 1.0
        volume.time_unit = "frame"
    else:
        volume.spacing["t"] = float(frame_interval_s)
        volume.time_unit = "s"


class _Cancelled(Exception):
    pass


class RunWorker(QObject):
    progress = Signal(int, str)  # percent, message
    finished = Signal(str)  # run directory
    failed = Signal(str, str)  # message, run directory ("" when nothing usable was written)
    cancelled = Signal()

    def __init__(
        self,
        recipe: Recipe,
        path: Path,
        pixel_size_um: float | None,
        frame_interval_s: float | None,
        out_dir: Path,
        *,
        max_frames: int | None = None,
        hash_input: bool = True,
    ) -> None:
        super().__init__()
        self.recipe = recipe
        self.path = Path(path)
        self.pixel_size_um = pixel_size_um
        self.frame_interval_s = frame_interval_s
        self.out_dir = Path(out_dir)
        self.max_frames = max_frames
        self.hash_input = hash_input
        self._ops = {s.id: s.op for s in recipe.steps}
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            volume = open_volume(self.path)
            apply_calibration(volume, self.pixel_size_um, self.frame_interval_s)
            run_recipe(
                self.recipe,
                volume,
                self.out_dir,
                max_frames=self.max_frames,
                hash_input=self.hash_input,
                progress=self._say,
                tick=self._tick,
            )
        except _Cancelled:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.cancelled.emit()
        except QCGateError as e:
            self.failed.emit(f"被质控规则中止：{e}", str(self.out_dir))
        except Exception as e:  # surfaced to the user, never swallowed
            usable = (self.out_dir / "summary.json").exists()
            self.failed.emit(f"{type(e).__name__}: {e}", str(self.out_dir) if usable else "")
        else:
            self.finished.emit(str(self.out_dir))

    def _say(self, message: str) -> None:
        if self._cancel:
            raise _Cancelled
        if message == "hashing input":
            self.progress.emit(99, "计算输入文件指纹…")

    def _tick(self, step_id: str, index: int, n_steps: int, done: int, total: int) -> None:
        if self._cancel:
            raise _Cancelled
        fraction = (index - 1 + (done / total if total else 1.0)) / n_steps
        label = STEP_LABELS.get(self._ops.get(step_id, ""), step_id)
        self.progress.emit(min(98, int(fraction * 98)), f"{label}（{index}/{n_steps}）")
