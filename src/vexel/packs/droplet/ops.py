"""Droplet pack operators (S1). Phase 1 covers size statistics, de-duplicated counting and
generation frequency; encapsulation, Poisson fits and fluorescence arrive in Phase 2."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op
from vexel.ops.stats import describe_values

DROPLET_COLUMNS = [
    "track_id",
    "n_frames",
    "first_frame",
    "last_frame",
    "crossing_frame",
    "crossing_time",
    "diameter",
    "diameter_px",
    "circularity",
    "velocity",
    "cross_position",
]


def crossing_frame(positions: np.ndarray, frames: np.ndarray, line: float) -> float | None:
    """Sub-frame time at which a trajectory first crosses ``line`` (either direction).

    Shared by the summary operator and by ground-truth validation so both count the
    same way.
    """
    side = positions >= line
    change = np.nonzero(side[1:] != side[:-1])[0]
    if change.size == 0:
        return None
    i = int(change[0])
    p0, p1 = positions[i], positions[i + 1]
    f0, f1 = frames[i], frames[i + 1]
    return float(f0 + (line - p0) / (p1 - p0) * (f1 - f0))


class DropletSummaryParams(Params):
    flow_axis: Literal["x", "y"] = Field("x", description="image axis along which droplets travel")
    count_line: float = Field(0.5, gt=0, lt=1, description="virtual counting line, as a fraction of the image extent along flow_axis")
    min_track_frames: int = Field(3, ge=2, description="ignore tracks shorter than this")
    min_circularity: float = Field(0.0, ge=0, le=1, description="ignore detections less circular than this when sizing")


@register_op(
    "droplet.summary",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE},
    outputs={"droplets": Kind.TABLE, "summary": Kind.SUMMARY},
    params=DropletSummaryParams,
)
def droplet_summary(params: DropletSummaryParams, ctx: Context, *, table: pd.DataFrame) -> OpResult:
    """One row per physical droplet (counted once, when its track crosses the counting line),
    then diameter mean/std/CV, generation frequency, spacing and velocity."""
    if "track_id" not in table.columns:
        raise KeyError("droplet.summary needs a tracked table (run track.link first)")
    h, w = ctx.frame_shape
    flow_is_x = params.flow_axis == "x"
    extent = w if flow_is_x else h
    step = ctx.spacing_yx[1] if flow_is_x else ctx.spacing_yx[0]
    pos_col = "centroid_x_px" if flow_is_x else "centroid_y_px"
    cross_col = "centroid_y" if flow_is_x else "centroid_x"
    line = params.count_line * extent

    rows = []
    n_short = n_not_crossing = n_unsized = 0
    for track_id, g in table.sort_values("frame").groupby("track_id", sort=True):
        if len(g) < params.min_track_frames:
            n_short += 1
            continue
        frames = g["frame"].to_numpy(dtype=np.float64)
        pos = g[pos_col].to_numpy(dtype=np.float64)
        cross = crossing_frame(pos, frames, line)
        if cross is None:
            n_not_crossing += 1
            continue
        whole = g[~g["touches_border"].astype(bool) & (g["circularity"] >= params.min_circularity)]
        if whole.empty:
            n_unsized += 1
            continue
        slope = np.polyfit(frames, pos, 1)[0] if np.ptp(frames) > 0 else 0.0
        rows.append(
            {
                "track_id": int(track_id),
                "n_frames": int(len(g)),
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "crossing_frame": cross,
                "crossing_time": cross * ctx.frame_interval,
                "diameter": float(whole["diameter"].median()),
                "diameter_px": float(whole["diameter_px"].median()),
                "circularity": float(whole["circularity"].median()),
                "velocity": float(slope * step / ctx.frame_interval),
                "cross_position": float(whole[cross_col].median()),
            }
        )
    droplets = pd.DataFrame(rows, columns=DROPLET_COLUMNS).sort_values("crossing_frame", kind="stable").reset_index(drop=True)

    n = len(droplets)
    diameter = describe_values(droplets["diameter"].to_numpy(dtype=np.float64))
    times = droplets["crossing_time"].to_numpy(dtype=np.float64)
    duration = float(times[-1] - times[0]) if n >= 2 else 0.0
    frequency = (n - 1) / duration if duration > 0 else None
    velocity = float(droplets["velocity"].mean()) if n else None
    time_unit = ctx.time_unit
    summary = {
        "n_droplets": n,
        "diameter_mean": diameter["mean"],
        "diameter_std": diameter["std"],
        "diameter_cv_percent": diameter["cv_percent"],
        "diameter_median": diameter["median"],
        "diameter_min": diameter["min"],
        "diameter_max": diameter["max"],
        "frequency": frequency,
        "velocity_mean": velocity,
        "spacing_mean": abs(velocity) / frequency if (frequency and velocity is not None) else None,
        "length_unit": ctx.space_unit,
        "frequency_unit": "Hz" if time_unit == "s" else f"1/{time_unit}",
        "velocity_unit": f"{ctx.space_unit}/{time_unit}",
        "flow_axis": params.flow_axis,
        "count_line_px": float(line),
        "n_tracks_total": int(table["track_id"].nunique()),
        "n_tracks_too_short": n_short,
        "n_tracks_not_crossing": n_not_crossing,
        "n_tracks_unsized": n_unsized,
        "calibrated": ctx.space_unit != "px",
    }
    qc = {
        "n_droplets": float(n),
        "diameter_cv_percent": float(diameter["cv_percent"]) if diameter["cv_percent"] is not None else float("nan"),
        "circularity_median": float(droplets["circularity"].median()) if n else float("nan"),
    }
    return OpResult({"droplets": droplets, "summary": summary}, qc)
