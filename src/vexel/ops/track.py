"""Tracking operators (linking is delegated to ``laptrack``; no home-grown tracker)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op


class LinkParams(Params):
    max_distance_px: float = Field(30.0, gt=0, description="maximum frame-to-frame displacement, pixels (after removing expected_shift_px)")
    expected_shift_px: tuple[float, float] = Field(
        (0.0, 0.0), description="expected (dy, dx) motion per frame, e.g. the mean flow; removed before linking"
    )
    max_gap_frames: int = Field(1, ge=0, description="bridge detections missing for up to this many frames; 0 disables gap closing")


@register_op(
    "track.link",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE},
    outputs={"table": Kind.TABLE},
    params=LinkParams,
)
def link(params: LinkParams, ctx: Context, *, table: pd.DataFrame) -> OpResult:
    """Link per-frame detections into tracks (linear assignment); adds a ``track_id`` column."""
    out = table.reset_index(drop=True).copy()
    if out.empty:
        out["track_id"] = pd.Series(dtype=np.int64)
        return OpResult({"table": out}, {"n_tracks": 0.0, "mean_track_length": 0.0})

    from laptrack import LapTrack

    frame = out["frame"].to_numpy(dtype=np.int64)
    dy, dx = params.expected_shift_px
    coords = pd.DataFrame(
        {
            "frame": frame,
            "y": out["centroid_y_px"].to_numpy(dtype=np.float64) - dy * frame,
            "x": out["centroid_x_px"].to_numpy(dtype=np.float64) - dx * frame,
            "row": np.arange(len(out)),
        }
    )
    cutoff = params.max_distance_px**2
    tracker = LapTrack(
        metric="sqeuclidean",
        cutoff=cutoff,
        gap_closing_metric="sqeuclidean",
        gap_closing_cutoff=cutoff if params.max_gap_frames > 0 else False,
        gap_closing_max_frame_count=max(1, params.max_gap_frames),
        splitting_cutoff=False,
        merging_cutoff=False,
    )
    tracked, _, _ = tracker.predict_dataframe(coords, coordinate_cols=["y", "x"], frame_col="frame", only_coordinate_cols=False)
    ids = tracked.set_index("row")["track_id"].reindex(np.arange(len(out))).to_numpy()
    # Renumber by order of first appearance so ids do not depend on solver internals.
    _, first = np.unique(ids, return_index=True)
    remap = {ids[i]: k for k, i in enumerate(np.sort(first))}
    out["track_id"] = np.array([remap[i] for i in ids], dtype=np.int64)
    lengths = out.groupby("track_id").size()
    return OpResult({"table": out}, {"n_tracks": float(len(lengths)), "mean_track_length": float(lengths.mean())})
