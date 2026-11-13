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
    estimate_shift: bool = Field(
        False,
        description="estimate the flow from the detections, frame by frame, and remove it before linking "
        "(replaces expected_shift_px); needed when objects move more than about half their spacing per frame",
    )


def _peaks2d(vectors: np.ndarray, radius: float) -> np.ndarray:
    """Local maxima (>= half the highest) of a smoothed 2D histogram of displacement vectors."""
    from scipy import ndimage as ndi

    r = int(np.ceil(radius))
    hist, _, _ = np.histogram2d(vectors[:, 0], vectors[:, 1], bins=2 * r + 1, range=[[-r - 0.5, r + 0.5]] * 2)
    hist = ndi.gaussian_filter(hist, 1.0)
    is_max = (hist == ndi.maximum_filter(hist, size=5)) & (hist >= 0.5 * hist.max()) & (hist > 0)
    return np.argwhere(is_max).astype(float) - r


def displacements(prev: np.ndarray, cur: np.ndarray, search: float) -> np.ndarray:
    """All (dy, dx) vectors from objects in ``prev`` to objects in ``cur`` no longer than ``search``."""
    if not len(prev) or not len(cur):
        return np.empty((0, 2))
    d = (cur[None, :, :] - prev[:, None, :]).reshape(-1, 2)
    return d[np.hypot(d[:, 0], d[:, 1]) <= search]


def global_flow(vectors: np.ndarray, search: float) -> np.ndarray | None:
    """The flow of a train of objects: the displacement peak smallest in magnitude (a
    regular train also peaks at the true shift minus one spacing)."""
    if not len(vectors):
        return None
    peaks = _peaks2d(vectors, search)
    return peaks[np.argmin(np.hypot(peaks[:, 0], peaks[:, 1]))]


def pair_shift(vectors: np.ndarray, flow: np.ndarray, max_distance: float) -> np.ndarray:
    """One frame pair's shift: the median of the displacements within ``max_distance`` of ``flow``."""
    close = vectors[np.hypot(*(vectors - flow).T) <= max_distance] if len(vectors) else vectors
    return np.median(close, axis=0) if len(close) else np.asarray(flow, dtype=float)


def estimate_flow(frame: np.ndarray, y: np.ndarray, x: np.ndarray, max_distance: float) -> dict[int, np.ndarray]:
    """Per-frame displacement (dy, dx) from frame t to t+1.

    Objects in a regular train look the same after moving by the true shift or by the
    shift minus one spacing, so the global flow is the peak of all frame-to-frame
    displacement vectors that is smallest in magnitude. Each frame pair then takes, for
    every object, the displacement closest to that global flow (tolerating changes in
    frame interval), and the result is median-filtered over time."""
    search = 2.0 * max_distance
    frames = np.unique(frame)
    pos = {t: np.column_stack([y[frame == t], x[frame == t]]) for t in frames}
    pairs: dict[int, np.ndarray] = {}
    for t in frames:
        if t + 1 in pos:
            d = displacements(pos[t], pos[t + 1], search)
            if len(d):
                pairs[int(t)] = d
    flow = global_flow(np.concatenate(list(pairs.values())), search) if pairs else None
    if flow is None:
        return {}
    shifts = {t: pair_shift(d, flow, max_distance) for t, d in pairs.items()}
    keys = sorted(shifts)
    arr = np.array([shifts[k] for k in keys])
    from scipy.ndimage import median_filter

    if len(arr) >= 3:
        arr = median_filter(arr, size=(3, 1), mode="nearest")
    return {k: arr[i] for i, k in enumerate(keys)}


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
    y = out["centroid_y_px"].to_numpy(dtype=np.float64)
    x = out["centroid_x_px"].to_numpy(dtype=np.float64)
    qc_flow: dict[str, float] = {}
    if params.estimate_shift:
        shifts = estimate_flow(frame, y, x, params.max_distance_px)
        # Cumulative offset of each frame in a frame of reference moving with the flow.
        t0, t1 = int(frame.min()), int(frame.max())
        step = np.array([shifts.get(t, np.zeros(2)) for t in range(t0, t1)]).reshape(-1, 2)
        if shifts:  # frames without an estimate move like the median frame
            step[[t - t0 not in {k - t0 for k in shifts} for t in range(t0, t1)]] = np.median(list(shifts.values()), axis=0)
        offset = np.vstack([np.zeros((1, 2)), np.cumsum(step, axis=0)])
        oy, ox = offset[frame - t0, 0], offset[frame - t0, 1]
        if shifts:
            med = np.median(list(shifts.values()), axis=0)
            qc_flow = {"flow_dy_px": float(med[0]), "flow_dx_px": float(med[1])}
    else:
        dy, dx = params.expected_shift_px
        oy, ox = dy * frame, dx * frame
    coords = pd.DataFrame({"frame": frame, "y": y - oy, "x": x - ox, "row": np.arange(len(out))})
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
    qc = {"n_tracks": float(len(lengths)), "mean_track_length": float(lengths.mean()), **qc_flow}
    return OpResult({"table": out}, qc)
