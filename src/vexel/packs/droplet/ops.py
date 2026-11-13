"""Droplet pack operators (S1): size statistics, de-duplicated counting, generation
frequency, and encapsulation (cells per droplet, Poisson statistics)."""

from __future__ import annotations

from typing import Literal

import math

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import ndimage as ndi
from scipy import stats as sps
from skimage.draw import disk as draw_disk
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries

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


def _mode(values: np.ndarray) -> tuple[int, float]:
    """Most frequent integer (ties: the smaller one) and the share of values equal to it."""
    uniq, counts = np.unique(values.astype(np.int64), return_counts=True)
    best = int(np.argmax(counts))
    return int(uniq[best]), float(counts[best] / counts.sum())


@register_op(
    "droplet.summary",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE, "inclusions": Kind.TABLE},
    optional_inputs=("inclusions",),
    outputs={"droplets": Kind.TABLE, "summary": Kind.SUMMARY},
    params=DropletSummaryParams,
)
def droplet_summary(
    params: DropletSummaryParams, ctx: Context, *, table: pd.DataFrame, inclusions: pd.DataFrame | None = None
) -> OpResult:
    """One row per physical droplet (counted once, when its track crosses the counting line),
    then diameter mean/std/CV, generation frequency, spacing and velocity. With an
    ``inclusions`` table (from droplet.inclusions) each droplet also gets its cell count:
    the most frequent per-frame count, and the share of frames that agree with it."""
    if "track_id" not in table.columns:
        raise KeyError("droplet.summary needs a tracked table (run track.link first)")
    if inclusions is not None:
        counts = inclusions[["frame", "label", "n_inclusions"]]
        table = table.merge(counts, on=["frame", "label"], how="left", validate="one_to_one")
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
        extra: dict[str, float | int] = {}
        if inclusions is not None:
            per_frame = whole["n_inclusions"].dropna().to_numpy()
            if per_frame.size:
                extra["n_inclusions"], extra["inclusion_agreement"] = _mode(per_frame)
            else:
                extra["n_inclusions"], extra["inclusion_agreement"] = -1, 0.0
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
                **extra,
            }
        )
    columns = DROPLET_COLUMNS + (["n_inclusions", "inclusion_agreement"] if inclusions is not None else [])
    droplets = pd.DataFrame(rows, columns=columns).sort_values("crossing_frame", kind="stable").reset_index(drop=True)
    if inclusions is not None:
        droplets["n_inclusions"] = droplets["n_inclusions"].astype(np.int64)

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


# -- encapsulation -------------------------------------------------------------------------


class InclusionParams(Params):
    polarity: Literal["dark", "bright"] = Field(
        "dark", description="dark: cells darker than the droplet interior (brightfield); bright: brighter (fluorescence)"
    )
    cell_diameter_px: float = Field(5.0, gt=0, description="typical cell diameter, pixels")
    min_contrast: float | None = Field(
        None,
        gt=0,
        description="minimum contrast of a cell against its droplet's interior, image units; "
        "None = automatic (noise_factor x the noise level estimated in each frame)",
    )
    noise_factor: float = Field(6.0, gt=0, description="automatic threshold, in multiples of the smoothed noise level")
    edge_margin_px: int = Field(6, ge=0, description="ignore this many pixels inside the droplet edge, where the dark outline is")
    min_separation_px: float | None = Field(None, gt=0, description="minimum distance between two cells; default 0.6 x cell diameter")


@register_op(
    "droplet.inclusions",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE, "labels": Kind.LABELS},
    outputs={"table": Kind.TABLE, "labels": Kind.LABELS},
    params=InclusionParams,
)
def inclusions(params: InclusionParams, ctx: Context, *, image: np.ndarray, labels: np.ndarray) -> OpResult:
    """Count cells (or beads) inside each droplet: blob-sized spots in the droplet interior
    that differ from that droplet's own median intensity by at least ``min_contrast``."""
    n = int(labels.max())
    marks = np.zeros(labels.shape, dtype=np.int32)
    idx = np.arange(1, n + 1)
    if n == 0:
        empty = pd.DataFrame({"label": pd.Series(dtype=np.int64), "n_inclusions": pd.Series(dtype=np.float64)})
        return OpResult({"table": empty, "labels": marks}, {"n_inclusions": 0.0})

    # Interior: each droplet shrunk by edge_margin_px, measured from its own boundary so
    # that touching droplets are shrunk apart too.
    inner = np.where(find_boundaries(labels, mode="inner"), 0, labels)
    interior = np.where(ndi.distance_transform_edt(inner > 0) > params.edge_margin_px, labels, 0)

    img = image.astype(np.float32)
    # Smoothing at a quarter of the cell diameter suppresses pixel noise but, unlike the
    # blob-matched sigma (d / 2*sqrt(2)), still resolves two touching cells as two peaks.
    smooth = ndi.gaussian_filter(img, params.cell_diameter_px / 4.0)  # = sigma below
    present = np.bincount(interior.ravel(), minlength=n + 1)[1:] > 0
    background = np.zeros(n + 1, dtype=np.float32)
    if present.any():
        background[idx[present]] = ndi.median(img, labels=interior, index=idx[present])
    contrast = background[interior] - smooth if params.polarity == "dark" else smooth - background[interior]
    contrast[interior == 0] = 0.0

    sigma = params.cell_diameter_px / 4.0
    if params.min_contrast is not None:
        threshold = params.min_contrast
    else:
        # Pixel noise from the median absolute deviation inside droplets (robust to the few
        # cell pixels), scaled to the noise left after Gaussian smoothing of white noise.
        inside = interior > 0
        if inside.any():
            residual = img[inside] - background[interior[inside]]
            noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        else:
            noise = 0.0
        threshold = max(params.noise_factor * noise / (2.0 * math.sqrt(math.pi) * sigma), 1e-6)

    separation = params.min_separation_px or 0.6 * params.cell_diameter_px
    peaks = peak_local_max(
        contrast,
        min_distance=max(1, int(round(separation))),
        threshold_abs=threshold,
        labels=interior,
        exclude_border=False,
    )
    owner = interior[tuple(peaks.T)] if len(peaks) else np.empty(0, dtype=np.int64)
    counts = np.bincount(owner, minlength=n + 1)[1:]
    radius = max(1.0, params.cell_diameter_px / 2.0)
    for j, (y, x) in enumerate(peaks, 1):
        rr, cc = draw_disk((y, x), radius, shape=labels.shape)
        marks[rr, cc] = j
    # float so that droplets whose interior is too small to judge can be NaN
    table = pd.DataFrame({"label": idx.astype(np.int64), "n_inclusions": np.where(present, counts, np.nan)})
    return OpResult({"table": table, "labels": marks}, {"n_inclusions": float(len(peaks)), "contrast_threshold": float(threshold)})


class PoissonParams(Params):
    column: str = Field("n_inclusions", description="per-droplet count column")
    min_agreement: float = Field(
        0.0, ge=0, le=1, description="exclude droplets whose count agrees across fewer than this share of frames"
    )


def _pooled_chi2(observed: np.ndarray, lam: float, n: int) -> tuple[float | None, int | None, float | None]:
    """Pearson chi-square of observed counts against Poisson(lam), pooling neighbouring
    classes until every expected count is at least 5. One parameter is estimated."""
    kmax = int(observed.max()) if observed.size else 0
    k = np.arange(kmax + 1)
    obs = np.array([np.sum(observed == i) for i in k], dtype=np.float64)
    exp = n * sps.poisson.pmf(k, lam)
    exp[-1] += n * sps.poisson.sf(kmax, lam)  # the last class collects the tail
    groups_o, groups_e, acc_o, acc_e = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):
        acc_o, acc_e = acc_o + o, acc_e + e
        if acc_e >= 5:
            groups_o.append(acc_o)
            groups_e.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:
        if groups_e:
            groups_o[-1] += acc_o
            groups_e[-1] += acc_e
        else:
            groups_o, groups_e = [acc_o], [acc_e]
    dof = len(groups_e) - 2
    if dof < 1:
        return None, None, None
    go, ge = np.array(groups_o), np.array(groups_e)
    chi2 = float(np.sum((go - ge) ** 2 / ge))
    return chi2, dof, float(sps.chi2.sf(chi2, dof))


@register_op(
    "droplet.poisson",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE},
    outputs={"distribution": Kind.TABLE, "summary": Kind.SUMMARY},
    params=PoissonParams,
)
def poisson(params: PoissonParams, ctx: Context, *, table: pd.DataFrame) -> OpResult:
    """Encapsulation statistics: shares of empty, single and multiple droplets, the Poisson
    expectation for the measured mean (lambda), dispersion index and chi-square fit."""
    if params.column not in table.columns:
        raise KeyError(f"droplet.poisson: column {params.column!r} not in table (run droplet.inclusions first)")
    counts = table[params.column].to_numpy(dtype=np.float64)
    keep = counts >= 0  # -1 marks droplets whose count could not be measured
    if "inclusion_agreement" in table.columns:
        keep &= table["inclusion_agreement"].to_numpy(dtype=np.float64) >= params.min_agreement
    excluded = int((~keep).sum())
    counts = counts[keep].astype(np.int64)
    n = int(counts.size)

    low_agreement = (
        int((table["inclusion_agreement"].to_numpy(dtype=np.float64)[keep] < 0.8).sum())
        if "inclusion_agreement" in table.columns
        else 0
    )
    summary: dict[str, float | int | str | None] = {
        "column": params.column,
        "n_droplets": n,
        "n_excluded": excluded,
        "n_low_agreement": low_agreement,
        "n_inclusions_total": int(counts.sum()),
    }
    empty_keys = [
        "lambda", "lambda_ci_low", "lambda_ci_high", "variance", "dispersion_index",
        "frac_empty", "frac_single", "frac_multi", "poisson_empty", "poisson_single", "poisson_multi",
        "single_among_occupied", "chi2", "chi2_dof", "chi2_p",
    ]  # fmt: skip
    if n == 0:
        summary.update({k: None for k in empty_keys})
        dist = pd.DataFrame(columns=["k", "observed", "observed_fraction", "poisson_fraction", "poisson_expected"])
        return OpResult({"distribution": dist, "summary": summary}, {"n_droplets": 0.0})

    lam = float(counts.mean())
    total = int(counts.sum())
    variance = float(counts.var(ddof=1)) if n > 1 else None
    frac = {k: float(np.mean(m)) for k, m in (("empty", counts == 0), ("single", counts == 1), ("multi", counts >= 2))}
    p0, p1 = math.exp(-lam), lam * math.exp(-lam)
    chi2, dof, p_value = _pooled_chi2(counts, lam, n)
    summary.update(
        {
            "lambda": lam,
            # exact (Garwood) 95% interval for a Poisson mean, from the total count
            "lambda_ci_low": float(sps.chi2.ppf(0.025, 2 * total) / 2 / n) if total > 0 else 0.0,
            "lambda_ci_high": float(sps.chi2.ppf(0.975, 2 * total + 2) / 2 / n),
            "variance": variance,
            "dispersion_index": (variance / lam) if (variance is not None and lam > 0) else None,
            "frac_empty": frac["empty"],
            "frac_single": frac["single"],
            "frac_multi": frac["multi"],
            "poisson_empty": p0,
            "poisson_single": p1,
            "poisson_multi": max(0.0, 1.0 - p0 - p1),
            "single_among_occupied": frac["single"] / (1.0 - frac["empty"]) if frac["empty"] < 1 else None,
            "chi2": chi2,
            "chi2_dof": dof,
            "chi2_p": p_value,
        }
    )
    kmax = max(int(counts.max()), int(sps.poisson.ppf(0.999, lam)) if lam > 0 else 0, 2)
    k = np.arange(kmax + 1)
    observed = np.array([int(np.sum(counts == i)) for i in k])
    pmf = sps.poisson.pmf(k, lam)
    dist = pd.DataFrame(
        {
            "k": k,
            "observed": observed,
            "observed_fraction": observed / n,
            "poisson_fraction": pmf,
            "poisson_expected": pmf * n,
        }
    )
    qc = {
        "n_droplets": float(n),
        "dispersion_index": float(summary["dispersion_index"]) if summary["dispersion_index"] is not None else float("nan"),
        "chi2_p": p_value if p_value is not None else float("nan"),
        "low_agreement_fraction": low_agreement / n,
    }
    return OpResult({"distribution": dist, "summary": summary}, qc)
