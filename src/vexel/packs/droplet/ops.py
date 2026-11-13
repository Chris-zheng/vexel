"""Droplet pack operators (S1): size statistics, de-duplicated counting, generation
frequency, encapsulation (cells per droplet, Poisson statistics) and fluorescence
(per-droplet intensity, positive/negative calls, digital copies per droplet)."""

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


_TO_UM = {"um": 1.0, "nm": 1e-3, "mm": 1e3}


def droplet_volume_um3(diameter_um: np.ndarray, channel_height_um: float | None) -> np.ndarray:
    """Droplet volume from its projected diameter D.

    Without a channel height, or when D <= h, the droplet is a sphere: pi D^3 / 6. A
    droplet squeezed between the channel floor and ceiling (D > h) is modelled as a
    cylinder of height h and radius a = (D - h) / 2 with a half-torus rim of tube radius
    h / 2 (Pappus): V = pi a^2 h + pi^2 h^2 a / 4 + pi h^3 / 6. Both forms agree at D = h.
    """
    d = np.asarray(diameter_um, dtype=np.float64)
    sphere = np.pi * d**3 / 6.0
    if channel_height_um is None:
        return sphere
    h = float(channel_height_um)
    a = np.maximum(d - h, 0.0) / 2.0
    flat = np.pi * a**2 * h + np.pi**2 * h**2 * a / 4.0 + np.pi * h**3 / 6.0
    return np.where(d > h, flat, sphere)


class DropletSummaryParams(Params):
    flow_axis: Literal["x", "y"] = Field("x", description="image axis along which droplets travel")
    count_line: float = Field(0.5, gt=0, lt=1, description="virtual counting line, as a fraction of the image extent along flow_axis")
    min_track_frames: int = Field(3, ge=2, description="ignore tracks spanning fewer frames than this (first to last detection)")
    min_circularity: float = Field(0.0, ge=0, le=1, description="ignore detections less circular than this when sizing")
    channel_height_um: float | None = Field(
        None, gt=0, description="channel height, um; droplets wider than it are treated as flattened. None: all spheres"
    )


def _mode(values: np.ndarray) -> tuple[int, float]:
    """Most frequent integer (ties: the smaller one) and the share of values equal to it."""
    uniq, counts = np.unique(values.astype(np.int64), return_counts=True)
    best = int(np.argmax(counts))
    return int(uniq[best]), float(counts[best] / counts.sum())


@register_op(
    "droplet.summary",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE, "inclusions": Kind.TABLE, "intensity": Kind.TABLE},
    optional_inputs=("inclusions", "intensity"),
    outputs={"droplets": Kind.TABLE, "summary": Kind.SUMMARY},
    params=DropletSummaryParams,
)
def droplet_summary(
    params: DropletSummaryParams,
    ctx: Context,
    *,
    table: pd.DataFrame,
    inclusions: pd.DataFrame | None = None,
    intensity: pd.DataFrame | None = None,
) -> OpResult:
    """One row per physical droplet (counted once, when its track crosses the counting line),
    then diameter mean/std/CV, generation frequency, spacing and velocity. With an
    ``inclusions`` table (from droplet.inclusions) each droplet also gets its cell count:
    the most frequent per-frame count, and the share of frames that agree with it. With an
    ``intensity`` table (from droplet.intensity) it gets the median of each fluorescence
    measure over its whole frames."""
    if "track_id" not in table.columns:
        raise KeyError("droplet.summary needs a tracked table (run track.link first)")
    if inclusions is not None:
        counts = inclusions[["frame", "label", "n_inclusions"]]
        table = table.merge(counts, on=["frame", "label"], how="left", validate="one_to_one")
    fluo_cols: list[str] = []
    if intensity is not None:
        fluo_cols = [c for c in intensity.columns if c.startswith("fluo_") and c != "fluo_background"]
        table = table.merge(intensity[["frame", "label", *fluo_cols]], on=["frame", "label"], how="left", validate="one_to_one")
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
        # the span the track covers, not how many frames saw it (a stream that thins frames
        # when behind sees every droplet in fewer frames); at least two detections
        span = int(g["frame"].iloc[-1]) - int(g["frame"].iloc[0]) + 1
        if len(g) < 2 or span < params.min_track_frames:
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
        for col in fluo_cols:
            extra[col] = float(whole[col].median())
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
    columns = DROPLET_COLUMNS + (["n_inclusions", "inclusion_agreement"] if inclusions is not None else []) + fluo_cols
    droplets = pd.DataFrame(rows, columns=columns).sort_values("crossing_frame", kind="stable").reset_index(drop=True)
    if inclusions is not None:
        droplets["n_inclusions"] = droplets["n_inclusions"].astype(np.int64)

    n = len(droplets)
    to_um = _TO_UM.get(ctx.space_unit)
    volume = {"mean": None, "cv_percent": None}
    n_flattened = 0
    if to_um is not None:
        d_um = droplets["diameter"].to_numpy(dtype=np.float64) * to_um
        droplets["volume_pl"] = droplet_volume_um3(d_um, params.channel_height_um) / 1000.0  # 1 pL = 1000 um^3
        volume = describe_values(droplets["volume_pl"].to_numpy(dtype=np.float64))
        if params.channel_height_um is not None:
            n_flattened = int((d_um > params.channel_height_um).sum())
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
        "volume_mean_pl": volume["mean"],
        "volume_cv_percent": volume["cv_percent"],
        "volume_model": None if to_um is None else ("flattened" if params.channel_height_um is not None else "sphere"),
        "channel_height_um": params.channel_height_um,
        "n_flattened": n_flattened,
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
    relative_floor: float = Field(
        0.0,
        ge=0,
        lt=1,
        description="when a frame has clear cells (contrast >= 2x threshold), also require this fraction of "
        "their median contrast; removes faint shading spots in clean images. 0 disables",
    )
    skip_border: bool = Field(
        True, description="do not count cells in droplets cut by the image border (reported as NaN for that frame)"
    )
    background: Literal["median", "radial"] = Field(
        "median",
        description="median: one interior level per droplet; radial: the droplet's own median profile by distance "
        "from its centre, which removes a rotationally symmetric outline (dark ring, bright rim) so cells "
        "near the edge can be counted with a small edge margin",
    )


def _radial_model(
    smooth: np.ndarray, raw: np.ndarray, labels: np.ndarray, n: int, inner_fraction: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-droplet background following the droplet's own outline.

    Inside ``inner_fraction`` of its radius a droplet has one level (the median of those
    pixels, which cells cannot dominate); further out, each whole-pixel distance from the
    centroid gets its own median, which follows the rim. Returns the model for the
    smoothed and the raw image (0 outside droplets), each pixel's distance from its
    droplet's centroid, and the centre level per droplet."""
    level_s = np.zeros(labels.shape, dtype=np.float32)
    level_r = np.zeros(labels.shape, dtype=np.float32)
    rad_img = np.zeros(labels.shape, dtype=np.float32)
    centre = np.zeros(n + 1, dtype=np.float64)
    mask = labels > 0
    if not mask.any():
        return level_s, level_r, rad_img, centre
    yy, xx = np.nonzero(mask)
    lab = labels[yy, xx]
    count = np.bincount(lab, minlength=n + 1).astype(np.float64)
    cy = np.bincount(lab, weights=yy, minlength=n + 1) / np.maximum(count, 1)
    cx = np.bincount(lab, weights=xx, minlength=n + 1) / np.maximum(count, 1)
    dist = np.hypot(yy - cy[lab], xx - cx[lab])
    r_max = np.zeros(n + 1)
    np.maximum.at(r_max, lab, dist)
    core = dist < inner_fraction * r_max[lab]
    ring = np.where(core, -1, np.rint(dist)).astype(np.int64)
    table = pd.DataFrame({"key": lab * 100_000 + ring, "s": smooth[yy, xx], "r": raw[yy, xx]})
    grouped = table.groupby("key")
    level_s[yy, xx] = grouped["s"].transform("median").to_numpy(np.float32)
    level_r[yy, xx] = grouped["r"].transform("median").to_numpy(np.float32)
    rad_img[yy, xx] = dist
    centre_keys = lab[core]
    if centre_keys.size:
        centre[np.unique(centre_keys)] = pd.Series(smooth[yy, xx][core]).groupby(centre_keys).median().to_numpy()
    return level_s, level_r, rad_img, centre


def _rim_limit(
    smooth: np.ndarray,
    rad: np.ndarray,
    labels: np.ndarray,
    centre: np.ndarray,
    threshold: float,
    clearance: float,
    n_sectors: int = 8,
    inner_fraction: float = 0.6,
) -> np.ndarray:
    """Pixels inside each droplet's outline ring, found separately in angular sectors
    because rings are often thicker on one side (illumination, focus).

    In each sector the median level at each whole-pixel radius is compared with the
    droplet's centre level; the ring is the band that, walking inward from the strongest
    deviation in the outer part of the droplet, keeps deviating by more than
    ``threshold`` and a third of that strongest deviation. A cell touching the ring is
    left out with it; one separated from it by background is kept. Pixels within
    ``clearance`` of the ring are left out too (smoothing spreads it inward)."""
    inside = np.zeros(labels.shape, dtype=bool)
    yy, xx = np.nonzero(labels > 0)
    if yy.size == 0:
        return inside
    lab = labels[yy, xx]
    n = centre.size - 1
    count = np.bincount(lab, minlength=n + 1).astype(np.float64)
    cy = np.bincount(lab, weights=yy, minlength=n + 1) / np.maximum(count, 1)
    cx = np.bincount(lab, weights=xx, minlength=n + 1) / np.maximum(count, 1)
    dist = rad[yy, xx]
    r = np.rint(dist).astype(np.int64)
    sector = ((np.arctan2(yy - cy[lab], xx - cx[lab]) + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64) % n_sectors
    r_top = int(r.max()) + 1
    key = (lab * n_sectors + sector) * r_top + r
    med = pd.Series(smooth[yy, xx]).groupby(key).median()
    dev = np.full((n + 1) * n_sectors * r_top, np.nan)
    dev[med.index.to_numpy()] = np.abs(med.to_numpy() - centre[med.index.to_numpy() // (n_sectors * r_top)])
    dev = dev.reshape(n + 1, n_sectors, r_top)
    r_max = np.zeros(n + 1)
    np.maximum.at(r_max, lab, dist)
    radii = np.arange(r_top)
    outer = radii[None, :] >= inner_fraction * r_max[:, None]  # (n+1, r_top)
    dev_outer = np.where(outer[:, None, :], dev, np.nan)
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(np.where(np.isnan(dev_outer), -np.inf, dev_outer), axis=2)  # (n+1, sectors)
    peak_r = np.argmax(np.where(np.isnan(dev_outer), -np.inf, dev_outer), axis=2)
    dmax = np.max(np.where(np.isfinite(peak), peak, 0.0), axis=1)  # per droplet
    cut = np.maximum(threshold, dmax / 3.0)[:, None]
    limit = np.full((n + 1, n_sectors), np.inf)
    ring = np.isfinite(peak) & (peak > cut)
    above = np.nan_to_num(dev, nan=0.0) > cut[:, :, None]
    for k, j in zip(*np.nonzero(ring)):
        start = int(peak_r[k, j])
        while start > 0 and above[k, j, start - 1]:
            start -= 1
        limit[k, j] = start
    inside[yy, xx] = dist < limit[lab, sector] - clearance
    return inside


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
    labels_in = labels
    if params.skip_border:
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        labels_in = np.where(np.isin(labels, edge[edge > 0]), 0, labels)
    inner = np.where(find_boundaries(labels_in, mode="inner"), 0, labels_in)
    interior = np.where(ndi.distance_transform_edt(inner > 0) > params.edge_margin_px, labels_in, 0)

    img = image.astype(np.float32)
    # Smoothing at a quarter of the cell diameter suppresses pixel noise but, unlike the
    # blob-matched sigma (d / 2*sqrt(2)), still resolves two touching cells as two peaks.
    sigma = params.cell_diameter_px / 4.0
    smooth = ndi.gaussian_filter(img, sigma)
    if params.background == "radial":
        level, level_raw, rad, centre = _radial_model(smooth, img, labels_in, n)
        residual_img = img - level_raw
        body = labels_in > 0
        noise_all = 1.4826 * float(np.median(np.abs(residual_img[body] - np.median(residual_img[body])))) if body.any() else 0.0
        # leave out each droplet's own outline ring (its width varies with optics and focus)
        interior = np.where(_rim_limit(smooth, rad, labels_in, centre, 4.0 * noise_all, 2.0 * sigma), interior, 0)
    else:
        present0 = np.bincount(interior.ravel(), minlength=n + 1)[1:] > 0
        background = np.zeros(n + 1, dtype=np.float32)
        if present0.any():
            background[idx[present0]] = ndi.median(img, labels=interior, index=idx[present0])
        level = background[interior]
        residual_img = img - level
    present = np.bincount(interior.ravel(), minlength=n + 1)[1:] > 0
    inside = interior > 0
    contrast = level - smooth if params.polarity == "dark" else smooth - level
    contrast[interior == 0] = 0.0

    if params.min_contrast is not None:
        threshold = params.min_contrast
    else:
        # Pixel noise from the median absolute deviation inside droplets (robust to the few
        # cell pixels), scaled to the noise left after Gaussian smoothing of white noise.
        if inside.any():
            residual = residual_img[inside]
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
    if params.relative_floor > 0 and len(peaks):
        # Faint shading spots pass a noise-based threshold in clean images; when the frame
        # has clear cells, require a fraction of their typical contrast.
        heights = contrast[tuple(peaks.T)]
        strong = heights[heights >= 2.0 * threshold]
        if strong.size:
            floor = params.relative_floor * float(np.median(strong))
            if floor > threshold:
                peaks = peaks[heights >= floor]
                threshold = floor
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
        "inclusions_per_ul", "inclusions_per_ul_ci_low", "inclusions_per_ul_ci_high",
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
            "inclusions_per_ul": per_microlitre(lam, table),
            "inclusions_per_ul_ci_low": per_microlitre(float(sps.chi2.ppf(0.025, 2 * total) / 2 / n) if total > 0 else 0.0, table),
            "inclusions_per_ul_ci_high": per_microlitre(float(sps.chi2.ppf(0.975, 2 * total + 2) / 2 / n), table),
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


# -- fluorescence --------------------------------------------------------------------------

FLUO_COLUMNS = ["label", "fluo_mean", "fluo_median", "fluo_p95", "fluo_integrated", "fluo_background"]


class IntensityParams(Params):
    edge_margin_px: int = Field(2, ge=0, description="ignore this many pixels inside the droplet edge when averaging")
    background: Literal["outside", "none"] = Field(
        "outside", description="outside: subtract the median of pixels away from all droplets in the same frame"
    )
    background_gap_px: int = Field(4, ge=0, description="pixels around droplets excluded from the background estimate")


@register_op(
    "droplet.intensity",
    scope=Scope.FRAME,
    inputs={"image": Kind.IMAGE, "labels": Kind.LABELS},
    outputs={"table": Kind.TABLE},
    params=IntensityParams,
)
def intensity(params: IntensityParams, ctx: Context, *, image: np.ndarray, labels: np.ndarray) -> OpResult:
    """Per-droplet fluorescence: background-subtracted mean, median and 95th percentile over
    the droplet interior, and the integrated signal over the whole droplet."""
    img = image.astype(np.float64)
    mask = labels > 0
    bg = 0.0
    if params.background == "outside":
        near = ndi.binary_dilation(mask, iterations=params.background_gap_px) if params.background_gap_px else mask
        outside = img[~near]
        bg = float(np.median(outside)) if outside.size else 0.0
    n = int(labels.max())
    if n == 0:
        return OpResult({"table": pd.DataFrame(columns=FLUO_COLUMNS)}, {"background": bg})
    idx = np.arange(1, n + 1)
    inner = np.where(find_boundaries(labels, mode="inner"), 0, labels)
    interior = np.where(ndi.distance_transform_edt(inner > 0) > params.edge_margin_px, labels, 0)
    present = np.bincount(interior.ravel(), minlength=n + 1)[1:] > 0
    # droplets too small for the margin fall back to their full mask
    interior = np.where(np.isin(labels, idx[~present]), labels, interior)
    corrected = img - bg
    mean = np.asarray(ndi.mean(corrected, labels=interior, index=idx))
    median = np.asarray(ndi.median(corrected, labels=interior, index=idx))
    p95 = np.asarray(ndi.labeled_comprehension(corrected, interior, idx, lambda v: np.percentile(v, 95), np.float64, np.nan))
    integrated = np.asarray(ndi.sum(corrected, labels=labels, index=idx))
    table = pd.DataFrame(
        {
            "label": idx.astype(np.int64),
            "fluo_mean": mean,
            "fluo_median": median,
            "fluo_p95": p95,
            "fluo_integrated": integrated,
            "fluo_background": bg,
        }
    )
    return OpResult({"table": table}, {"background": bg, "droplet_mean": float(np.nanmean(mean))})


class PositivityParams(Params):
    column: str = Field("fluo_mean", description="per-droplet intensity column to classify")
    method: Literal["auto", "otsu", "outlier", "fixed"] = Field(
        "auto",
        description="auto: split into two populations only when a two-population Gaussian mixture has a clearly "
        "lower BIC and the groups are separated (a density dip below max_dip_ratio with Ashman's D >= min_separation, "
        "or a dip below strong_max_dip_ratio with D >= strong_separation); otherwise treat all droplets as one "
        "population and use the outlier rule; otsu: always split at the best Otsu gap; outlier: median + outlier_k x "
        "robust spread; fixed: use 'threshold'",
    )
    threshold: float | None = Field(None, description="intensity threshold for method='fixed'")
    min_bic_gain: float = Field(10.0, ge=0, description="auto: how much lower the two-population BIC must be")
    max_dip_ratio: float = Field(0.6, gt=0, le=1, description="auto: density at the dip / density of the lower peak")
    min_separation: float = Field(3.0, gt=0, description="auto: minimum Ashman's D between the two populations")
    strong_separation: float = Field(
        3.6, gt=0, description="auto: with at least this D, a shallower dip (up to strong_max_dip_ratio) is enough"
    )
    strong_max_dip_ratio: float = Field(0.8, gt=0, le=1, description="auto: dip limit for strongly separated populations")
    outlier_k: float = Field(6.0, gt=0, description="outlier rule: robust standard deviations above the median")
    min_droplets: int = Field(10, ge=2, description="auto: below this many droplets, use the outlier rule")


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _ashman_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    spread = math.sqrt(float(a.var(ddof=1)) + float(b.var(ddof=1)))
    return math.sqrt(2.0) * abs(float(a.mean()) - float(b.mean())) / spread if spread > 0 else float("inf")


def otsu_1d(values: np.ndarray) -> float | None:
    """Exact Otsu split of 1-D values: the gap between consecutive sorted values that
    maximises the between-class variance; returns the middle of that gap."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size < 2 or v[0] == v[-1]:
        return None
    n = v.size
    k = np.arange(1, n)  # sizes of the lower class
    csum = np.cumsum(v)[:-1]
    mean_low = csum / k
    mean_high = (v.sum() - csum) / (n - k)
    between = k * (n - k) * (mean_low - mean_high) ** 2
    between[v[1:] == v[:-1]] = -1.0  # cannot split between equal values
    i = int(np.argmax(between))
    return float((v[i] + v[i + 1]) / 2.0)


def dip_ratio(v: np.ndarray, threshold: float) -> float:
    """Depth of the density valley between the highest density peak on each side of
    ``threshold``: valley height / lower peak height. 1.0 when one side has no peak.

    The kernel bandwidth follows Silverman's rule with the robust spread (min of standard
    deviation and IQR/1.34), so a small population is not smoothed away."""
    sd = float(np.std(v, ddof=1))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    if sd <= 0 or spread <= 0:
        return 1.0
    kde = sps.gaussian_kde(v, bw_method=0.9 * spread * v.size ** -0.2 / sd)
    grid = np.linspace(float(v.min()), float(v.max()), 1024)
    d = kde(grid)
    peaks = np.nonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]))[0] + 1
    left = [i for i in peaks if grid[i] <= threshold]
    right = [i for i in peaks if grid[i] > threshold]
    if not left or not right:
        return 1.0
    il = max(left, key=lambda i: d[i])
    ih = max(right, key=lambda i: d[i])
    return float(d[il : ih + 1].min() / min(d[il], d[ih]))


def _mixture_threshold(v: np.ndarray) -> tuple[float, float, float]:
    """Two-population Gaussian mixture. Returns (threshold, BIC with one population, BIC
    with two); the threshold is where both components are equally likely, searched between
    the two component means. Deterministic (fixed random_state)."""
    from sklearn.mixture import GaussianMixture

    x = v.reshape(-1, 1)
    scale = max(float(np.std(v)), 1e-9)
    reg = (1e-3 * scale) ** 2  # keeps a one-droplet component from collapsing
    one = GaussianMixture(1, reg_covar=reg, random_state=0).fit(x)
    two = GaussianMixture(2, reg_covar=reg, n_init=3, random_state=0).fit(x)
    bic1, bic2 = float(one.bic(x)), float(two.bic(x))
    means = two.means_.ravel()
    lo, hi = int(np.argmin(means)), int(np.argmax(means))
    grid = np.linspace(means[lo], means[hi], 2001).reshape(-1, 1)
    post = two.predict_proba(grid)
    cross = np.nonzero(post[:, hi] >= post[:, lo])[0]
    threshold = float(grid[cross[0], 0]) if cross.size else float((means[lo] + means[hi]) / 2)
    return threshold, bic1, bic2


def per_microlitre(value: float | None, table: pd.DataFrame, column: str = "volume_pl") -> float | None:
    """Convert a per-droplet mean (e.g. lambda) to a concentration per microlitre using the
    mean droplet volume (1 pL = 1e-6 uL). None without volumes."""
    if value is None or column not in table.columns:
        return None
    volume = table[column].to_numpy(dtype=np.float64)
    volume = volume[np.isfinite(volume)]
    if volume.size == 0 or volume.mean() <= 0:
        return None
    return float(value) / (float(volume.mean()) * 1e-6)


def _copies(p: float | None) -> float | None:
    """Mean copies per droplet from the positive fraction (Poisson): -ln(1 - p)."""
    if p is None or not math.isfinite(p) or p >= 1.0:
        return None
    return -math.log1p(-p)


@register_op(
    "droplet.positivity",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE},
    outputs={"droplets": Kind.TABLE, "summary": Kind.SUMMARY},
    params=PositivityParams,
)
def positivity(params: PositivityParams, ctx: Context, *, table: pd.DataFrame) -> OpResult:
    """Call each droplet positive or negative by its fluorescence, then report the positive
    fraction (Wilson interval) and, for digital assays, mean copies per droplet
    lambda = -ln(1 - positive fraction)."""
    if params.column not in table.columns:
        raise KeyError(f"droplet.positivity: column {params.column!r} not in table (run droplet.intensity first)")
    if params.method == "fixed" and params.threshold is None:
        raise ValueError("droplet.positivity: method='fixed' requires 'threshold'")
    values = table[params.column].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    v = values[finite]
    n = int(v.size)

    method_used, threshold, separation = params.method, None, None
    bic1 = bic2 = dip = None
    if n:
        median = float(np.median(v))
        spread = 1.4826 * float(np.median(np.abs(v - median)))
        outlier_t = median + params.outlier_k * max(spread, 1e-9)
        if params.method == "fixed":
            threshold = float(params.threshold)
        elif params.method == "otsu":
            threshold = otsu_1d(v)
            if threshold is None:
                threshold, method_used = outlier_t, "outlier"
        elif params.method == "outlier":
            threshold = outlier_t
        else:
            accepted = None
            if n >= params.min_droplets and np.ptp(v) > 0:
                mixture_t, bic1, bic2 = _mixture_threshold(v)
                dip = dip_ratio(v, mixture_t)
                d = _ashman_d(v[v <= mixture_t], v[v > mixture_t])
                separated = d is None or d >= params.min_separation  # None: a one-droplet population
                strong = d is None or d >= params.strong_separation
                if (
                    bic2 + params.min_bic_gain < bic1
                    and ((dip < params.max_dip_ratio and separated) or (dip < params.strong_max_dip_ratio and strong))
                    and (v > mixture_t).any()
                    and (v <= mixture_t).any()
                ):
                    accepted = mixture_t
            threshold, method_used = (accepted, "mixture") if accepted is not None else (outlier_t, "outlier")
        separation = _ashman_d(v[v <= threshold], v[v > threshold])

    out = table.copy()
    out["positive"] = (values > threshold) if threshold is not None else False
    out.loc[~finite, "positive"] = False
    if threshold is not None and n:
        hi_v, lo_v = v[v > threshold], v[v <= threshold]
        scale = float(hi_v.mean() - lo_v.mean()) if (hi_v.size and lo_v.size) else params.outlier_k * max(spread, 1e-9)
        # signed distance to the threshold in units of the gap between the two groups;
        # values near 0 are the uncertain calls
        out["call_margin"] = (values - threshold) / max(scale, 1e-9)
    k = int(out.loc[finite, "positive"].sum())
    p = k / n if n else None
    lo, hi = _wilson(k, n) if n else (None, None)
    pos, neg = v[v > threshold] if threshold is not None else v[:0], v[v <= threshold] if threshold is not None else v
    summary = {
        "column": params.column,
        "n_droplets": n,
        "n_positive": k,
        "n_unmeasured": int((~finite).sum()),
        "positive_fraction": p,
        "positive_fraction_ci_low": lo,
        "positive_fraction_ci_high": hi,
        "copies_per_droplet": _copies(p),
        "copies_ci_low": _copies(lo),
        "copies_ci_high": _copies(hi),
        # concentration in the droplets, from the mean droplet volume; the interval reflects
        # counting statistics only, not volume uncertainty
        "copies_per_ul": per_microlitre(_copies(p), table),
        "copies_per_ul_ci_low": per_microlitre(_copies(lo), table),
        "copies_per_ul_ci_high": per_microlitre(_copies(hi), table),
        "threshold": threshold,
        "threshold_method": method_used,
        "separation_d": separation,
        "populations": 2 if method_used in ("mixture", "otsu") else (1 if method_used == "outlier" else None),
        "dip_ratio": dip,
        "bic_one_population": bic1,
        "bic_two_populations": bic2,
        "negative_mean": float(neg.mean()) if neg.size else None,
        "positive_mean": float(pos.mean()) if pos.size else None,
        "saturated": bool(n and k == n),
    }
    qc = {
        "n_droplets": float(n),
        "positive_fraction": float(p) if p is not None else float("nan"),
        "separation_d": float(separation) if separation is not None and math.isfinite(separation) else float("nan"),
    }
    return OpResult({"droplets": out, "summary": summary}, qc)


# -- review flags ----------------------------------------------------------------------------

FLAG_REASONS = {
    "low_circularity": "形状不圆，可能是粘连或碎片",
    "size_outlier": "粒径明显偏离其他液滴",
    "count_unstable": "细胞数在各帧之间不一致",
    "near_threshold": "荧光接近阴阳阈值",
    "few_frames": "在视野中停留的帧数少，测量依据少",
}
FLAG_COLUMNS = ["track_id", "frame", "centroid_y_px", "centroid_x_px", "diameter_px", "reasons", "reason_text"]


class FlagParams(Params):
    min_circularity: float = Field(0.85, ge=0, le=1, description="flag droplets whose median circularity is below this")
    size_outlier_k: float = Field(4.0, gt=0, description="flag diameters this many robust standard deviations from the median")
    min_inclusion_agreement: float = Field(0.8, ge=0, le=1, description="flag cell counts that agree across fewer frames")
    threshold_margin: float = Field(0.1, ge=0, description="flag fluorescence within this share of the group gap from the threshold")
    min_frames_fraction: float = Field(0.5, ge=0, le=1, description="flag droplets in view (first to last frame) for less than this x the median")


@register_op(
    "droplet.flags",
    scope=Scope.WINDOW,
    inputs={"table": Kind.TABLE, "tracks": Kind.TABLE},
    outputs={"table": Kind.TABLE, "summary": Kind.SUMMARY},
    params=FlagParams,
)
def flags(params: FlagParams, ctx: Context, *, table: pd.DataFrame, tracks: pd.DataFrame) -> OpResult:
    """List droplets whose result deserves a human look (odd shape or size, unstable cell
    count, fluorescence near the threshold, few frames), with the frame and position to
    inspect: the whole-droplet frame closest to the counting line."""
    n = len(table)
    reasons: dict[int, list[str]] = {i: [] for i in range(n)}
    if n:
        circ = table["circularity"].to_numpy(dtype=np.float64)
        for i in np.nonzero(circ < params.min_circularity)[0]:
            reasons[i].append("low_circularity")
        d = table["diameter"].to_numpy(dtype=np.float64)
        med = float(np.median(d))
        spread = 1.4826 * float(np.median(np.abs(d - med)))
        if spread > 0:
            for i in np.nonzero(np.abs(d - med) > params.size_outlier_k * spread)[0]:
                reasons[i].append("size_outlier")
        if "inclusion_agreement" in table.columns:
            agree = table["inclusion_agreement"].to_numpy(dtype=np.float64)
            for i in np.nonzero(agree < params.min_inclusion_agreement)[0]:
                reasons[i].append("count_unstable")
        if "call_margin" in table.columns:
            margin = table["call_margin"].to_numpy(dtype=np.float64)
            for i in np.nonzero(np.abs(margin) < params.threshold_margin)[0]:
                reasons[i].append("near_threshold")
        # Judged on the span a droplet stayed in view (first to last frame), not on how many
        # frames saw it: a stream that thins frames when it falls behind sees every droplet
        # in fewer frames, which says nothing about the droplet.
        if {"first_frame", "last_frame"} <= set(table.columns):
            frames = table["last_frame"].to_numpy(dtype=np.float64) - table["first_frame"].to_numpy(dtype=np.float64) + 1
        else:
            frames = table["n_frames"].to_numpy(dtype=np.float64)
        for i in np.nonzero(frames < params.min_frames_fraction * float(np.median(frames)))[0]:
            reasons[i].append("few_frames")

    rows = []
    by_track = {tid: g for tid, g in tracks.groupby("track_id")} if "track_id" in tracks.columns else {}
    for i, why in reasons.items():
        if not why:
            continue
        r = table.iloc[i]
        g = by_track.get(int(r["track_id"]))
        if g is None or g.empty:
            continue
        whole = g[~g["touches_border"].astype(bool)] if "touches_border" in g.columns else g
        pick = (whole if len(whole) else g).iloc[
            int(np.argmin(np.abs((whole if len(whole) else g)["frame"].to_numpy(dtype=np.float64) - float(r["crossing_frame"]))))
        ]
        rows.append(
            {
                "track_id": int(r["track_id"]),
                "frame": int(pick["frame"]),
                "centroid_y_px": float(pick["centroid_y_px"]),
                "centroid_x_px": float(pick["centroid_x_px"]),
                "diameter_px": float(pick["diameter_px"]),
                "reasons": ";".join(why),
                "reason_text": "；".join(FLAG_REASONS[w] for w in why),
            }
        )
    out = pd.DataFrame(rows, columns=FLAG_COLUMNS)
    counts = {k: int(sum(k in w for w in reasons.values())) for k in FLAG_REASONS}
    flagged = int(sum(bool(w) for w in reasons.values()))
    summary = {
        "n_droplets": n,
        "n_flagged": flagged,
        "flagged_fraction": flagged / n if n else None,
        **{f"n_{k}": v for k, v in counts.items()},
    }
    return OpResult({"table": out, "summary": summary}, {"flagged_fraction": flagged / n if n else 0.0, "n_flagged": float(flagged)})
