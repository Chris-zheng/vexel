"""Probe: a fast, deterministic fingerprint of a dataset, for planning and recipe memory.

The fingerprint has two parts (roadmap, section 5.2):

* metadata: modality (reader), channels, frame size and count, calibration;
* image statistics, per channel, from a few evenly spaced frames: intensity range and
  skewness, bimodality, noise and signal-to-noise ratio, the share of the field that is
  static background, motion (the flow, from frame-to-frame correlation) and camera
  jitter (the shift of the static scene).

For droplet data it also runs the circle detector and the cell counter on a few frames
(object count, size, roundness, polarity; whether small dark spots sit inside the
droplets) and compares channels, to tell a brightfield droplet channel from a
fluorescence one. A 2 x 2 diagnostic montage (downsampled) can be written for a
person, or a vision model when the user allows it.

Nothing here changes data or writes outside the given montage path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vexel.core.volume import SpatialVolume

N_SAMPLES = 24  # frames sampled for the statistics
N_OBJECT_FRAMES = 6  # frames on which objects are detected


@dataclass
class ChannelStats:
    index: int
    name: str | None
    p1: float
    p50: float
    p99: float
    skewness: float
    bimodality: float  # Ashman D between the two Otsu classes of pixel values
    noise: float  # temporal noise, from differences of consecutive frames
    snr: float  # (p99 - p1) / noise
    static_fraction: float  # share of pixels that hardly change over time
    moving_fraction: float  # share of pixels that change clearly
    flow_px: tuple[float, float]  # (dy, dx) per frame of the moving content
    jitter_px: float  # shift of the static scene between frames (camera / stage)
    objects_per_frame: float = 0.0
    object_diameter_px: float | None = None
    object_circularity: float | None = None
    object_contrast: float | None = None  # mean |difference from background| inside objects
    interior_minus_rim: float | None = None  # > 0: bright centre / dark outline
    spots_per_object: float | None = None  # strong small dark spots inside objects (cells, beads)
    spot_stability: float | None = None  # spots kept at a strict / a normal threshold (cells ~1, shading << 1)
    spot_contrast: float | None = None
    outside_level: float | None = None  # median level outside objects
    inside_level: float | None = None  # median level inside objects


@dataclass
class Fingerprint:
    source: str | None
    reader: str | None
    shape: dict[str, int]
    dtype: str
    calibrated: bool
    pixel_size: float | None
    space_unit: str
    frame_interval_s: float | None
    channels: list[ChannelStats] = field(default_factory=list)
    droplet_channel: int | None = None
    fluorescence_channel: int | None = None
    flow_axis: str | None = None  # "x" or "y"
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def features(self) -> dict[str, float]:
        """Numeric features for recipe memory (of the droplet channel)."""
        ch = self.channels[self.droplet_channel or 0] if self.channels else None
        if ch is None:
            return {}
        out = {
            "log_snr": float(np.log10(max(ch.snr, 1e-3))),
            "skewness": ch.skewness,
            "bimodality": ch.bimodality,
            "static_fraction": ch.static_fraction,
            "moving_fraction": ch.moving_fraction,
            "speed_px": float(np.hypot(*ch.flow_px)),
            "jitter_px": ch.jitter_px,
            "objects_per_frame": ch.objects_per_frame,
            "diameter_px": ch.object_diameter_px or 0.0,
            "circularity": ch.object_circularity or 0.0,
            "interior_minus_rim": ch.interior_minus_rim or 0.0,
            "spots_per_object": ch.spots_per_object or 0.0,
        }
        return {k: float(v) for k, v in out.items()}


def _ashman_d(values: np.ndarray) -> float:
    from vexel.packs.droplet.ops import _ashman_d as d, otsu_1d

    v = values[np.isfinite(values)]
    if v.size < 10 or np.ptp(v) == 0:
        return 0.0
    t = otsu_1d(v)
    if t is None:
        return 0.0
    out = d(v[v <= t], v[v > t])
    return float(out) if out is not None and np.isfinite(out) else 0.0


def _shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float, float]:
    """(dy, dx) that best aligns b to a, by phase correlation (whole pixels)."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = fa.conj() * fb
    r /= np.abs(r) + 1e-9
    c = np.fft.ifft2(r).real
    h, w = c.shape
    y, x = np.unravel_index(int(np.argmax(c)), c.shape)
    dy, dx = (y if y <= h // 2 else y - h), (x if x <= w // 2 else x - w)
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return 0.0, 0.0
    return float(dy), float(dx)


def _channel_stats(index: int, name: str | None, frames: np.ndarray, consecutive: np.ndarray) -> ChannelStats:
    sample = frames.astype(np.float32)
    flat = sample[:, ::2, ::2].ravel()
    p1, p50, p99 = (float(v) for v in np.percentile(flat, (1, 50, 99)))
    sd = float(flat.std()) or 1.0
    skew = float(((flat - flat.mean()) ** 3).mean() / sd**3)
    diffs = np.diff(consecutive.astype(np.float32), axis=0)
    noise = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) / np.sqrt(2) if diffs.size else 0.0
    noise = max(noise, 1e-3)
    temporal = sample.std(axis=0)
    static = float((temporal < 3 * noise).mean())
    moving = float((temporal > 8 * noise).mean())
    # flow: shift of the moving content between consecutive frames, from the
    # background-removed frames; jitter: shift of the static scene itself
    background = np.median(sample, axis=0)
    flows, jitters = [], []
    max_shift = min(consecutive.shape[1:]) // 3
    for a, b in zip(consecutive[:-1], consecutive[1:]):
        a, b = a.astype(np.float32), b.astype(np.float32)
        flows.append(_shift(np.abs(a - background), np.abs(b - background), max_shift))
        jitters.append(np.hypot(*_shift(np.where(temporal < 3 * noise, a, 0), np.where(temporal < 3 * noise, b, 0), 5)))
    flow = tuple(float(v) for v in np.median(np.array(flows), axis=0)) if flows else (0.0, 0.0)
    return ChannelStats(
        index=index, name=name, p1=p1, p50=p50, p99=p99, skewness=skew, bimodality=_ashman_d(flat), noise=noise,
        snr=(p99 - p1) / noise, static_fraction=static, moving_fraction=moving, flow_px=flow,  # type: ignore[arg-type]
        jitter_px=float(np.median(jitters)) if jitters else 0.0,
    )  # fmt: skip


def _objects(stats: ChannelStats, frames: np.ndarray, background: np.ndarray) -> None:
    """Fill the object fields of ``stats`` from circle detection and the cell counter."""
    from skimage.measure import regionprops

    from vexel.ops import get_op
    from vexel.ops.registry import Context

    seg, inc = get_op("segment.circles"), get_op("droplet.inclusions")
    ctx = Context(frame_shape=frames.shape[1:], n_frames=1)
    counts, diam, circ, contrast, centre_rim, spots, strict, spot_c, outside, inside = [], [], [], [], [], [], [], [], [], []
    for f in frames.astype(np.float32):
        lab = seg.fn(seg.params(), ctx, image=f, contrast=np.abs(f - background)).outputs["labels"]
        props = [p for p in regionprops(lab) if 0 not in p.bbox[:2] and p.bbox[2] < lab.shape[0] and p.bbox[3] < lab.shape[1]]
        counts.append(len(props))
        if not props:
            continue
        diam += [p.equivalent_diameter_area for p in props]
        circ += [4 * np.pi * p.area / max(p.perimeter, 1) ** 2 for p in props]
        inside_mask = lab > 0
        contrast.append(float(np.abs(f - background)[inside_mask].mean()))
        outside.append(float(np.median(f[~inside_mask])))
        inside.append(float(np.median(f[inside_mask])))
        for p in props[:20]:
            coords = p.coords
            r = np.hypot(coords[:, 0] - p.centroid[0], coords[:, 1] - p.centroid[1])
            rmax = r.max() or 1
            core, rim = f[tuple(coords[r < 0.5 * rmax].T)], f[tuple(coords[r > 0.8 * rmax].T)]
            if core.size and rim.size:
                centre_rim.append(float(np.median(core) - np.median(rim)))
        # Cells and beads are strong spots: their count hardly changes when the threshold is
        # raised; shading and reflections inside droplets fade away.
        for noise_factor, bucket in ((6.0, spots), (15.0, strict)):
            res = inc.fn(inc.params(edge_margin_px=3, background="radial", relative_floor=0.3, noise_factor=noise_factor), ctx, image=f, labels=lab)
            valid = res.outputs["table"]["n_inclusions"].dropna()
            if len(valid):
                bucket.append(float(valid.mean()))
                if bucket is spots:
                    spot_c.append(float(res.qc.get("contrast_threshold", 0.0)))
    stats.objects_per_frame = float(np.mean(counts)) if counts else 0.0
    stats.object_diameter_px = float(np.median(diam)) if diam else None
    stats.object_circularity = float(np.median(circ)) if circ else None
    stats.object_contrast = float(np.mean(contrast)) if contrast else None
    stats.interior_minus_rim = float(np.median(centre_rim)) if centre_rim else None
    normal, hard = (float(np.mean(spots)) if spots else 0.0), (float(np.mean(strict)) if strict else 0.0)
    stats.spot_stability = hard / normal if normal > 0 else None
    stats.spots_per_object = (hard if (stats.spot_stability or 0) >= 0.6 else 0.0) if spots else None
    stats.spot_contrast = float(np.mean(spot_c)) if spot_c else None
    stats.outside_level = float(np.median(outside)) if outside else None
    stats.inside_level = float(np.median(inside)) if inside else None


def probe(volume: SpatialVolume, *, z: int = 0, max_frames: int | None = None) -> Fingerprint:
    sizes = volume.sizes
    n = min(sizes["T"], max_frames) if max_frames else sizes["T"]
    picks = np.unique(np.linspace(0, n - 1, min(n, N_SAMPLES)).round().astype(int))
    start = max(0, n // 2 - 4)
    run = np.arange(start, min(n, start + 8))  # consecutive frames: noise, flow, jitter
    fp = Fingerprint(
        source=volume.source, reader=volume.metadata.get("reader"), shape={k: int(v) for k, v in sizes.items()},
        dtype=str(volume.dtype), calibrated=volume.space_unit not in ("px", "pixel"),
        pixel_size=float(volume.spacing["x"]) if volume.space_unit not in ("px", "pixel") else None, space_unit=volume.space_unit,
        frame_interval_s=float(volume.spacing["t"]) if volume.time_unit == "s" else None,
    )  # fmt: skip
    names = volume.channel_names or [None] * sizes["C"]
    for c in range(sizes["C"]):
        lazy = volume.frames(c, z)
        frames = np.asarray(lazy[picks.tolist()])
        consecutive = np.asarray(lazy[run.tolist()])
        stats = _channel_stats(c, names[c] if c < len(names) else None, frames, consecutive)
        if n >= 3:  # whether objects are there is for the detector to say (its threshold follows the noise)
            objs = frames[np.unique(np.linspace(0, len(frames) - 1, min(len(frames), N_OBJECT_FRAMES)).round().astype(int))]
            _objects(stats, objs, np.median(frames.astype(np.float32), axis=0))
        fp.channels.append(stats)
    _interpret(fp)
    return fp


def _interpret(fp: Fingerprint) -> None:
    """Which channel shows the droplets, which (if any) is fluorescence, the flow axis."""
    chans = fp.channels
    if not chans:
        return
    def dark_field(c: ChannelStats) -> bool:  # most pixels near the floor: fluorescence
        return (c.p50 - c.p1) < 0.25 * max(c.p99 - c.p1, 1e-9) and c.skewness > 1.0

    def score(c: ChannelStats) -> float:
        return c.objects_per_frame * (c.object_circularity or 0)

    bright = [c for c in chans if not dark_field(c) and score(c) > 0]
    pool = bright or [c for c in chans if score(c) > 0]
    fp.droplet_channel = max(pool, key=score).index if pool else 0
    d = chans[fp.droplet_channel]
    if d.objects_per_frame < 0.5:
        fp.notes.append("没有找到圆形的运动目标：可能不是流动的液滴，或需要画笔训练")
    for c in chans:
        if c.index != fp.droplet_channel and dark_field(c):
            fp.fluorescence_channel = c.index
            break
    dy, dx = d.flow_px
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        fp.flow_axis = "x" if abs(dx) >= abs(dy) else "y"
    elif d.moving_fraction > 0.002:
        fp.notes.append("目标在动，但帧间位移没有测出：可能太快（超过画面三分之一）或方向不定")
    if d.jitter_px >= 1.0:
        fp.notes.append(f"静止背景有约 {d.jitter_px:.1f} 像素的帧间抖动：背景扣除可能留下残影")
    if not fp.calibrated:
        fp.notes.append("没有像素尺寸：结果将以像素为单位")
    if fp.frame_interval_s is None:
        fp.notes.append("没有帧间隔：频率将以每帧为单位")
    if d.spots_per_object is not None and d.spots_per_object >= 0.15:
        fp.notes.append(f"液滴里有小的深色斑点（平均每滴约 {d.spots_per_object:.1f} 个）：可能是细胞或珠子")


def montage(volume: SpatialVolume, fp: Fingerprint, path: str | Path, *, size: int = 512) -> Path:
    """2 x 2 diagnostic picture: a frame, the temporal median, the temporal spread and the
    circle detection on the frame, each downsampled to fit ``size`` / 2 pixels."""
    import imageio.v3 as iio
    from skimage.segmentation import find_boundaries
    from skimage.transform import resize

    from vexel.ops import get_op
    from vexel.ops.registry import Context

    c = fp.droplet_channel or 0
    lazy = volume.frames(c, 0)
    n = int(lazy.shape[0])
    sample = np.asarray(lazy[np.unique(np.linspace(0, n - 1, min(n, N_SAMPLES)).round().astype(int)).tolist()]).astype(np.float32)
    frame, med, spread = sample[len(sample) // 2], np.median(sample, axis=0), sample.std(axis=0)

    def gray(a: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(a, (1, 99))
        g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        return np.dstack([g] * 3)

    seg = get_op("segment.circles")
    lab = seg.fn(seg.params(), Context(frame_shape=frame.shape, n_frames=1), image=frame, contrast=np.abs(frame - med)).outputs["labels"]
    det = gray(frame)
    det[find_boundaries(lab)] = (1.0, 0.15, 0.15)
    half = size // 2
    scale = half / max(frame.shape)
    shape = (max(1, int(frame.shape[0] * scale)), max(1, int(frame.shape[1] * scale)))
    tiles = [resize(t, (*shape, 3), anti_aliasing=True) for t in (gray(frame), gray(med), gray(spread), det)]
    pad = np.ones((shape[0], 4, 3))
    top, bottom = np.hstack([tiles[0], pad, tiles[1]]), np.hstack([tiles[2], pad, tiles[3]])
    out = np.vstack([top, np.ones((4, top.shape[1], 3)), bottom])
    path = Path(path)
    iio.imwrite(path, (out * 255).round().astype(np.uint8))
    return path
