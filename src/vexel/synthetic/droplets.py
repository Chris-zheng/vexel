"""Synthetic brightfield droplet videos with exact ground truth.

Public droplet datasets lack size, tracking and frequency truth (roadmap §12), so these
videos are the reference for diameter CV, de-duplicated counting and generation
frequency. Droplets are dark rings on a bright, unevenly lit channel, moving at constant
speed along +x. The ring's outer edge is the true droplet boundary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

VIDEO_NAME = "droplets.ome.tif"


@dataclass(frozen=True)
class DropletSynthConfig:
    n_frames: int = 400
    height: int = 256
    width: int = 768
    pixel_size_um: float = 2.0
    frame_interval_s: float = 0.001
    diameter_mean_um: float = 60.0
    diameter_cv: float = 0.03
    frequency_hz: float = 100.0
    speed_um_s: float = 20000.0
    period_jitter: float = 0.02  # std of birth-time jitter, as a fraction of the period
    y_jitter_um: float = 4.0
    ring_width_px: float = 3.0
    ring_contrast: float = 90.0
    noise_sigma: float = 3.0
    seed: int = 0

    @property
    def speed_px_per_frame(self) -> float:
        return self.speed_um_s * self.frame_interval_s / self.pixel_size_um


def _coverage(x: np.ndarray) -> np.ndarray:
    """Linear 1-pixel edge ramp: fraction of a pixel on the positive side."""
    return np.clip(x + 0.5, 0.0, 1.0)


def _background(cfg: DropletSynthConfig) -> np.ndarray:
    yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float32)
    bg = 180.0 + 12.0 * (xx / cfg.width - 0.5) - 10.0 * ((yy / cfg.height - 0.5) ** 2) * 4
    wall = max(8, cfg.height // 9)
    bg[:wall] -= 60.0
    bg[cfg.height - wall :] -= 60.0
    return bg


def generate(cfg: DropletSynthConfig = DropletSynthConfig()) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return ``(video uint8 (T, Y, X), droplets, positions)``.

    ``droplets`` has one row per droplet; ``positions`` one row per droplet per frame in
    which any part of it is inside the image.
    """
    rng = np.random.default_rng(cfg.seed)
    v = cfg.speed_px_per_frame
    period = 1.0 / (cfg.frequency_hz * cfg.frame_interval_s)  # frames between droplets
    r_mean = cfg.diameter_mean_um / cfg.pixel_size_um / 2.0
    start_x = -(r_mean * 1.5 + 2.0)
    lead = int(np.ceil((cfg.width - start_x) / v / period)) + 2  # droplets already in view at t=0
    n = lead + int(np.ceil(cfg.n_frames / period)) + 1

    index = np.arange(n) - lead
    birth = (index + rng.normal(0.0, cfg.period_jitter, n)) * period
    diameter_um = rng.normal(cfg.diameter_mean_um, cfg.diameter_mean_um * cfg.diameter_cv, n)
    radius = diameter_um / cfg.pixel_size_um / 2.0
    y = cfg.height / 2.0 + rng.normal(0.0, cfg.y_jitter_um / cfg.pixel_size_um, n)
    droplets = pd.DataFrame(
        {
            "droplet_id": np.arange(n),
            "diameter_um": diameter_um,
            "birth_frame": birth,
            "y_px": y,
            "speed_px_per_frame": v,
        }
    )

    background = _background(cfg)
    video = np.empty((cfg.n_frames, cfg.height, cfg.width), dtype=np.uint8)
    rows = []
    for t in range(cfg.n_frames):
        frame = background.copy()
        x = start_x + v * (t - birth)
        for i in np.nonzero((x + radius > 0) & (x - radius < cfg.width - 1))[0]:
            r, cx, cy = radius[i], x[i], y[i]
            x0, x1 = max(0, int(cx - r - 2)), min(cfg.width, int(cx + r + 3))
            y0, y1 = max(0, int(cy - r - 2)), min(cfg.height, int(cy + r + 3))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.hypot(yy - cy, xx - cx)
            inner = _coverage(r - cfg.ring_width_px - d)
            ring = _coverage(r - d) - inner
            frame[y0:y1, x0:x1] += 8.0 * inner - cfg.ring_contrast * ring
            visible = (cx - r >= 0) and (cx + r <= cfg.width - 1)
            rows.append((t, i, cx, cy, visible))
        frame += rng.normal(0.0, cfg.noise_sigma, frame.shape)
        video[t] = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    positions = pd.DataFrame(rows, columns=["frame", "droplet_id", "x_px", "y_px", "fully_visible"])
    return video, droplets, positions


def write_dataset(out_dir: str | Path, cfg: DropletSynthConfig = DropletSynthConfig()) -> Path:
    """Generate a dataset: OME-TIFF video (calibrated) plus ``truth/`` tables. Returns the video path."""
    out_dir = Path(out_dir)
    truth = out_dir / "truth"
    truth.mkdir(parents=True, exist_ok=True)
    video, droplets, positions = generate(cfg)
    path = out_dir / VIDEO_NAME
    tifffile.imwrite(
        path,
        video,
        metadata={
            "axes": "TYX",
            "PhysicalSizeX": cfg.pixel_size_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": cfg.pixel_size_um,
            "PhysicalSizeYUnit": "µm",
            "TimeIncrement": cfg.frame_interval_s,
            "TimeIncrementUnit": "s",
        },
    )
    droplets.to_csv(truth / "droplets.csv", index=False)
    positions.to_csv(truth / "positions.csv", index=False)
    (truth / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return path
