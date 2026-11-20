"""SEM tilt 3D acceptance (plan §2.6, roadmap 0.0.5 M6).

    uv run python tests/benchmarks/sem_tilt3d_eval.py [--out table.md]

Known height maps (a truncated pyramid, terraces, spherical caps, a corrugation, a random
rough surface) on a textured surface are rendered as SEM views at given stage tilts
(``vexel.synthetic.sem.tilt_view``: parallel projection, eucentric tilt, a slope-dependent
secondary-electron brightness, blur and noise) and reconstructed.

Required (plan): height error < 5 % on textured areas - here the RMS error over the whole
map as a fraction of its height range, after removing a constant (heights are relative),
at a 10° tilt difference and image noise 0.02 (0..1 grey); step heights (ISO 5436-1
two-plane fit) within 5 % over several textures, and within twice their reported uncertainty; a textureless patch marked unreliable; the reported uncertainty
within a factor of 2 of the actual error in most cases. Also reported: heavier noise,
larger tilt differences, more views, a vertical tilt axis, a tilted sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vexel.packs.sem import tilt3d as T
from vexel.synthetic import sem as S

ROWS: list[dict] = []


def check(group: str, what: str, value: float, limit: float | None, unit: str = "%", lower: bool = True) -> None:
    ok = True if limit is None else (value <= limit if lower else value >= limit)
    ROWS.append({"group": group, "what": what, "value": value, "limit": limit, "unit": unit, "pass": bool(ok)})


def rms_error(hm: T.HeightMap, z: np.ndarray, border: int = 14) -> float:
    c = slice(border, -border)
    d = (hm.z - z)[c, c]
    m = hm.valid[c, c]
    d = d - np.median(d[m])
    return float(np.sqrt(np.mean(d[m] ** 2)))


def views(z: np.ndarray, tilts: tuple[float, ...], noise: float, seed: int = 0, flat: np.ndarray | None = None) -> list[np.ndarray]:
    tex = S.surface_texture(*z.shape, seed=seed + 2)
    return [S.tilt_view(z, tex, t, noise=noise, flat=flat, seed=seed + i) for i, t in enumerate(tilts)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for kind in ("pyramid", "steps", "spheres", "sine", "rough"):
        z = S.height_map(kind, seed=1)
        for noise, limit in ((0.02, 5.0), (0.05, None)):
            hm = T.reconstruct(views(z, (-5, 5), noise), [-5, 5])
            check("height map", f"{kind}, ±5°, noise {noise}: RMS error / height range", 100 * rms_error(hm, z) / np.ptp(z), limit)
    z = S.height_map("pyramid", seed=1)
    for tilts in ((-10, 10), (-10, -5, 0, 5, 10)):
        hm = T.reconstruct(views(z, tilts, 0.05), list(tilts), reference=len(tilts) // 2)
        check("height map", f"pyramid, tilts {tilts}, noise 0.05: RMS error / height range", 100 * rms_error(hm, z) / np.ptp(z), None)
    hm = T.reconstruct([np.rot90(v) for v in views(z, (-5, 5), 0.02)], [-5, 5], axis=90)
    check("height map", "pyramid, vertical tilt axis: RMS error / height range", 100 * rms_error(hm, np.rot90(z)) / np.ptp(z), 5.0)
    # steps: two-plane fit, also on a tilted sample; four textures each
    zs = S.height_map("steps", seed=1)
    rows, cols = zs.shape
    yy, xx = np.mgrid[:rows, :cols]
    true = zs[:, -10].mean() - zs[:, 10].mean()
    for name, zz in (("level sample", zs), ("tilted sample", zs + 0.03 * xx + 0.02 * yy)):
        errs, cover = [], []
        for seed in range(4):
            hm = T.reconstruct(views(zz, (-5, 5), 0.02, seed=seed), [-5, 5])
            st = T.step_height(hm, hm.z, [5, 20, int(cols * 0.15), rows - 20], [int(cols * 0.75), 20, cols - 5, rows - 20])
            errs.append(abs(st["step"] / true - 1))
            cover.append(abs(st["step"] - true) <= 2 * st["uncertainty"])
        check("step height", f"{name}: two-plane step error, worst of 4 textures", 100 * max(errs), 5.0)
        check("step height", f"{name}: error within 2 × the reported uncertainty", 100 * float(np.mean(cover)), 75.0, lower=False)
    # a textureless patch
    flat = np.zeros_like(z, bool)
    flat[40:120, 40:140] = True
    hm = T.reconstruct(views(z, (-5, 5), 0.02, flat=flat), [-5, 5])
    check("confidence", "textureless patch: share marked unreliable", 100 * (1 - hm.valid[flat].mean()), 70.0, lower=False)
    check("confidence", "textured surface: share marked reliable", 100 * hm.valid[~flat].mean(), 95.0, lower=False)
    # the uncertainty against the actual error
    ratios = []
    for kind in ("pyramid", "rough", "sine"):
        zz = S.height_map(kind, seed=3)
        for noise in (0.01, 0.03, 0.06):
            for dt in (6, 10, 16):
                hm = T.reconstruct(views(zz, (-dt / 2, dt / 2), noise, seed=4), [-dt / 2, dt / 2])
                c = slice(14, -14)
                ratios.append(rms_error(hm, zz) / float(np.median(hm.uncertainty[c, c])))
    r = np.array(ratios)
    check("uncertainty", "actual / reported, median", float(np.median(r)), None, "×")
    check("uncertainty", "cases within a factor of 2", 100 * float(np.mean((r >= 0.5) & (r <= 2))), 70.0, lower=False)
    lines = ["| 项目 | 检查 | 结果 | 要求 | |", "| :-- | :-- | --: | --: | :-- |"]
    for row in ROWS:
        lim = "（记录）" if row["limit"] is None else (f"≤ {row['limit']:g}" if "unreliable" not in row["what"] and "reliable" not in row["what"] and "within" not in row["what"] else f"≥ {row['limit']:g}") + f" {row['unit']}"
        lines.append(f"| {row['group']} | {row['what']} | {row['value']:.3g} {row['unit']} | {lim} | {'✓' if row['pass'] else '✗'} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    ok = all(row["pass"] for row in ROWS)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
