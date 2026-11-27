"""SEM materials measurement acceptance: fibres, pores, particle classes, profile roughness, nested regions (cells and nuclei).

    uv run python tests/benchmarks/sem_materials_eval.py [--out table.md]

Synthetic images with exact geometry (``vexel.synthetic.sem``):

- fibre mats (``fibre_mat``): straight and gently curved fibres drawn over one another, with
  and without secondary-electron edge brightening (yield ~ 1/cos of the surface tilt, as on
  a cylinder), blur and noise. Every diameter sample is compared with the fibre whose
  centre line is nearest; orientation with the fibres' directions.
- pores: random discs darker than the matrix; porosity against the discs' area fraction.
- particle classes: separate discs plus a broken, a touching pair and a bright foreign one.
- profile roughness: sines of known amplitude with waviness and tilt (ISO 21920 Ra, Rq, Rz).

Requirements: fibre diameter bias (median) ≤ 2 % at 8-40 px without edge brightening,
≤ 6 % at 8-12 px and ≤ 3 % at ≥ 16 px with it (gradient edges), ≤ 6 % at 4-6 px; spread
(median |error|) ≤ 6 % at ≥ 8 px; aligned fibres: order ≥ 0.95, direction ≤ 2°; porosity
≤ 0.01 absolute; classes all right; Ra, Rq, Rz ≤ 3 %.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from vexel.packs.sem import objects as O
from vexel.packs.sem.fibres import fibres
from vexel.packs.sem.tilt3d import profile_roughness
from vexel.synthetic import sem as S

ROWS: list[dict] = []


def check(group: str, what: str, value: float, limit: float | None, unit: str = "%", lower: bool = True) -> bool:
    ok = True if limit is None else bool(value <= limit) if lower else bool(value >= limit)
    ROWS.append({"group": group, "what": what, "value": value, "limit": limit, "unit": unit, "pass": ok, "lower": lower})
    return ok


def _errors(img: np.ndarray, truth: list, **kw) -> np.ndarray:
    s = fibres(img, **kw)["samples"]
    pts = [t["points"] for t in truth]
    out = []
    for x, y, d in zip(s["x"], s["y"], s["diameter"]):
        k = int(np.argmin([np.min(np.hypot(p[:, 0] - x, p[:, 1] - y)) for p in pts]))
        out.append(d / truth[k]["diameter"] - 1)
    return np.asarray(out)


def fibre_diameters() -> None:
    cases = [((4, 6), 0.0, 6.0, None), ((8, 12), 0.0, 2.0, 6.0), ((16, 24), 0.0, 2.0, 6.0), ((30, 40), 0.0, 2.0, 6.0),
             ((8, 12), 0.25, 6.0, 6.0), ((16, 24), 0.25, 3.0, 6.0), ((30, 40), 0.25, 3.0, 6.0)]  # fmt: skip
    for dia, edge, lim_bias, lim_spread in cases:
        for curve in (0.0, 0.15):
            errs = []
            for seed in range(4):
                img, truth = S.fibre_mat((512, 512), n=8, diameter=dia, seed=seed, edge=edge, curve=curve)
                errs.append(_errors(img, truth, smooth=0.5 if dia[0] < 8 else 1.0))
            e = np.concatenate(errs) * 100
            label = f"{dia[0]}-{dia[1]} px, {'edge brightening' if edge else 'plain edges'}, {'curved' if curve else 'straight'}"
            check("fibres", f"{label}: diameter bias (median)", abs(float(np.median(e))), lim_bias)
            check("fibres", f"{label}: spread (median |error|)", float(np.median(np.abs(e))), lim_spread if lim_spread else None)
    # the threshold edge for comparison (recorded, no requirement)
    errs = []
    for seed in range(4):
        img, truth = S.fibre_mat((512, 512), n=8, diameter=(8, 12), seed=seed, edge=0.25)
        errs.append(_errors(img, truth, method="threshold"))
    check("fibres", "8-12 px, edge brightening: threshold edge bias (for comparison)", abs(float(np.median(np.concatenate(errs)))) * 100, None)


def fibre_orientation() -> None:
    orders, dirs = [], []
    for seed in range(4):
        a = 20.0 + 35 * seed
        img, truth = S.fibre_mat((512, 512), n=8, diameter=(8, 12), seed=seed, angle=(a - 2, a + 2))
        sm = fibres(img)["summary"]
        orders.append(sm["order"])
        d = abs(sm["orientation"] - np.mean([t["angle"] for t in truth]))
        dirs.append(min(d, 180 - d))
    check("fibres", "aligned mats: order parameter (worst of 4)", min(orders), 0.95, "", lower=False)
    check("fibres", "aligned mats: main direction error (worst of 4)", max(dirs), 2.0, "°")
    rnd = [fibres(S.fibre_mat((512, 512), n=14, diameter=(8, 12), seed=10 + s)[0])["summary"]["order"] for s in range(4)]
    check("fibres", "random mats: order parameter (mean, recorded)", float(np.mean(rnd)), None, "")


def pores() -> None:
    worst = 0.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        discs = [(float(rng.uniform(0, 500)), float(rng.uniform(0, 400)), float(rng.uniform(5, 16))) for _ in range(30 + 15 * seed)]
        cov = S._disks(400, 500, discs)
        img = ndi.gaussian_filter(0.8 - 0.6 * cov, 0.8) + rng.normal(0, 0.03, cov.shape)
        worst = max(worst, abs(O.pores(img)["summary"]["porosity"] - cov.mean()))
    check("pores", "porosity error (worst of 5, 5-40 % porosity)", worst, 0.01, "")


def particle_classes() -> None:
    rows, cols = 300, 420
    wrong = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        discs = [(50 + 70 * (i % 5), 50 + 70 * (i // 5), float(rng.uniform(9, 11))) for i in range(12)]
        cov = S._disks(rows, cols, discs)
        yy, xx = np.mgrid[0:rows, 0:cols]
        broken = ((xx - 260) ** 2 + (yy - 260) ** 2 <= 14**2) & ~((xx - 270) ** 2 + (yy - 260) ** 2 <= 9**2)
        pair = S._disks(rows, cols, [(370.0, 60.0, 10.0), (370.0, 79.0, 10.0)])
        img = ndi.gaussian_filter(0.2 + 0.6 * np.clip(cov + broken + pair, 0, 1), 0.8) + rng.normal(0, 0.02, cov.shape)
        img += 0.5 * ndi.gaussian_filter(S._disks(rows, cols, [(370.0, 200.0, 10.0)]), 0.8)
        r = O.particles(img)
        want = {(258, 260): "irregular", (370, 60): "agglomerate", (370, 79): "agglomerate", (370, 200): "outlier"}
        for (x, y), cls in want.items():
            p = min(r["particles"], key=lambda q: math.hypot(q["x"] - x, q["y"] - y))
            wrong += p["class"] != cls
        wrong += sum(1 for p in r["particles"] if p["class"] != "normal") - len(want)
    check("particles", "wrongly classed particles (5 images, 17 each)", wrong, 0, "")


def profile_rough() -> None:
    worst = 0.0
    for lam, A in ((12.0, 1.0), (20.0, 3.0), (35.0, 5.0)):
        x = np.arange(3000.0)
        z = A * np.sin(2 * np.pi * x / lam) + 30 * np.sin(2 * np.pi * x / 3000) + 0.02 * x
        r = profile_roughness(z, 1.0, cutoff=5 * lam)
        for got, want in ((r["Ra"], 2 * A / math.pi), (r["Rq"], A / math.sqrt(2)), (r["Rz"], 2 * A)):
            worst = max(worst, abs(got / want - 1) * 100)
    check("roughness", "Ra, Rq, Rz of sines with waviness and tilt (worst)", worst, 3.0)


def _nested_errors(kw: dict, mk: dict) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Cells found / in the truth, and per matched cell the area error, nucleus area error (%)
    and nucleus/cell ratio error (percentage points)."""
    from scipy.spatial import cKDTree

    img, tr = S.nested_cells(**kw)
    r = O.nested(img, **mk)
    found, truth = r["parents"], tr["cells"]
    if not found:
        return 0, len(truth), np.array([np.inf]), np.array([np.inf]), np.array([np.inf])
    dist, idx = cKDTree([[c["x"], c["y"]] for c in found]).query([[c["x"], c["y"]] for c in truth])
    ea, en, er, n = [], [], [], 0
    for c, dd, i in zip(truth, dist, idx):
        if dd > 5:
            continue
        f = found[i]
        n += 1
        ea.append(abs(f["area"] / c["area"] - 1) * 100)
        if f["child_area"]:
            en.append(abs(f["child_area"] / c["nucleus_area"] - 1) * 100)
            er.append(abs(f["ratio"] - c["ratio"]) * 100)
        else:
            en.append(np.inf)
            er.append(np.inf)
    return n, len(truth), np.array(ea), np.array(en), np.array(er)


def nested_cells() -> None:
    """Cells and nuclei: tissue (touching cells sharing membranes), cells with a membrane on a
    slide, cells without a membrane (plain dark objects); 3 images each, normal and high noise."""
    cases = (("tissue", {"layout": "tissue", "rows": 600, "cols": 700, "n": 30}, {}, 1.0),
             ("tissue, membrane 5 px", {"layout": "tissue", "rows": 600, "cols": 700, "n": 30, "membrane": 5.0}, {"membrane": 5.0}, 1.0),
             ("cells with membrane on bright slide", {"layout": "isolated"}, {}, 2.0),
             ("cells without membrane (dark objects)", {"layout": "isolated", "membrane": 0.0}, {"parent": "dark"}, 2.0))  # fmt: skip
    for label, kw, mk, area_lim in cases:
        for noise in (0.03, 0.08):
            found = total = 0
            ea, en, er = [], [], []
            for seed in (1, 2, 3):
                n, t, a, b, c = _nested_errors({**kw, "noise": noise, "seed": seed}, mk)
                found, total = found + n, total + t
                ea.append(a)
                en.append(b)
                er.append(c)
            ea, en, er = np.concatenate(ea), np.concatenate(en), np.concatenate(er)
            name = f"{label}, noise {noise}"
            check("nested", f"{name}: cells found", total - found, 0, "")
            check("nested", f"{name}: cell area error (mean |e|)", float(np.mean(ea)), area_lim)
            check("nested", f"{name}: nucleus area error (mean |e|)", float(np.mean(en)), 2.5)
            check("nested", f"{name}: nucleus/cell ratio error (worst)", float(np.max(er)), 1.0, "pp")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for f in (fibre_diameters, fibre_orientation, pores, particle_classes, profile_rough, nested_cells):
        f()
    out = ["| 项目 | 检查 | 结果 | 要求 | |", "| :-- | :-- | --: | --: | :-- |"]
    for r in ROWS:
        lim = "（记录）" if r["limit"] is None else f"{'≤' if r['lower'] else '≥'} {r['limit']:g} {r['unit']}"
        out.append(f"| {r['group']} | {r['what']} | {r['value']:.3g} {r['unit']} | {lim} | {'✓' if r['pass'] else '✗'} |")
    text = "\n".join(out)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    ok = all(r["pass"] for r in ROWS)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
