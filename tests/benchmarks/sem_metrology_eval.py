"""SEM automatic measurement acceptance (plan §2.4 and §7 A, roadmap 0.0.5 M4).

    uv run python tests/benchmarks/sem_metrology_eval.py [--out table.md]

Synthetic images with exact geometry (``vexel.synthetic.sem``): rough line/space
patterns, contact-hole arrays, particle fields, layer stacks, line and trench cross-sections
and tips. Each recipe is compared with the truth; repeatability is the 3σ of a result over
10 images of the same pattern with fresh noise (what repeated shots of one place give).

Requirements: CD ≤ 0.5 px, pitch ≤ 0.2 px; noise-corrected LER and LWR within 10 % of the
same pattern measured without noise; hole diameter ≤ 0.3 px (no rim), missing holes all
found; particle diameter < 2 % (median) and count within 5 %; layer thickness ≤ 0.5 px;
sidewall angle ≤ 0.5°, widths ≤ 0.5 px; tip radius < 5 %, cone angle ≤ 0.5°.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from vexel.packs.sem import metrology as M
from vexel.packs.sem import objects as O
from vexel.packs.sem.tips import tip
from vexel.synthetic import sem as S

ROWS: list[dict] = []


def check(group: str, what: str, value: float, limit: float, unit: str = "px", lower: bool = True) -> bool:
    ok = bool(value <= limit) if lower else bool(value >= limit)
    ROWS.append({"group": group, "what": what, "value": value, "limit": limit, "unit": unit, "pass": ok})
    return ok


def _detrended(s: np.ndarray) -> float:
    y = np.arange(len(s))
    return float(np.std(s - np.polyval(np.polyfit(y, s, 1), y)))


def lines() -> None:
    for name, kw in (("bright lines", {}), ("dark lines", {"feature": "dark"}), ("slanted 2.9°", {"slant": -0.05}), ("edge peaks", {"edge_peaks": True}),
                     ("noisy (σ 0.08)", {"noise": 0.08})):  # fmt: skip
        feat = kw.pop("feature", "bright")
        img, tr = S.rough_lines(seed=3, **kw)
        if feat == "dark":
            img = 1 - img
        r = M.lines_spaces(img, feature=feat, method="peak" if kw.get("edge_peaks") else "threshold")
        across = math.cos(math.atan(kw.get("slant", 0.0)))
        cd_true = np.mean([np.mean(ln["right"] - ln["left"]) for ln in tr["lines"]]) * across
        if kw.get("edge_peaks"):
            continue_ok = abs(r.summary["cd_mean"] - cd_true)
            ROWS.append({"group": "lines", "what": f"{name}: CD offset of the edge-peak definition (peaks sit inside the line)",
                         "value": continue_ok, "limit": None, "unit": "px", "pass": True})  # fmt: skip
        else:
            check("lines", f"{name}: CD error", abs(r.summary["cd_mean"] - cd_true), 0.5)
        check("lines", f"{name}: pitch error", abs(r.summary["pitch"] - tr["pitch"] * across), 0.2)
        check("lines", f"{name}: lines found", abs(r.summary["lines"] - len(tr["lines"])), 0, "lines")
    # roughness: noise-corrected against the same pattern without noise
    for corr in (3, 8, 25):
        ref = M.lines_spaces(S.rough_lines(seed=5, corr=corr, noise=0.0)[0]).summary
        out = M.lines_spaces(S.rough_lines(seed=5, corr=corr, noise=0.08, noise_seed=99)[0]).summary
        check("lines", f"LER corrected vs noise-free (ξ {corr} lines)", abs(out["ler_3sigma_unbiased"] / ref["ler_3sigma"] - 1) * 100, 10, "%")
        check("lines", f"LWR corrected vs noise-free (ξ {corr} lines)", abs(out["lwr_3sigma_unbiased"] / ref["lwr_3sigma"] - 1) * 100, 10, "%")
        ROWS.append({"group": "lines", "what": f"LER without correction vs noise-free (ξ {corr})", "value": (out["ler_3sigma"] / ref["ler_3sigma"] - 1) * 100,
                     "limit": None, "unit": "%", "pass": True})  # fmt: skip
    # against the true edges (no blur: the truth is exact), noise up to very high
    for noise in (0.08, 0.15, 0.25):
        b, c = [], []
        for seed in (5, 6, 7):
            img, tr = S.rough_lines(seed=seed, corr=8, noise=noise, blur=0.0, noise_seed=99)
            true = 3 * math.sqrt(np.mean([_detrended(ln[k]) ** 2 for ln in tr["lines"] for k in ("left", "right")]))
            s = M.lines_spaces(img).summary
            b.append(s["ler_3sigma"] / true - 1)
            c.append(s["ler_3sigma_unbiased"] / true - 1)
        check("lines", f"LER corrected vs true edges (noise σ {noise})", abs(np.mean(c)) * 100, 5, "%")
        ROWS.append({"group": "lines", "what": f"LER without correction vs true edges (noise σ {noise})", "value": np.mean(b) * 100, "limit": None,
                     "unit": "%", "pass": True})  # fmt: skip
    st, _ = S.rough_lines(seed=5, noise=0.03)
    check("lines", "CD change under charging stripes (σ 0.08)", abs(M.lines_spaces(S.add_stripes(st, 0.08)).summary["cd_mean"] - M.lines_spaces(st).summary["cd_mean"]), 0.05)
    # repeatability: one pattern, ten noise realisations
    cds = [M.lines_spaces(S.rough_lines(seed=7, noise=0.05, noise_seed=k)[0]).summary for k in range(10)]
    ROWS.append({"group": "repeatability", "what": "CD 3σ over 10 shots (σ noise 0.05)", "value": 3 * np.std([c["cd_mean"] for c in cds], ddof=1),
                 "limit": None, "unit": "px", "pass": True})  # fmt: skip
    ROWS.append({"group": "repeatability", "what": "LER (corrected) 3σ over 10 shots", "value": 3 * np.std([c["ler_3sigma_unbiased"] for c in cds], ddof=1),
                 "limit": None, "unit": "px", "pass": True})  # fmt: skip


def holes() -> None:
    for name, kw in (("no rim", {"rim": False}), ("no rim, σ 0.06", {"rim": False, "noise": 0.06})):
        img, tr = S.hole_array(seed=4, sd=0.8, **kw)
        r = O.holes(img)
        tc = np.array([[x - 0.5, y - 0.5] for x, y, _ in tr["holes"]])  # pixel-centre coordinates
        td = np.array([d for *_, d in tr["holes"]])
        mc = np.array([[h["x"], h["y"]] for h in r["holes"]])
        dist, idx = cKDTree(tc).query(mc)
        err = np.array([h["diameter"] for h in r["holes"]]) - td[idx]
        check("holes", f"{name}: diameter error (mean)", abs(err.mean()), 0.3)
        check("holes", f"{name}: diameter error (worst)", np.abs(err).max(), 1.0)
        check("holes", f"{name}: centre error (worst)", dist.max(), 0.3)
        check("holes", f"{name}: missing holes not found", abs(r["summary"]["missing"] - len(tr["missing"])), 0, "holes")
        check("holes", f"{name}: pitch error", abs(r["summary"]["pitch_a"] - tr["pitch"][0]), 0.05)
    img, tr = S.hole_array(seed=4, sd=0.8, rim=True)
    r = O.holes(img)
    td = np.mean([d for *_, d in tr["holes"]])
    for m in ("threshold", "gradient"):
        r = O.holes(img, method=m)
        ROWS.append({"group": "holes", "what": f"bright rims: diameter offset, {m} (edge definition)", "value": r["summary"]["diameter_mean"] - td,
                     "limit": None, "unit": "px", "pass": True})  # fmt: skip
    ds = [O.holes(S.hole_array(seed=4, rim=False, noise=0.05, noise_seed=k)[0])["summary"]["diameter_mean"] for k in range(10)]
    ROWS.append({"group": "repeatability", "what": "mean hole diameter 3σ over 10 shots", "value": 3 * np.std(ds, ddof=1), "limit": None, "unit": "px", "pass": True})


def particles() -> None:
    for name, kw in (("isolated", {"touching": 0.0}), ("20 % touching", {"touching": 0.2}), ("small (d 12)", {"d_mean": 12, "d_sd": 2, "touching": 0.0})):
        img, tr = S.particle_field(seed=6, **kw)
        r = O.particles(img)
        inside = [(x, y, d) for x, y, d in tr["particles"]]
        tc = np.array([[x - 0.5, y - 0.5] for x, y, _ in inside])
        td = np.array([d for *_, d in inside])
        mc = np.array([[p["x"], p["y"]] for p in r["particles"]])
        dist, idx = cKDTree(tc).query(mc)
        ok = dist < 2
        rel = np.abs(np.array([p["eq_diameter"] for p in r["particles"]])[ok] / td[idx[ok]] - 1) * 100
        check("particles", f"{name}: diameter error (median)", float(np.median(rel)), 2.0, "%")
        check("particles", f"{name}: count error", abs(len(r["particles"]) + r["summary"]["at_border"] - len(inside)) / len(inside) * 100, 5.0, "%")
        d50_true = O.distribution(td)["d50_n"]
        check("particles", f"{name}: d50 error", abs(r["summary"]["d50_n"] / d50_true - 1) * 100, 2.0, "%")


def cross_sections() -> None:
    img, tr = S.layer_stack(seed=1)
    r = M.layers(img)
    for ly, t in zip(r["layers"], tr["thicknesses"]):
        check("cross-section", f"layer {ly['layer']} thickness error", abs(ly["thickness"] - t), 0.5)
    # tilted and bent conformal stacks (thickness along the normal), straightened before measuring
    for name, kw in (("tilt 6°", {"tilt": 6}), ("tilt 12°", {"tilt": 12}), ("bow 30 px", {"bow": 30}),
                     ("bow 60 px + tilt 4°", {"bow": 60, "tilt": 4})):  # fmt: skip
        img, tr = S.bent_layer_stack(seed=3, **kw)
        r = M.stack_layers(img)
        check("cross-section", f"{name}: layers found", abs(len(r["layers"]) - len(tr["thicknesses"])), 0, "")
        for ly, t in zip(r["layers"], tr["thicknesses"]):
            check("cross-section", f"{name}: layer {ly['layer']} normal thickness error", abs(ly["thickness"] - t), 0.5)
    img, tr = S.bent_layer_stack(seed=3, bow=30, tilt=5)
    r = M.stack_layers(img.T, orientation="auto")
    check("cross-section", "vertical stack (bow 30 px): worst thickness error",
          max(abs(ly["thickness"] - t) for ly, t in zip(r["layers"], tr["thicknesses"])) if len(r["layers"]) == 3 else 99, 0.5)  # fmt: skip
    # a fine alternating stack (40 layers of 6 and 8 px, like a 3D NAND word-line stack), bent; each
    # thickness carries the error of two interfaces only a few pixels apart
    for noise in (0.03, 0.08):
        img, tr = S.bent_layer_stack(rows=420, thicknesses=(6.0, 8.0) * 20, levels=(0.2, 0.8, 0.45), top=40.5, bow=25, tilt=3,
                                     noise=noise, seed=3)  # fmt: skip
        r = M.stack_layers(img)
        check("cross-section", f"40 thin layers, noise {noise}: layers found", abs(len(r["layers"]) - 40), 0, "")
        err = np.array([abs(ly["thickness"] - t) for ly, t in zip(r["layers"], tr["thicknesses"])])
        check("cross-section", f"40 thin layers, noise {noise}: mean thickness error", float(err.mean()), 0.25)
        check("cross-section", f"40 thin layers, noise {noise}: worst thickness error", float(err.max()), 0.75)
    for feat in ("bright", "dark"):
        for ang in (80.0, 88.0, 95.0):
            img, tr = S.xsection_feature(angle=ang, feature=feat, seed=2)
            f = M.feature_profile(img, feature=feat)
            check("cross-section", f"{feat} {ang:.0f}°: sidewall angle error", max(abs(f["angle_left"] - ang), abs(f["angle_right"] - ang)), 0.5, "°")
            check("cross-section", f"{feat} {ang:.0f}°: middle width error", abs(f["width_middle"] - tr["width_middle"]), 0.5)
            check("cross-section", f"{feat} {ang:.0f}°: height error", abs(f["height"] - tr["height"]), 1.0)


def tips() -> None:
    for radius in (8.0, 20.0, 40.0):
        for half in (8.0, 20.0):
            errs, cones = [], []
            for seed in range(3):
                r = tip(S.tip(radius=radius, half_angle=half, seed=seed)[0])
                errs.append(abs(r["radius"] / radius - 1) * 100)
                cones.append(abs(r["cone_angle"] - 2 * half))
            check("tips", f"R {radius:.0f} px, half-angle {half:.0f}°: radius error (worst of 3)", max(errs), 5.0, "%")
            check("tips", f"R {radius:.0f} px, half-angle {half:.0f}°: cone angle error", max(cones), 0.5, "°")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for f in (lines, holes, particles, cross_sections, tips):
        f()
    out = ["| 项目 | 检查 | 结果 | 要求 | |", "| :-- | :-- | --: | --: | :-- |"]
    for r in ROWS:
        lim = f"≤ {r['limit']:g} {r['unit']}" if r["limit"] is not None else "（记录）"
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
