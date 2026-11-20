"""SEM image recognition and planning acceptance (roadmap 0.0.5 M5, first half).

    uv run python tests/benchmarks/sem_recognition_eval.py [FOLDER ...] [--n 20] [--out table.md]

1. **Recognition**: random variants of every kind (lines in either direction, dark or
   bright, with edge peaks, slanted; hole arrays with and without rims; particle fields;
   layer stacks; line and trench cross-sections; tips pointing any way; textures and grains
   as "other"), each with its own size, noise and geometry. Reported: a confusion table.
   Required: >= 95 % recognised, and no image confidently given a wrong kind ("unknown" or
   "other" for an image that has one is a miss, not an error: the planner then asks).
2. **Planning**: the plan made for each recognised variant (rules only, offline) is run and
   its result compared with the geometry: line width (of the narrower stripes, the
   planner's convention), hole diameter, particle d50, layer thickness, sidewall angle, tip
   radius. Required: >= 95 % of plans pass their own trial with a result within tolerance.
3. **Real images** (folders): what each is recognised as; nothing is required.
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from vexel.packs.sem.measure import Scale
from vexel.packs.sem.plan import make_plan
from vexel.packs.sem.recognize import KINDS, probe
from vexel.synthetic import sem as S


def variant(kind: str, k: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """One random image of a kind and its truth (pixels)."""
    noise = float(rng.uniform(0.01, 0.1))
    if kind == "lines":
        w = rng.uniform(8, 30)
        pitch = w * rng.uniform(1.6, 3.5)
        peaks = bool(rng.random() < 0.3)
        slant = rng.uniform(-0.03, 0.03)
        img, _ = S.rough_lines(rows=int(rng.integers(256, 600)), cols=int(rng.integers(300, 800)), width=w, pitch=pitch, offset=rng.uniform(1, 10),
                               sigma=rng.uniform(0.3, 2), noise=noise, edge_peaks=peaks, slant=slant, seed=k)  # fmt: skip
        inverted = bool(rng.random() < 0.3) and not peaks  # edges are bright in SEM: no inverted edge peaks
        if inverted:
            img = 1 - img
        if rng.random() < 0.4:
            img = img.T
        across = math.cos(math.atan(slant))
        widths = {"bright": w * across, "dark": (pitch - w) * across}
        if inverted:
            widths = {"bright": widths["dark"], "dark": widths["bright"]}
        return img, {"widths": widths, "edge_peaks": peaks}
    if kind == "holes":
        p = rng.uniform(30, 70)
        d = p * rng.uniform(0.35, 0.6)
        rim = bool(rng.random() < 0.6)
        img, tr = S.hole_array(rows=int(rng.integers(300, 600)), cols=int(rng.integers(300, 700)), pitch=(p, p * rng.uniform(0.9, 1.3)), diameter=d,
                               sd=rng.uniform(0, 1.5), noise=noise, rim=rim, seed=k)  # fmt: skip
        inverted = bool(rng.random() < 0.3) and not rim  # bright rims around dark holes only
        if inverted:
            img = 1 - img
        return img, {"diameter": float(np.mean([q for *_, q in tr["holes"]])), "rim": rim}
    if kind == "particles":
        img, tr = S.particle_field(rows=int(rng.integers(300, 600)), cols=int(rng.integers(300, 700)), n=int(rng.integers(8, 70)), d_mean=rng.uniform(10, 40),
                                   d_sd=rng.uniform(1, 8), touching=rng.uniform(0, 0.3), noise=noise, seed=k)  # fmt: skip
        dark = bool(rng.random() < 0.3)
        if dark:
            img = 1 - img
        return img, {"d50": float(np.median([d for *_, d in tr["particles"]])), "feature": "dark" if dark else "bright"}
    if kind == "layers":
        n = int(rng.integers(2, 6))
        th = tuple(float(t) for t in rng.uniform(15, 70, n))
        levels = [float(rng.uniform(0.1, 0.9))]
        while len(levels) < n + 2:  # neighbouring layers differ enough to show an interface
            v = float(rng.uniform(0.1, 0.9))
            if abs(v - levels[-1]) >= 0.15:
                levels.append(v)
        img, _ = S.layer_stack(rows=int(rng.integers(250, 500)), cols=int(rng.integers(300, 700)), thicknesses=th, top=rng.uniform(20, 60),
                               levels=tuple(levels), noise=noise, seed=k)  # fmt: skip
        return img, {"thicknesses": th}
    if kind == "profile":
        ang = rng.uniform(78, 98)
        top, height = rng.uniform(30, 70), rng.uniform(80, 150)
        img, _ = S.xsection_feature(rows=int(rng.integers(int(top + height + 30), 420)), cols=int(rng.integers(250, 500)), top=top, height=height,
                                    width_top=rng.uniform(30, 90), angle=ang, feature="bright" if rng.random() < 0.5 else "dark", noise=noise, seed=k)  # fmt: skip
        return img, {"angle": ang}
    if kind == "tip":
        r = rng.uniform(6, 40)
        img, _ = S.tip(rows=400, cols=300, radius=r, half_angle=rng.uniform(6, 25), noise=noise, seed=k)
        return np.rot90(img, int(rng.integers(0, 4))), {"radius": r}
    t = int(rng.integers(0, 3))  # other: textures
    if t == 0:
        img = rng.normal(0.5, 0.1, (300, 400))
    elif t == 1:
        img = ndi.gaussian_filter(rng.normal(0, 1, (300, 400)), rng.uniform(2, 8))
        img = (img - img.min()) / np.ptp(img) + rng.normal(0, noise, img.shape)
    else:  # grains (a polycrystal)
        pts = rng.uniform(0, 1, (40, 2)) * [300, 400]
        yy, xx = np.mgrid[:300, :400]
        lab = np.argmin((yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2, axis=2)
        img = rng.uniform(0.2, 0.8, 40)[lab] + rng.normal(0, noise, (300, 400))
    return img, {}


plan_image: list[Any] = [None]
TOLERANCE = {"lines": 0.5, "holes": 0.5, "particles": 0.03, "layers": 0.75, "profile": 0.5, "tip": 0.05}
# layers: 0.75 px here (0.5 px in sem_metrology_eval.py): these random stacks have grey steps down to
# 0.15 between neighbouring layers in noise up to 0.1, and a thickness takes the error of two interfaces
# lines: against the stripe the plan measured (bright or dark: a convention the plan states);
# particles: against the same recipe with the right polarity (the recipe's own accuracy, for
# small and touching particles, is sem_metrology_eval.py's business, not the planner's)


def plan_error(kind: str, plan: Any, truth: dict[str, Any]) -> tuple[float | None, float]:
    """The planned measurement's error against the truth, and the tolerance it is held to."""
    b = plan.best
    if b is None:
        return None, 0.0
    v = b.summary
    if kind == "lines":
        tol = 1.2 if truth["edge_peaks"] else TOLERANCE["lines"]  # the edge-peak definition sits ~0.9 px inside a line
        cd = truth["widths"][plan.params.get("feature", "bright")]
        return (abs(v["cd_mean"] - cd) if v.get("cd_mean") else None), tol
    if kind == "holes":
        tol = 2.0 if truth["rim"] else TOLERANCE["holes"]  # bright rims: the edge definition decides, more so for small holes (sem_metrology_eval.py)
        return (abs(v["diameter_mean"] - truth["diameter"]) if v.get("diameter_mean") else None), tol
    if kind == "particles":
        from vexel.packs.sem.objects import particles

        ref = particles(plan_image[0], feature=truth["feature"])["summary"].get("d50_n")
        return (abs(v["d50_n"] / ref - 1) if v.get("d50_n") and ref else None), TOLERANCE["particles"]
    if kind == "layers":
        got = [v.get(f"layer{i + 1}") for i in range(len(truth["thicknesses"]))]
        if any(g is None for g in got):
            return None, TOLERANCE["layers"]
        return max(abs(g - t) for g, t in zip(got, truth["thicknesses"])), TOLERANCE["layers"]
    if kind == "profile":
        a, b2 = v.get("angle_left"), v.get("angle_right")
        return (max(abs(a - truth["angle"]), abs(b2 - truth["angle"])) if a and b2 else None), TOLERANCE["profile"]
    if kind == "tip":
        tol = TOLERANCE["tip"] if truth["radius"] >= 10 else 0.10  # under 10 px the recipe warns: may exceed 5 %
        return (abs(v["radius"] / truth["radius"] - 1) if v.get("radius") else None), tol
    return None, 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--n", type=int, default=20, help="variants per kind")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    conf: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    planned, plan_ok, failures = 0, 0, []
    for kind in KINDS:
        for k in range(args.n):
            img, truth = variant(kind, k, rng)
            fp = probe(img)
            conf[kind][fp.kind] += 1
            if kind == "other" or fp.kind != kind:
                continue
            plan = make_plan(img, scale=Scale(), models=False)
            plan_image[0] = img
            err, tol = plan_error(kind, plan, truth)
            planned += 1
            ok = plan.status in ("ready", "check") and err is not None and err <= tol
            plan_ok += ok
            if not ok:
                failures.append(f"{kind} #{k}: status {plan.status}, error {err if err is None else round(err, 3)} (tolerance {tol}); {plan.reasons[-1]}")
    total = sum(sum(c.values()) for c in conf.values())
    right = sum(conf[k][k] for k in KINDS)
    wrong = sum(n for k in KINDS for got, n in conf[k].items() if got not in (k, "unknown", "other"))
    lines = ["| 实际 \\\\ 识别为 | " + " | ".join(KINDS) + " | unknown |", "| :-- |" + " --: |" * (len(KINDS) + 1)]
    for k in KINDS:
        lines.append(f"| {k} | " + " | ".join(str(conf[k][g]) for g in [*KINDS, "unknown"]) + " |")
    lines += ["", f"recognised: {right}/{total} ({right / total:.1%}); confidently wrong: {wrong}",
              f"plans: {plan_ok}/{planned} passed their trial within tolerance ({plan_ok / max(planned, 1):.1%})"]  # fmt: skip
    lines += [f"- {f}" for f in failures]
    for folder in args.folders:
        from vexel.packs.sem import calibrate as C

        for f in sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".bmp"}):
            try:
                cal = C.calibrate(f, use_ocr=False)
                img = C.micrograph(f, cal)
            except Exception as e:  # noqa: BLE001
                lines.append(f"real {f.name}: unreadable ({e})")
                continue
            if np.ptp(img) == 0:
                continue
            fp = probe(img, raw=img, cal=cal)
            lines.append(f"real {f.name}: {fp.kind} ({fp.confidence:.2f}); suggested processing: {[s['op'] for s in fp.suggested_steps]}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    ok = right / total >= 0.95 and wrong == 0 and plan_ok / max(planned, 1) >= 0.95
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
