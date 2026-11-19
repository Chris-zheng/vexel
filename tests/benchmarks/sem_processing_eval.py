"""SEM image processing acceptance (plan §2.3, roadmap 0.0.5 M3): injected artefacts, recovered.

    uv run python tests/benchmarks/sem_processing_eval.py [FOLDER ...] [--out table.md]

A clean picture gets one artefact at a time (scan-line stripes, uneven brightness, noise,
jumping scan lines, clipped bright spots); the matching processing step is applied, and
the error against the clean picture is compared before and after (RMS of the difference,
mean removed, a border of 8 pixels left out). "Ratio" is after / before: lower is better.

The clean pictures are synthetic (particles; a line/space pattern): the acceptance.

With folders, real SEM micrographs (data bar cropped off) get the same artefacts. A real
picture is not clean - a step also corrects its own noise and stripes - so it is not
compared with the original. Reported instead, per step: **removed**, the artefact left
after processing (processed damaged picture against the processed undamaged one, as a
fraction of the artefact); and **change**, how much the step alters the undamaged
picture (RMS change / the picture's standard deviation). A step that removes artefacts
but also changes a picture a lot (line levelling on horizontal structure, say) shows
up there. Images stay where they are; nothing is copied or sent anywhere.

Also checked: every clipped pixel is marked as saturated; frames with known sub-pixel
drift are realigned and averaged; a line width measured through stripes, uneven
brightness and noise is back within 0.5 pixel of the truth after processing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vexel.packs.sem import process as P
from vexel.synthetic import sem as S

BORDER = 8

# (name, artefact, steps, required ratio, applies to)
CASES = [
    ("stripes, constant along the line", "stripes", [P.step("line_level")], 0.10, "all"),
    ("stripes, fading along the line", "stripes_vary", [P.step("destripe")], 0.30, "lines"),
    ("stripes, fading along the line", "stripes_vary", [P.step("destripe")], 0.50, "particles"),
    ("stripes, fading along the line", "stripes_vary", [P.step("line_level"), P.step("destripe")], 0.45, "all"),
    ("uneven brightness", "gradient", [P.step("background")], 0.15, "all"),
    ("uneven brightness", "gradient", [P.step("background", method="gaussian")], 0.35, "all"),
    ("uneven brightness", "gradient", [P.step("background", method="rolling_ball", size=60)], 0.35, "particles"),
    ("noise", "noise", [P.step("denoise")], 0.40, "all"),
    ("noise", "noise", [P.step("denoise", method="bilateral")], 0.50, "all"),
    ("noise", "noise", [P.step("denoise", method="median")], 0.60, "all"),
    ("jumping scan lines", "jitter", [P.step("align_rows")], 0.40, "lines"),
    ("jumping scan lines", "jitter", [P.step("align_rows")], 1.00, "particles"),  # few horizontal features: must not get worse
    ("all of the above", "all", [P.step("align_rows"), P.step("line_level"), P.step("background"), P.step("denoise")], 0.40, "all"),
]


def rms(a: np.ndarray, b: np.ndarray) -> float:
    c = slice(BORDER, -BORDER)
    d = a[c, c] - b[c, c]
    return float(np.sqrt(((d - d.mean()) ** 2).mean()))


def damage(clean: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    if kind == "stripes":
        return S.add_stripes(clean, seed=seed)
    if kind == "stripes_vary":
        return S.add_stripes(clean, vary=0.9, seed=seed)
    if kind == "gradient":
        return S.add_gradient(clean)
    if kind == "noise":
        return S.add_noise(clean, 0.05, seed=seed)
    if kind == "jitter":
        return S.add_row_jitter(clean, seed=seed)[0]
    if kind == "all":
        j = S.add_row_jitter(clean, seed=seed)[0]
        return S.add_noise(S.add_gradient(S.add_stripes(j, 0.04, seed=seed)), 0.03, seed=seed)
    raise ValueError(kind)


def synthetic() -> list[tuple[str, str, np.ndarray]]:
    return [("synthetic particles", "particles", S.micrograph(512, 768, seed=1, noise=0)),
            ("synthetic lines/spaces", "lines", S.line_space(512, 768, noise=0))]  # fmt: skip


def real(folders: list[Path]) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = []
    if folders:
        from vexel.packs.sem import calibrate as C

        for folder in folders:
            for f in sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".bmp"}):
                try:
                    cal = C.calibrate(f, use_ocr=False)
                    img = np.asarray(C.micrograph(f, cal), np.float64)
                except Exception as e:  # noqa: BLE001 - report and go on
                    print(f"skipped {f.name}: {e}", file=sys.stderr)
                    continue
                lo, hi = np.percentile(img, (0.5, 99.5))
                if min(img.shape) < 4 * BORDER or hi <= lo:
                    print(f"skipped {f.name}: too small or blank", file=sys.stderr)
                    continue
                out.append((f.name, "particles", (img - lo) / max(hi - lo, 1e-12)))  # real: treated like the general case
    return out


def run_real(folders: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for name, _kind, pic in real(folders):
        key = hash(pic.tobytes())
        if key in seen:  # the same picture under another name or format
            continue
        seen.add(key)
        sd = float(pic[BORDER:-BORDER, BORDER:-BORDER].std())
        for label, art, steps, _need, applies in CASES:
            if applies not in ("all", "particles"):
                continue
            proc = P.Processing(steps=steps, saturation=P.Saturation(enabled=False))
            bad = damage(pic, art)
            base = P.apply(pic, proc).analysis
            left = rms(P.apply(bad, proc).analysis, base) / rms(bad, pic)
            rows.append({"picture": name, "artefact": label, "steps": " + ".join(s.op for s in steps), "left": left, "change": rms(base, pic) / sd})
    return rows


def run_cases() -> list[dict]:
    rows = []
    for name, kind, clean in synthetic():
        for label, art, steps, need, applies in CASES:
            if applies not in ("all", kind):
                continue
            bad = damage(clean, art)
            fixed = P.apply(bad, P.Processing(steps=steps, saturation=P.Saturation(enabled=False))).analysis
            before, after = rms(bad, clean), rms(fixed, clean)
            rows.append({"picture": name, "artefact": label, "steps": " + ".join(s.op for s in steps), "before": before, "after": after,
                         "ratio": after / before, "required": need, "pass": after / before <= need})  # fmt: skip
    return rows


def saturation_check() -> dict:
    rng = np.random.default_rng(7)
    clean = S.micrograph(512, 768, seed=1, noise=0) * 0.8 + rng.normal(0, 0.01, (512, 768))  # nothing clipped in the clean picture
    sat, truth = S.add_saturation(clean, spots=4)
    out = {}
    for name, img, t in (("float", sat, truth), ("8-bit", np.round(sat * 255).astype(np.uint8), None),
                         ("16-bit, 12-bit detector", np.round(sat * 4095).astype(np.uint16), None)):  # fmt: skip
        t = t if t is not None else img == img.max()
        mask, info = P.saturation_mask(img)
        clean_mask, _ = P.saturation_mask(np.round(clean * 4095).astype(np.uint16))
        out[name] = {"clipped": int(t.sum()), "marked": int((t & mask).sum()), "false_on_clean": int(clean_mask.sum())}
    return out


def averaging_check() -> dict:
    from scipy import ndimage as ndi

    clean = S.micrograph(400, 600, seed=4, noise=0)
    true = [(0.0, 0.0), (0.7, -1.3), (-2.4, 0.45), (1.15, 2.6)]
    frames = [S.add_noise(ndi.shift(clean, s, order=3, mode="nearest"), 0.05, seed=i) for i, s in enumerate(true)]
    mean, shifts = P.average_frames(frames)
    e = np.asarray(shifts) + np.asarray(true)  # the correction undoes the drift...
    e -= e.mean(0)  # ...up to a common offset (the frames' mean position is the reference)
    err = float(np.hypot(e[:, 0], e[:, 1]).max())
    return {"frames": len(frames), "max_shift_error_px": err, "noise_ratio": rms(mean, clean) / rms(frames[0], clean)}


def cd_check() -> dict:
    from vexel.packs.sem.measure import MeasurementSet, compute

    width, pitch = 13.3, 31.7
    clean = S.line_space(300, 800, width=width, pitch=pitch, noise=0)
    bad = S.add_noise(S.add_gradient(S.add_stripes(clean, 0.05)), 0.04)
    out = {}
    for name, img, steps in (("damaged", bad, []), ("processed", bad, [P.step("line_level"), P.step("background"), P.step("denoise")])):
        ms = MeasurementSet()
        res = P.apply(img, P.Processing(steps=steps))
        item = ms.add("profile", [(20, 150), (780, 150)], params={"width": 9})
        v = compute(item, ms.scale, res.analysis, res.mask)
        out[name] = {"width_error_px": abs(v["width_mean"] - width) if v["width_mean"] else None, "width_sd_px": v["width_sd"],
                     "pitch_error_px": abs(v["pitch"] - pitch) if v["pitch"] else None}  # fmt: skip
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = run_cases()
    lines = ["| 图 | 伪影 | 处理 | 处理前 | 处理后 | 比值 | 要求 | |", "| :-- | :-- | :-- | --: | --: | --: | --: | :-- |"]
    for r in rows:
        lines.append(f"| {r['picture']} | {r['artefact']} | {r['steps']} | {r['before']:.4f} | {r['after']:.4f} | {r['ratio']:.2f} | ≤ {r['required']:.2f} | "
                     f"{'✓' if r['pass'] else '✗'} |")  # fmt: skip
    real_rows = run_real(args.folders)
    if real_rows:
        lines += ["", "| 真实图 | 伪影 | 处理 | 伪影残留 | 对无伪影图的改变 |", "| :-- | :-- | :-- | --: | --: |"]
        lines += [f"| {r['picture']} | {r['artefact']} | {r['steps']} | {r['left']:.0%} | {r['change']:.0%} |" for r in real_rows]
    sat = saturation_check()
    avg = averaging_check()
    cd = cd_check()
    lines += ["", "saturation: " + "; ".join(f"{k}: {v['marked']}/{v['clipped']} clipped pixels marked, {v['false_on_clean']} on a clean image" for k, v in sat.items()),
              f"averaging: {avg['frames']} frames, drift error ≤ {avg['max_shift_error_px']:.3f} px, noise × {avg['noise_ratio']:.2f}",
              "line width through artefacts: " + "; ".join(f"{k}: width error {v['width_error_px']:.2f} px (sd {v['width_sd_px']:.2f}), pitch error {v['pitch_error_px']:.2f} px"
                                                           for k, v in cd.items())]  # fmt: skip
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    synthetic_ok = all(r["pass"] for r in rows if r["picture"].startswith("synthetic"))
    sat_ok = all(v["marked"] == v["clipped"] and v["false_on_clean"] == 0 for v in sat.values())
    ok = synthetic_ok and sat_ok and avg["max_shift_error_px"] < 0.1 and avg["noise_ratio"] < 0.6 and cd["processed"]["width_error_px"] < 0.5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
