"""SEM acceptance on real images (first round: Phenom gallery images and FIB-SEM stacks).

    uv run python tests/benchmarks/sem_real_eval.py --data <folder> [--out table.md]

The images are not in the repository (``tests/benchmarks/real/sem_gallery_truth.csv`` lists
them with what was read by eye from each data bar and the kind of measurement that fits).
Every image found in ``--data`` is opened, calibrated, recognised and planned (offline
rules, with the trial run); TIFF stacks calibrated by ImageJ and images too large for the
SEM window are checked too.

Requirements:
- no crash on any file;
- calibration: the pixel size within 1 % of field width ÷ image width, or flagged (a
  cross-check that disagrees) - never silently wrong; the field width is cross-checked on
  every image whose data bar shows one;
- recognition never names a wrong kind with confidence (``other`` / ``unknown`` is allowed;
  how many of the images with a kind it names is recorded);
- under 3 s per image.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

ROWS: list[dict] = []
TRUTH = Path(__file__).parent / "real" / "sem_gallery_truth.csv"
STACKS = {"Figure_S3C_FIB-SEM_HeLa_G1_20x20x20nm_xy.tif": 20e-9, "Figure_S3D_FIB-SEM_HeLa_G1_20x20x20nm_Illastik_xy.tif": 20e-9,
          "Movie_S5_FIB-SEM_HeLa_G1_4x4x3nm_xz.tif": 4e-9}  # fmt: skip
HUGE = ("6064-364a.ome.tiff", "Figure1_Multichannel.ome.tif")


def check(group: str, what: str, value: float, limit: float | None, unit: str = "", lower: bool = True) -> bool:
    ok = True if limit is None else bool(value <= limit) if lower else bool(value >= limit)
    ROWS.append({"group": group, "what": what, "value": value, "limit": limit, "unit": unit, "pass": ok, "lower": lower})
    return ok


def truth() -> list[dict[str, str]]:
    with open(TRUTH, encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def gallery(folder: Path) -> list[str]:
    from PIL import Image

    from vexel.packs.sem import calibrate as C
    from vexel.packs.sem.measure import scale_from
    from vexel.packs.sem.ocr import lengths_in
    from vexel.packs.sem.plan import make_plan

    rows = [r for r in truth() if (folder / r["file"]).is_file()]
    if not rows:
        return [f"no gallery images in {folder}"]
    notes, crashes, slow = [], 0, []
    within = flagged = silent = checked = has_fw = 0
    named = right = wrong = 0
    for r in rows:
        f = folder / r["file"]
        t0 = time.time()
        try:
            cal = C.calibrate(f)
            img = C.load_image(f)
            plan = make_plan(img, cal=cal, scale=scale_from(cal, 0.0), crop_rows=cal.crop_rows, models=False)
        except Exception as e:  # noqa: BLE001 - counted
            crashes += 1
            notes.append(f"{f.name}: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        if dt > 3:
            slow.append(f"{f.name} {dt:.1f}s")
        truth_px = lengths_in(r["field_width"])[0] / Image.open(f).width
        err = abs(cal.pixel_size / truth_px - 1) if cal.ok else float("inf")
        fw = [c for c in cal.checks if c["check"] == "field width"]
        has_fw += 1
        checked += bool(fw)
        if err <= 0.01:
            within += 1
        elif fw and not all(c["agree"] for c in fw):
            flagged += 1
            notes.append(f"{f.name}: pixel size off by {err:+.1%}, flagged by the field width")
        else:
            silent += 1
            notes.append(f"{f.name}: pixel size off by {err:+.1%}, not flagged")
        want = r["kind"].split("|")
        kind = plan.fingerprint.kind
        if kind not in ("other", "unknown"):
            if kind in want:
                right += 1
            else:
                wrong += 1
                notes.append(f"{f.name}: recognised as {kind}, truth {r['kind']}")
        named += want[0] != "other"
    n = len(rows)
    check("gallery", f"images opened, calibrated and planned without a crash ({n})", crashes, 0, "crashes")
    check("calibration", "pixel size within 1 % of field width ÷ width", within / n * 100, None, "%")
    check("calibration", "off by more than 1 % and not flagged by a cross-check", silent, 0, "images")
    check("calibration", "field width cross-checked (data bars that show it)", checked / has_fw * 100, 100, "%", lower=False)
    check("recognition", "named a wrong kind", wrong, 0, "images")
    check("recognition", f"right kind named, of the {named} images with one (recorded)", right, None, "images")
    check("speed", "images over 3 s", len(slow), 0, "images")
    return notes + slow


def stacks(folder: Path) -> list[str]:
    from vexel.packs.sem import calibrate as C

    notes = []
    for name, px in STACKS.items():
        f = folder / name
        if not f.is_file():
            continue
        cal = C.calibrate(f)
        err = abs(cal.pixel_size / px - 1) * 100 if cal.ok else float("inf")
        check("FIB-SEM stacks", f"{name[:34]}…: ImageJ pixel size error", err, 0.01, "%")
    for name in HUGE:
        f = folder / name
        if not f.is_file():
            continue
        try:
            C.load_image(f)
            ok = False
        except C.ImageTooLarge:
            ok = True
        check("large files", f"{name}: refused before reading its pixels", 0 if ok else 1, 0, "")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="folder with the images")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("VEXEL_HOME", tempfile.mkdtemp(prefix="vexel-real-"))  # no records left behind
    os.environ.setdefault("VEXEL_OFFLINE", "1")
    notes = gallery(args.data) + stacks(args.data)
    out = ["| 项目 | 检查 | 结果 | 要求 | |", "| :-- | :-- | --: | --: | :-- |"]
    for r in ROWS:
        lim = "（记录）" if r["limit"] is None else f"{'≤' if r['lower'] else '≥'} {r['limit']:g} {r['unit']}"
        out.append(f"| {r['group']} | {r['what']} | {r['value']:.3g} {r['unit']} | {lim} | {'✓' if r['pass'] else '✗'} |")
    text = "\n".join(out) + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "")
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    ok = all(r["pass"] for r in ROWS)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
