"""SEM calibration acceptance (plan §2.1, roadmap 0.0.5 M1): the data-bar route against the instrument.

    uv run --extra sem python tests/benchmarks/sem_calibration_eval.py [FOLDER ...] [--out table.md]

For every image that carries instrument metadata, calibrate twice: from the metadata, and
from the data bar alone (metadata ignored). Report the difference, whether the data bar's
crop matches the metadata's, and every image where the data bar route gave no answer.

Without folders it uses a synthetic set (one image per vendor layout). With folders (real
SEM images, e.g. ``$VEXEL_SEM_DATA``) it walks them recursively. Images stay where they are;
nothing is copied or sent anywhere.

Acceptance: |difference| < 1 % for every image where the bar was read; no image where the
data-bar route returned a pixel size that differs by more than 2 % without a warning.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from vexel.packs.sem import calibrate as semcal
from vexel.packs.sem.metadata import read_metadata

IMAGES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def files(folders: list[Path]) -> list[Path]:
    out = []
    for folder in folders:
        out += sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGES and p.is_file())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    folders = args.folders or ([Path(os.environ["VEXEL_SEM_DATA"])] if os.environ.get("VEXEL_SEM_DATA") else [])
    if not folders:
        from vexel.synthetic.sem import write_set

        tmp = Path(tempfile.mkdtemp(prefix="vexel-sem-"))
        write_set(tmp)
        folders = [tmp]
    rows, failures, silent = [], 0, 0
    for path in files(folders):
        try:
            md = read_metadata(path)
        except Exception as e:  # noqa: BLE001 - reported
            rows.append(f"| {path.name} | - | - | - | unreadable: {type(e).__name__} |")
            continue
        if md is None or md.pixel_size is None:
            continue
        bar = semcal.calibrate(path, use_metadata=False)
        meta = semcal.calibrate(path, use_ocr=False)
        if not bar.ok:
            failures += 1
            rows.append(f"| {path.name} | {md.vendor_name} | {semcal.format_length(md.pixel_size)} | - | no answer: {bar.reason} |")
            continue
        diff = bar.pixel_size / md.pixel_size - 1
        crop = "same" if bar.crop_rows == meta.crop_rows or meta.crop_source != "metadata" else f"{bar.crop_rows} vs {meta.crop_rows}"
        flag = "" if abs(diff) < 0.01 else (" **> 1 %**" + ("" if bar.warnings else " **no warning**"))
        if abs(diff) > 0.02 and not bar.warnings and bar.confidence != "low":
            silent += 1
        rows.append(f"| {path.name} | {md.vendor_name} | {semcal.format_length(md.pixel_size)} | {semcal.format_length(bar.pixel_size)} "
                    f"({bar.label}, {bar.scale_bar['length_px']:.1f} px) | {diff * 100:+.2f}%{flag}; crop {crop}; {bar.confidence} |")  # fmt: skip
    table = "\n".join(["| file | vendor | metadata | data bar | difference |", "| :-- | :-- | --: | --: | :-- |", *rows])
    summary = f"\n{len(rows)} images with metadata; data bar gave no answer for {failures}; silent errors > 2 %: {silent}"
    print(table + summary)
    if args.out:
        args.out.write_text(table + "\n" + summary + "\n", encoding="utf-8")
    worst = [r for r in rows if "**" in r]
    return 1 if worst or silent else 0


if __name__ == "__main__":
    sys.exit(main())
