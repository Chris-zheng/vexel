"""Real-data check of in-droplet counting: the Droplet-Bead dataset (Roboflow Universe,
hrithikxoml/droplet-bead v1, CC BY 4.0): 35 consecutive frames of one flow-focusing video,
beads and droplets labelled with boxes. The truth count of a droplet is the number of bead
boxes whose centre lies inside its box.

    uv run python tests/benchmarks/bead_counting_eval.py [--out table.md]

Looks for ``Droplet-Bead.v1i.yolo26.zip`` in ``$VEXEL_REALDATA`` (default: the repository's
``microfluidics dataset/``). The frames were exported stretched to 640x640 from a 16:9
letterboxed video; the stretch is undone before analysis. The tuned settings were chosen on
Roboflow's train frames only; the held-out score is on its valid and test frames (the frames
are consecutive, so droplets can appear in both: an optimistic held-out estimate).
See docs/eval_cell_counting.md.
"""

import argparse
import os
import re
import tempfile
import zipfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.transform import resize

from vexel.core.volume import SpatialVolume
from vexel.engine import Recipe, run_recipe

ROOT = Path(__file__).resolve().parents[2]
ZIP = Path(os.environ.get("VEXEL_REALDATA", ROOT / "microfluidics dataset")) / "Droplet-Bead.v1i.yolo26.zip"
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path)
args = parser.parse_args()
WORK = Path(tempfile.mkdtemp(prefix="vexel-beads-"))
with zipfile.ZipFile(ZIP) as zf:
    zf.extractall(WORK / "data")
DATA = WORK / "data"

TOP, BOTTOM = 220, 420  # content band of the letterboxed, 640x640-stretched frames
H_OUT = round((BOTTOM - TOP) * 9 / 16)  # undo the 16:9 -> square stretch
SY = H_OUT / (BOTTOM - TOP)

# -- frames and labels ------------------------------------------------------------------------
frames, truth = [], []
paths = sorted(DATA.glob("*/images/*.jpg"), key=lambda p: int(re.search(r"mp4-(\d+)", p.name).group(1)))
for t, im in enumerate(paths):
    g = iio.imread(im).astype(np.float32).mean(-1)[TOP:BOTTOM]
    frames.append(resize(g, (H_OUT, g.shape[1]), order=1, anti_aliasing=True, preserve_range=True))
    lab = im.parent.parent / "labels" / (im.stem + ".txt")
    for line in lab.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = line.split()
        x, y, w, h = (float(v) * 640 for v in (x, y, w, h))
        truth.append({"frame": t, "cls": "bead" if c == "0" else "droplet", "x": x, "y": (y - TOP) * SY, "w": w, "h": h * SY})
stack = np.stack(frames)
truth = pd.DataFrame(truth)
W = stack.shape[2]

drops = truth[truth.cls == "droplet"].copy()
beads = truth[truth.cls == "bead"].copy()
drops["x0"], drops["x1"] = drops.x - drops.w / 2, drops.x + drops.w / 2
drops["y0"], drops["y1"] = drops.y - drops.h / 2, drops.y + drops.h / 2
drops["full"] = (drops.x0 > 2) & (drops.x1 < W - 3)  # Vexel does not count droplets cut by the border
drops["n_true"] = [
    int(((beads.frame == d.frame) & (beads.x > d.x0) & (beads.x < d.x1) & (beads.y > d.y0) & (beads.y < d.y1)).sum())
    for d in drops.itertuples()
]
drops = drops.reset_index(drop=True)
split = {t: im.parent.parent.name for t, im in enumerate(paths)}
drops["split"] = drops.frame.map(split)
beads_outside = len(beads) - int(drops.n_true.sum())


def compare(pred: pd.Series, name: str) -> dict:
    m = drops[drops.full].copy()
    m["n_pred"] = pred.reindex(m.index)
    found = m.n_pred.notna()
    k = m[found]
    cls = lambda v: np.minimum(v, 3).astype(int)  # noqa: E731
    confusion = pd.crosstab(cls(k.n_true).rename("truth"), cls(k.n_pred).rename("vexel"))
    return {
        "variant": name,
        "droplets_labelled_full": int(len(m)),
        "droplets_matched": int(found.sum()),
        "exact": float((k.n_true == k.n_pred).mean()),
        "exact_held_out": float((k[k.split != "train"].n_true == k[k.split != "train"].n_pred).mean()),
        "within_1": float(((k.n_true - k.n_pred).abs() <= 1).mean()),
        "empty_correct": float(((k.n_true == 0) & (k.n_pred == 0)).sum() / max((k.n_true == 0).sum(), 1)),
        "mean_true": float(k.n_true.mean()),
        "mean_pred": float(k.n_pred.mean()),
        "over": int((k.n_pred > k.n_true).sum()),
        "under": int((k.n_pred < k.n_true).sum()),
        "confusion": confusion,
        "rows": k,
    }


# -- the pipeline on these frames ----------------------------------------------------------------
results = []
SEG = {"seg.min_diameter_px": 50, "seg.max_diameter_px": 80, "seg.motion_threshold": 0}  # sampled frames: droplets hardly move
from vexel.agent.planner import make_plan

PLAN = make_plan(SpatialVolume.from_array(stack.astype(np.float32), "TYX"), "数每个液滴里的珠子", memory=None).overrides
VARIANTS = [
    ("automatic plan (vexel plan, no hand settings): " + ", ".join(f"{k}={v}" for k, v in PLAN.items()), PLAN),
    ("automatic (default settings)", {}),
    ("droplet size given, motion gate off", SEG),
    ("size given, motion off, cell_diameter_px=8", {**SEG, "inclusions.cell_diameter_px": 8.0}),
    ("tuned on train frames: + cell 8 px, noise_factor 40, edge_margin 6, relative_floor 0.6",
     {**SEG, "inclusions.cell_diameter_px": 8.0, "inclusions.noise_factor": 40.0, "inclusions.edge_margin_px": 6, "inclusions.relative_floor": 0.6}),
]
for name, over in VARIANTS:
    out = WORK / ("run_" + re.sub(r"[^a-z0-9]+", "_", name))
    vol = SpatialVolume.from_array(stack.astype(np.float32), "TYX")
    recipe = Recipe.load("droplet_encapsulation").with_overrides(over)
    run_recipe(recipe, vol, out, overwrite=True, hash_input=False, report=False)
    meas = pd.read_csv(out / "tables" / "measure.table.csv")
    inc = pd.read_csv(out / "tables" / "inclusions.table.csv")
    det = meas.merge(inc, on=["frame", "label"])
    pred = {}
    for i, d in drops.iterrows():
        cand = det[(det.frame == d.frame) & (det.centroid_x_px > d.x0) & (det.centroid_x_px < d.x1) & (det.centroid_y_px > d.y0) & (det.centroid_y_px < d.y1)]
        if len(cand):
            c = cand.iloc[((cand.centroid_x_px - d.x) ** 2 + (cand.centroid_y_px - d.y) ** 2).argmin()]
            if np.isfinite(c.n_inclusions):
                pred[i] = c.n_inclusions
    r = compare(pd.Series(pred, dtype=float), name)
    r["detections"] = int(len(meas))
    r["out"] = out
    results.append(r)

# -- report ---------------------------------------------------------------------------------------
dist = drops[drops.full].n_true.value_counts().sort_index().to_dict()
lines = [f"{len(stack)} frames, {len(drops)} labelled droplets ({int(drops.full.sum())} fully in view; true counts {dist}), "
         f"{len(beads)} labelled beads ({beads_outside} outside droplets).", "",
         "| settings | droplets found / labelled | exact | exact (held-out frames) | within ±1 | mean true | mean counted |",
         "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in results:
    lines.append(f"| {r['variant']} | {r['droplets_matched']} / {r['droplets_labelled_full']} | {r['exact']:.0%} | {r['exact_held_out']:.0%} | "
                 f"{r['within_1']:.0%} | {r['mean_true']:.2f} | {r['mean_pred']:.2f} |")
lines += ["", "Truth (rows) against counted (columns), 3 = three or more, last settings:", "", "```", results[-1]["confusion"].to_string(), "```"]
table = "\n".join(lines)
print(table)
if args.out:
    args.out.write_text(table + "\n", encoding="utf-8")
