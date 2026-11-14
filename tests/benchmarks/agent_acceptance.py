"""Phase 5 acceptance: how often a planned recipe passes QC without being changed by hand,
and whether planning works fully offline.

    uv run python tests/benchmarks/agent_acceptance.py [--model rules|claude|local|auto] [--out table.md]

For each dataset (synthetic variants with known truth, plus the two real recordings when
present) the planner drafts a recipe from the data and a goal, tries it on the first
frames, and the accepted recipe then runs on all frames. A case passes when the full run
raises no QC warning and, where truth exists, matches it (droplet count, cells, positives).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from vexel.agent import models
from vexel.agent.memory import RecipeMemory
from vexel.agent.planner import make_plan
from vexel.core.volume import SpatialVolume
from vexel.engine import run_recipe
from vexel.packs.droplet.validate import validate_run
from vexel.readers import open_volume
from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("VEXEL_REALDATA", ROOT / "microfluidics dataset"))

# name, goal, synthetic configuration
SYNTHETIC = [
    ("basic", "粒径和生成频率", dict(seed=1)),
    ("small_fast", "", dict(seed=2, diameter_mean_um=40, speed_um_s=30000, frequency_hz=150)),
    ("large_slow", "measure droplet size", dict(seed=3, diameter_mean_um=80, speed_um_s=12000, frequency_hz=60)),
    ("noisy", "", dict(seed=4, noise_sigma=8.0)),
    ("polydisperse", "", dict(seed=5, diameter_cv=0.08)),
    ("cells", "数每个液滴里的细胞", dict(seed=6, cell_lambda=1.0)),
    ("cells_unasked", "", dict(seed=7, cell_lambda=0.8)),
    ("cells_dense", "包裹率", dict(seed=8, cell_lambda=2.0)),
    ("fluo", "ddPCR 阳性比例", dict(seed=9, fluorescence=True, fluo_lambda=0.5)),
    ("fluo_unasked", "", dict(seed=10, fluorescence=True, fluo_lambda=1.0)),
    ("fluo_cells", "荧光阳性和每滴细胞数", dict(seed=11, fluorescence=True, fluo_lambda=0.6, cell_lambda=1.0)),
    ("vertical", "", dict(seed=12)),  # transposed below: droplets move down
]


def _real() -> list[tuple[str, str, SpatialVolume]]:
    out = []
    mov = DATA / "DropletMicrofluidics.mov"
    if mov.exists():
        import imageio.v3 as iio

        frames = iio.imread(mov, plugin="pyav").mean(-1).astype(np.float32)
        out.append(("real: DropletMicrofluidics.mov", "", SpatialVolume.from_array(frames, "TYX", spacing={"y": 2.27, "x": 2.27}, space_unit="um")))
    archive = DATA / "WSCNet2-master.zip"
    if archive.exists():
        import imageio.v3 as iio

        images = {}
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                m = re.search(r"examples/example-folder/(\d+)\.jpg$", name)
                if m:
                    images[int(m.group(1))] = iio.imread(io.BytesIO(zf.read(name))).mean(-1)
        stack = np.stack([images[k] for k in sorted(images)]).astype(np.float32)
        out.append(("real: WSCNet sequence", "每个液滴里的细胞数", SpatialVolume.from_array(stack, "TYX")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["auto", "claude", "deepseek", "qwen", "gemini", "custom", "local", "rules"], default="rules")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="vexel-agent-accept-"))
    os.environ.setdefault("VEXEL_HOME", str(work / "home"))  # a clean recipe memory: no help from earlier runs
    cfg = models.load_config()
    if args.model != "auto":
        cfg["order"] = [args.model]
    chain = models.tiers(cfg)

    cases: list[tuple[str, str, SpatialVolume, Path | None]] = []
    for name, goal, conf in SYNTHETIC:
        folder = work / name
        video = write_dataset(folder, DropletSynthConfig(n_frames=400, **conf))
        volume = open_volume(video)
        if name == "vertical":
            data = np.swapaxes(np.asarray(volume.frames(0, 0)), 1, 2)
            volume = SpatialVolume.from_array(data, "TYX", spacing={"y": 2.0, "x": 2.0, "t": 0.001}, space_unit="um", time_unit="s")
            cases.append((name, goal, volume, None))  # truth positions are along x: not comparable
        else:
            cases.append((name, goal, volume, folder / "truth"))
    cases += [(n, g, v, None) for n, g, v in _real()]

    rows = []
    for name, goal, volume, truth in cases:
        plan = make_plan(volume, goal, memory=RecipeMemory(), chain=chain)
        row = {"case": name, "goal": goal or "（未写）", "recipe": plan.base, "status": plan.status, "planner": plan.planner,
               "adjusted": len(plan.attempts) > 1, "qc_ok": None, "truth_ok": None}  # fmt: skip
        if plan.status != "needs_user":
            result = run_recipe(plan.recipe(), volume, work / f"{name}-run", hash_input=False, report=False)
            row["qc_ok"] = not result.violations
            row["warnings"] = [v.message() for v in result.violations]
            if truth is not None:
                checks = validate_run(result.out_dir, truth)
                row["truth_ok"] = all(c.passed for c in checks)
                row["failed_checks"] = [c.name for c in checks if not c.passed]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    passed = [r for r in rows if r["qc_ok"] and r["truth_ok"] is not False]
    synthetic = [r for r in rows if not r["case"].startswith("real")]
    lines = [
        f"规划层级：{args.model}（实际回答：{', '.join(sorted({r['planner'] for r in rows}))}）。机器：{platform.machine()}，{platform.system()}。",
        "",
        "| 数据 | 目标 | 规划出的配方 | 规划状态 | 试运行后调整 | 全量运行无质控提示 | 与真值一致 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        yes = {True: "是", False: "否", None: "–"}
        lines.append(f"| {r['case']} | {r['goal']} | {r['recipe']} | {r['status']} | {yes[r['adjusted']]} | {yes[r['qc_ok']]} | {yes[r['truth_ok']]} |")
    lines += [
        "",
        f"**无需人工修改即通过质控（且与真值一致）：{len(passed)} / {len(rows)}**；其中合成数据 "
        f"{sum(1 for r in synthetic if r in passed)} / {len(synthetic)}。",
    ]
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
