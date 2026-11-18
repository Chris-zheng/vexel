"""Phase 6 acceptance: 100 files unattended; the exported script reproduces them on its own.

    uv run python tests/benchmarks/batch_acceptance.py [--out table.md] [--work DIR]

Design: 4 conditions x 5 chips (replicates) x 5 recordings per chip = 100 synthetic
recordings with exact truth, plus one corrupt file and one missing file that must be
reported without stopping the batch. The run is interrupted after 30 files and started
again (resume). Then ``reproduce.py`` re-analyses every file in a fresh process and must
find identical result digests and statistics. The statistics must find the planted
differences (diameter, cells per droplet) and not the absent one.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from vexel.batch import Design, run_batch
from vexel.engine import Recipe
from vexel.packs.droplet.validate import validate_run
from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

CONDITIONS = {  # name: (diameter um, cells per droplet)
    "ctrl": (56.0, 0.5),
    "ctrl_repeat": (56.0, 0.5),  # same as ctrl: no difference may be found
    "bigger": (64.0, 0.5),
    "more_cells": (56.0, 1.2),
}
CHIPS, PER_CHIP = 5, 5


def make_data(root: Path) -> Path:
    rows = []
    seed = 0
    for cond, (diameter, lam) in CONDITIONS.items():
        for chip in range(CHIPS):
            chip_offset = (chip - 2) * 0.6  # chips differ a little (the reason replicates matter)
            for k in range(PER_CHIP):
                seed += 1
                folder = root / cond / f"chip{chip + 1}" / f"rec{k + 1}"
                if not (folder / "droplets.ome.tif").exists():
                    write_dataset(folder, DropletSynthConfig(n_frames=150, height=128, width=448, diameter_mean_um=diameter + chip_offset,
                                                             cell_lambda=lam, seed=seed))  # fmt: skip
                rows.append({"file": str(folder.relative_to(root) / "droplets.ome.tif"), "group": cond, "replicate": f"chip{chip + 1}"})
    (root / "corrupt.avi").write_bytes(b"this is not a video")
    rows.append({"file": "corrupt.avi", "group": "ctrl", "replicate": "chipX"})
    rows.append({"file": "deleted_by_someone.avi", "group": "ctrl", "replicate": "chipY"})
    design = root / "design.csv"
    pd.DataFrame(rows).to_csv(design, index=False)
    return design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--work", type=Path, help="keep data and results here (default: a temporary folder)")
    args = parser.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="vexel-batch-accept-"))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    design_path = make_data(work / "data")
    t_data = time.perf_counter() - t0
    recipe = Recipe.load("droplet_encapsulation")
    out = work / "batch"

    # first pass, interrupted after 30 files
    done = []
    t1 = time.perf_counter()
    first = run_batch(Design.load(design_path), recipe, out, on_file=lambda i, n, o: done.append(o.status), should_stop=lambda: len(done) >= 30)
    t_first = time.perf_counter() - t1
    interrupted = json.loads((out / "batch.json").read_text(encoding="utf-8"))["status"]
    # second pass: resume and finish
    t2 = time.perf_counter()
    result = run_batch(Design.load(design_path), recipe, out)
    t_second = time.perf_counter() - t2
    statuses = pd.Series([o.status for o in result.outcomes]).value_counts().to_dict()
    per_file = [o.duration_s for o in result.outcomes if o.status == "completed"]

    # truth per file
    truth_ok = 0
    for o in result.outcomes:
        if o.ok:
            truth = Path(o.file).parent / "truth"
            truth_ok += all(c.passed for c in validate_run(out / o.run_dir, truth))

    # reproduce in a fresh process
    t3 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(out / "reproduce.py"), "--out", str(work / "reproduced")], capture_output=True, text=True)
    t_repro = time.perf_counter() - t3
    line = next((ln for ln in proc.stdout.splitlines() if "reproduced with identical results" in ln), proc.stdout[-300:] + proc.stderr[-300:])
    stats_same = all(
        pd.read_csv(out / name).equals(pd.read_csv(work / "reproduced" / name)) for name in ("replicates.csv", "groups.csv", "comparisons.csv")
    ) if proc.returncode == 0 else False  # fmt: skip

    comp = pd.read_csv(out / "comparisons.csv")

    def p(metric: str, a: str, b: str) -> float:
        row = comp[(comp.metric == metric) & (((comp.a == a) & (comp.b == b)) | ((comp.a == b) & (comp.b == a)))]
        return float(row.p_holm.iloc[0])

    findings = [
        ("平均直径：bigger vs ctrl", p("droplets.diameter_mean", "ctrl", "bigger"), True),
        ("平均直径：ctrl vs ctrl_repeat", p("droplets.diameter_mean", "ctrl", "ctrl_repeat"), False),
        ("λ：more_cells vs ctrl", p("encapsulation.lambda", "ctrl", "more_cells"), True),
        ("λ：ctrl vs ctrl_repeat", p("encapsulation.lambda", "ctrl", "ctrl_repeat"), False),
        ("λ：bigger vs ctrl", p("encapsulation.lambda", "ctrl", "bigger"), False),
    ]
    units = pd.read_csv(out / "replicates.csv")

    lines = [
        f"机器：{platform.machine()}，{platform.system()} {platform.release()}，Python {platform.python_version()}。",
        "",
        "| 项 | 结果 |",
        "| :--- | :--- |",
        f"| 设计表 | {len(Design.load(design_path).rows)} 行：4 组 × 5 芯片 × 5 段录像 = 100 段（150 帧，448×128，每段约 30 个液滴，含细胞），另有 1 个损坏文件、1 个不存在的文件 |",
        f"| 第一次运行 | 处理 30 个文件后要求停止，状态 `{interrupted}`，用时 {t_first:.0f} 秒 |",
        f"| 再次运行（续跑） | 沿用 {statuses.get('resumed', 0)} 个、新分析 {statuses.get('completed', 0)} 个；损坏 {statuses.get('failed', 0)} 个、缺失 {statuses.get('missing', 0)} 个被记录，批量未中断；用时 {t_second:.0f} 秒 |",
        f"| 每个文件分析用时 | 中位数 {pd.Series(per_file).median():.2f} 秒（不含合成数据 {t_data:.0f} 秒） |",
        f"| 与真值一致（液滴数、粒径、细胞数，`vexel validate` 标准） | {truth_ok} / {result.n_ok} |",
        f"| 统计单位 | 重复（芯片）：{len(units)} 个，每个由 {PER_CHIP} 段录像平均 |",
        f"| `reproduce.py` 在新进程中重跑 | {line.strip()}；统计表{'完全相同' if stats_same else '不同'}；用时 {t_repro:.0f} 秒；返回码 {proc.returncode} |",
        "",
        "| 比较（Welch t，Holm 校正） | p | 应有差异 | 判断 |",
        "| :--- | ---: | :--- | :--- |",
    ]
    for name, pv, expected in findings:
        lines.append(f"| {name} | {pv:.3g} | {'是' if expected else '否'} | {'✓' if (pv < 0.05) == expected else '✗'} |")
    ok = (result.n_ok == 100 and truth_ok == 100 and proc.returncode == 0 and stats_same and all((pv < 0.05) == e for _, pv, e in findings))
    lines += ["", f"**验收：{'通过' if ok else '未通过'}**（100 个文件无人值守完成；导出脚本独立复现全部结果；统计找出了设定的差异且没有报出不存在的差异）。"]
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
