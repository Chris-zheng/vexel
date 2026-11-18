"""Phase 3 acceptance: online (stream) results against the full batch analysis, processing
frame rate, and frame drops at several delivery rates.

    uv run python tests/benchmarks/stream_acceptance.py [--out docs/phase3_acceptance_table.md]

Real recordings are read from ``$VEXEL_REALDATA`` (default ``<repo>/microfluidics dataset``)
and skipped when absent. Timings depend on the machine; the table records which one.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from vexel.core.volume import SpatialVolume
from vexel.engine import Recipe, run_recipe, run_stream
from vexel.sources import ReplaySource

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("VEXEL_REALDATA", ROOT / "microfluidics dataset"))


def _mov() -> SpatialVolume | None:
    path = DATA / "DropletMicrofluidics.mov"
    if not path.exists():
        return None
    import imageio.v3 as iio

    frames = iio.imread(path, plugin="pyav").mean(-1).astype(np.float32)
    return SpatialVolume.from_array(frames, "TYX", spacing={"y": 2.2727, "x": 2.2727, "t": 1 / 7}, space_unit="um", time_unit="s")


def _wscnet() -> SpatialVolume | None:
    path = DATA / "WSCNet2-master.zip"
    if not path.exists():
        return None
    import imageio.v3 as iio

    images = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            m = re.search(r"examples/example-folder/(\d+)\.jpg$", name)
            if m:
                images[int(m.group(1))] = iio.imread(io.BytesIO(zf.read(name))).mean(-1)
    frames = np.stack([images[k] for k in sorted(images)]).astype(np.float32)
    return SpatialVolume.from_array(frames, "TYX", spacing={"t": 0.01}, time_unit="s")  # interval unknown: nominal 10 ms


def _synthetic(n: int) -> SpatialVolume:
    from vexel.synthetic.droplets import DropletSynthConfig, generate

    video, _, _ = generate(DropletSynthConfig(n_frames=n, seed=21, cell_lambda=1.0))
    return SpatialVolume.from_array(video, "TYX", spacing={"y": 2.0, "x": 2.0, "t": 0.001}, space_unit="um", time_unit="s")


def _key(name: str) -> str:
    return "encapsulation.summary" if "encapsulation" in name else "droplets.summary"


def measure(label: str, volume: SpatialVolume, recipe_name: str, rates: list[float | None], work: Path) -> list[dict]:
    recipe = Recipe.load(recipe_name)
    batch = run_recipe(recipe, volume, work / f"{label}-batch", hash_input=False, report=False)
    b_table = batch.tables["droplets.droplets"].sort_values("crossing_frame")
    rows = []
    for rate in rates:
        speed = None if rate is None else rate * volume.spacing["t"]
        src = ReplaySource(volume, recipe.resolve().input.sources(), speed=speed)
        s = run_stream(recipe, src, work / f"{label}-{rate}", report=False)
        st = s.provenance["stream"]
        s_table = s.tables["droplets.droplets"].sort_values("crossing_frame")
        span = s.provenance["input"]["frames_used"]
        in_span = b_table[b_table["crossing_frame"] < span - 1]  # batch droplets inside the span the stream covered
        same = len(s_table) == len(in_span) or abs(len(s_table) - len(in_span)) <= 1
        row = {
            "data": label,
            "rate": "不限速" if rate is None else f"{rate:g}",
            "fps": st["processing_fps"],
            "dropped": st["frames_dropped"] / st["frames_received"],
            "latency": (st["latency_ms"] or {}).get("median"),
            "n_stream": len(s_table),
            "n_batch": len(in_span),
            "agree": same,
            "alarm": bool((s.alarms["kind"] == "throughput").any()) if len(s.alarms) else False,
        }
        if "n_inclusions" in s_table and len(s_table) == len(in_span):
            row["cells_same"] = float((s_table["n_inclusions"].to_numpy() == in_span["n_inclusions"].to_numpy()).mean())
        rows.append(row)
        print(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="write the results as a Markdown table here")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="vexel-accept-"))
    rows: list[dict] = []
    mov, wsc = _mov(), _wscnet()
    if mov is not None:
        rows += measure("DropletMicrofluidics.mov（101 帧，512×104）", mov, "droplet_basic", [None, 7, 30, 100], work)
    if wsc is not None:
        rows += measure("WSCNet 序列（300 帧，290×116，含细胞计数）", wsc, "droplet_encapsulation", [None, 30, 100], work)
    rows += measure("合成视频（2000 帧，768×256，含细胞计数）", _synthetic(2000), "droplet_encapsulation", [None, 20, 40, 80, 160], work)

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    lines = [
        f"机器：{chip}，{os.cpu_count()} 线程，{platform.system()} {platform.release()}。",
        "",
        "| 数据 | 送帧速率（帧/秒） | 处理速度（帧/秒） | 丢帧 | 延迟中位数 | 在线液滴数 | 全量液滴数 | 细胞数一致 | 处理能力报警 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in rows:
        lat = "–" if r["latency"] is None else f"{r['latency']:.0f} ms"
        cells = "–" if "cells_same" not in r else f"{100 * r['cells_same']:.0f}%"
        alarm = "有" if r["alarm"] else "无"
        lines.append(f"| {r['data']} | {r['rate']} | {r['fps']:.1f} | {100 * r['dropped']:.0f}% | {lat} | {r['n_stream']} | {r['n_batch']} | {cells} | {alarm} |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
