"""Performance budgets (0.8 plan, gate 2), measured in the desktop window as a person uses it.

    uv run python tests/benchmarks/performance_eval.py [--size 8192x6144] [--out table.md]

A synthetic 16-bit SEM image with instrument metadata (8192 x 6144 = 50 MP by default, plus
its data bar) is opened in the SEM window; then every automatic measurement runs on the whole
image, and the planner once. A droplet video (2000 frames of 1024 x 512) gets a quick
preview and a full analysis in the droplet window.

Budgets: open (read, calibrate, process, show) < 3 s; each automatic measurement < 10 s;
peak memory of the process < 4 GB. Droplet times are recorded (no budget in the plan).
Each part runs in its own process so the memory peak is its own.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
OPEN_S, ANALYSE_S, MEMORY_GB = 3.0, 10.0, 4.0


def _peak_gb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20  # bytes on macOS, KiB on Linux


def _peak() -> float:
    try:
        return _peak_gb()
    except ImportError:  # Windows: no resource module
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]  # fmt: skip

        c = PMC()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**30


def _say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)  # progress, in the log as it happens


def child_sem(size: str, work: Path) -> dict[str, Any]:
    import numpy as np  # noqa: F401
    from qtpy.QtWidgets import QApplication

    from vexel.synthetic.sem import write_sem

    cols, rows = (int(v) for v in size.lower().split("x"))
    path = work / "big.tif"
    write_sem(path, "thermo", pixel_size=2e-9, rows=rows, cols=cols, seed=1)
    app = QApplication.instance() or QApplication([])
    from vexel.packs.sem.analysis import RECIPES
    from vexel.ui.sem_window import SemWindow

    w = SemWindow()
    w._sync = True
    w.show()
    t = time.perf_counter()
    ok = w.open_path(path, wait=True)
    app.processEvents()
    out: dict[str, Any] = {"open_s": time.perf_counter() - t, "opened": bool(ok), "analyses": {}}
    _say(f"open {out['open_s']:.1f} s")
    for name in RECIPES:
        t = time.perf_counter()
        an = w.run_analysis(recipe=name, region=None)
        app.processEvents()
        out["analyses"][name] = {"s": time.perf_counter() - t, "error": (an.error if an is not None else "not run")}
        _say(f"{name} {out['analyses'][name]['s']:.1f} s")
    t = time.perf_counter()
    w.auto_plan(wait=True, accept=False)
    out["plan_s"] = time.perf_counter() - t
    _say(f"plan {out['plan_s']:.1f} s")
    out["peak_gb"] = _peak()
    return out


def child_droplet(work: Path) -> dict[str, Any]:
    from qtpy.QtWidgets import QApplication

    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    path = write_dataset(work / "video", DropletSynthConfig(seed=1, n_frames=2000, width=1024, height=512, cell_lambda=1.0))
    app = QApplication.instance() or QApplication([])
    from vexel.ui.main_window import MainWindow

    w = MainWindow()
    w.warn = lambda *a: None
    w.show()
    out: dict[str, Any] = {}
    t = time.perf_counter()
    w.open_path(path)
    out["open_s"] = time.perf_counter() - t
    for name, preview in (("preview_s", True), ("full_s", False)):
        t = time.perf_counter()
        w.cells_check.setChecked(True)
        w.start(preview=preview)
        while w.is_running():
            app.processEvents()
            time.sleep(0.01)
        out[name] = time.perf_counter() - t
        _say(f"droplet {name} {out[name]:.1f} s")
    out["peak_gb"] = _peak()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="8192x6144", help="SEM image width x height")
    parser.add_argument("--part", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.part:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        work = Path(tempfile.mkdtemp(prefix="vexel-perf-"))
        res = child_sem(args.size, work) if args.part == "sem" else child_droplet(work)
        print("@@RESULT@@" + json.dumps(res))
        sys.stdout.flush()
        os._exit(0)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", VEXEL_HOME=tempfile.mkdtemp(prefix="vexel-perf-home-"))
    res = {}
    for part in ("sem", "droplet"):
        p = subprocess.run([sys.executable, str(HERE), "--part", part, "--size", args.size], stdout=subprocess.PIPE, text=True, env=env, timeout=3600)
        if "@@RESULT@@" not in p.stdout:
            print(f"{part}: no result (exit code {p.returncode})")
            return 2
        res[part] = json.loads(p.stdout.rsplit("@@RESULT@@", 1)[1])
    s, d = res["sem"], res["droplet"]
    rows = ["| 项目 | 结果 | 要求 | |", "| :-- | --: | --: | :-- |"]

    def row(what: str, v: float, lim: float | None, unit: str = "s") -> bool:
        ok = lim is None or v <= lim
        rows.append(f"| {what} | {v:.2f} {unit} | {'≤ ' + format(lim, 'g') + ' ' + unit if lim is not None else '（记录）'} | {'✓' if ok else '✗'} |")
        return ok

    oks = [row(f"SEM {args.size}：打开（读取、标定、处理、显示）", s["open_s"], OPEN_S)]
    for name, a in s["analyses"].items():
        oks.append(row(f"SEM 自动测量 {name}" + (f"（{a['error'][:40]}）" if a["error"] else ""), a["s"], ANALYSE_S))
    row("SEM 自动规划（识别、规划、试运行）", s["plan_s"], None)
    oks.append(row("SEM 进程内存峰值", s["peak_gb"], MEMORY_GB, "GB"))
    row("液滴 2000 帧 1024×512：打开", d["open_s"], None)
    row("液滴：快速预览（前 200 帧）", d["preview_s"], None)
    row("液滴：完整分析（含包裹计数）", d["full_s"], None)
    oks.append(row("液滴进程内存峰值", d["peak_gb"], MEMORY_GB, "GB"))
    text = "\n".join(rows) + f"\n\n{sum(oks)}/{len(oks)} 项达标。"
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
