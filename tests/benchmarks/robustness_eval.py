"""Robustness acceptance (0.8 plan, gate 2): awkward files through the desktop windows.

    uv run python tests/benchmarks/robustness_eval.py [--out table.md] [--keep DIR] [--only sem|droplet] [--timeout 180]

About 100 files are made here: every SEM metadata style; unusual pixel types (signed,
float with NaN and inf, bool, 32-bit); extreme shapes (one pixel, one row, a thin strip, a
48 MP image, one over the 100 MP limit); colour, alpha, two channels, multi-page TIFF and
stacks; blank, saturated, noise, ramps and checkerboards; JPEG, BMP, 16-bit PNG, GIF;
broken files (empty, random bytes, truncated, wrong extension, text, a folder named like an
image); odd metadata (absurd pixel sizes, garbage in the instrument tag); odd names; and
droplet videos (normal, cells, fluorescence, one frame, static, blank, noise, 16-bit,
float, large frames, tiny frames, MP4, broken AVI, image-sequence folders - empty, mixed
sizes, not images).

Each file runs in its own process (a hard crash or a hang is caught, with a time limit):

- SEM: open in the SEM window; run every automatic measurement on the whole image;
  plan; add every image-processing step; add a manual length; export Excel, CSV, HTML and
  an annotated PNG.
- Droplets: open in the droplet window; quick preview, waiting for it to finish.

A case **fails** when an exception escapes to the program (a handler that raised), the
process crashes, or it hangs. Refusing a file, or a measurement reporting an error, is the
right behaviour for many of these files and is listed, not failed.
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

import numpy as np

HERE = Path(__file__).resolve()


# -- the corpus -----------------------------------------------------------------------------


def _tif(path: Path, a: np.ndarray, **kw: Any) -> Path:
    import tifffile

    tifffile.imwrite(path, a, **kw)
    return path


def make_corpus(d: Path) -> list[dict[str, str]]:
    import imageio.v3 as iio

    from vexel.synthetic import sem as S
    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cases: list[dict[str, str]] = []

    def add(kind: str, path: Path, note: str) -> None:
        cases.append({"id": f"{kind}-{len(cases) + 1:03d}", "kind": kind, "path": str(path), "note": note})

    base = (np.clip(S.particle_field(rows=300, cols=400, n=25, seed=1)[0], 0, 1))
    # metadata styles
    for style in S.STYLES:
        S.write_sem(d / f"style_{style}.tif", style, pixel_size=4e-9, rows=300, cols=400, seed=2)
        add("sem", d / f"style_{style}.tif", f"SEM metadata style {style}")
    # pixel types
    add("sem", _tif(d / "u8.tif", (base * 255).astype(np.uint8)), "8-bit")
    add("sem", _tif(d / "u16.tif", (base * 65535).astype(np.uint16)), "16-bit")
    add("sem", _tif(d / "i16.tif", ((base - 0.5) * 60000).astype(np.int16)), "signed 16-bit with negative values")
    add("sem", _tif(d / "u32.tif", (base * 4e9).astype(np.uint32)), "32-bit unsigned")
    add("sem", _tif(d / "f32.tif", base.astype(np.float32)), "float32 0..1")
    add("sem", _tif(d / "f64_large.tif", (base * 1e6).astype(np.float64)), "float64 with large values")
    nan = base.astype(np.float32).copy()
    nan[50:80, 60:120] = np.nan
    add("sem", _tif(d / "f32_nan.tif", nan), "float with a block of NaN")
    inf = base.astype(np.float32).copy()
    inf[10, 10] = np.inf
    inf[20, 20] = -np.inf
    add("sem", _tif(d / "f32_inf.tif", inf), "float with +inf and -inf")
    add("sem", _tif(d / "bool.tif", base > 0.5), "boolean image")
    # shapes
    add("sem", _tif(d / "one_pixel.tif", np.array([[128]], np.uint8)), "1 x 1 pixel")
    add("sem", _tif(d / "one_row.tif", (base[:1] * 255).astype(np.uint8)), "one row")
    add("sem", _tif(d / "one_col.tif", (base[:, :1] * 255).astype(np.uint8)), "one column")
    add("sem", _tif(d / "tiny.tif", (base[:5, :5] * 255).astype(np.uint8)), "5 x 5 pixels")
    add("sem", _tif(d / "strip.tif", (np.tile(base[:33], (1, 10)) * 255).astype(np.uint8)), "33 x 4000 strip")
    big = (np.clip(np.tile(base, (20, 20)) + rng.normal(0, 0.02, (6000, 8000)), 0, 1) * 255).astype(np.uint8)
    add("sem", _tif(d / "large_48mp.tif", big), "48 MP (6000 x 8000)")
    del big
    add("sem", _tif(d / "over_limit.tif", np.zeros((10001, 10001), np.uint8)), "100.02 MP: over the 100 MP limit")
    # channels and pages
    rgb = (np.stack([base, base * 0.8, base * 0.6], -1) * 255).astype(np.uint8)
    add("sem", _tif(d / "rgb.tif", rgb, photometric="rgb"), "RGB TIFF")
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], -1)
    iio.imwrite(d / "rgba.png", rgba)
    add("sem", d / "rgba.png", "RGBA PNG")
    add("sem", _tif(d / "two_channel.tif", (np.stack([base, 1 - base]) * 255).astype(np.uint8)), "2 channels")
    import tifffile

    with tifffile.TiffWriter(d / "multipage.tif") as tw:
        for k in range(3):
            tw.write((np.roll(base, k * 10, 1) * 255).astype(np.uint8))
    add("sem", d / "multipage.tif", "3 pages")
    add("sem", _tif(d / "stack3d.tif", (np.stack([base] * 5) * 255).astype(np.uint8), imagej=True), "5-slice stack (ImageJ)")
    # content
    add("sem", _tif(d / "zeros.tif", np.zeros((300, 400), np.uint8)), "all black")
    add("sem", _tif(d / "white.tif", np.full((300, 400), 255, np.uint8)), "all white (saturated)")
    add("sem", _tif(d / "constant.tif", np.full((300, 400), 128, np.uint8)), "constant grey")
    add("sem", _tif(d / "noise.tif", rng.integers(0, 256, (300, 400)).astype(np.uint8)), "uniform noise")
    add("sem", _tif(d / "ramp.tif", np.tile(np.linspace(0, 255, 400), (300, 1)).astype(np.uint8)), "horizontal ramp")
    cb = ((np.indices((300, 400)).sum(0) // 20) % 2 * 255).astype(np.uint8)
    add("sem", _tif(d / "checker.tif", cb), "checkerboard")
    dot = np.zeros((300, 400), np.uint8)
    dot[150, 200] = 255
    add("sem", _tif(d / "one_dot.tif", dot), "one bright pixel")
    half = (base * 255).astype(np.uint8)
    half[:, 200:] = 255
    add("sem", _tif(d / "half_saturated.tif", half), "half the image saturated")
    sp = (base * 255).astype(np.uint8)
    m = rng.random(sp.shape)
    sp[m < 0.05] = 0
    sp[m > 0.95] = 255
    add("sem", _tif(d / "salt_pepper.tif", sp), "salt and pepper noise")
    # formats
    iio.imwrite(d / "photo.jpg", (base * 255).astype(np.uint8))
    add("sem", d / "photo.jpg", "JPEG")
    iio.imwrite(d / "bitmap.bmp", (base * 255).astype(np.uint8))
    add("sem", d / "bitmap.bmp", "BMP")
    iio.imwrite(d / "png16.png", (base * 65535).astype(np.uint16))
    add("sem", d / "png16.png", "16-bit PNG")
    iio.imwrite(d / "anim.gif", (np.stack([base] * 2) * 255).astype(np.uint8))
    add("sem", d / "anim.gif", "GIF (2 frames)")
    # broken
    (d / "empty.tif").write_bytes(b"")
    add("sem", d / "empty.tif", "empty file")
    (d / "random.tif").write_bytes(rng.integers(0, 256, 5000).astype(np.uint8).tobytes())
    add("sem", d / "random.tif", "random bytes named .tif")
    full = (d / "u16.tif").read_bytes()
    (d / "truncated.tif").write_bytes(full[: len(full) // 2])
    add("sem", d / "truncated.tif", "TIFF cut in half")
    (d / "png_as_tif.tif").write_bytes((d / "rgba.png").read_bytes())
    add("sem", d / "png_as_tif.tif", "PNG content named .tif")
    (d / "tif_as_png.png").write_bytes((d / "u8.tif").read_bytes())
    add("sem", d / "tif_as_png.png", "TIFF content named .png")
    (d / "text.tif").write_text("not an image\n" * 50, encoding="utf-8")
    add("sem", d / "text.tif", "text file named .tif")
    (d / "folder.tif").mkdir(exist_ok=True)
    add("sem", d / "folder.tif", "a folder named like an image")
    (d / "noext").write_bytes((d / "u8.tif").read_bytes())
    add("sem", d / "noext", "TIFF without an extension")
    add("sem", d / "does_not_exist.tif", "a file that does not exist")
    # metadata extremes
    for ps, name in ((1e-15, "femto"), (1.0, "metre"), (5e-3, "five_mm")):
        S.write_sem(d / f"px_{name}.tif", "thermo", pixel_size=ps, rows=300, cols=400, seed=3)
        add("sem", d / f"px_{name}.tif", f"metadata pixel size {ps:g} m")
    _tif(d / "garbage_tag.tif", (base * 255).astype(np.uint8), extratags=[(34682, "s", 0, "[Scan]\r\nPixelWidth=abc\r\nHFW=-1\r\n[[[" + "x" * 5000, True)])
    add("sem", d / "garbage_tag.tif", "Thermo tag with garbage values")
    _tif(d / "zero_px_tag.tif", (base * 255).astype(np.uint8), extratags=[(34682, "s", 0, "[Scan]\r\nPixelWidth=0\r\nPixelHeight=0\r\n", True)])
    add("sem", d / "zero_px_tag.tif", "Thermo tag with pixel size 0")
    _tif(d / "neg_px_tag.tif", (base * 255).astype(np.uint8), extratags=[(34682, "s", 0, "[Scan]\r\nPixelWidth=-4e-9\r\nPixelHeight=-4e-9\r\n", True)])
    add("sem", d / "neg_px_tag.tif", "Thermo tag with a negative pixel size")
    _tif(d / "nan_px_tag.tif", (base * 255).astype(np.uint8), extratags=[(34682, "s", 0, "[Scan]\r\nPixelWidth=nan\r\nPixelHeight=inf\r\n", True)])
    add("sem", d / "nan_px_tag.tif", "Thermo tag with NaN and inf pixel sizes")
    (d / "sidecar.tif").write_bytes((d / "u8.tif").read_bytes())
    (d / "sidecar.txt").write_text("[SemImageFile]\nPixelSize=garbage\nDataSize=0x0\n" + "\x00" * 20, encoding="utf-8")
    add("sem", d / "sidecar.tif", "Hitachi-style sidecar with garbage")
    # names
    for name in ("样品 #1 (α).tif", "a" * 180 + ".tif", "with spaces and ;,'&.tif"):
        (d / name).write_bytes((d / "style_thermo.tif").read_bytes())
        add("sem", d / name, f"file name: {name[:30]}")

    # TIFF variants and edge cases of size and value
    u8 = (base * 255).astype(np.uint8)
    add("sem", _tif(d / "lzw.tif", u8, compression="lzw"), "LZW-compressed TIFF")
    add("sem", _tif(d / "deflate.tif", (base * 65535).astype(np.uint16), compression="zlib"), "Deflate-compressed 16-bit TIFF")
    add("sem", _tif(d / "jpeg_in_tiff.tif", u8, compression="jpeg"), "JPEG-compressed TIFF")
    add("sem", _tif(d / "tiled.tif", u8, tile=(64, 64)), "tiled TIFF")
    add("sem", _tif(d / "bigtiff.tif", u8, bigtiff=True), "BigTIFF")
    add("sem", _tif(d / "u16_12bit.tif", (base * 4095).astype(np.uint16)), "12-bit data in 16 bits (saturates at 4095)")
    add("sem", _tif(d / "f32_negative.tif", (base.astype(np.float32) - 0.5) * 1e-3), "float, negative and tiny values")
    add("sem", _tif(d / "all_nan.tif", np.full((300, 400), np.nan, np.float32)), "float, all NaN")
    add("sem", _tif(d / "exact16.tif", u8[:16, :16]), "16 x 16 (the smallest accepted)")
    add("sem", _tif(d / "exact15.tif", u8[:15, :400]), "15 rows (just under the limit)")
    add("sem", _tif(d / "thin16.tif", np.tile(u8[:16], (1, 20))), "16 x 8000 strip")
    add("sem", _tif(d / "rotated.tif", np.ascontiguousarray(u8.T)), "rotated 90 degrees")
    add("sem", _tif(d / "rgb16.tif", (np.stack([base] * 3, -1) * 65535).astype(np.uint16), photometric="rgb"), "16-bit RGB")
    ga = np.stack([u8, np.full_like(u8, 255)], -1)
    iio.imwrite(d / "grey_alpha.png", ga)
    add("sem", d / "grey_alpha.png", "grey + alpha PNG")
    bar_top = np.vstack([np.zeros((60, 400), np.uint8), u8])
    add("sem", _tif(d / "bar_on_top.tif", bar_top), "black band on top (data bar above)")
    add("sem", _tif(d / "readonly.tif", u8), "read-only file")
    os.chmod(d / "readonly.tif", 0o444)

    # droplet videos
    def dataset(name: str, **cfg: Any) -> Path:
        return write_dataset(d / name, DropletSynthConfig(**cfg))

    add("droplet", dataset("drop_basic", seed=1, n_frames=120), "synthetic droplets")
    add("droplet", dataset("drop_cells", seed=2, n_frames=120, cell_lambda=1.0), "droplets with cells")
    add("droplet", dataset("drop_fluo", seed=3, n_frames=120, fluorescence=True), "two channels (fluorescence)")
    vid = iio.imread(dataset("drop_src", seed=4, n_frames=60))
    vid = np.asarray(vid)
    frames = vid if vid.ndim == 3 else vid[:, 0]
    add("droplet", _tif(d / "drop_one_frame.tif", frames[:1]), "one frame")
    add("droplet", _tif(d / "drop_static.tif", np.repeat(frames[:1], 60, 0)), "60 identical frames (nothing moves)")
    add("droplet", _tif(d / "drop_black.tif", np.zeros((60, 200, 300), np.uint8)), "blank video")
    add("droplet", _tif(d / "drop_noise.tif", rng.integers(0, 256, (60, 200, 300)).astype(np.uint8)), "noise video")
    add("droplet", _tif(d / "drop_u16.tif", frames.astype(np.uint16) * 200), "16-bit video")
    add("droplet", _tif(d / "drop_f32.tif", frames.astype(np.float32) / 255), "float video")
    add("droplet", _tif(d / "drop_tiny.tif", frames[:, :8, :8]), "8 x 8 frames")
    add("droplet", _tif(d / "drop_large.tif", np.tile(frames[:5], (1, 12, 6))), "5 frames of 3072 x 4608")
    add("droplet", _tif(d / "drop_rgb.tif", np.repeat(frames[..., None], 3, -1), photometric="rgb"), "RGB video")
    try:
        iio.imwrite(d / "drop.mp4", np.repeat(frames[..., None], 3, -1), plugin="pyav", codec="libx264", fps=30)
        add("droplet", d / "drop.mp4", "MP4 (H.264)")
        mp4 = (d / "drop.mp4").read_bytes()
        (d / "drop_cut.mp4").write_bytes(mp4[: len(mp4) // 3])
        add("droplet", d / "drop_cut.mp4", "MP4 cut to a third")
    except Exception:  # noqa: BLE001 - no encoder here: those cases are left out
        pass
    (d / "broken.avi").write_bytes(rng.integers(0, 256, 20000).astype(np.uint8).tobytes())
    add("droplet", d / "broken.avi", "random bytes named .avi")
    (d / "empty.mov").write_bytes(b"")
    add("droplet", d / "empty.mov", "empty .mov")
    seq = d / "seq_ok"
    seq.mkdir(exist_ok=True)
    for i, f in enumerate(frames[:40]):
        iio.imwrite(seq / f"f{i:04d}.png", f)
    add("droplet", seq, "image sequence folder (PNG)")
    mixed = d / "seq_mixed"
    mixed.mkdir(exist_ok=True)
    for i, f in enumerate(frames[:20]):
        iio.imwrite(mixed / f"f{i:04d}.png", f if i % 2 else f[:100, :150])
    add("droplet", mixed, "image sequence with mixed frame sizes")
    (d / "seq_empty").mkdir(exist_ok=True)
    add("droplet", d / "seq_empty", "empty folder")
    junk = d / "seq_junk"
    junk.mkdir(exist_ok=True)
    for i in range(5):
        (junk / f"note{i}.txt").write_text("hello", encoding="utf-8")
    add("droplet", junk, "folder of text files")
    add("droplet", d / "style_thermo.tif", "an SEM image opened as a droplet video")
    add("droplet", dataset("drop_long_small", seed=5, n_frames=1500, width=256, height=96), "1500 small frames")
    add("droplet", dataset("drop_vertical_src", seed=6, n_frames=80), "droplets (second synthetic set)")
    add("droplet", _tif(d / "drop_lzw.tif", frames, compression="lzw"), "LZW-compressed video")
    return cases


# -- one case, in its own process -----------------------------------------------------------


def _collect() -> tuple[list[str], list[str]]:
    """Escaped exceptions (sys/threading hooks) and messages shown to the person."""
    import threading
    import traceback

    from qtpy.QtWidgets import QMessageBox

    escaped: list[str] = []
    shown: list[str] = []
    sys.excepthook = lambda t, e, tb: escaped.append("".join(traceback.format_exception(t, e, tb))[-1500:])
    threading.excepthook = lambda a: escaped.append(f"thread: {a.exc_type.__name__}: {a.exc_value}")
    for name in ("warning", "critical", "information"):
        setattr(QMessageBox, name, staticmethod(lambda *a, _n=name, **k: shown.append(f"{_n}: {a[2] if len(a) > 2 else ''}"[:300]) or QMessageBox.StandardButton.Ok))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
    return escaped, shown


def _pump(app: Any, seconds: float, done: Any = None) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if done is not None and done():
            return
        time.sleep(0.01)


def run_sem(path: str, work: Path) -> dict[str, Any]:
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    escaped, shown = _collect()
    from vexel.packs.sem.analysis import RECIPES
    from vexel.packs.sem.process import OPS
    from vexel.ui.sem_window import SemWindow

    out: dict[str, Any] = {"steps": []}
    w = SemWindow()
    w._sync = True
    w.show()

    def step(name: str, fn: Any) -> Any:
        try:
            r = fn()
            out["steps"].append(name)
            return r
        except Exception as e:  # noqa: BLE001 - an exception from a window method is an escape
            import traceback

            escaped.append(f"{name}: " + "".join(traceback.format_exception(e))[-1500:])
            return None

    opened = step("open", lambda: w.open_path(path, wait=True))
    _pump(app, 0.2)
    out["opened"] = bool(opened)
    if opened:
        errors = []
        for name in RECIPES:
            an = step(f"analyse {name}", lambda name=name: w.run_analysis(recipe=name, region=None))
            if an is not None and an.error:
                errors.append(name)
        out["analysis_errors"] = errors
        step("plan", lambda: w.auto_plan(wait=True, accept=False))
        for op in OPS:
            step(f"process {op}", lambda op=op: w.add_step(op))
        _pump(app, 0.2)
        h, wd = (w.image.shape[:2] if w.image is not None else (2, 2))
        step("manual length", lambda: w._shape_done("line", [[0.0, 0.0], [max(wd - 1, 1) * 0.5, max(h - 1, 1) * 0.5]]))
        for ext in ("xlsx", "csv", "html", "png"):
            step(f"export {ext}", lambda ext=ext: w.export(work / f"out.{ext}"))
    out["escaped"], out["shown"] = escaped, shown
    w.close()
    return out


def run_droplet(path: str, work: Path) -> dict[str, Any]:
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    escaped, shown = _collect()
    from vexel.ui.main_window import MainWindow

    w = MainWindow()
    w.warn = lambda title, text: shown.append(f"warn: {title}: {text}"[:300])
    w.show()
    out: dict[str, Any] = {"steps": []}
    try:
        out["opened"] = bool(w.open_path(path))
        out["steps"].append("open")
        if out["opened"]:
            out["started"] = bool(w.start(preview=True))
            _pump(app, 150, lambda: not w.is_running())
            out["finished"] = not w.is_running()
            out["steps"].append("preview")
    except Exception as e:  # noqa: BLE001
        import traceback

        escaped.append("".join(traceback.format_exception(e))[-1500:])
    _pump(app, 0.3)
    out["escaped"], out["shown"] = escaped, shown
    w.close()
    return out


# -- driver -----------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--kind", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--keep", type=Path, help="make the corpus in this folder and keep it")
    parser.add_argument("--only", choices=("sem", "droplet"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.case:  # child process: one case, JSON on the last line
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        work = Path(tempfile.mkdtemp(prefix="vexel-robust-"))
        res = run_sem(args.case, work) if args.kind == "sem" else run_droplet(args.case, work)
        print("\n@@RESULT@@" + json.dumps(res, ensure_ascii=False))
        sys.stdout.flush()
        os._exit(0)  # no teardown surprises: the result is out
    root = args.keep or Path(tempfile.mkdtemp(prefix="vexel-corpus-"))
    home = Path(tempfile.mkdtemp(prefix="vexel-robust-home-"))
    cases = [c for c in make_corpus(root) if not args.only or c["kind"] == args.only]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", VEXEL_HOME=str(home))
    env.pop("VEXEL_PREVIEW", None)
    rows = []
    for c in cases:
        t0 = time.monotonic()
        try:
            limit = args.timeout * (8 if "48 MP" in c["note"] else 1)  # every step on 48 MP, synchronously: minutes, not a hang
            p = subprocess.run([sys.executable, str(HERE), "--case", c["path"], "--kind", c["kind"]], capture_output=True, text=True,
                               timeout=limit, env=env)  # fmt: skip
            tail = p.stdout.rsplit("@@RESULT@@", 1)
            if p.returncode != 0 or len(tail) < 2:
                res = {"crash": f"exit code {p.returncode}: {p.stderr[-800:]}"}
            else:
                res = json.loads(tail[1])
        except subprocess.TimeoutExpired:
            res = {"crash": f"hung (no result in {limit:.0f} s)"}
        res["seconds"] = round(time.monotonic() - t0, 1)
        failed = bool(res.get("crash") or res.get("escaped") or (c["kind"] == "droplet" and res.get("started") and not res.get("finished")))
        rows.append(c | res | {"failed": failed})
        mark = "✗" if failed else "✓"
        print(f"{mark} {c['id']} {c['note']} ({res['seconds']} s)" + ("" if not failed else f"\n    {(res.get('crash') or (res.get('escaped') or ['did not finish'])[0])[:400]}"), flush=True)
    out = ["| 用例 | 类型 | 文件 | 打开 | 结果 | 用时 (s) |", "| :-- | :-- | :-- | :-- | :-- | --: |"]
    for r in rows:
        result = "✗ " + (r.get("crash") or (r.get("escaped") or ["未完成"])[0]).splitlines()[-1][:120] if r["failed"] else "✓"
        if not r["failed"] and r.get("shown"):
            result += f"（提示：{r['shown'][0][:80]}）"
        out.append(f"| {r['id']} | {r['kind']} | {r['note']} | {'是' if r.get('opened') else '否'} | {result} | {r['seconds']} |")
    n_fail = sum(r["failed"] for r in rows)
    out.append(f"\n{len(rows)} 个文件，{n_fail} 个失败（异常逃逸、崩溃或卡住）。")
    text = "\n".join(out)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
