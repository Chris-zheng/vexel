"""``vexel`` command line. Every capability lives in the engine; this is a thin shell over it."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def _calibration(args: argparse.Namespace) -> dict[str, Any]:
    return {"pixel_size": args.pixel_size, "space_unit": args.unit, "frame_interval": args.frame_interval}


def _add_calibration(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("calibration (overrides file metadata)")
    g.add_argument("--pixel-size", type=float, help="pixel size, in --unit")
    g.add_argument("--unit", default=None, help="length unit of --pixel-size (default: um)")
    g.add_argument("--frame-interval", type=float, help="time between frames, seconds")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# -- commands -------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    from vexel.readers import open_volume

    info = open_volume(args.input, **_calibration(args)).describe()
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    print(f"source   {info['source']}")
    print(f"reader   {info['metadata'].get('reader')}")
    print(f"shape    {dict(zip(info['axes'], info['shape']))}  {info['dtype']}  ({info['nbytes'] / 1e6:.1f} MB)")
    sp = info["spacing"]
    print(f"spacing  y={sp['y']:g} x={sp['x']:g} z={sp['z']:g} {info['space_unit']};  t={sp['t']:g} {info['time_unit']}")
    if info["space_unit"] == "px":
        print("note     no pixel size found: results will be in pixels (pass --pixel-size to calibrate)")
    return 0


def cmd_ops(args: argparse.Namespace) -> int:
    from vexel.ops import list_ops

    ops = [o.describe() for o in list_ops()]
    if args.json:
        print(json.dumps(ops, indent=2, ensure_ascii=False))
        return 0
    for o in ops:
        ins = ", ".join(f"{k}:{v}" + ("?" if k in o["optional_inputs"] else "") for k, v in o["inputs"].items())
        outs = ", ".join(f"{k}:{v}" for k, v in o["outputs"].items())
        print(f"{o['name']}  [{o['scope']}]  ({ins}) -> ({outs})")
        print(f"    {o['doc']}")
        for name, schema in o["params"].items():
            default = schema.get("default")
            desc = schema.get("description", "")
            print(f"      {name} = {json.dumps(default)}" + (f"   # {desc}" if desc else ""))
    return 0


def cmd_recipes(args: argparse.Namespace) -> int:
    from vexel.engine import Recipe
    from vexel.packs import builtin_recipes

    for name, text in builtin_recipes().items():
        if args.show == name:
            print(text.rstrip())
            return 0
        if not args.show:
            r = Recipe.from_json(text)
            print(f"{name}: {r.description}")
    if args.show:
        print(f"no built-in recipe named {args.show!r}", file=sys.stderr)
        return 2
    return 0


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects step.param=value, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def cmd_run(args: argparse.Namespace) -> int:
    from vexel.engine import QCGateError, Recipe, run_recipe
    from vexel.readers import open_volume

    recipe = Recipe.load(args.recipe)
    if args.set:
        recipe = recipe.with_overrides(_parse_overrides(args.set))
    volume = open_volume(args.input, **_calibration(args))
    out = Path(args.out) if args.out else Path("runs") / f"{Path(args.input).name.split('.')[0]}-{recipe.name}-{datetime.now():%Y%m%d-%H%M%S}"
    sizes = volume.sizes
    print(f"input   {args.input}  T={sizes['T']} {sizes['Y']}x{sizes['X']} {volume.dtype}")
    print(f"recipe  {recipe.name}")
    try:
        result = run_recipe(
            recipe,
            volume,
            out,
            overwrite=args.overwrite,
            keep_intermediates=args.keep_intermediates,
            max_frames=args.max_frames,
            workers=args.workers,
            hash_input=not args.no_input_hash,
            progress=print,
        )
    except QCGateError as e:
        print(f"stopped by QC gate: {e}\npartial results and provenance are in {out}", file=sys.stderr)
        return 3
    for ref, summary in result.summary.items():
        print(f"\n{ref}")
        for key, value in summary.items():
            print(f"  {key:<24} {_fmt(value)}")
    if result.violations:
        print(f"\n{len(result.violations)} QC warning(s):")
        for v in result.violations:
            print(f"  ! {v.message()}")
    print(f"\nresult digest  {result.result_digest}")
    print(f"output         {out}")
    if (out / "report.html").exists():
        print(f"report         {out / 'report.html'}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from vexel.packs.droplet.validate import Tolerances, validate_run

    tol = Tolerances(
        args.count_tol,
        args.diameter_tol,
        args.cv_tol,
        args.frequency_tol,
        args.volume_tol,
        args.mismatch_tol,
        args.lambda_tol,
        args.fraction_tol,
        args.positive_mismatch_tol,
        args.positive_fraction_tol,
    )
    checks = validate_run(args.run, args.truth, tol)
    print(f"{'metric':<22}{'measured':>12}{'truth':>12}{'error':>12}{'tolerance':>14}  result")
    for c in checks:
        pct = c.kind in ("rel", "share")
        err = "-" if c.error is None else (f"{100 * c.error:.2f}%" if pct else f"{c.error:.3g}")
        tolerance = f"{100 * c.tolerance:g}%" if pct else f"{c.tolerance:g}"
        print(f"{c.name:<22}{_fmt(c.measured):>12}{_fmt(c.truth):>12}{err:>12}{tolerance:>14}  {'PASS' if c.passed else 'FAIL'}")
    ok = all(c.passed for c in checks)
    print("\nvalidation " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_synth(args: argparse.Namespace) -> int:
    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    cfg = DropletSynthConfig(
        n_frames=args.frames,
        seed=args.seed,
        diameter_mean_um=args.diameter,
        diameter_cv=args.cv / 100.0,
        frequency_hz=args.frequency,
        noise_sigma=args.noise,
        cell_lambda=args.cells,
        fluorescence=args.fluo is not None or args.cells_in == "fluorescence",
        fluo_lambda=args.fluo if args.fluo is not None else 0.5,
        cells_in=args.cells_in,
    )
    path = write_dataset(args.out, cfg)
    print(f"video  {path}")
    print(f"truth  {Path(args.out) / 'truth'}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from vexel.core.cache import write_ome_zarr
    from vexel.readers import open_volume

    volume = open_volume(args.input, **_calibration(args))
    path = write_ome_zarr(volume, args.out, levels=args.levels, overwrite=args.overwrite)
    print(f"wrote {path}")
    return 0


_FLAGGED = ("gpl", "agpl", "non-commercial", "noncommercial", "cc-by-nc", "cc by-nc")


def cmd_licenses(args: argparse.Namespace) -> int:
    """License inventory of the installed environment, with copyleft / non-commercial terms flagged."""
    rows = []
    for dist in metadata.distributions():
        meta = dist.metadata
        lic = meta.get("License-Expression") or meta.get("License") or ""
        classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License")]
        text = (lic.strip().splitlines() or [""])[0][:60] or "; ".join(classifiers) or "UNKNOWN"
        probe = (text + " " + " ".join(classifiers)).lower()
        flag = "CHECK" if any(f in probe for f in _FLAGGED) and "lgpl" not in probe else ("UNKNOWN" if text == "UNKNOWN" else "")
        rows.append((meta["Name"] or "?", dist.version, text, flag))
    rows.sort(key=lambda r: r[0].lower())
    if args.markdown:
        print("| package | version | license | flag |\n| :--- | :--- | :--- | :--- |")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        for name, version, text, flag in rows:
            print(f"{name:<28}{version:<14}{text:<62}{flag}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from vexel.report import write_report

    run = Path(args.run)
    if not (run / "summary.json").exists():
        raise FileNotFoundError(f"{run} is not a run directory (no summary.json)")
    print(f"wrote {write_report(run, args.out)}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from vexel.ui.app import main as gui_main

    return gui_main([args.input] if args.input else [])


def _speed(text: str) -> float | None:
    if text in ("max", "0"):
        return None
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("speed must be positive, or 'max'")
    return value


def cmd_replay(args: argparse.Namespace) -> int:
    from vexel.engine import Recipe, StreamSettings, run_stream
    from vexel.readers import open_volume
    from vexel.sources import ReplaySource

    recipe = Recipe.load(args.recipe)
    if args.set:
        recipe = recipe.with_overrides(_parse_overrides(args.set))
    resolved = recipe.resolve()
    volume = open_volume(args.input, **_calibration(args))
    source = ReplaySource(volume, resolved.input.sources(), resolved.input.z, speed=args.speed, max_frames=args.max_frames)
    out = Path(args.out) if args.out else Path("runs") / f"{Path(args.input).name.split('.')[0]}-{recipe.name}-replay-{datetime.now():%Y%m%d-%H%M%S}"
    if args.speed is None:
        pace = "as fast as possible"
    else:
        fps = args.speed / source.frame_interval_s
        pace = f"{args.speed:g}x real time: {fps:.4g} frames/s (recorded at {1 / source.frame_interval_s:.4g} frames/s)"
    print(f"input   {args.input}  {source.n_frames} frames, replayed {pace}")
    print(f"recipe  {recipe.name}")
    settings = StreamSettings(queue_size=args.queue, update_every=args.update_every, window=args.window)

    def show(snap: dict[str, Any]) -> None:
        parts = [f"frame {snap['frame']:>6}", f"{_fmt(snap.get('fps'))} fps", f"dropped {snap['dropped']}"]
        for key, label in (("droplets.n_droplets", "n"), ("droplets.diameter_mean", "d"), ("droplets.diameter_cv_percent", "CV%"),
                           ("droplets.frequency", "f"), ("encapsulation.lambda", "lambda"), ("positivity.positive_fraction", "pos")):  # fmt: skip
            if snap.get(key) is not None:
                parts.append(f"{label} {_fmt(snap[key])}")
        print("  " + "  ".join(parts))

    result = run_stream(recipe, source, out, overwrite=args.overwrite, settings=settings, progress=print, on_update=show, workers=args.workers)
    st = result.provenance["stream"]
    print(f"\nreceived {st['frames_received']}  processed {st['frames_processed']}  dropped {st['frames_dropped']}"
          f"  processing {_fmt(st['processing_fps'])} fps")  # fmt: skip
    if st["latency_ms"]:
        lat = st["latency_ms"]
        print(f"latency  median {lat['median']:.1f} ms  p95 {lat['p95']:.1f} ms  max {lat['max']:.1f} ms (software only, no camera transfer)")
    for ref, summary in result.summary.items():
        print(f"\n{ref}")
        for key, value in summary.items():
            print(f"  {key:<24} {_fmt(value)}")
    if len(result.alarms):
        print(f"\n{len(result.alarms)} alarm(s), see {out / 'stream' / 'alarms.csv'}")
    print(f"\noutput         {out}")
    return 0


# -- parser -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vexel", description="Vexel: recipe-driven, reproducible scientific image analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="(re)write the HTML report of a run")
    p.add_argument("run", help="run directory")
    p.add_argument("--out", help="output file (default: <run>/report.html)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("gui", help="open the desktop interface")
    p.add_argument("input", nargs="?", help="data to open on start")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("run", help="run a recipe on a recorded acquisition")
    p.add_argument("input", help="TIFF / OME-TIFF, video, image-sequence folder, .npy or OME-Zarr")
    p.add_argument("--recipe", "-r", required=True, help="recipe JSON file or built-in recipe name (see `vexel recipes`)")
    p.add_argument("--out", "-o", help="output directory (default: runs/<input>-<recipe>-<timestamp>)")
    p.add_argument("--set", action="append", default=[], metavar="STEP.PARAM=VALUE", help="override a recipe parameter (repeatable)")
    p.add_argument("--max-frames", type=int, help="only process the first N frames")
    p.add_argument("--workers", type=int, help="threads for frame-scope operators")
    p.add_argument("--keep-intermediates", action="store_true", help="keep scratch arrays (Zarr) of every step")
    p.add_argument("--no-input-hash", action="store_true", help="skip hashing the input (faster on very large files)")
    p.add_argument("--overwrite", action="store_true", help="replace a previous run in --out")
    _add_calibration(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("replay", help="replay a recording frame by frame through the stream executor")
    p.add_argument("input", help="TIFF / OME-TIFF, video, image-sequence folder, .npy or OME-Zarr")
    p.add_argument("--recipe", "-r", required=True, help="recipe JSON file or built-in recipe name")
    p.add_argument("--out", "-o", help="output directory (default: runs/<input>-<recipe>-replay-<timestamp>)")
    p.add_argument("--set", action="append", default=[], metavar="STEP.PARAM=VALUE", help="override a recipe parameter (repeatable)")
    p.add_argument("--speed", type=_speed, default=1.0, help="replay speed relative to the recording, or 'max' (default 1)")
    p.add_argument("--queue", type=int, default=64, help="frames that may wait; further frames are dropped when paced")
    p.add_argument("--update-every", type=int, default=50, help="frames between rolling updates")
    p.add_argument("--window", type=int, default=500, help="frames covered by the rolling statistics")
    p.add_argument("--max-frames", type=int, help="only replay the first N frames")
    p.add_argument("--workers", type=int, help="threads for frames processed together")
    p.add_argument("--overwrite", action="store_true", help="replace a previous run in --out")
    _add_calibration(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("info", help="show metadata without reading pixels")
    p.add_argument("input")
    p.add_argument("--json", action="store_true")
    _add_calibration(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("ops", help="list registered operators")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ops)

    p = sub.add_parser("recipes", help="list built-in recipes")
    p.add_argument("--show", metavar="NAME", help="print one recipe as JSON")
    p.set_defaults(func=cmd_recipes)

    p = sub.add_parser("validate", help="compare a droplet run against ground truth")
    p.add_argument("run", help="run directory")
    p.add_argument("--truth", required=True, help="truth directory (droplets.csv, positions.csv, config.json)")
    p.add_argument("--count-tol", type=int, default=0, help="allowed absolute error in droplet count")
    p.add_argument("--diameter-tol", type=float, default=0.02, help="allowed relative error of mean diameter")
    p.add_argument("--cv-tol", type=float, default=1.0, help="allowed absolute error of CV, percentage points")
    p.add_argument("--frequency-tol", type=float, default=0.02, help="allowed relative error of generation frequency")
    p.add_argument("--volume-tol", type=float, default=0.06, help="allowed relative error of mean droplet volume")
    p.add_argument("--mismatch-tol", type=float, default=0.05, help="allowed share of droplets with a wrong cell count")
    p.add_argument("--lambda-tol", type=float, default=0.05, help="allowed relative error of lambda (mean cells per droplet)")
    p.add_argument("--fraction-tol", type=float, default=0.03, help="allowed absolute error of empty/single/multiple shares")
    p.add_argument("--positive-mismatch-tol", type=float, default=0.02, help="allowed share of droplets called wrongly")
    p.add_argument("--positive-fraction-tol", type=float, default=0.03, help="allowed absolute error of the positive fraction")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("synth", help="generate a synthetic droplet video with exact ground truth")
    p.add_argument("out", help="output directory")
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diameter", type=float, default=60.0, help="mean diameter, um")
    p.add_argument("--cv", type=float, default=3.0, help="diameter CV, percent")
    p.add_argument("--frequency", type=float, default=100.0, help="generation frequency, Hz")
    p.add_argument("--noise", type=float, default=3.0, help="camera noise sigma, gray levels")
    p.add_argument("--cells", type=float, default=0.0, help="mean cells per droplet (Poisson); 0 = no cells")
    p.add_argument("--fluo", type=float, metavar="LAMBDA", help="add a fluorescence channel; mean target copies per droplet")
    p.add_argument(
        "--cells-in", choices=["brightfield", "fluorescence"], default="brightfield", help="channel in which cells are visible"
    )
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("convert", help="convert an input to an OME-Zarr pyramid")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    _add_calibration(p)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("licenses", help="license inventory of installed dependencies")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=cmd_licenses)
    return parser


def main(argv: list[str] | None = None) -> int:
    from vexel.engine.recipe import RecipeError

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (RecipeError, FileNotFoundError, FileExistsError, NotImplementedError, ValueError, KeyError) as e:
        message = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
