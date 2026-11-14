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


def cmd_batch_init(args: argparse.Namespace) -> int:
    from vexel.batch.design import scan_folder

    out = Path(args.output) if args.output else Path(args.folder) / "design.csv"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    table = scan_folder(args.folder)
    if table.empty:
        print(f"no recordings found under {args.folder}", file=sys.stderr)
        return 2
    if out.resolve().parent != Path(args.folder).resolve():
        table["file"] = [str((Path(args.folder) / f).resolve()) for f in table["file"]]
    table.to_csv(out, index=False, encoding="utf-8-sig")  # with BOM: Excel opens it with the right encoding
    print(f"{len(table)} recordings -> {out}")
    print("fill in group (condition) and replicate (independent chip / day / preparation), then:")
    print(f"  vexel batch run {out} -r <recipe> -o <output folder>")
    return 0


def cmd_batch_run(args: argparse.Namespace) -> int:
    from vexel.batch import Design, run_batch
    from vexel.batch.design import scan_folder
    from vexel.engine import Recipe

    source = Path(args.design)
    design = Design.from_frame(scan_folder(source), base=source) if source.is_dir() else Design.load(source)
    missing = design.missing()
    if missing:
        print(f"warning: {len(missing)} file(s) not found, they will be reported as missing:", file=sys.stderr)
        for r in missing[:10]:
            print(f"  row {r.index}: {r.file}", file=sys.stderr)
    recipe = Recipe.load(args.recipe)
    out = Path(args.out) if args.out else Path("runs") / f"batch-{recipe.name}-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"batch   {len(design.rows)} files, {len(design.groups)} group(s)  recipe {recipe.name}  ->  {out}")
    result = run_batch(design, recipe, out, settings=_parse_overrides(args.set) if args.set else None, resume=not args.no_resume,
                       hash_input=not args.no_input_hash, max_frames=args.max_frames, workers=args.workers, progress=print)  # fmt: skip
    a = result.stats
    print(f"\n{result.n_ok} / {len(result.outcomes)} files completed")
    for o in result.failed:
        print(f"  ! row {o.row} {Path(o.file).name}: {o.status} {o.error}")
    if a is not None and not a.comparisons.empty:
        print("\ncomparisons (unit: " + a.unit_column + ")")
        for r in a.comparisons.itertuples():
            p = r.p_holm if isinstance(r.p_holm, float) and r.p_holm == r.p_holm else r.p
            print(f"  {r.metric:<32} {r.comparison:<24} p={_fmt(p)}  {r.effect_name}={_fmt(r.effect)}")
    for z, e in (a.notes if a is not None else []):
        print(f"  note: {e}")
    print(f"\nreport   {out / 'report.html'}\ntables   {out / 'results.csv'}, groups.csv, comparisons.csv\nreproduce  python {out / 'reproduce.py'}")
    return 0 if not result.failed else 4


def cmd_plugins(args: argparse.Namespace) -> int:
    from vexel.ops import list_ops
    from vexel.plugins import file_status, load_plugins, model_cards

    list_ops()  # loads built-in operators and plugins
    plugins, cards = load_plugins(), model_cards()
    if args.json:
        print(json.dumps({"plugins": [p.__dict__ for p in plugins],
                          "models": [c.to_json() | {"file_status": file_status(c)} for c in cards]}, indent=2, ensure_ascii=False))  # fmt: skip
        return 0
    if not plugins and not cards:
        print("no plugins installed (entry point group 'vexel.plugins'); see docs/plugin_interface.md")
        return 0
    for p in plugins:
        head = f"{p.name}  {p.distribution} {p.version}".rstrip()
        print(f"{head}  [{p.status}]" + (f"  {p.error}" if p.error else ""))
        for op in p.ops:
            print(f"  op     {op}")
        for m in p.models:
            print(f"  model  {m}")
    if cards:
        print("\nmodels")
        for c in cards:
            use = {True: "commercial use ok", False: "NON-COMMERCIAL", None: "licence to check"}[c.commercial_use]
            data = ", ".join(f"{d.name} ({d.license})" for d in c.training_data) or "training data not declared"
            print(f"  {c.ref:<32} {c.task:<18} {c.framework:<8} weights: {c.license}; {data}; {use}")
            print(f"  {'':<32} file: {file_status(c)}")
    return 1 if any(p.status == "failed" for p in plugins) else 0


def cmd_bugreport(args: argparse.Namespace) -> int:
    from vexel.bugreport import write_bundle

    out = write_bundle(args.output, args.runs, note=args.note or "", tables=args.tables, keep_paths=args.keep_paths)
    print(f"wrote {out}")
    print("contains versions, settings, recipes and summaries; no images, no API keys. Look inside before sending it.")
    return 0


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


def cmd_probe(args: argparse.Namespace) -> int:
    from vexel.agent.probe import montage, probe
    from vexel.readers import open_volume

    volume = open_volume(args.input, **_calibration(args))
    fp = probe(volume)
    if args.montage:
        print(f"montage  {montage(volume, fp, args.montage)}")
    if args.json:
        print(json.dumps(fp.to_json(), indent=2, ensure_ascii=False))
        return 0
    print(f"input     {args.input}  {fp.shape}  {fp.dtype}  reader {fp.reader}")
    print(f"calibrated {fp.calibrated}  pixel {_fmt(fp.pixel_size)} {fp.space_unit}  interval {_fmt(fp.frame_interval_s)} s")
    for c in fp.channels:
        role = " (droplets)" if c.index == fp.droplet_channel else (" (fluorescence)" if c.index == fp.fluorescence_channel else "")
        print(f"\nchannel {c.index}{role}")
        print(f"  intensity p1 {_fmt(c.p1)}  p50 {_fmt(c.p50)}  p99 {_fmt(c.p99)}  skew {_fmt(c.skewness)}  bimodality D {_fmt(c.bimodality)}")
        print(f"  noise {_fmt(c.noise)}  SNR {_fmt(c.snr)}  static {c.static_fraction:.0%}  moving {c.moving_fraction:.0%}")
        print(f"  flow (dy, dx) {c.flow_px} px/frame  jitter {_fmt(c.jitter_px)} px")
        if c.objects_per_frame:
            print(f"  objects/frame {_fmt(c.objects_per_frame)}  diameter {_fmt(c.object_diameter_px)} px  circularity {_fmt(c.object_circularity)}"
                  f"  spots/object {_fmt(c.spots_per_object)} (stability {_fmt(c.spot_stability)})")  # fmt: skip
    print(f"\nflow axis {fp.flow_axis}")
    for n in fp.notes:
        print(f"note      {n}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from vexel.agent.memory import RecipeMemory
    from vexel.agent.planner import make_plan, remember
    from vexel.engine import run_recipe
    from vexel.readers import open_volume

    from vexel.agent import models

    volume = open_volume(args.input, **_calibration(args))
    memory = None if args.no_memory else RecipeMemory()
    cfg = models.load_config()
    if args.model != "auto":
        cfg["order"] = [args.model, "rules"]
    chain = models.tiers(cfg)
    plan = make_plan(volume, args.goal, memory=memory, trial=not args.no_trial, chain=chain)
    status = {"ready": "ready: passed the trial run", "check": "check: runs, with QC warnings", "needs_user": "needs you",
              "draft": "draft (not tried)"}[plan.status]  # fmt: skip
    planner = plan.planner + (f" ({plan.model})" if plan.model and plan.model != "rules" else "")
    print(f"plan     {plan.base}  {plan.overrides or ''}\nstatus   {status}\nplanner  {planner}")
    for s in plan.skipped:
        print(f"         skipped {s}")
    print()
    for r in plan.reasons:
        print(f"  - {r}")
    for i, a in enumerate(plan.attempts, 1):
        print(f"  trial {i}: {a.change}  ->  droplets {a.metrics.get('n_droplets')}" + (f", {a.failure}" if a.failure else "")
              + (f", {len(a.violations)} QC warning(s)" if a.violations else ""))  # fmt: skip
    for h in plan.memory:
        print(f"  memory: {h['base']} {h['overrides']} distance {h['distance']:.2f}, passed {h['passed']}/{h['runs']}")
    out = Path(args.out) if args.out else Path("runs") / f"{Path(args.input).name.split('.')[0]}-plan-{datetime.now():%Y%m%d-%H%M%S}"
    recipe_path = plan.save(out)
    print(f"\nrecipe   {recipe_path}   (plan details: {out / 'plan.json'})")
    if not args.run:
        print(f"accept and run:  vexel run {args.input} -r {recipe_path}   (or re-plan with --run)")
        return 0 if plan.status != "needs_user" else 1
    if plan.status == "needs_user":
        print("not run: the plan needs you (see above)", file=sys.stderr)
        return 1
    result = run_recipe(plan.recipe(), volume, out / "run", progress=print, hash_input=not args.no_input_hash)
    if memory is not None:
        remember(memory, plan, result)
    print(f"\nrun      {out / 'run'}   ({len(result.violations)} QC warning(s))")
    return 0


def cmd_brush(args: argparse.Namespace) -> int:
    import sys as _sys

    from qtpy.QtWidgets import QApplication

    from vexel.engine import Recipe
    from vexel.ops.classify import use_in_recipe
    from vexel.ui.app import STYLE
    from vexel.ui.brush import BrushWindow

    base = Recipe.load(args.recipe)
    app = QApplication.instance() or QApplication(_sys.argv[:1])
    app.setStyleSheet(STYLE)
    window = BrushWindow(args.input, channel=args.channel)
    written: list[Path] = []

    def save(path: str, sha: str, n: int) -> None:
        recipe_path = Path(path).with_suffix(".recipe.json")
        recipe_path.write_text(use_in_recipe(base, path, sha).model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(recipe_path)
        print(f"training  {path}  ({n} painted frame(s), SHA-256 {sha[:12]}…)")
        print(f"recipe    {recipe_path}")
        print(f"run with  vexel run {args.input} -r {recipe_path}")

    window.trained.connect(save)
    window.show()
    app.exec()
    return 0 if written else 1


def cmd_review(args: argparse.Namespace) -> int:
    from vexel.ui.review import main as review_main

    return review_main([args.run])


def cmd_correct(args: argparse.Namespace) -> int:
    from vexel.engine.corrections import Corrections
    from vexel.engine.review import CORRECTIONS_FILE, rerun_with_corrections

    run = Path(args.run)
    source = Path(args.corrections) if args.corrections else run / CORRECTIONS_FILE
    if not source.exists():
        raise FileNotFoundError(f"no corrections at {source} (make them with `vexel review {run}`)")
    corrections = Corrections.load(source)
    print(f"run          {run}\ncorrections  {source}  ({len(corrections.edits)} edit(s))")
    result = rerun_with_corrections(run, corrections, args.out, progress=print)
    c = result.provenance["corrections"]
    for item in c["edits"]:
        e = item["edit"]
        mark = "ok " if item["status"] == "applied" else "!! "
        print(f"  {mark}frame {e['frame']:>5}  {e['kind']:<14} {e['target']:<20} {item['status']}")
    print(f"\napplied {c['n_applied']} of {c['n_edits']} edit(s)")
    for ref, summary in result.summary.items():
        print(f"\n{ref}")
        for key, value in summary.items():
            print(f"  {key:<24} {_fmt(value)}")
    print(f"\noutput       {result.out_dir}")
    return 0 if c["n_applied"] == c["n_edits"] else 1


def cmd_live(args: argparse.Namespace) -> int:
    from vexel.ui.live import main as live_main

    return live_main([args.input] if args.input else [])


def cmd_replay(args: argparse.Namespace) -> int:
    from vexel.engine import Recipe
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
    _stream_and_print(recipe, source, out, args)
    print(f"\noutput         {out}")
    return 0


def _stream_and_print(recipe: Any, source: Any, out: Path, args: argparse.Namespace) -> Any:
    from vexel.engine import StreamSettings, run_stream

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
    return result


def cmd_watch(args: argparse.Namespace) -> int:
    from vexel.engine import Recipe, run_recipe
    from vexel.readers import open_volume
    from vexel.sources.watch import WatchSource

    recipe = Recipe.load(args.recipe)
    if args.set:
        recipe = recipe.with_overrides(_parse_overrides(args.set))
    resolved = recipe.resolve()
    folder = Path(args.folder)
    out = Path(args.out) if args.out else Path("runs") / f"{folder.name}-{recipe.name}-watch-{datetime.now():%Y%m%d-%H%M%S}"
    source = WatchSource(
        folder, resolved.input.sources(), pixel_size=args.pixel_size, space_unit=args.unit, frame_interval_s=args.frame_interval,
        poll_s=args.poll, idle_timeout_s=args.idle_timeout, include_existing=not args.new_only, max_frames=args.max_frames,
        zarr_path=None if args.no_zarr else out / "frames.zarr",
    )  # fmt: skip
    print(f"watching {folder}  (ends after {args.idle_timeout:g} s without a new image, or when a file named STOP appears)")
    print(f"recipe   {recipe.name}")
    result = _stream_and_print(recipe, source, out, args)
    print(f"\nended by       {source.ended_by}  ({len(source.files)} image(s))")
    if source.zarr_path is not None and source.files:
        print(f"frames         {source.zarr_path}")
        if not args.no_final:
            print("\nfinal pass: batch analysis of the whole acquisition")
            final = run_recipe(recipe, open_volume(source.zarr_path), out / "final", hash_input=False, progress=None)
            online = result.summary.get("droplets.summary", {}).get("n_droplets")
            batch = final.summary.get("droplets.summary", {}).get("n_droplets")
            if online is not None:
                print(f"  droplets online {online}, final {batch}")
            print(f"  final results  {out / 'final'}")
    print(f"output         {out}")
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

    p = sub.add_parser("probe", help="fingerprint a dataset: statistics, motion, objects, channel roles")
    p.add_argument("input")
    p.add_argument("--json", action="store_true")
    p.add_argument("--montage", help="also write a 2 x 2 diagnostic picture here (PNG)")
    _add_calibration(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("plan", help="draft a recipe for a dataset and a goal, check it on a trial run")
    p.add_argument("input")
    p.add_argument("--goal", "-g", default="", help="what to measure, in words (e.g. '数每个液滴里的细胞', 'positive droplets')")
    p.add_argument("--out", "-o", help="folder for plan.json and recipe.json (default: runs/<input>-plan-<timestamp>)")
    p.add_argument("--no-trial", action="store_true", help="draft only; do not try the recipe")
    p.add_argument("--no-memory", action="store_true", help="neither use nor update recipe memory")
    p.add_argument("--model", choices=["auto", "claude", "deepseek", "qwen", "gemini", "custom", "local", "rules"], default="auto",
                   help="planning model: auto follows ~/.vexel/config.json; rules is offline; others fall back to rules")  # fmt: skip
    p.add_argument("--run", action="store_true", help="accept the plan and run it on all frames (remembered in recipe memory)")
    p.add_argument("--no-input-hash", action="store_true")
    _add_calibration(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("mcp", help="serve the operator registry and recipe engine to AI assistants (MCP, stdio)")
    p.set_defaults(func=lambda _a: __import__("vexel.agent.mcp_server", fromlist=["main"]).main())

    p = sub.add_parser("brush", help="train a pixel classifier by painting strokes; writes a recipe that uses it")
    p.add_argument("input", help="data to paint on")
    p.add_argument("--recipe", "-r", default="droplet_basic", help="recipe whose segmentation the classifier replaces")
    p.add_argument("--channel", type=int, default=0, help="channel to paint on and classify")
    p.set_defaults(func=cmd_brush)

    p = sub.add_parser("review", help="open the review and correction window for a run")
    p.add_argument("run", help="run directory")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("correct", help="re-run a run with its corrections (corrections.json) into <run>/corrected")
    p.add_argument("run", help="run directory")
    p.add_argument("--corrections", help="corrections file (default: <run>/corrections.json)")
    p.add_argument("--out", "-o", help="output directory (default: <run>/corrected)")
    p.set_defaults(func=cmd_correct)

    p = sub.add_parser("live", help="open the live dashboard (replay a recording or watch an acquisition folder)")
    p.add_argument("input", nargs="?", help="recording to replay, or folder to watch")
    p.set_defaults(func=cmd_live)

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

    p = sub.add_parser("watch", help="analyse images as an acquisition writes them into a folder")
    p.add_argument("folder", help="folder the acquisition software writes one image per frame into")
    p.add_argument("--recipe", "-r", required=True, help="recipe JSON file or built-in recipe name")
    p.add_argument("--out", "-o", help="output directory (default: runs/<folder>-<recipe>-watch-<timestamp>)")
    p.add_argument("--set", action="append", default=[], metavar="STEP.PARAM=VALUE", help="override a recipe parameter (repeatable)")
    p.add_argument("--idle-timeout", type=float, default=10.0, help="stop after this many seconds without a new image")
    p.add_argument("--poll", type=float, default=0.2, help="seconds between folder checks")
    p.add_argument("--new-only", action="store_true", help="ignore images already in the folder when watching starts")
    p.add_argument("--no-zarr", action="store_true", help="do not store the frames as OME-Zarr (implies --no-final)")
    p.add_argument("--no-final", action="store_true", help="skip the batch analysis of the whole acquisition at the end")
    p.add_argument("--queue", type=int, default=64, help="images that may wait; further images are thinned or dropped")
    p.add_argument("--update-every", type=int, default=50, help="images between rolling updates")
    p.add_argument("--window", type=int, default=500, help="images covered by the rolling statistics")
    p.add_argument("--max-frames", type=int, help="stop after N images")
    p.add_argument("--workers", type=int, help="threads for images processed together")
    p.add_argument("--overwrite", action="store_true", help="replace a previous run in --out")
    _add_calibration(p)
    p.set_defaults(func=cmd_watch)

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

    p = sub.add_parser("batch", help="analyse many recordings with one recipe; group statistics from a design table")
    bsub = p.add_subparsers(dest="batch_command", required=True)
    b = bsub.add_parser("init", help="draft a design table (CSV) for every recording under a folder")
    b.add_argument("folder")
    b.add_argument("-o", "--output", help="design table to write (default: <folder>/design.csv)")
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=cmd_batch_init)
    b = bsub.add_parser("run", help="run a recipe on every file of a design table (or every recording under a folder)")
    b.add_argument("design", help="design table (CSV) or a folder")
    b.add_argument("-r", "--recipe", required=True, help="built-in recipe name or recipe JSON (e.g. from vexel plan)")
    b.add_argument("-o", "--out", help="batch output folder; run the same command again to resume")
    b.add_argument("--set", action="append", metavar="KEY=VALUE", help="setting for every file (repeatable)")
    b.add_argument("--no-resume", action="store_true", help="re-run files that already completed")
    b.add_argument("--no-input-hash", action="store_true")
    b.add_argument("--max-frames", type=int)
    b.add_argument("--workers", type=int)
    b.set_defaults(func=cmd_batch_run)

    p = sub.add_parser("plugins", help="installed plugins and models: operators, licences, training data, weight files")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plugins)

    p = sub.add_parser("bugreport", help="write a diagnostics zip for a problem report (no images, no keys)")
    p.add_argument("runs", nargs="*", help="run folders the problem is about")
    p.add_argument("-o", "--output", default="vexel-bugreport.zip")
    p.add_argument("--note", help="what happened, in your words")
    p.add_argument("--tables", action="store_true", help="also include the per-droplet tables (CSV)")
    p.add_argument("--keep-paths", action="store_true", help="keep full file paths (default: file names only)")
    p.set_defaults(func=cmd_bugreport)

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
