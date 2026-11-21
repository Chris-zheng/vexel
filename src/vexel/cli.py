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

    return gui_main(([args.input] if args.input else []) + (["--module", args.module] if args.module else []))


def cmd_policy(args: argparse.Namespace) -> int:
    from vexel.agent import models
    from vexel.policy import default_path, load_policy

    policy = load_policy()
    print(policy.describe())
    if policy.source is None:
        print(f"(an administrator can place one at {default_path()} or point VEXEL_POLICY at a file)")
    cfg = models.load_config()
    print("planning models in effect: " + " -> ".join(cfg["order"]))
    return 0


def cmd_modules(args: argparse.Namespace) -> int:
    from vexel.modules import for_path, modules

    if args.file:
        best, scores = for_path(args.file)
        for mid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"{mid:<10} {score:.2f}")
        print(f"-> {best.id if best else 'unclear: choose in the launcher or pass --module'}")
        return 0
    for m in modules():
        print(f"{m.id:<10} {m.name}（{m.name_en}）{'' if m.ready else '  [in development]'}")
        print(f"{'':<10} {m.description}")
    return 0


def cmd_sem_calibrate(args: argparse.Namespace) -> int:
    from vexel.packs.sem import calibrate as semcal

    manual = args.pixel_size or args.bar_length
    if manual and len(args.files) != 1:
        raise ValueError("--pixel-size / --bar-length apply to one file at a time")
    results = []
    for f in args.files:
        path = Path(f)
        if manual:
            cal = semcal.manual(path, pixel_size=semcal.parse_quantity(args.pixel_size) if args.pixel_size else None,
                                bar_length=semcal.parse_quantity(args.bar_length) if args.bar_length else None, bar_px=args.bar_pixels)  # fmt: skip
        else:
            saved = None if args.fresh else semcal.load(semcal.file_sha256(path))
            cal = saved if saved is not None and saved.ok and saved.confirmed and not args.no_metadata else \
                semcal.calibrate(path, use_ocr=not args.no_ocr, use_metadata=not args.no_metadata)  # fmt: skip
            if args.confirm and cal.needs_confirmation:
                cal.confirm("person (command line)")
        if manual or (args.confirm and cal.ok):
            semcal.save(cal)
        results.append(cal)
    if args.json:
        print(json.dumps([c.to_dict() for c in results], ensure_ascii=False, indent=2))
    else:
        for c in results:
            crop = f"; analysis area rows 0-{c.crop_rows - 1} (data bar {c.crop_source})" if c.crop_rows else ""
            print(f"{c.file}: {c.describe()}{crop}")
            if c.scale_bar:
                print(f"  scale bar {c.scale_bar['length_px']:.1f} px" + (f" = {c.label}" if c.label else ""))
            for chk in c.checks:
                print(f"  check ({chk['check']}): {semcal.format_length(chk['pixel_size'])}/px, {chk['difference'] * 100:+.2f}%" + ("" if chk["agree"] else "  DISAGREES"))
            for w in c.warnings:
                print(f"  ! {w}")
            if c.needs_confirmation:
                print("  needs confirmation: check it, then run again with --confirm (or confirm in the SEM window)")
    return 0 if all(c.ok for c in results) else 1


def _xy(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y in pixels, got {text!r}") from None
    return x, y


def _sem_measurements(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import measure_io

    cal = semcal.for_file(args.image, use_ocr=not getattr(args, "no_ocr", False))
    if getattr(args, "record", None):
        from vexel.packs.sem.measure import MeasurementSet

        ms = MeasurementSet.load(args.record)
        if ms.image.get("sha256") and ms.image["sha256"] != cal.sha256:
            print(f"warning: the record was made on {ms.image.get('file')}, not on this image", file=sys.stderr)
    else:
        ms = measure_io.load_record(cal.sha256) or measure_io.new_set(args.image, cal)
    tilt = args.tilt if getattr(args, "tilt", None) is not None else ms.scale.tilt
    from vexel.packs.sem.measure import scale_from

    ms.scale = scale_from(cal, tilt)  # values follow the calibration in force
    return cal, ms, semcal.load_image(args.image)


def cmd_sem_measure(args: argparse.Namespace) -> int:
    """Recompute and export the measurements of an image (its saved record, or --record)."""
    from vexel.packs.sem import measure_io

    cal, ms, img = _sem_measurements(args)
    if not ms.items:
        print("no measurements for this image (draw them in the SEM window, or pass --record)")
        return 1
    t = measure_io.tables(ms, img)
    if args.json:
        print(json.dumps([{k: v for k, v in r.items() if k not in ("profile", "distance", "outline")} for r in measure_io.values_of(ms, img)],
                         ensure_ascii=False, indent=2))  # fmt: skip
    else:
        import pandas as pd

        with pd.option_context("display.max_columns", 50, "display.width", 200):
            print(t["measurements"].dropna(axis=1, how="all").to_string(index=False))
        print(f"calibration: {cal.describe()}" + (f"; tilt correction {ms.scale.tilt:.1f}°" if ms.scale.tilt else ""))
    for out in args.output or []:
        p = Path(out)
        suffix = p.suffix.lower()
        if suffix == ".xlsx":
            written = [measure_io.export_xlsx(ms, img, p)]
        elif suffix == ".csv":
            written = measure_io.export_csv(ms, img, p)
        elif suffix == ".png":
            written = [measure_io.export_png(img, ms, p, crop_rows=cal.crop_rows)]
        elif suffix == ".svg":
            written = [measure_io.export_svg(img, ms, p, crop_rows=cal.crop_rows)]
        elif suffix == ".json":
            written = [ms.save(p)]
        else:
            raise ValueError(f"unknown output type {suffix!r}: use .xlsx, .csv, .png, .svg or .json")
        for w in written:
            print(f"wrote {w}")
    return 0


def cmd_sem_profile(args: argparse.Namespace) -> int:
    """One line profile from the command line: edges, widths, spaces and pitch (e.g. CD on a line/space pattern)."""
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.measure import MeasurementSet, compute

    cal, base, img = _sem_measurements(args)
    ms = MeasurementSet(image=base.image, scale=base.scale, processing=base.processing)  # on the image as the record processes it
    item = ms.add("profile", [args.start, args.end], params={"width": args.width, "method": args.method, "level": args.level, "smooth": args.smooth})
    res = measure_io.processed(ms, img)
    v = compute(item, ms.scale, res.analysis, res.mask)
    unit, f = measure_io.display_unit(ms)
    out = {k: v[k] for k in ("method", "level", "width_px", "length", "width_mean", "width_sd", "space_mean", "pitch", "pitch_edges", "peak_spacing")}
    out |= {"unit_of_lengths": "m" if ms.scale.pixel_size else "px", "edges": v["edges"], "polarity": v["polarity"], "widths": v["widths"], "spaces": v["spaces"]}
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    num = measure_io.num
    print(f"profile {args.start} -> {args.end}, {args.width} px wide; edge definition: {args.method}" + (f" {args.level:.0%}" if args.method != "gradient" else ""))
    print(f"edges: {len(v['edges'])}  (" + ", ".join(f"{num(e / f)}{'↑' if p > 0 else '↓'}" for e, p in zip(v["edges"], v["polarity"])) + f") {unit}")
    if v["widths"]:
        print(f"width (bright features): {num(v['width_mean'] / f)} ± {num(v['width_sd'] / f)} {unit}  (n={len(v['widths'])})")
    if v["spaces"]:
        print(f"space: {num(v['space_mean'] / f)} {unit}")
    if v["pitch"]:
        print(f"pitch (autocorrelation): {num(v['pitch'] / f)} {unit}" + (f"; from edges {num(v['pitch_edges'] / f)} {unit}" if v.get("pitch_edges") else ""))
    print(f"calibration: {cal.describe()}")
    return 0


def cmd_sem_process(args: argparse.Namespace) -> int:
    """Apply processing steps to an SEM image: report what changed, write the corrected image."""
    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import measure_io
    from vexel.packs.sem import process as proc

    if args.list:
        for name, op in proc.OPS.items():
            print(f"{name:11s} {op.label}（{'display only' if op.role == 'display' else 'correction'}）: {op.help}")
            for prm in op.params:
                extra = " | ".join(c[0] for c in prm.choices) if prm.choices else f"{prm.lo:g}..{prm.hi:g}" if prm.kind in ("int", "float") else "true|false"
                print(f"    {prm.name}={prm.default}  ({extra}) {prm.label}")
        return 0
    if not args.image:
        print("give an image (or --list)")
        return 2
    import numpy as np

    cal, ms, img = _sem_measurements(args)
    if args.step:
        ms.processing.steps = [proc.parse_step(t) for t in args.step]
    if args.no_saturation:
        ms.processing.saturation.enabled = False
    res = measure_io.processed(ms, img)
    crop = ms.image.get("crop_rows") or None
    micro_before = np.asarray(img[:crop] if crop else img, np.float64)
    micro_after = res.analysis[:crop] if crop else res.analysis
    report = {"image": str(args.image), "steps": proc.describe(ms.processing), "saturation": res.saturation,
              "noise_before": proc.noise_sigma(micro_before), "noise_after": proc.noise_sigma(micro_after), "notes": res.notes}  # fmt: skip
    if args.save and ms.image.get("sha256") == cal.sha256:
        measure_io.autosave(ms)
        report["record"] = str(measure_io.record_path(cal.sha256))
    if args.output:
        report["output"] = str(proc.write_derived(args.output, res.analysis, cal, {"operation": "图像处理", "source": Path(args.image).name,
                                                                                   "steps": report["steps"]}))  # fmt: skip
    if args.picture:
        from PIL import Image

        Image.fromarray(res.display).save(args.picture)
        report["picture"] = str(args.picture)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=float))
        return 0
    print(f"{Path(args.image).name}: calibration {cal.describe()}")
    for line in report["steps"] or ["no processing steps"]:
        print(f"  {line}")
    sat = res.saturation
    if sat:
        print(f"saturated pixels: {sat.get('pixels', 0)} ({sat.get('fraction', 0):.3%}) in {sat.get('regions', 0)} areas" if sat.get("pixels") else "no saturated areas")
    print(f"pixel noise (sd): {report['noise_before']:.4g} -> {report['noise_after']:.4g}")
    for n in res.notes:
        print(f"note: {n}")
    for key in ("record", "output", "picture"):
        if key in report:
            print(f"wrote {report[key]}")
    return 0


def cmd_sem_average(args: argparse.Namespace) -> int:
    """Align and average frames of the same place into one image (calibration of the first)."""
    from vexel.packs.sem import process as proc

    info = proc.average_files(args.frames, args.output, align=not args.no_align, use_ocr=not args.no_ocr)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2, default=float))
        return 0
    print(f"averaged {info['frames']} frames into {info['output']}")
    for f, (dy, dx) in zip(args.frames, info["shifts"]):
        print(f"  {Path(f).name}: moved by dx {dx:+.2f}, dy {dy:+.2f} px")
    print(f"pixel noise (sd): {info['noise_single']:.4g} (first frame) -> {info['noise_mean']:.4g}")
    print(f"calibration: {info['calibration'] or 'none (the first frame is not calibrated)'}")
    return 0


def _region(text: str) -> list[float]:
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected X0,Y0,X1,Y1 (pixels)") from e
    if len(v) != 4:
        raise argparse.ArgumentTypeError("expected X0,Y0,X1,Y1 (pixels)")
    return v


def cmd_sem_analyze(args: argparse.Namespace) -> int:
    """Automatic measurements (CD/LER, holes, particles, layers, profiles, tips) on one or many images.

    Several images: one row per image and the spread across images (mean, SD, 3σ) per
    folder - repeated images of the same place give the measurement's repeatability."""
    import pandas as pd

    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.analysis import RECIPES
    from vexel.packs.sem.measure import scale_from
    from vexel.packs.sem.process import parse_step

    if args.list:
        for name, rec in RECIPES.items():
            print(f"{name:10s} {rec.label}: {rec.help}")
            for prm in rec.params:
                extra = " | ".join(c[0] for c in prm.choices) if prm.choices else f"{prm.lo:g}..{prm.hi:g}" if prm.kind in ("int", "float") else "true|false"
                print(f"    {prm.name}={prm.default}  ({extra}) {prm.label}")
        return 0
    if not args.images:
        print("give images (or --list)")
        return 2
    params: dict[str, Any] = {}
    for kv in args.param or []:
        k, eq, v = kv.partition("=")
        if not eq:
            raise ValueError(f"expected key=value, got {kv!r}")
        params[k.strip()] = v.strip()
    if args.recipe:
        typed = RECIPES[args.recipe].resolve(params)
        params = {k: typed[k] for k in params}
    rows, objects, reports, pool_in = [], [], [], []
    for f in args.images:
        path = Path(f)
        cal = semcal.for_file(path, use_ocr=not args.no_ocr)
        ms = measure_io.load_record(cal.sha256) or measure_io.new_set(path, cal)
        ms.scale = scale_from(cal, args.tilt if args.tilt is not None else ms.scale.tilt)
        if args.step:
            ms.processing.steps = [parse_step(t) for t in args.step]
        if args.recipe:
            ms.analyses = [a for a in ms.analyses if not args.replace] if args.save else []
            ms.add_analysis(args.recipe, args.region, **params)
        if not ms.analyses and args.auto:
            from vexel.packs.sem.plan import make_plan, memory_store, remember

            img0 = semcal.load_image(path)
            plan = make_plan(img0, cal=cal, goal=args.goal or "", scale=ms.scale, crop_rows=cal.crop_rows, base_steps=ms.processing.steps,
                             memory=memory_store(), chain=_sem_chain(args))  # fmt: skip
            if plan.recipe and plan.status != "needs_user":
                plan.apply(ms)
                if args.save:
                    remember(memory_store(), plan, True, str(path))
                print(f"{path.name}: planned {plan.recipe} ({plan.status})", file=sys.stderr)
            else:
                print(f"{path.name}: could not plan a measurement - {plan.reasons[-1] if plan.reasons else ''}", file=sys.stderr)
        if not ms.analyses:
            print(f"{path.name}: no automatic measurements (give --recipe, or --auto)")
            continue
        img = semcal.load_image(path)
        unit, fct = measure_io.display_unit(ms)
        ans = measure_io.analyses_of(ms, img)
        if args.save:
            measure_io.autosave(ms)
        for an in ans:
            entry = {"file": path.name, "group": path.parent.name, "analysis": an.spec.id, "recipe": an.spec.recipe,
                     "calibration": cal.describe(), "error": an.error, "warnings": an.warnings}  # fmt: skip
            for key, label, v, kind in an.summary:
                if kind == "length" and v is not None and cal.ok:
                    v = v / fct
                entry[key] = v
            rows.append(entry)
            if not an.objects.empty:
                df = an.objects.copy()
                df.insert(0, "file", path.name)
                df.insert(1, "analysis", an.spec.id)
                objects.append(df)
            reports.append((path, an, unit, fct, cal))
            pool_in.append((path.parent.name, an, cal.ok))
    if not rows:
        return 1
    table = pd.DataFrame(rows)
    # axial directions (0..180°) do not average linearly: left out of the spread
    numeric = [c for c in table.columns if c not in ("file", "group", "analysis", "recipe", "calibration", "error", "warnings", "orientation")
               and pd.api.types.is_numeric_dtype(table[c])]  # fmt: skip
    spread = []
    if len(table) > 1:
        for (grp, rec), g in table.groupby(["group", "recipe"]):
            for c in numeric:
                v = g[c].dropna()
                if len(v) >= 2:
                    spread.append({"group": grp, "recipe": rec, "quantity": c, "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1), "3sigma": 3 * v.std(ddof=1),
                                   "min": v.min(), "max": v.max()})  # fmt: skip
    spread_df = pd.DataFrame(spread)
    from vexel.packs.sem.population import pooled

    pooled_df = pooled(pool_in, *(measure_io.display_unit(ms) if rows else ("m", 1.0)))
    if args.json:
        print(json.dumps({"results": rows, "repeatability": spread, "pooled": pooled_df.to_dict("records")}, ensure_ascii=False, indent=2, default=float))
    else:
        for path, an, unit, fct, cal in reports:
            print(f"{path.name}  A{an.spec.id} {RECIPES[an.spec.recipe].label}  ({cal.describe()})")
            if an.error:
                print(f"  error: {an.error}")
            for key, label, v, kind in an.summary:
                if v is None:
                    continue
                if kind == "length":
                    text = f"{v / fct:.4g} {unit}" if cal.ok else f"{v:.4g} px"
                elif kind == "angle":
                    text = f"{v:.2f}°"
                elif kind == "ratio":
                    text = f"{v:.3f}"
                else:
                    text = str(v)
                print(f"  {label}: {text}")
            for w in an.warnings:
                print(f"  ! {w}")
        if len(spread_df):
            print("\nspread across images (repeatability):")
            with pd.option_context("display.max_rows", 200, "display.width", 200):
                print(spread_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        if len(pooled_df) and len(reports) > 1:
            print("\nall objects of all images together (pooled):")
            with pd.option_context("display.max_rows", 200, "display.width", 200):
                print(pooled_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(out, engine="openpyxl") as xw:
                table.assign(warnings=table["warnings"].map("; ".join)).to_excel(xw, sheet_name="结果", index=False)
                (spread_df if len(spread_df) else pd.DataFrame({"": []})).to_excel(xw, sheet_name="重复性", index=False)
                if len(pooled_df):
                    pooled_df.to_excel(xw, sheet_name="合并统计", index=False)
                (pd.concat(objects, ignore_index=True) if objects else pd.DataFrame({"": []})).to_excel(xw, sheet_name="对象", index=False)
        else:
            table.assign(warnings=table["warnings"].map("; ".join)).to_csv(out, index=False, encoding="utf-8-sig")
            if len(spread_df):
                spread_df.to_csv(out.with_name(out.stem + "_repeatability.csv"), index=False, encoding="utf-8-sig")
            if len(pooled_df):
                pooled_df.to_csv(out.with_name(out.stem + "_pooled.csv"), index=False, encoding="utf-8-sig")
        print(f"wrote {out}")
    return 0


def _sem_chain(args: argparse.Namespace) -> list[Any] | None:
    from vexel.agent.models import RulesTier

    return [RulesTier()] if getattr(args, "offline", False) else None


def _print_sem_plan(path: Path, plan: Any) -> None:
    from vexel.packs.sem.recognize import KIND_NAMES

    fp = plan.fingerprint
    status = {"ready": "ready (passed every check)", "check": "check (runs, see the notes)", "needs_user": "needs you"}[plan.status]
    who = "rules (offline)" if plan.planner == "rules" else plan.model or plan.planner
    print(f"{path.name}: {KIND_NAMES.get(fp.kind, fp.kind)} ({fp.confidence:.2f}) -> {plan.recipe or '-'}  {status}  [{who}]")
    if plan.params:
        print(f"  settings: {plan.params}")
    if plan.region:
        print(f"  region: {[round(v) for v in plan.region]}")
    if plan.steps:
        print(f"  processing: {', '.join(s['op'] for s in plan.steps)}")
    for r in plan.reasons:
        print(f"  - {r}")
    for a in plan.attempts:
        print(f"  trial ({a.change}): " + ("ok" if not a.problems and not a.error else "; ".join(a.problems or [a.error or ""])))
    for s in plan.skipped:
        print(f"  (model tier skipped: {s})")


def cmd_sem_plan(args: argparse.Namespace) -> int:
    """Recognise each image and plan its measurement (recipe, settings, region, processing), tried first."""
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.plan import memory_store, plan_file, remember

    memory = memory_store()
    out = []
    code = 0
    for f in args.images:
        path = Path(f)
        plan, cal, ms = plan_file(path, args.goal or "", use_ocr=not args.no_ocr, memory=memory, trial_run=not args.no_trial,
                                  chain=_sem_chain(args))  # fmt: skip
        out.append({"file": path.name, **plan.to_json()})
        if not args.json:
            _print_sem_plan(path, plan)
        if args.apply:
            if plan.recipe and plan.status != "needs_user":
                spec = plan.apply(ms)
                measure_io.autosave(ms)
                remember(memory, plan, True, str(path))
                if not args.json:
                    print(f"  applied: A{spec.id} added to the image's record")
            else:
                code = 1
                if not args.json:
                    print("  not applied: the plan needs you (choose the measurement in the SEM window, or give --goal)")
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    return code


def _floats(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected numbers separated by commas") from e


def cmd_sem_tilt3d(args: argparse.Namespace) -> int:
    """A 2.5D height map from SEM images of one place at different stage tilts."""
    import numpy as np
    from PIL import Image

    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import measure_io
    from vexel.packs.sem import tilt3d as T

    files = [Path(f) for f in args.images]
    if len(files) < 2:
        print("give the reference image and at least one tilted view")
        return 2
    if args.tilts:
        tilts = args.tilts
        if len(tilts) != len(files):
            print(f"give one tilt per image ({len(files)}), got {len(tilts)}")
            return 2
    else:
        tilts = [T.stage_tilt(f) for f in files]
        if any(t is None for t in tilts):
            print("the stage tilt is not in every file's metadata: give --tilts (degrees, one per image)")
            return 2
    k = 1e-9 if semcal.for_file(files[0], use_ocr=False).ok else 1.0
    spec = T.Tilt3DSpec(reference_tilt=tilts[0], axis=args.axis, level=args.level, smooth=args.smooth, min_confidence=args.min_confidence,
                        cutoff=args.cutoff * k, short_cutoff=args.short_cutoff * k,
                        views=[T.TiltView(file=str(f.resolve()), sha256=semcal.file_sha256(f), tilt=t) for f, t in zip(files[1:], tilts[1:])],
                        profiles=[[[p[0], p[1]], [p[2], p[3]]] for p in args.profile or []],
                        steps=[{"lower": s[:4], "upper": s[4:]} for s in args.step or []])  # fmt: skip
    res = T.run_spec(spec, files[0], use_ocr=not args.no_ocr)
    hm = res.hm
    unit = "nm" if hm.pixel_size else "px"
    f = 1e-9 if hm.pixel_size else 1.0
    report = {"reference": files[0].name, "tilts": tilts, "axis_deg": args.axis, "unit": "m" if hm.pixel_size else "px",
              "height_range": float(np.nanpercentile(res.z[hm.valid], 99) - np.nanpercentile(res.z[hm.valid], 1)) if hm.valid.any() else None,
              "reliable_fraction": float(hm.valid.mean()), "median_uncertainty": float(np.median(hm.uncertainty[hm.valid])) if hm.valid.any() else None,
              "roughness": res.roughness, "steps": res.steps, "pairs": hm.pairs, "warnings": res.warnings,
              "profiles": [{k: v for k, v in p.items() if k in ("from", "to", "roughness")} | {"height_range": float(np.ptp(p["height"]))} for p in res.profiles]}  # fmt: skip
    if args.output:
        report["output"] = str(T.save_tiff(hm, args.output, res.z, {"tilts": tilts, "axis": args.axis}))
    if args.mesh:
        report["mesh"] = str(T.save_mesh(hm, args.mesh, res.z, exaggeration=args.exaggeration))
    if args.picture:
        Image.fromarray(T.colour(res.z, hm.valid)).save(args.picture)
        report["picture"] = str(args.picture)
    if args.view3d:
        Image.fromarray(T.render_3d(res.z, mask=hm.valid, exaggeration=args.exaggeration, sx=hm.sx, sy=hm.sy)).save(args.view3d)
        report["view3d"] = str(args.view3d)
    if args.save:
        cal = semcal.for_file(files[0], use_ocr=not args.no_ocr)
        ms = measure_io.load_record(cal.sha256) or measure_io.new_set(files[0], cal)
        ms.tilt3d = spec
        measure_io.autosave(ms)
        report["record"] = str(measure_io.record_path(cal.sha256))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=float))
        return 0
    print(f"{files[0].name}: height map from {len(files)} views (tilts {', '.join(f'{t:g}°' for t in tilts)}; axis {args.axis:g}°)")
    if report["height_range"] is not None:
        print(f"  height range (1-99 %): {report['height_range'] / f:.4g} {unit}; reliable pixels {report['reliable_fraction']:.0%}; "
              f"median uncertainty {report['median_uncertainty'] / f:.3g} {unit}")  # fmt: skip
    r = res.roughness
    if r:
        print("  roughness (levelled): " + ", ".join(f"{k} {r[k] / f:.4g} {unit}" for k in ("Sa", "Sq", "Sz")) + f", Ssk {r['Ssk']:.3f}, Sku {r['Sku']:.3f}")
    for st in res.steps:
        box = lambda b: ",".join(f"{v:.0f}" for v in b)  # noqa: E731
        print(f"  step [{box(st['lower'])}] -> [{box(st['upper'])}]: "
              + (st["error"] if "error" in st else f"{st['step'] / f:.4g} ± {st['uncertainty'] / f:.2g} {unit} (two-plane fit, ISO 5436-1)"))  # fmt: skip
    for p in report["profiles"]:
        print(f"  profile {p['from']} -> {p['to']}: height range {p['height_range'] / f:.4g} {unit}")
        r = p.get("roughness") or {}
        if r:
            print("    " + ", ".join(f"{q} {r[q] / f:.4g} {unit}" for q in ("Ra", "Rq", "Rz", "Rt")) + f", Rsk {r['Rsk']:.3f}, Rku {r['Rku']:.3f}"
                  f" (ISO 21920, λc {r['cutoff'] / f:.4g} {unit}, {r['sections']} section(s))")  # fmt: skip
            for n in r.get("notes", []):
                print(f"    ! {n}")
    for w in res.warnings:
        print(f"  ! {w}")
    for key in ("output", "mesh", "picture", "view3d", "record"):
        if key in report:
            print(f"wrote {report[key]}")
    return 0


def cmd_sem_report(args: argparse.Namespace) -> int:
    """An HTML report of one image's record."""
    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import measure_io
    from vexel.packs.sem.report import write_report

    path = Path(args.image)
    cal = semcal.for_file(path, use_ocr=not args.no_ocr)
    ms = measure_io.load_record(cal.sha256)
    if ms is None:
        print(f"{path.name}: no measurement record yet (measure it in the SEM window, or vexel sem analyze --save)")
        return 1
    tilt = None
    if ms.tilt3d is not None and ms.tilt3d.views and not args.no_3d:
        from vexel.packs.sem.tilt3d import run_spec

        tilt = run_spec(ms.tilt3d, path, use_ocr=not args.no_ocr)
    out = write_report(ms, semcal.load_image(path), args.output or path.with_name(path.stem + "_report.html"), tilt=tilt)
    print(f"wrote {out}")
    return 0


def cmd_sem_info(args: argparse.Namespace) -> int:
    from vexel.packs.sem.metadata import read_metadata

    md = read_metadata(args.file)
    if md is None:
        print("no SEM instrument metadata in this file (or next to it)")
        return 1
    d = md.to_dict()
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return 0
    print(f"vendor: {md.vendor_name}")
    for k, v in d.items():
        if k not in ("vendor", "raw", "warnings") and v not in (None, "", [], 0):
            print(f"{k}: {_fmt(v)}")
    for k, v in md.raw.items():
        print(f"  [{k}] {v}")
    for w in md.warnings:
        print(f"! {w}")
    return 0


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
    p.add_argument("--module", help="open this module directly (see `vexel modules`)")
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

    p = sub.add_parser("sem", help="SEM module: calibration, measurement, image processing")
    ssub = p.add_subparsers(dest="sem_command", required=True)
    c = ssub.add_parser("calibrate", help="pixel size of SEM images: instrument metadata, else the data bar's scale bar (OCR)")
    c.add_argument("files", nargs="+")
    c.add_argument("--json", action="store_true")
    c.add_argument("--confirm", action="store_true", help="record a data-bar calibration as confirmed (after checking it)")
    c.add_argument("--pixel-size", help="manual: the pixel size, e.g. 2.5nm")
    c.add_argument("--bar-length", help="manual: what the scale bar stands for, e.g. 500nm (its pixels are measured)")
    c.add_argument("--bar-pixels", type=float, help="manual: the scale bar's length in pixels, when it cannot be found")
    c.add_argument("--no-ocr", action="store_true", help="do not read text in the data bar")
    c.add_argument("--no-metadata", action="store_true", help="ignore instrument metadata (to check the data-bar route against it)")
    c.add_argument("--fresh", action="store_true", help="ignore a recorded confirmation")
    c.set_defaults(func=cmd_sem_calibrate)
    c = ssub.add_parser("measure", help="recompute and export an image's measurements (table, Excel, annotated PNG/SVG, record)")
    c.add_argument("image")
    c.add_argument("--record", help="measurement record (JSON) to use instead of the one saved for this image")
    c.add_argument("-o", "--output", action="append", help="write .xlsx, .csv, .png, .svg or .json (repeatable)")
    c.add_argument("--tilt", type=float, help="tilt correction, degrees (vertical distances / cos(tilt)); 0 turns it off")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_measure)
    c = ssub.add_parser("profile", help="a line profile: edges, line widths, spaces and pitch (e.g. CD on lines and spaces)")
    c.add_argument("image")
    c.add_argument("--from", dest="start", type=_xy, required=True, metavar="X,Y", help="start, pixels")
    c.add_argument("--to", dest="end", type=_xy, required=True, metavar="X,Y", help="end, pixels")
    c.add_argument("--width", type=int, default=5, help="pixels averaged across the line (default 5)")
    c.add_argument("--method", choices=["threshold", "gradient", "peak"], default="threshold", help="edge definition (default threshold)")
    c.add_argument("--level", type=float, default=0.5, help="threshold level between low and high grey (default 0.5)")
    c.add_argument("--smooth", type=float, default=1.0, help="Gaussian smoothing along the profile, samples (default 1)")
    c.add_argument("--tilt", type=float)
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_profile)
    c = ssub.add_parser("process", help="image processing: line levelling, destriping, background, scan-line alignment, denoising, contrast")
    c.add_argument("image", nargs="?")
    c.add_argument("-s", "--step", action="append", metavar="OP[:KEY=VALUE,...]",
                   help="a processing step, in order (repeatable), e.g. -s line_level -s denoise:method=nlm,strength=1.2; "
                        "without -s: the steps saved in the image's record")  # fmt: skip
    c.add_argument("--list", action="store_true", help="list the steps and their settings")
    c.add_argument("-o", "--output", help="write the corrected image (32-bit float TIFF; Vexel opens it with the same calibration)")
    c.add_argument("--picture", help="write the display picture (PNG), contrast and sharpening applied")
    c.add_argument("--save", action="store_true", help="save the steps in the image's record (the SEM window then uses them)")
    c.add_argument("--no-saturation", action="store_true", help="do not mark saturated pixels")
    c.add_argument("--record", help="measurement record (JSON) whose steps to use")
    c.add_argument("--tilt", type=float)
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_process)
    c = ssub.add_parser("average", help="align (sub-pixel) and average frames of the same place into one image")
    c.add_argument("frames", nargs="+")
    c.add_argument("-o", "--output", required=True, help="the averaged image (32-bit float TIFF)")
    c.add_argument("--no-align", action="store_true", help="average without aligning")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_average)
    c = ssub.add_parser("analyze", help="automatic measurements: CD/LER/LWR, contact holes, particles, pores, fibres, layer thickness, sidewall angle, tips")
    c.add_argument("images", nargs="*")
    c.add_argument("-r", "--recipe", choices=["lines", "holes", "particles", "pores", "fibres", "layers", "profile", "tip"],
                   help="what to measure; without it: the automatic measurements saved in each image's record")
    c.add_argument("--region", type=_region, metavar="X0,Y0,X1,Y1", help="pixels (default: the whole micrograph)")
    c.add_argument("-p", "--param", action="append", metavar="KEY=VALUE", help="a recipe setting (repeatable; --list shows them)")
    c.add_argument("-s", "--step", action="append", metavar="OP[:KEY=VALUE,...]", help="processing steps first (default: the record's)")
    c.add_argument("--list", action="store_true", help="list the recipes and their settings")
    c.add_argument("-o", "--output", help=".xlsx (results, repeatability, pooled, objects) or .csv")
    c.add_argument("--save", action="store_true", help="add the measurement to each image's record (the SEM window shows it)")
    c.add_argument("--replace", action="store_true", help="with --save: replace the record's automatic measurements")
    c.add_argument("--auto", action="store_true", help="images without automatic measurements: recognise and plan them first (vexel sem plan)")
    c.add_argument("-g", "--goal", help="with --auto: what to measure, in words")
    c.add_argument("--offline", action="store_true", help="with --auto: rules only, no language model")
    c.add_argument("--tilt", type=float)
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_analyze)
    c = ssub.add_parser("plan", help="recognise images (lines, holes, particles, layers, profile, tip) and plan their measurement, tried first")
    c.add_argument("images", nargs="+")
    c.add_argument("-g", "--goal", help="what to measure, in words (e.g. 线宽和粗糙度, hole diameter); default: the image decides")
    c.add_argument("--apply", action="store_true", help="add each ready plan to the image's record (then: vexel sem analyze)")
    c.add_argument("--offline", action="store_true", help="rules only, no language model")
    c.add_argument("--no-trial", action="store_true", help="draft only, do not try it")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_plan)
    c = ssub.add_parser("tilt3d", help="a 2.5D height map from images of one place at different stage tilts (stereo)")
    c.add_argument("images", nargs="+", help="the reference image (the map is drawn on it) and one or more tilted views")
    c.add_argument("--tilts", type=_floats, metavar="T0,T1,...", help="stage tilt of each image, degrees (default: from the metadata); write --tilts=-5,5 when the first is negative")
    c.add_argument("--axis", type=float, default=0.0, help="tilt axis direction in the image, degrees from horizontal (default 0; 90: vertical)")
    c.add_argument("--level", type=int, choices=[0, 1, 2], default=1, help="remove a plane (1, default) or a quadric (2); 0: none")
    c.add_argument("--smooth", type=float, default=1.5, help="Gaussian on the heights, pixels (default 1.5)")
    c.add_argument("--min-confidence", type=float, default=0.3, help="pixels below are marked unreliable (default 0.3)")
    c.add_argument("--profile", type=_floats, action="append", metavar="X0,Y0,X1,Y1", help="a height profile (repeatable)")
    c.add_argument("--cutoff", type=float, default=0.0, help="profile roughness λc in nm (pixels when uncalibrated); default: one fifth of each profile")
    c.add_argument("--short-cutoff", type=float, default=0.0, help="profile roughness λs in nm; default: none")
    c.add_argument("--step", type=_floats, action="append", metavar="X0,Y0,X1,Y1,X0,Y0,X1,Y1", help="step height: lower box, then upper box (repeatable)")
    c.add_argument("-o", "--output", help="height map TIFF (height, confidence, uncertainty; float32)")
    c.add_argument("--mesh", help="the surface as .stl or .obj")
    c.add_argument("--picture", help="coloured height map (PNG)")
    c.add_argument("--view3d", help="a shaded 3D preview (PNG)")
    c.add_argument("--exaggeration", type=float, default=1.0, help="height exaggeration for the mesh and 3D preview")
    c.add_argument("--save", action="store_true", help="keep the reconstruction in the reference image's record (the SEM window shows it)")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_tilt3d)
    c = ssub.add_parser("report", help="an HTML report of an image's measurements (calibration, processing, annotated image, results, distributions)")
    c.add_argument("image")
    c.add_argument("-o", "--output", help="the report (.html; default: next to the image)")
    c.add_argument("--no-3d", action="store_true", help="leave out the tilt 3D reconstruction even when the record has one")
    c.add_argument("--no-ocr", action="store_true")
    c.set_defaults(func=cmd_sem_report)
    c = ssub.add_parser("info", help="instrument metadata of an SEM image")
    c.add_argument("file")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_sem_info)

    p = sub.add_parser("policy", help="show the organisation policy in force (offline mode, allowed hosts)")
    p.set_defaults(func=cmd_policy)

    p = sub.add_parser("modules", help="list application modules; with a file, which module it belongs to")
    p.add_argument("file", nargs="?")
    p.set_defaults(func=cmd_modules)

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


def _safe_console() -> None:
    """Never crash on output a console cannot encode (e.g. "µm" on a GBK Windows console)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    from vexel.engine.recipe import RecipeError

    _safe_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (RecipeError, FileNotFoundError, FileExistsError, NotImplementedError, ValueError, KeyError) as e:
        message = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
