"""MCP server: Vexel's operator registry and recipe engine for AI assistants.

Run with ``vexel mcp`` (stdio). An assistant (Claude Desktop, Claude Code, any MCP client)
can then list operators and recipes, fingerprint data, ask the planner for a recipe draft,
run a recipe, and read results. Everything runs locally; data never leaves the machine
through this server. Planning and running write only into the output folders named.

Tools that only read are marked read-only; ``run_analysis`` writes a run folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INSTRUCTIONS = """Vexel analyses microfluidic droplet recordings and SEM images with validated, reproducible recipes.
Droplets: probe_data (what is in the file) -> plan_analysis (recipe draft, tried on the first
frames; show its reasons and status to the user) -> run_analysis once the user agrees ->
read_results.
SEM images: sem_calibrate (pixel size; a calibration read from the data bar needs the user's
confirmation) -> sem_plan (what the image shows and a tried measurement plan; show its reasons
and status) -> sem_analyze (apply=True only once the user agrees; or name a recipe) -> report
the values as given, with their edge definition and warnings.
Results come from deterministic operators; never estimate or adjust a value yourself."""


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("vexel", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @server.tool(annotations=read_only)
    def list_recipes() -> dict:
        """Built-in analysis recipes: name and what each measures."""
        from vexel.engine import Recipe
        from vexel.packs import builtin_recipes

        return {"recipes": [{"name": n, "description": Recipe.from_json(t).description} for n, t in builtin_recipes().items()]}

    @server.tool(annotations=read_only)
    def show_recipe(name: str) -> dict:
        """A recipe (built-in name or JSON file path) with every parameter made explicit."""
        from vexel.engine import Recipe

        return Recipe.load(name).resolve().model_dump(mode="json")

    @server.tool(annotations=read_only)
    def list_operators() -> dict:
        """Registered operators: scope, typed inputs and outputs, parameters with defaults."""
        from vexel.ops import list_ops

        return {"operators": [o.describe() for o in list_ops()]}

    @server.tool(annotations=read_only)
    def probe_data(path: str, pixel_size_um: float | None = None, frame_interval_s: float | None = None) -> dict:
        """Fingerprint a recording without changing it: size, calibration, noise, motion,
        droplet size and count, cells inside droplets, which channel is fluorescence."""
        from vexel.agent.probe import probe
        from vexel.readers import open_volume

        fp = probe(open_volume(path, pixel_size=pixel_size_um, frame_interval=frame_interval_s))
        return fp.to_json()

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
    def plan_analysis(path: str, goal: str = "", out_dir: str | None = None, trial: bool = True,
                      pixel_size_um: float | None = None, frame_interval_s: float | None = None) -> dict:  # fmt: skip
        """Draft a recipe for a recording and a goal in words (e.g. "cells per droplet",
        "ddPCR positive fraction"), check it on the first few hundred frames, and write
        plan.json and recipe.json to out_dir. Status: ready, check (QC warnings) or
        needs_user. The plan is a draft: show the reasons to the user before running it."""
        from vexel.agent.memory import RecipeMemory
        from vexel.agent.planner import make_plan
        from vexel.readers import open_volume

        volume = open_volume(path, pixel_size=pixel_size_um, frame_interval=frame_interval_s)
        plan = make_plan(volume, goal, memory=RecipeMemory(), trial=trial)
        out = Path(out_dir) if out_dir else Path(path).parent / f"{Path(path).name.split('.')[0]}_vexel" / "plan"
        recipe_path = plan.save(out)
        return {"status": plan.status, "recipe": plan.base, "settings": plan.overrides, "reasons": plan.reasons,
                "recipe_path": str(recipe_path), "planner": plan.planner, "model": plan.model,
                "trials": [{"change": a.change, "metrics": a.metrics, "qc_warnings": a.violations, "failure": a.failure} for a in plan.attempts]}  # fmt: skip

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
    def run_analysis(path: str, recipe: str, out_dir: str, settings: dict | None = None, max_frames: int | None = None,
                     pixel_size_um: float | None = None, frame_interval_s: float | None = None) -> dict:  # fmt: skip
        """Run a recipe (built-in name or recipe.json path, e.g. from plan_analysis) on a
        recording; writes tables, summary, report and provenance into out_dir (a new folder)."""
        from vexel.engine import Recipe, run_recipe
        from vexel.readers import open_volume

        r = Recipe.load(recipe)
        if settings:
            r = r.with_overrides(settings)
        result = run_recipe(r, open_volume(path, pixel_size=pixel_size_um, frame_interval=frame_interval_s), out_dir,
                            max_frames=max_frames, hash_input=False)  # fmt: skip
        return {"run_dir": str(result.out_dir), "status": result.status, "summary": json.loads((result.out_dir / "summary.json").read_text(encoding="utf-8")),
                "qc_warnings": [v.message() for v in result.violations], "report": str(result.out_dir / "report.html")}  # fmt: skip

    @server.tool(annotations=read_only)
    def read_results(run_dir: str) -> dict:
        """Summary, QC warnings and provenance highlights of a finished run."""
        run = Path(run_dir)
        prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        warnings = [f"{s['id']}.{v['metric']}" for s in prov.get("steps", []) for v in s.get("qc_violations", [])]
        return {"summary": json.loads((run / "summary.json").read_text(encoding="utf-8")), "status": prov.get("status"),
                "recipe": prov.get("recipe"), "result_digest": prov.get("result_digest"), "qc_warnings": warnings,
                "corrections": (prov.get("corrections") or {}).get("n_applied"), "report": str(run / "report.html")}  # fmt: skip

    # -- SEM -------------------------------------------------------------------------------------

    @server.tool(annotations=read_only)
    def sem_calibrate(path: str) -> dict:
        """Pixel size of an SEM image: from the instrument's metadata, else from the data bar's
        scale bar (text recognition; then it needs the user's confirmation in the SEM window or
        with `vexel sem calibrate --confirm`). Never guessed: ok is false with a reason."""
        from vexel.packs.sem.calibrate import for_file

        cal = for_file(path)
        return {"ok": cal.ok, "pixel_size_m": cal.pixel_size, "source": cal.source, "confirmed": cal.confirmed,
                "needs_confirmation": cal.needs_confirmation, "describe": cal.describe(), "crop_rows": cal.crop_rows,
                "warnings": cal.warnings, "reason": cal.reason, "vendor": cal.vendor}  # fmt: skip

    @server.tool(annotations=read_only)
    def sem_plan(path: str, goal: str = "") -> dict:
        """Recognise what an SEM image shows (lines, holes, particles, layers, profile, tip,
        other) and draft its measurement (recipe, settings, region, processing), tried on the
        image. Status: ready, check (warnings) or needs_user (unclear image or no usable
        result). Changes nothing: apply it with sem_analyze(apply=True) once the user agrees."""
        from vexel.packs.sem.plan import memory_store, plan_file

        plan, cal, _ = plan_file(path, goal, memory=memory_store())
        fp = plan.fingerprint
        return {"kind": fp.kind if fp else None, "confidence": fp.confidence if fp else None, "recipe": plan.recipe, "status": plan.status,
                "settings": plan.params, "region": plan.region, "processing": [s["op"] for s in plan.steps], "reasons": plan.reasons,
                "trials": [{"change": a.change, "problems": a.problems, "warnings": a.warnings} for a in plan.attempts],
                "planner": plan.planner, "calibration": cal.describe()}  # fmt: skip

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
    def sem_analyze(path: str, recipe: str | None = None, region: list[float] | None = None, settings: dict | None = None,
                    apply: bool = False, goal: str = "") -> dict:  # fmt: skip
        """Measure an SEM image. With `recipe` (lines, holes, particles, pores, fibres, layers, profile, tip;
        `region` [x0, y0, x1, y1] pixels; `settings` as in the recipe): that measurement. With
        apply=True and no recipe: the sem_plan plan, added to the image's record (only after the
        user agreed to it). Without either: the measurements already in the record. Values are in
        metres when calibrated, else pixels."""
        from vexel.packs.sem import calibrate as C
        from vexel.packs.sem import measure_io
        from vexel.packs.sem.measure import scale_from
        from vexel.packs.sem.plan import memory_store, plan_file, remember

        cal = C.for_file(path)
        ms = measure_io.load_record(cal.sha256) or measure_io.new_set(path, cal)
        ms.scale = scale_from(cal, ms.scale.tilt)
        if recipe:
            ms.add_analysis(recipe, region, **(settings or {}))
            measure_io.autosave(ms)
        elif apply:
            plan, _, ms = plan_file(path, goal, memory=memory_store())
            if not plan.recipe or plan.status == "needs_user":
                return {"applied": False, "status": plan.status, "reasons": plan.reasons}
            plan.apply(ms)
            measure_io.autosave(ms)
            remember(memory_store(), plan, True, path)
        if not ms.analyses:
            return {"analyses": [], "note": "no automatic measurements: give a recipe, or apply a plan"}
        out = []
        for an in measure_io.analyses_of(ms, C.load_image(path)):
            out.append({"id": an.spec.id, "recipe": an.spec.recipe, "region": an.spec.region, "settings": an.spec.params, "error": an.error,
                        "values": [{"key": k, "label": label, "value": v, "kind": kind} for k, label, v, kind in an.summary],
                        "warnings": an.warnings, "objects": len(an.objects)})  # fmt: skip
        return {"calibration": cal.describe(), "unit": "m" if cal.ok else "px", "analyses": out}

    return server


def main() -> int:
    build_server().run("stdio")
    return 0
