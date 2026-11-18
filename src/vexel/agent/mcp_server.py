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

INSTRUCTIONS = """Vexel analyses microfluidic droplet recordings with validated, reproducible recipes.
Typical flow: probe_data (what is in the file) -> plan_analysis (recipe draft, tried on the first
frames; show its reasons and status to the user) -> run_analysis once the user agrees ->
read_results. Results come from deterministic operators; report them as given."""


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

    return server


def main() -> int:
    build_server().run("stdio")
    return 0
