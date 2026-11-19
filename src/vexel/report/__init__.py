"""Run reports: plain-language notes and the single-file HTML report.

``write_report`` hands a run to the report of the module its recipe belongs to
(:mod:`vexel.modules`); ``build_report`` is the droplet module's report.
"""

from __future__ import annotations

import json
from pathlib import Path

from vexel.report.html_reporter import REPORT_NAME, build_report


def write_report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    """Write the HTML report of a run with its module's report writer."""
    from vexel.modules import for_recipe, resolve
    from vexel.report import html_reporter

    recipe_path = Path(run_dir) / "recipe.json"
    writer = html_reporter.write_report
    if recipe_path.exists():
        module = for_recipe(json.loads(recipe_path.read_text(encoding="utf-8")))
        if module.report:
            writer = resolve(module.report)
    return writer(run_dir, out)


__all__ = ["REPORT_NAME", "build_report", "write_report"]
