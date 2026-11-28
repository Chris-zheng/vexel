"""Planning evaluation set (0.8 plan, gate 3): which measurement the planner picks, per case.

    uv run python tests/benchmarks/plan_eval.py [--tier rules|auto|<tier>] [--only sem|droplet] [--out table.md]

Cases are in ``evalsets/planning/cases.json`` (format: ``evalsets/README.md``). Synthetic
cases are generated here; real files are read from ``VEXEL_EVAL_DATA`` and skipped (with a
note) when that folder or the file is missing, or its SHA-256 differs from the manifest.

A case is right when the planned method is the expected one (or one of ``accept``) and the
expected settings match; ``expect.recipe = null`` means "nothing to measure": right when the
planner leaves the choice to the person. The planners run without trial runs: this scores
the choice, the trial runs are scored by the module acceptance scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evalsets" / "planning" / "cases.json"


def _chain(tier: str) -> list[Any] | None:
    """The planner tiers to use: only the offline rules, the configured order, or one tier."""
    from vexel.agent import models

    if tier == "auto":
        return None
    if tier == "rules":
        return [models.RulesTier()]
    chain = [t for t in models.tiers() if t.name == tier]
    if not chain:
        raise SystemExit(f"tier {tier!r} is not configured (see 帮助 → 规划模型设置)")
    return chain + [models.RulesTier()]


def _sem_image(src: dict[str, Any]) -> np.ndarray:
    from vexel.synthetic import sem as S

    name, args = src["synthetic"], dict(src.get("args", {}))
    if name == "noise":
        img = 0.5 + np.random.default_rng(args.get("seed", 0)).normal(0, 0.05, (400, 500))
    else:
        if "pitch" in args and isinstance(args["pitch"], list):
            args["pitch"] = tuple(args["pitch"])
        out = getattr(S, name)(**args)
        img = out[0] if isinstance(out, tuple) else out
    img = np.asarray(img, np.float64)
    if src.get("invert"):
        img = 1.0 - img
    if src.get("transpose"):
        img = img.T
    return img


def _droplet_volume(src: dict[str, Any], work: Path) -> Any:
    from vexel.core.volume import SpatialVolume
    from vexel.readers import open_volume
    from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

    path = write_dataset(work, DropletSynthConfig(**src.get("args", {})))
    vol = open_volume(path)
    if src.get("transpose"):  # droplets move down instead of right (one channel, as the acceptance script does)
        data = np.swapaxes(np.asarray(vol.frames(0, 0)), 1, 2)
        vol = SpatialVolume.from_array(data, "TYX", spacing={"y": 2.0, "x": 2.0, "t": 0.001}, space_unit="um", time_unit="s")
    return vol


def _real(src: dict[str, Any]) -> tuple[Path | None, str]:
    base = os.environ.get("VEXEL_EVAL_DATA")
    if not base:
        return None, "VEXEL_EVAL_DATA not set"
    p = Path(base) / src["file"]
    if not p.exists():
        return None, f"missing: {src['file']}"
    if src.get("sha256") and hashlib.sha256(p.read_bytes()).hexdigest() != src["sha256"]:
        return None, f"changed (SHA-256 differs): {src['file']}"
    return p, ""


def _score(expect: dict[str, Any], accept: list[str], recipe: str | None, params: dict[str, Any]) -> bool:
    want = expect.get("recipe")
    if want is None:
        return recipe is None
    if recipe != want and recipe not in accept:
        return False
    return all(params.get(k) == v for k, v in (expect.get("params") or {}).items()) if recipe == want else True


def run_case(case: dict[str, Any], chain: list[Any] | None, work: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"id": case["id"], "module": case["module"], "goal": case.get("goal", ""), "expect": case["expect"].get("recipe")}
    src = case["source"]
    if "file" in src:
        path, why = _real(src)
        if path is None:
            return row | {"skipped": why}
    if case["module"] == "sem":
        from vexel.packs.sem.plan import make_plan

        if "file" in src:
            from vexel.packs.sem.calibrate import load_image

            img = np.asarray(load_image(path), np.float64)
        else:
            img = _sem_image(src)
        plan = make_plan(img, goal=case.get("goal", ""), trial_run=False, chain=chain, models=chain is None or any(t.name != "rules" for t in chain))
        recipe, params = plan.recipe, dict(plan.params)
    else:
        from vexel.agent.planner import make_plan

        if "file" in src:
            from vexel.readers import open_volume

            vol = open_volume(path)
        else:
            vol = _droplet_volume(src, work / case["id"])
        plan = make_plan(vol, case.get("goal", ""), trial=False, chain=chain)
        recipe, params = plan.base, dict(plan.overrides)
    ok = _score(case["expect"], case.get("accept", []), recipe, params)
    return row | {"got": recipe, "ok": ok, "tier": plan.planner, "model": plan.model, "skipped_tiers": "; ".join(plan.skipped)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="rules", help="rules (default), auto (the configured order) or one tier's name")
    parser.add_argument("--only", choices=("sem", "droplet"))
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = [c for c in json.loads(args.cases.read_text(encoding="utf-8"))["cases"] if not args.only or c["module"] == args.only]
    chain = _chain(args.tier)
    rows = []
    with tempfile.TemporaryDirectory(prefix="vexel-plan-eval-") as tmp:
        os.environ.setdefault("VEXEL_HOME", str(Path(tmp) / "home"))  # a clean recipe memory: no help from earlier runs
        for case in cases:
            try:
                rows.append(run_case(case, chain, Path(tmp)))
            except Exception as e:  # noqa: BLE001 - one broken case must not hide the others
                rows.append({"id": case["id"], "module": case["module"], "goal": case.get("goal", ""), "expect": case["expect"].get("recipe"),
                             "got": f"error: {type(e).__name__}: {e}", "ok": False})  # fmt: skip
    out = ["| 用例 | 模块 | 目标 | 应为 | 规划 | 档位 | 结果 |", "| :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
    for r in rows:
        res = f"跳过（{r['skipped']}）" if "skipped" in r else ("✓" if r["ok"] else "✗")
        out.append(f"| {r['id']} | {r['module']} | {r['goal'] or '（未写）'} | {r['expect']} | {r.get('got', '')} | {r.get('tier', '')} {r.get('model', '')} | {res} |")
    scored = [r for r in rows if "skipped" not in r]
    for mod in ("sem", "droplet"):
        m = [r for r in scored if r["module"] == mod]
        if m:
            out.append(f"\n{mod}: {sum(r['ok'] for r in m)}/{len(m)} 正确")
    text = "\n".join(out)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
