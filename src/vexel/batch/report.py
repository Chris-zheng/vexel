"""Batch deliverables: HTML report, methods text (Chinese and English), reproduce script."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vexel.batch.design import Design
from vexel.batch.stats import Analysis, Metric, describe
from vexel.report.html_reporter import CSS, H, PAD_B, PAD_L, PAD_R, PAD_T, W, _e, _fmt, _svg, _text, methods_text, methods_text_en

_UNIT = {"um": "µm", "px": "px"}


def _unit(m: Metric) -> str:
    return _UNIT.get(m.unit, m.unit)


def _p(p: Any) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def svg_groups(units: pd.DataFrame, m: Metric, groups: list[str]) -> str:
    """Each replicate as a point, per group; group mean with its 95% CI."""
    data = [(g, units.loc[units["group"] == g, m.key].to_numpy(dtype=float)) for g in groups]
    data = [(g, v[np.isfinite(v)]) for g, v in data]
    allv = np.concatenate([v for _, v in data]) if data else np.array([])
    if allv.size == 0:
        return _svg(_text(W / 2, H / 2, "没有数据", "middle"), m.zh)
    lo, hi = float(allv.min()), float(allv.max())
    for _, v in data:
        if len(v) >= 2:
            d = describe(v)
            lo, hi = min(lo, d["ci_low"]), max(hi, d["ci_high"])
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    lo, hi = lo - 0.08 * span, hi + 0.08 * span
    x0, x1, y0, y1 = PAD_L + 10, W - PAD_R, PAD_T, H - PAD_B

    def Y(v: float) -> float:
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    parts = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="grid"/>', f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="grid"/>']
    parts += [_text(x0 - 6, y0 + 4, _fmt(hi), "end"), _text(x0 - 6, y1, _fmt(lo), "end")]
    step = (x1 - x0) / max(len(data), 1)
    for i, (g, v) in enumerate(data):
        cx = x0 + step * (i + 0.5)
        for j, val in enumerate(v):
            dx = ((j * 0.618) % 1 - 0.5) * min(step * 0.4, 40)  # deterministic spread
            parts.append(f'<circle cx="{cx + dx:.1f}" cy="{Y(val):.1f}" r="3.5" class="pos"/>')
        if len(v):
            mean = float(v.mean())
            parts.append(f'<line x1="{cx - 16:.1f}" y1="{Y(mean):.1f}" x2="{cx + 16:.1f}" y2="{Y(mean):.1f}" class="line"/>')
            if len(v) >= 2:
                d = describe(v)
                parts.append(f'<line x1="{cx:.1f}" y1="{Y(d["ci_low"]):.1f}" x2="{cx:.1f}" y2="{Y(d["ci_high"]):.1f}" class="mean"/>')
        label = g or "（未分组）"
        parts.append(_text(cx, H - 18, label if len(label) <= 14 else label[:13] + "…", "middle", "label"))
        parts.append(_text(cx, H - 5, f"n = {len(v)}", "middle"))
    return _svg("".join(parts), m.zh)


# -- methods ------------------------------------------------------------------------------------


def batch_methods(out: Path, design: Design, a: Analysis, record: dict[str, Any]) -> tuple[list[str], list[str]]:
    ok = [f for f in record.get("files", []) if f["status"] in ("completed", "resumed")]
    n_groups = len({f["group"] for f in ok})
    unit_zh = "重复（设计表中的 replicate；同组同一重复的多个文件先取平均）" if a.unit_column == "replicate" else "文件（每个文件视为一个独立重复）"
    unit_en = "replicate (the design table's replicate column; files of the same group and replicate were averaged first)" if a.unit_column == "replicate" else "file (each file treated as an independent replicate)"
    zh = [
        f"{len(ok)} 段录像（共 {n_groups} 组）以同一配方 {record['recipe']['name']}（SHA-256 {record['recipe']['sha256'][:16]}…）逐个分析；"
        f"每段录像得到一组汇总指标。统计单位为{unit_zh}。",
        "各组以均值、标准差和均值的 95% 置信区间（t 分布）描述。两组比较用 Welch t 检验（不假定方差相等），报告均值差及其 95% 置信区间、"
        "Hedges' g 效应量，并以 Mann–Whitney U 检验作为不依赖正态假设的核对；三组及以上用 Welch 单因素方差分析与 Kruskal–Wallis 检验，"
        "报告 η²，再以 Welch t 检验做两两比较，并在两两比较之间做 Holm 校正。少于 2 个重复的组只作描述。检验均为双侧，α = 0.05。",
    ]
    en = [
        f"{len(ok)} recordings ({n_groups} groups) were analysed one by one with the same recipe {record['recipe']['name']} "
        f"(SHA-256 {record['recipe']['sha256'][:16]}…); each recording yields one set of summary metrics. The statistical unit is the {unit_en}.",
        "Groups are described by mean, standard deviation and the 95% confidence interval of the mean (t distribution). Two groups were compared "
        "by Welch's t-test (unequal variances), reporting the mean difference with its 95% confidence interval and Hedges' g, with the Mann–Whitney "
        "U test as a check that does not assume normality; three or more groups by Welch's one-way ANOVA and the Kruskal–Wallis test, reporting "
        "eta squared, followed by pairwise Welch t-tests with Holm's correction across the pairs. Groups with fewer than two replicates were only "
        "described. All tests are two-sided at α = 0.05.",
    ]
    per_row = [r for r in design.rows if r.settings]
    if per_row:
        zh.append(f"其中 {len(per_row)} 个文件使用了单独的设置（见设计表 settings 列）。")
        en.append(f"{len(per_row)} files used file-specific settings (design table, settings column).")
    first = next((out / f["run_dir"] for f in ok if (out / f["run_dir"] / "provenance.json").exists()), None)
    if first is not None:
        recipe = json.loads((first / "recipe.json").read_text()) if (first / "recipe.json").exists() else json.loads((out / "recipe.json").read_text())
        prov = json.loads((first / "provenance.json").read_text())
        zh += methods_text(recipe, prov)
        en += methods_text_en(recipe, prov)
    return zh, en


# -- reproduce script ---------------------------------------------------------------------------

SCRIPT = '''"""Re-run this batch from scratch and check that every result is identical.

Written by Vexel {version} on {date}. Needs Vexel installed (same version for identical
results). Usage:

    python reproduce.py                 # writes into ./reproduced next to this script
    python reproduce.py --out OTHER_DIR

Every file is analysed again with the frozen recipe (recipe.json in this folder) and the
calibration and settings below; the statistics are recomputed from the new results. At the
end each file's result digest (SHA-256 over all output tables and summaries) is compared
with the one recorded in batch.json.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vexel.batch.runner import FileOutcome, flatten_summary, results_frame
from vexel.batch.design import Design, Row
from vexel.batch import stats
from vexel.engine import Recipe, run_recipe
from vexel.readers import open_volume

HERE = Path(__file__).resolve().parent

# file, group, replicate, pixel size (um), frame interval (s), file-specific settings, metadata
FILES = {files}
MAX_FRAMES = {max_frames}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "reproduced")
    args = parser.parse_args()
    recipe = Recipe.load(HERE / "recipe.json")
    recorded = {{f["id"]: f for f in json.loads((HERE / "batch.json").read_text(encoding="utf-8"))["files"]}}
    rows, outcomes = [], []
    for item in FILES:
        row = Row(index=item["row"], file=Path(item["file"]), group=item["group"], replicate=item["replicate"],
                  pixel_size_um=item["pixel_size_um"], frame_interval_s=item["frame_interval_s"],
                  settings=item["settings"], meta=item["meta"], label=item["label"])
        rows.append(row)
        r = recipe.with_overrides(row.settings) if row.settings else recipe
        run_dir = args.out / "runs" / row.id
        print(f"{{row.id}}  {{row.file}}", flush=True)
        volume = open_volume(row.file, pixel_size=row.pixel_size_um, frame_interval=row.frame_interval_s)
        result = run_recipe(r, volume, run_dir, max_frames=MAX_FRAMES, hash_input=False, report=False, overwrite=True)
        o = FileOutcome(row.index, row.id, str(row.file), row.group, row.replicate, "completed", str(run_dir),
                        result_digest=result.result_digest,
                        metrics=flatten_summary(json.loads((run_dir / "summary.json").read_text())))
        outcomes.append(o)
    analysis = stats.analyse(results_frame(Design(rows=rows), outcomes))
    stats.write_tables(analysis, args.out)
    differ = [o.id for o in outcomes if o.result_digest != recorded[o.id]["result_digest"]]
    print()
    print(f"{{len(outcomes) - len(differ)}} / {{len(outcomes)}} files reproduced with identical results")
    for i in differ:
        print(f"  differs: {{i}}")
    print(f"statistics: {{args.out / 'groups.csv'}}, {{args.out / 'comparisons.csv'}}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def reproduce_script(design: Design, record: dict[str, Any]) -> str:
    import vexel

    ok = {f["id"] for f in record.get("files", []) if f["status"] in ("completed", "resumed")}
    files = [
        {"row": r.index, "id": r.id, "label": r.label, "file": str(r.file.resolve()), "group": r.group, "replicate": r.replicate,
         "pixel_size_um": r.pixel_size_um, "frame_interval_s": r.frame_interval_s, "settings": r.settings, "meta": r.meta}  # fmt: skip
        for r in design.rows
        if r.id in ok
    ]
    body = "[\n" + "".join(f"    {json.dumps(f, ensure_ascii=False)},\n" for f in files) + "]"
    body = body.replace(": null", ": None").replace(": true", ": True").replace(": false", ": False")
    return SCRIPT.format(version=vexel.__version__, date=datetime.now().strftime("%Y-%m-%d"), files=body, max_frames=record.get("max_frames"))


# -- report -------------------------------------------------------------------------------------


def build_report(out: Path, design: Design, a: Analysis, record: dict[str, Any], zh: list[str], en: list[str]) -> str:
    files = record.get("files", [])
    ok = [f for f in files if f["status"] in ("completed", "resumed")]
    bad = [f for f in files if f["status"] not in ("completed", "resumed")]
    groups = list(dict.fromkeys(a.units["group"])) if not a.units.empty else []
    body = [f"<h1>批量分析：{_e(record['recipe']['name'])}</h1>",
            f'<p class="meta">{len(files)} 个文件：完成 {len(ok)}，失败 {len(bad)}；{len(groups)} 组；统计单位：'
            f'{"重复（replicate）" if a.unit_column == "replicate" else "文件"}。开始于 {_e(record.get("started_at", ""))}，'
            f'本次运行用时 {_fmt(record.get("duration_s"))} 秒。</p>']  # fmt: skip
    notes = list(a.notes)
    if bad:
        notes.insert(0, (f"{len(bad)} 个文件没有完成，不参与统计（见下方“文件”表）。", ""))
    warn = [f for f in ok if f.get("qc_warnings")]
    if warn:
        notes.append((f"{len(warn)} 个文件有质控提示，建议打开它们各自的报告看一眼。", ""))
    if notes:
        body.append('<div class="notes">' + "".join(f"<p>{_e(z)}</p>" for z, _ in notes) + "</div>")

    if a.metrics and groups:
        body.append("<h2>分组结果</h2>")
        tested = [m for m in a.metrics if m.test]
        for m in tested + [m for m in a.metrics if not m.test]:
            g = a.groups[a.groups["metric"] == m.key]
            unit = _unit(m)
            title = f"{m.zh}" + (f"（{unit}）" if unit else "")
            rows = "".join(
                f"<tr><td>{_e(r.group or '（未分组）')}</td><td>{int(r.n)}</td><td>{_fmt(r.mean)}</td><td>{_fmt(r.sd)}</td>"
                f"<td>{_fmt(r.ci_low)} – {_fmt(r.ci_high)}</td><td>{_fmt(r.median)}</td></tr>"
                for r in g.itertuples()
            )
            table = f"<table><tr><th>组</th><th>n</th><th>均值</th><th>标准差</th><th>95% 置信区间</th><th>中位数</th></tr>{rows}</table>"
            comp = a.comparisons[a.comparisons["metric"] == m.key] if not a.comparisons.empty else pd.DataFrame()
            ctable = ""
            if not comp.empty:
                crow = []
                for r in comp.itertuples():
                    if r.comparison == "all groups":
                        crow.append(f"<tr><td>所有组</td><td>–</td><td>{_p(r.p)}（Welch ANOVA）</td><td>{_p(r.p_nonparametric)}（Kruskal–Wallis）</td><td>η² = {_fmt(r.effect, 2)}</td></tr>")
                    else:
                        p = _p(r.p_holm) + "（Holm 校正）" if len(comp) > 1 and math.isfinite(r.p_holm) else _p(r.p)
                        crow.append(f"<tr><td>{_e(r.a)} − {_e(r.b)}</td><td>{_fmt(r.diff)}（{_fmt(r.diff_ci_low)} – {_fmt(r.diff_ci_high)}）</td>"
                                    f"<td>{p}</td><td>{_p(r.p_nonparametric)}</td><td>g = {_fmt(r.effect, 2)}</td></tr>")  # fmt: skip
                ctable = "<table><tr><th>比较</th><th>均值差（95% CI）</th><th>p（Welch）</th><th>p（非参数）</th><th>效应量</th></tr>" + "".join(crow) + "</table>"
            elif not m.test:
                ctable = '<p class="meta">只作描述，不做检验（取决于录像长度或属于质控指标）。</p>'
            body.append(f'<section class="metric"><h3>{_e(title)}</h3><div class="charts"><figure>{svg_groups(a.units, m, groups)}'
                        f'<figcaption class="meta">每个点是一个{"重复" if a.unit_column == "replicate" else "文件"}；横线为均值，竖虚线为均值的 95% 置信区间。</figcaption></figure>'
                        f"<div>{table}{ctable}</div></div></section>")  # fmt: skip

    # files
    head = ["#", "文件", "组", "重复", "状态", "直径", "CV %", "λ", "阳性比例", "质控", "报告"]
    rows = []
    status_zh = {"completed": "完成", "resumed": "完成（沿用）", "failed": "失败", "missing": "找不到文件", "stopped_by_qc": "质控中止"}
    res = pd.read_csv(out / "results.csv") if (out / "results.csv").exists() else pd.DataFrame()
    by_id = {r["id"]: r for r in res.to_dict("records")} if not res.empty else {}
    for f in files:
        r = by_id.get(f["id"], {})
        link = f'<a href="{_e(f["run_dir"])}/report.html">打开</a>' if (out / f["run_dir"] / "report.html").exists() else ""
        status = status_zh.get(f["status"], f["status"]) + (f"：{_e(f['error'][:160])}" if f.get("error") else "")
        rows.append(f"<tr><td>{f['row']}</td><td>{_e(Path(f['file']).name)}</td><td>{_e(f['group'])}</td><td>{_e(f['replicate'])}</td><td>{status}</td>"
                    f"<td>{_fmt(r.get('droplets.diameter_mean'))}</td><td>{_fmt(r.get('droplets.diameter_cv_percent'), 3)}</td>"
                    f"<td>{_fmt(r.get('encapsulation.lambda'), 3)}</td><td>{_fmt(r.get('positivity.positive_fraction'), 3)}</td>"
                    f"<td>{len(f.get('qc_warnings') or [])}</td><td>{link}</td></tr>")  # fmt: skip
    body.append("<h2>文件</h2><table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(rows) + "</table>")

    body.append('<h2>方法</h2><div class="methods">' + "".join(f"<p>{_e(t)}</p>" for t in zh) + "</div>")
    body.append('<details class="methods"><summary>Methods (English)</summary>' + "".join(f"<p>{_e(t)}</p>" for t in en) + "</details>")
    env = record.get("environment", {})
    prov_rows = [("配方", f"{record['recipe']['name']}（SHA-256 {record['recipe']['sha256']}）"), ("设计表 SHA-256", record["design"].get("sha256") or "–"),
                 ("Vexel", env.get("vexel", "")), ("Python", env.get("python", "")), ("平台", env.get("platform", "")),
                 ("复现", "python reproduce.py（重新分析全部文件并核对每个文件的结果摘要）")]  # fmt: skip
    body.append("<h2>溯源</h2><table>" + "".join(f"<tr><th>{_e(k)}</th><td><code>{_e(v)}</code></td></tr>" for k, v in prov_rows) + "</table>")
    body.append(f'<footer>由 Vexel 生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}。表格：results.csv（每文件一行）、replicates.csv、groups.csv、comparisons.csv。</footer>')
    extra = ".metric .charts { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.2fr); gap:16px; align-items:start; } @media (max-width:760px) { .metric .charts { grid-template-columns:1fr; } }"
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Vexel 批量分析</title><style>{CSS}{extra}</style></head><body><main>{''.join(body)}</main></body></html>")  # fmt: skip


def write_all(out: Path, design: Design, a: Analysis) -> None:
    record = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    zh, en = batch_methods(out, design, a, record)
    (out / "methods_zh.txt").write_text("\n\n".join(zh) + "\n", encoding="utf-8")
    (out / "methods_en.txt").write_text("\n\n".join(en) + "\n", encoding="utf-8")
    (out / "reproduce.py").write_text(reproduce_script(design, record), encoding="utf-8")
    (out / "report.html").write_text(build_report(out, design, a, record, zh, en), encoding="utf-8")
