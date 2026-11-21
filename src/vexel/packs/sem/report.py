"""An HTML report of one SEM image's record: calibration, processing, the annotated image,
manual and automatic measurements with their distributions, and the tilt 3D results when
given. One self-contained file (images embedded) that opens in any browser and prints to
PDF; the full tables are in the Excel export.

Text follows the interface language (``vexel.i18n``).
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vexel.i18n import tr

W, H, PL, PR, PT, PB = 560, 200, 46, 16, 22, 34

CSS = """
body{font:14px/1.5 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:#1f2937;background:#fff;max-width:1000px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}h3{font-size:15px;margin:18px 0 6px}
.muted{color:#6b7280}table{border-collapse:collapse;margin:6px 0 10px;font-size:13px}td,th{border:1px solid #e5e7eb;padding:3px 8px;text-align:left}
th{background:#f9fafb}td.num{text-align:right;font-variant-numeric:tabular-nums}.warn{background:#fffbeb;border-left:3px solid #f59e0b;padding:6px 10px;margin:6px 0}
img{max-width:100%;height:auto;border:1px solid #e5e7eb}svg{max-width:100%;height:auto}.bar{fill:#3b82f6}.axis{font-size:11px;fill:#6b7280}.mean{stroke:#111827;stroke-dasharray:4 3}
.grid{stroke:#d1d5db}.charts{display:flex;flex-wrap:wrap;gap:12px}
@media print{body{margin:0}h2{break-after:avoid}}
"""


def _e(v: Any) -> str:
    return html.escape(str(v))


def _num(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return tr("是") if v else tr("否")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return _e(tr(str(v)))


def _table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return ""
    d = df.head(max_rows)
    head = "".join(f"<th>{_e(c)}</th>" for c in d.columns)
    body = []
    for r in d.itertuples(index=False):
        cells = "".join(f'<td class="num">{_num(v)}</td>' if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else f"<td>{_num(v)}</td>" for v in r)
        body.append(f"<tr>{cells}</tr>")
    more = f'<p class="muted">{_e(tr(f"只列出前 {max_rows} 行，共 {len(df)} 行；完整表格见 Excel 导出。"))}</p>' if len(df) > max_rows else ""
    return f"<table><tr>{head}</tr>{''.join(body)}</table>{more}"


def svg_histogram(values: np.ndarray, label: str, unit: str, weights: np.ndarray | None = None, *, mark: float | None = None,
                  mark_label: str | None = None, span: tuple[float, float] | None = None) -> str:  # fmt: skip
    """A bar histogram (counts, or the weights' sum per bin) with the mean - or ``mark`` - as a
    dashed line; ``span`` fixes the axis (e.g. 0..180° for directions)."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(v)
    v, w = v[ok], w[ok]
    body = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="{_e(label)}" xmlns="http://www.w3.org/2000/svg">']
    body.append(f'<text x="{PL}" y="14" class="axis">{_e(label)}</text>')
    if v.size == 0:
        body.append(f'<text x="{W / 2}" y="{H / 2}" text-anchor="middle" class="axis">{_e(tr("没有数据"))}</text></svg>')
        return "".join(body)
    lo, hi = span if span else (float(v.min()), float(v.max()))
    if hi - lo < 1e-12 * max(abs(hi), 1):
        lo, hi = lo - 0.5, hi + 0.5
    bins = int(np.clip(np.sqrt(v.size) * 1.5, 5, 40))
    counts, _ = np.histogram(v, bins=bins, range=(lo, hi), weights=w)
    x0, x1, y0, y1 = PL, W - PR, PT, H - PB
    bw = (x1 - x0) / bins
    body.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="grid"/>')
    top = counts.max() or 1
    for i, c in enumerate(counts):
        hh = (y1 - y0) * c / top
        body.append(f'<rect x="{x0 + i * bw + 0.5:.1f}" y="{y1 - hh:.1f}" width="{max(bw - 1, 0.5):.1f}" height="{hh:.1f}" class="bar"/>')
    mean = float(np.average(v, weights=w)) if mark is None else mark
    mx = x0 + (mean - lo) / (hi - lo) * (x1 - x0)
    body.append(f'<line x1="{mx:.1f}" y1="{y0}" x2="{mx:.1f}" y2="{y1}" class="mean"/>')
    anchor = "start" if mx < W - 160 else "end"
    body.append(f'<text x="{mx + (5 if anchor == "start" else -5):.1f}" y="{y0 + 10}" text-anchor="{anchor}" class="axis">{_e(mark_label or tr("平均"))} {mean:.4g} {_e(unit)}</text>')
    body.append(f'<text x="{x0}" y="{H - 14}" class="axis">{lo:.4g}</text>')
    body.append(f'<text x="{x1}" y="{H - 14}" text-anchor="end" class="axis">{hi:.4g} {_e(unit)}</text>')
    body.append(f'<text x="{(x0 + x1) / 2}" y="{H - 14}" text-anchor="middle" class="axis">n = {v.size}</text>')
    body.append("</svg>")
    return "".join(body)


def _png(img: Any, max_side: int = 1600) -> str:
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _fmt(v: Any, kind: str, unit: str, f: float) -> str:
    if v is None:
        return "–"
    if kind == "length":
        return f"{v / f:.4g} {unit}"
    if kind == "area":
        return f"{v / f**2:.4g} {unit}²"
    if kind == "angle":
        return f"{v:.2f}°"
    if kind == "ratio":
        return f"{v:.3f}"
    return _num(v)


def _charts(an: Any, unit: str, f: float) -> str:
    from vexel.packs.sem.population import SIZE, sizes

    out = []
    if an.spec.recipe in SIZE:
        v, w = sizes(an)
        name = {"particles": "等效圆直径", "pores": "孔的等效圆直径", "holes": "孔径", "fibres": "纤维直径（按长度）"}[an.spec.recipe]
        out.append(svg_histogram(v / f, tr(name), unit, w))
        if an.spec.recipe == "particles":
            out.append(svg_histogram(v / f, tr("等效圆直径（体积加权）"), unit, w * v**3))
    if an.spec.recipe == "fibres" and an.extra.get("samples"):
        s = an.extra["samples"]
        out.append(svg_histogram(np.asarray(s["angle"]), tr("纤维方向（0–180°，逆时针）"), "°", np.asarray(s["length"]), mark=an.value("orientation"),
                                 mark_label=tr("主方向"), span=(0.0, 180.0)))  # fmt: skip
    return f'<div class="charts">{"".join(out)}</div>' if out else ""


def build_report(ms: Any, image: np.ndarray, *, tilt: Any = None, result: Any = None) -> str:
    from vexel import __version__
    from vexel.packs.sem import measure_io, process
    from vexel.packs.sem.analysis import RECIPES

    unit, f = measure_io.display_unit(ms)
    res = measure_io.processed(ms, image, result)
    ans = measure_io.analyses_of(ms, image, res)
    t = measure_io.tables(ms, image, res, analyses=ans)
    name = ms.image.get("file", "")
    sc = ms.scale
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{_e(tr('SEM 测量报告'))} – {_e(Path(name).name)}</title><style>{CSS}</style></head><body>"]  # fmt: skip
    parts.append(f"<h1>{_e(tr('SEM 测量报告'))}</h1>")
    parts.append(f'<p class="muted">{_e(Path(name).name)} · {_dt.datetime.now():%Y-%m-%d %H:%M} · Vexel {__version__} · '
                 f'{_e(tr("记录摘要"))} {_e(ms.digest()[:12])}</p>')  # fmt: skip
    # calibration
    parts.append(f"<h2>{_e(tr('标定'))}</h2>")
    if sc.pixel_size:
        ps = f"{sc.pixel_size * 1e9:.4g} nm" + (f" × {sc.pixel_size_y * 1e9:.4g} nm" if sc.pixel_size_y and abs(sc.pixel_size_y - sc.pixel_size) > 1e-15 else "")
        cal = [(tr("像素尺寸"), ps), (tr("来源"), tr(str(sc.source))), (tr("已人工确认"), tr("是") if sc.confirmed else tr("否"))]
        if sc.tilt:
            cal.append((tr("倾转校正"), f"{sc.tilt:g}°"))
    else:
        cal = [(tr("像素尺寸"), tr("未标定：长度以像素为单位"))]
    parts.append("<table>" + "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in cal) + "</table>")
    if not sc.confirmed and sc.pixel_size:
        parts.append(f'<div class="warn">{_e(tr("标定尚未人工确认：报告中的长度依赖自动读取的比例尺。"))}</div>')
    # processing
    steps = process.describe(ms.processing)
    if steps:
        parts.append(f"<h2>{_e(tr('图像处理'))}</h2><ol>" + "".join(f"<li>{_e(tr(s))}</li>" for s in steps) + "</ol>")
    if res is not None and res.saturation and res.saturation.get("pixels"):
        parts.append(f'<div class="warn">{_e(tr(f"饱和像素：{res.saturation.get('fraction', 0):.2%}（已标记，未修复；这些位置的测量不可靠）"))}</div>')
    # picture
    parts.append(f"<h2>{_e(tr('标注图'))}</h2>")
    parts.append(_png(measure_io.render(image, ms, crop_rows=ms.image.get("crop_rows"), result=res)))
    # manual
    if ms.items:
        parts.append(f"<h2>{_e(tr('手动测量'))}</h2>")
        parts.append(_table(t["measurements"]))
        if len(t["summary"]):
            parts.append(f"<h3>{_e(tr('汇总'))}</h3>" + _table(t["summary"]))
    # automatic
    if ans:
        parts.append(f"<h2>{_e(tr('自动测量'))}</h2>")
        for an in ans:
            rec = RECIPES.get(an.spec.recipe)
            title = f"A{an.spec.id} {tr(rec.label) if rec else an.spec.recipe}" + (f" – {an.spec.name}" if an.spec.name else "")
            region = tr(f"区域 {an.extra.get('region')}") if an.spec.region else tr("整张图")
            parts.append(f"<h3>{_e(title)}</h3><p class='muted'>{_e(region)}</p>")
            if an.error:
                parts.append(f'<div class="warn">{_e(tr(f"无法测量：{an.error}"))}</div>')
                continue
            parts.append("<table>" + "".join(f"<tr><th>{_e(tr(lab))}</th><td class='num'>{_e(_fmt(v, kind, unit, f))}</td></tr>"
                                             for _, lab, v, kind in an.summary if v is not None) + "</table>")  # fmt: skip
            for w in an.warnings:
                parts.append(f'<div class="warn">{_e(tr(w))}</div>')
            parts.append(_charts(an, unit, f))
            from vexel.packs.sem.analysis import objects_table

            parts.append(_table(objects_table([an], unit, f).drop(columns=["analysis", "recipe"], errors="ignore"), 40))
            if rec:
                params = ", ".join(f"{tr(p.label)} = {tr(str(dict(p.choices).get(v, v)))}" for p in rec.params for k, v in rec.resolve(an.spec.params).items() if k == p.name)
                parts.append(f'<p class="muted">{_e(tr(rec.help))}<br>{_e(tr("设置："))}{_e(params)}</p>')
    # tilt 3D
    if tilt is not None:
        from vexel.packs.sem.tilt3d import colour

        hm = tilt.hm
        u3, f3 = ("nm", 1e-9) if hm.pixel_size else (tr("像素"), 1.0)
        parts.append(f"<h2>{_e(tr('倾转三维（2.5D 高度图）'))}</h2>")
        parts.append(_png(colour(tilt.z, hm.valid), 900))
        r = tilt.roughness
        if r:
            parts.append(f"<h3>{_e(tr('面粗糙度（ISO 25178，校平后）'))}</h3><table>"
                         + "".join(f"<tr><th>{k}</th><td class='num'>{r[k] / f3:.4g} {_e(u3)}</td></tr>" for k in ("Sa", "Sq", "Sp", "Sv", "Sz"))
                         + f"<tr><th>Ssk</th><td class='num'>{r['Ssk']:.3f}</td></tr><tr><th>Sku</th><td class='num'>{r['Sku']:.3f}</td></tr></table>")  # fmt: skip
        for i, p in enumerate(tilt.profiles, start=1):
            pr = p.get("roughness") or {}
            parts.append(f"<h3>{_e(tr(f'剖面 {i}'))}</h3>")
            if pr:
                parts.append("<table>" + "".join(f"<tr><th>{k}</th><td class='num'>{pr[k] / f3:.4g} {_e(u3)}</td></tr>" for k in ("Ra", "Rq", "Rz", "Rp", "Rv", "Rt"))
                             + f"<tr><th>Rsk</th><td class='num'>{pr['Rsk']:.3f}</td></tr><tr><th>Rku</th><td class='num'>{pr['Rku']:.3f}</td></tr>"
                             + f"<tr><th>λc</th><td class='num'>{pr['cutoff'] / f3:.4g} {_e(u3)}</td></tr></table>")  # fmt: skip
                for n in pr.get("notes", []):
                    parts.append(f'<div class="warn">{_e(tr(n))}</div>')
        for i, st in enumerate(tilt.steps, start=1):
            txt = st["error"] if "error" in st else f"{st['step'] / f3:.4g} ± {st['uncertainty'] / f3:.2g} {u3}"
            parts.append(f"<p>{_e(tr(f'台阶 {i}：'))}{_e(txt)}</p>")
        for w in tilt.warnings:
            parts.append(f'<div class="warn">{_e(tr(w))}</div>')
    parts.append(f'<p class="muted" style="margin-top:32px">{_e(tr("由 Vexel 生成。数值来自记录中的标定、图像处理和测量设置，重新打开同一图像可以复现。"))}</p>')
    parts.append("</body></html>")
    return "".join(parts)


def write_report(ms: Any, image: np.ndarray, path: str | Path, *, tilt: Any = None, result: Any = None) -> Path:
    path = Path(path)
    if path.suffix.lower() not in (".html", ".htm"):
        path = path.with_suffix(".html")
    path.write_text(build_report(ms, image, tilt=tilt, result=result), encoding="utf-8")
    return path


__all__ = ["build_report", "svg_histogram", "write_report"]
