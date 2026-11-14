"""Single-file HTML report of a run: results, charts, review images, methods, provenance.

Everything (styles, charts, images) is embedded, so the file can be e-mailed or archived
on its own. Charts are inline SVG drawn here, without a plotting library.
"""

from __future__ import annotations

import base64
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vexel.report.notes import pick_summaries, run_notes

REPORT_NAME = "report.html"
_UNIT = {"um": "µm", "px": "px"}
_REASON_SHORT = {
    "low_circularity": "形状不圆",
    "size_outlier": "粒径异常",
    "count_unstable": "计数不稳",
    "near_threshold": "近阈值",
    "few_frames": "帧数少",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.{digits}g}"
    return str(value)


def _pct(value: Any) -> str:
    return "–" if value is None else f"{100 * value:.1f} %"


def _png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _table_with(run_dir: Path, columns: set[str]) -> pd.DataFrame | None:
    for path in sorted((run_dir / "tables").glob("*.csv")):
        header = set(path.open(encoding="utf-8").readline().strip().split(","))
        if columns <= header:
            return pd.read_csv(path)
    return None


# -- charts (inline SVG) ---------------------------------------------------------------------

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 480, 230, 44, 12, 26, 34


def _svg(body: str, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_e(label)}" xmlns="http://www.w3.org/2000/svg">'
        f"{body}</svg>"
    )


def _text(x: float, y: float, s: str, anchor: str = "start", cls: str = "axis") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{_e(s)}</text>'


def svg_histogram(values: np.ndarray, unit: str) -> str:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return _svg(_text(W / 2, H / 2, "没有数据", "middle"), "粒径分布")
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    bins = int(np.clip(np.sqrt(v.size) * 1.5, 5, 30))
    counts, _ = np.histogram(v, bins=bins, range=(lo, hi))
    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T, H - PAD_B
    bw = (x1 - x0) / bins
    parts = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="grid"/>']
    for i, c in enumerate(counts):
        h = (y1 - y0) * c / counts.max()
        parts.append(f'<rect x="{x0 + i * bw + 1:.1f}" y="{y1 - h:.1f}" width="{max(bw - 2, 1):.1f}" height="{h:.1f}" rx="2" class="bar"/>')
    mean = float(v.mean())
    mx = x0 + (mean - lo) / (hi - lo) * (x1 - x0)
    parts.append(f'<line x1="{mx:.1f}" y1="{y0}" x2="{mx:.1f}" y2="{y1}" class="mean"/>')
    parts.append(_text(mx + (6 if mx < W - 150 else -6), y0 - 8, f"均值 {mean:.4g} {unit}", "start" if mx < W - 150 else "end", "label"))
    parts.append(_text(x0, H - 12, f"{lo:.4g}"))
    parts.append(_text(x1, H - 12, f"{hi:.4g} {unit}", "end"))
    parts.append(_text((x0 + x1) / 2, H - 12, f"n = {v.size}", "middle"))
    parts.append(_text(x0 - 6, y0 + 4, str(int(counts.max())), "end"))
    parts.append(_text(x0 - 6, y1, "0", "end"))
    return _svg("".join(parts), "粒径分布")


def svg_counts(dist: pd.DataFrame, max_classes: int = 8) -> str:
    if dist is None or dist.empty:
        return _svg(_text(W / 2, H / 2, "没有数据", "middle"), "包裹数分布")
    k = dist["k"].to_numpy(dtype=int)
    obs = dist["observed_fraction"].to_numpy(dtype=float)
    exp = dist["poisson_fraction"].to_numpy(dtype=float)
    folded = k.size > max_classes
    if folded:
        obs = np.append(obs[: max_classes - 1], obs[max_classes - 1 :].sum())
        exp = np.append(exp[: max_classes - 1], exp[max_classes - 1 :].sum())
        k = k[:max_classes]
    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T + 14, H - PAD_B
    top = max(obs.max(), exp.max(), 1e-9)
    slot = (x1 - x0) / k.size
    bar = min(26.0, slot * 0.36)
    parts = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="grid"/>']
    for i, kk in enumerate(k):
        cx = x0 + (i + 0.5) * slot
        ho, he = (y1 - y0) * obs[i] / top, (y1 - y0) * exp[i] / top
        parts.append(f'<rect x="{cx - bar - 1:.1f}" y="{y1 - ho:.1f}" width="{bar:.1f}" height="{ho:.1f}" rx="2" class="bar"/>')
        parts.append(f'<rect x="{cx + 1:.1f}" y="{y1 - he:.1f}" width="{bar:.1f}" height="{he:.1f}" rx="2" class="ref"/>')
        parts.append(_text(cx - bar / 2, y1 - ho - 5, f"{100 * obs[i]:.0f}%", "middle", "label"))
        parts.append(_text(cx, H - 12, f"≥{kk}" if folded and i == k.size - 1 else str(kk), "middle"))
    parts.append(f'<rect x="{x0}" y="6" width="10" height="10" rx="2" class="bar"/>')
    parts.append(_text(x0 + 14, 15, "实测", "start", "label"))
    parts.append(f'<rect x="{x0 + 56}" y="6" width="10" height="10" rx="2" class="ref"/>')
    parts.append(_text(x0 + 70, 15, "泊松预期", "start", "label"))
    parts.append(_text(x1, 15, "横轴：每滴细胞数", "end"))
    return _svg("".join(parts), "包裹数分布")


def svg_scatter(x: np.ndarray, y: np.ndarray, positive: np.ndarray, threshold: float | None, unit: str) -> str:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, positive = x[keep], y[keep], np.asarray(positive, dtype=bool)[keep]
    if x.size == 0:
        return _svg(_text(W / 2, H / 2, "没有数据", "middle"), "粒径与荧光")
    ys = y if threshold is None else np.append(y, threshold)
    xa, xb, ya, yb = float(x.min()), float(x.max()), float(ys.min()), float(ys.max())
    px, py = max((xb - xa) * 0.05, 1e-6), max((yb - ya) * 0.05, 1e-6)
    xa, xb, ya, yb = xa - px, xb + px, ya - py, yb + py
    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T, H - PAD_B

    def X(v: float) -> float:
        return x0 + (v - xa) / (xb - xa) * (x1 - x0)

    def Y(v: float) -> float:
        return y1 - (v - ya) / (yb - ya) * (y1 - y0)

    parts = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="grid"/>', f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="grid"/>']
    for xv, yv, pos in zip(x, y, positive):
        parts.append(f'<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="2.6" class="{"pos" if pos else "neg"}"/>')
    if threshold is not None:
        ty = Y(threshold)
        parts.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x1}" y2="{ty:.1f}" class="mean"/>')
        parts.append(_text(x1, ty - 5, f"阈值 {threshold:.4g}", "end", "label"))
    parts.append(_text(x0, 14, "纵轴：荧光强度", "start"))
    parts.append(_text(x0 - 6, y0 + 4, f"{yb:.3g}", "end"))
    parts.append(_text(x0 - 6, y1, f"{ya:.3g}", "end"))
    parts.append(_text(x0, H - 12, f"{xa:.4g}"))
    parts.append(_text(x1, H - 12, f"{xb:.4g} {unit}", "end"))
    parts.append(_text((x0 + x1) / 2, H - 12, "横轴：粒径", "middle"))
    return _svg("".join(parts), "粒径与荧光")


# -- methods ----------------------------------------------------------------------------------


def svg_series(x: np.ndarray, y: np.ndarray, unit: str, label: str, marks: list[float] | None = None) -> str:
    """A rolling statistic over the stream; vertical marks at alarm frames."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return _svg(_text(W / 2, H / 2, "没有数据", "middle"), label)
    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T, H - PAD_B
    xlo, xhi = float(x.min()), float(x.max()) if x.max() > x.min() else float(x.min()) + 1
    ylo, yhi = float(y.min()), float(y.max())
    pad = (yhi - ylo) * 0.1 or abs(yhi) * 0.05 or 1.0
    ylo, yhi = ylo - pad, yhi + pad

    def px(v: float) -> float:
        return x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)

    def py(v: float) -> float:
        return y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)

    parts = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="grid"/>']
    for m in marks or []:
        if xlo <= m <= xhi:
            parts.append(f'<line x1="{px(m):.1f}" y1="{y0}" x2="{px(m):.1f}" y2="{y1}" class="threshold"/>')
    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y))
    parts.append(f'<polyline points="{pts}" fill="none" class="line"/>')
    parts += [f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.5" class="dot"/>' for a, b in zip(x, y)]
    parts.append(_text(x0, H - 12, f"第 {xlo:.0f} 帧"))
    parts.append(_text(x1, H - 12, f"第 {xhi:.0f} 帧", "end"))
    parts.append(_text(x0 - 6, y0 + 4, f"{yhi:.4g}", "end"))
    parts.append(_text(x0 - 6, y1, f"{ylo:.4g}", "end"))
    if unit:
        parts.append(_text(x0, y0 - 8, unit, "start", "label"))
    return _svg("".join(parts), label)


def stream_section(run_dir: Path, prov: dict[str, Any], unit: str) -> list[str]:
    """Throughput, frame drops, rolling statistics and alarms of a stream run."""
    st = prov.get("stream") or {}
    src = st.get("source") or {}
    speed = src.get("speed")
    pace = "不限速（测软件最大处理速度）" if speed is None else f"按原始帧率的 {speed:g} 倍回放"
    out = [f'<h2>在线回放</h2><p class="meta">{_e(pace)}。计时不含相机读出与传输，是软件本身的速度，不是端到端延迟。</p><div class="cards">']
    out.append(_card("处理速度", f"{_fmt(st.get('processing_fps'), 3)} 帧/秒", f"共 {_fmt(st.get('wall_s'), 3)} 秒"))
    dropped = st.get("frames_dropped") or 0
    detail = f"收到 {st.get('frames_received')} · 处理 {st.get('frames_processed')}"
    if dropped:
        detail += f"（抽样跳过 {st.get('frames_sampled_out', 0)}，队列满丢弃 {st.get('frames_lost_queue_full', 0)}）"
    out.append(_card("丢帧", str(dropped), detail))
    lat = st.get("latency_ms")
    if lat:
        out.append(_card("延迟", f"{lat['median']:.0f} ms", f"p95 {lat['p95']:.0f} ms · 最大 {lat['max']:.0f} ms"))
    out.append(_card("警报", str(st.get("n_alarms", 0)), f"滚动统计 {st.get('n_rolling_updates', 0)} 次"))
    out.append("</div>")
    rolling = pd.read_csv(run_dir / "stream" / "rolling.csv") if (run_dir / "stream" / "rolling.csv").exists() else None
    alarms = pd.read_csv(run_dir / "stream" / "alarms.csv") if (run_dir / "stream" / "alarms.csv").exists() else None
    marks = alarms["frame"].tolist() if alarms is not None and not alarms.empty else []
    if rolling is not None and not rolling.empty:
        charts = []
        for col, label, u in (("droplets.diameter_mean", "滚动平均粒径", unit), ("droplets.frequency", "滚动生成频率", ""),
                              ("encapsulation.lambda", "滚动 λ", ""), ("positivity.positive_fraction", "滚动阳性比例", "")):  # fmt: skip
            if col in rolling and rolling[col].notna().any():
                charts.append((label, svg_series(rolling["frame"].to_numpy(), rolling[col].to_numpy(), u, label, marks)))
        if charts:
            out.append('<div class="charts">' + "".join(f"<figure><h3>{_e(t)}</h3>{svg}</figure>" for t, svg in charts) + "</div>")
    if alarms is not None and not alarms.empty:
        out.append("<h3>警报</h3><table><tr><th>帧</th><th>类型</th><th>内容</th><th>画面</th></tr>")
        for r in alarms.itertuples(index=False):
            img = run_dir / "stream" / str(r.image) if isinstance(r.image, str) and r.image else None
            pic = f'<img class="event" src="{_png(img)}" alt="第 {int(r.frame)} 帧">' if img is not None and img.exists() else ""
            kind = {"drift": "漂移", "qc": "质控", "throughput": "处理能力"}.get(r.kind, r.kind)
            out.append(f'<tr><td class="num">{int(r.frame)}</td><td>{_e(kind)}</td><td>{_e(r.message)}</td><td>{pic}</td></tr>')
        out.append("</table>")
    return out


_EDIT_TEXT = {"labels.delete": "删除轮廓", "labels.merge": "合并", "labels.paint": "补画", "labels.erase": "擦除",
              "table.set": "修改", "table.exclude": "排除"}  # fmt: skip


def corrections_section(c: dict[str, Any]) -> list[str]:
    edits = c.get("edits", [])
    out = [f'<h2>人工修正</h2><p class="meta">本结果按 {c.get("n_edits", 0)} 处人工修正重新计算，其中 {c.get("n_applied", 0)} 处生效。'
           f"修改记录与原结果放在一起（corrections.json，SHA-256 {_e(str(c.get('sha256', ''))[:16])}…），原结果未改动。</p>"]  # fmt: skip
    if edits:
        out.append("<table><tr><th>帧</th><th>修改</th><th>对象</th><th>内容</th><th>结果</th></tr>")
        for item in edits:
            e = item["edit"]
            what = "，".join(f"{k} = {v}" for k, v in (e.get("values") or {}).items()) or (e.get("note") or "")
            ok = "生效" if item["status"] == "applied" else f"未生效：{item['status']}"
            out.append(f'<tr><td class="num">{e["frame"]}</td><td>{_e(_EDIT_TEXT.get(e["kind"], e["kind"]))}</td>'
                       f"<td><code>{_e(e['target'])}</code></td><td>{_e(what)}</td><td>{_e(ok)}</td></tr>")  # fmt: skip
        out.append("</table>")
    return out


def methods_text(recipe: dict[str, Any], provenance: dict[str, Any]) -> list[str]:
    """A methods paragraph per step, generated from the frozen recipe."""
    out: list[str] = []
    inp = recipe.get("input", {})
    channels = inp.get("channels", {})
    if channels:
        names = "，".join(f"{k} 为通道 {v}" for k, v in channels.items())
        out.append(f"液滴在通道 {inp.get('channel', 0)} 中分割；{names}（通道号从 0 起）。")
    for step in recipe.get("steps", []):
        p, op = step.get("params", {}), step["op"]
        if op == "preprocess.static_background":
            out.append(f"以均匀抽取的 {p['max_frames']} 帧的时间中值估计静态背景（流道壁、照明不均），逐帧与背景作差（{p['mode']}）。")
        elif op == "segment.threshold_watershed":
            thr = f"固定阈值 {p['threshold']}" if p["method"] == "fixed" else f"{p['method']} 自动阈值（下限 {p['min_threshold']}）"
            out.append(
                f"经 σ = {p['smooth_sigma']} 像素的高斯平滑后按{thr}分割，填充孔洞，以半径 {p['opening_radius']} 像素作开运算，"
                f"去除面积小于 {p['min_area']} 像素的目标，并以距离变换分水岭分离相互接触的液滴（最小中心距 {p['min_distance']} 像素）。"
            )
        elif op == "segment.circles":
            if p.get("min_diameter_px") and p.get("max_diameter_px"):
                rng = f"直径 {p['min_diameter_px']}–{p['max_diameter_px']} 像素"
            else:
                rng = "按每帧液滴典型大小自动确定的直径范围"
            out.append(
                f"在原始图像上以 Canny 算子（σ = {p['edge_sigma']} 像素）提取边缘，只保留与背景差异超过阈值"
                f"（{p['motion_threshold']}，按噪声自适应抬高）的区域；以 Hough 变换在{rng}内检测圆，"
                f"要求圆周至少 {p['min_score']:.0%} 落在边缘上，重叠的圆保留得分高者；"
                "圆心以边缘点最小二乘拟合精修，半径取液滴外缘（对比度径向剖面下降最快处）。"
            )
        elif op == "measure.regionprops":
            out.append("测量每个液滴的质心、面积、等效面积直径与圆度（4πA/P²）。")
        elif op == "droplet.inclusions":
            thr = "取每帧由液滴内部估计的噪声的 " + f"{p['noise_factor']} 倍" if p.get("min_contrast") is None else f"固定为 {p['min_contrast']}"
            look = "暗" if p["polarity"] == "dark" else "亮"
            if p.get("background") == "radial":
                bg = (
                    f"以该液滴自身的径向灰度剖面为背景（中心区取中位灰度，外圈逐个半径取中位数），并按 8 个扇区排除液滴的轮廓暗环"
                    f"（另距边缘 {p['edge_margin_px']} 像素以内不计）"
                )
            else:
                bg = f"以该液滴的中位灰度为背景（距边缘 {p['edge_margin_px']} 像素以内不计）"
            floor = (
                f"；同一帧中有明显细胞时，另要求达到其中位对比度的 {p['relative_floor']:.0%}"
                if p.get("relative_floor")
                else ""
            )
            out.append(
                f"在每个液滴内部{bg}，将比背景{look}、对比度超过阈值（{thr}{floor}）的局部极值记为细胞"
                f"（细胞直径设为 {p['cell_diameter_px']} 像素）；被画面边缘截断的液滴不计，"
                "每个液滴取其完整出现在视场内各帧计数的众数。"
            )
        elif op == "droplet.intensity":
            bg = "减去同一帧液滴外像素的中位数作为背景，" if p["background"] == "outside" else ""
            out.append(f"在荧光通道中{bg}取液滴内部（距边缘 {p['edge_margin_px']} 像素以内不计）的平均荧光，每个液滴取各完整帧的中位数。")
        elif op == "track.link":
            flow = "先由各帧检测位置估计逐帧整体流动位移并扣除，再" if p.get("estimate_shift") else ""
            out.append(f"{flow}以线性指派（laptrack）跨帧关联液滴，帧间最大位移 {p['max_distance_px']} 像素，允许缺失 {p['max_gap_frames']} 帧。")
        elif op == "droplet.summary":
            vol = (
                f"液滴体积按流道高度 {p['channel_height_um']} µm 换算（直径大于流道高度者按压扁形状：中央圆柱加半圆环边）"
                if p.get("channel_height_um")
                else "液滴体积按球形换算"
            )
            out.append(
                f"每个液滴在其轨迹越过视场 {p['count_line'] * 100:.0f}% 处的虚拟计数线时计数一次；粒径取其完整出现在视场内各帧的中位数；"
                f"生成频率为（液滴数 − 1）除以首末液滴越线的时间差；{vol}。"
            )
        elif op == "droplet.poisson":
            out.append(
                "以平均每滴细胞数 λ（最大似然估计，95% 精确区间）为参数与泊松分布比较，"
                "并报告离散指数（方差/均值）与卡方拟合优度检验（相邻类别合并至预期数不少于 5）。"
            )
        elif op == "droplet.positivity":
            method = {
                "auto": f"自动判读：仅当两成分高斯混合模型的 BIC 比单成分低 {p['min_bic_gain']} 以上、两群间密度低谷比小于 {p['max_dip_ratio']}、"
                f"Ashman's D 不小于 {p['min_separation']} 时，以两成分概率相等处为阈值；否则以中位数加 {p['outlier_k']} 倍稳健标准差为阈值",
                "fixed": f"固定阈值 {p['threshold']}",
                "otsu": "精确大津法",
                "outlier": f"中位数加 {p['outlier_k']} 倍稳健标准差",
            }[p["method"]]
            out.append(
                f"以 {p['column']} 判定阴阳性（{method}）。阳性比例 p 的置信区间为 Wilson 95% 区间，"
                "每滴平均拷贝数按 λ = −ln(1 − p) 计算，浓度为 λ 除以平均液滴体积。"
            )
    env = provenance.get("environment", {})
    pk = env.get("packages", {})
    libs = "、".join(f"{k} {pk[k]}" for k in ("numpy", "scipy", "scikit-image", "laptrack", "scikit-learn") if k in pk)
    out.append(f"分析使用 Vexel {env.get('vexel', '')}（Python {env.get('python', '')}；{libs}），配方 SHA-256 为 {provenance.get('recipe', {}).get('sha256', '')[:16]}…。")
    corr = provenance.get("corrections")
    if corr:
        out.append(
            f"自动分析结果经人工复核：在复核界面中对分割轮廓与逐液滴结果做了 {corr.get('n_applied', 0)} 处修正"
            "（删除误检、补画漏检、修改细胞数或阴阳性、排除杂质等），并按修正重新运行同一配方；修改记录随结果保存，可复现。"
        )
    return out


# -- page ---------------------------------------------------------------------------------------

CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#6b7280; --line:#e5e7eb; --card:#f8f9fb; --accent:#2563eb; --ref:#b8bec7; --warn-bg:#fff6e0; --warn-fg:#7a4b00; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181d; --fg:#e6e8eb; --muted:#9aa1ab; --line:#2c3038; --card:#1d2027; --accent:#5b8cff; --ref:#5d636e; --warn-bg:#3a2f12; --warn-fg:#f3d38a; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
main { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:36px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line); }
h3 { font-size:15px; margin:20px 0 8px; }
.meta { color:var(--muted); font-size:13px; }
.notes { background:var(--warn-bg); color:var(--warn-fg); border-radius:8px; padding:10px 14px; margin:16px 0; }
.notes li { margin:2px 0; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.card .t { color:var(--muted); font-size:13px; } .card .v { font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; } .card .d { color:var(--muted); font-size:12px; }
.charts { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:20px; }
svg { width:100%; height:auto; } svg .bar { fill:var(--accent); } svg .ref { fill:var(--ref); } svg .grid { stroke:var(--line); }
svg .mean { stroke:var(--fg); stroke-dasharray:4 3; } svg .axis { fill:var(--muted); font-size:11px; } svg .label { fill:var(--fg); font-size:11px; }
svg .pos { fill:var(--accent); fill-opacity:.8; } svg .neg { fill:var(--ref); fill-opacity:.8; }
svg .line { stroke:var(--accent); stroke-width:2; } svg .dot { fill:var(--accent); } svg .threshold { stroke:var(--warn-fg); stroke-dasharray:3 3; }
img.event { width:220px; max-width:100%; border-radius:6px; image-rendering:pixelated; }
.review img { width:100%; border-radius:6px; margin:4px 0 2px; image-rendering:pixelated; } .cap { color:var(--muted); font-size:12px; margin-bottom:10px; }
.gallery { display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:10px; }
.gallery figure { margin:0; } .gallery img { width:100%; border-radius:6px; image-rendering:pixelated; } .gallery figcaption { font-size:12px; color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:13px; } td, th { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:500; } td.num { font-variant-numeric:tabular-nums; } code { font-size:12px; word-break:break-all; }
.methods p { margin:6px 0; }
footer { margin-top:40px; color:var(--muted); font-size:12px; }
@media print { body { background:#fff; color:#000; } main { padding:0; } h2 { break-after:avoid; } .card, figure, svg { break-inside:avoid; } }
"""


def _card(title: str, value: str, detail: str = "") -> str:
    return f'<div class="card"><div class="t">{_e(title)}</div><div class="v">{_e(value)}</div><div class="d">{_e(detail)}</div></div>'


def build_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    summaries = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    recipe = json.loads((run_dir / "recipe.json").read_text(encoding="utf-8"))
    parts = pick_summaries(summaries)
    s, enc, fluo, flg = parts["droplets"] or {}, parts["encapsulation"], parts["fluorescence"], parts["flags"]
    unit = _UNIT.get(s.get("length_unit", "px"), s.get("length_unit", "px"))
    inp = prov.get("input", {})
    source_name = Path(str(inp["source"])).name if inp.get("source") else "内存中的数组"
    status = {"completed": "已完成", "stopped_by_qc": "被质控规则中止", "stopped": "已停止"}.get(prov.get("status"), prov.get("status"))
    if prov.get("mode") == "stream":
        status = f"在线回放，{status}"
    try:  # stored in UTC; shown in the reader's local time
        started = datetime.fromisoformat(prov["started_at"]).astimezone().strftime("%Y-%m-%d %H:%M")
    except (KeyError, ValueError):
        started = prov.get("started_at", "")

    body: list[str] = [
        "<header>",
        "<h1>液滴分析报告</h1>",
        f'<div class="meta">{_e(source_name)} · 配方 {_e(recipe.get("name", ""))} · {_e(status)} · 分析时间 {_e(started)}</div>',
        "</header>",
    ]
    notes = run_notes(summaries, prov)
    if notes:
        body.append('<div class="notes"><ul>' + "".join(f"<li>{_e(n)}</li>" for n in notes) + "</ul></div>")

    # results
    body.append("<h2>结果</h2><h3>液滴</h3><div class=\"cards\">")
    freq_unit = "次/帧" if s.get("frequency_unit") == "1/frame" else s.get("frequency_unit", "")
    body += [
        _card("液滴数", _fmt(s.get("n_droplets")), f"共追踪到 {_fmt(s.get('n_tracks_total'))} 条轨迹"),
        _card("平均粒径", f"{_fmt(s.get('diameter_mean'))} {unit}", f"标准差 {_fmt(s.get('diameter_std'), 3)} · 中位数 {_fmt(s.get('diameter_median'))}"),
        _card("粒径 CV", "–" if s.get("diameter_cv_percent") is None else f"{s['diameter_cv_percent']:.2f} %", f"范围 {_fmt(s.get('diameter_min'))} – {_fmt(s.get('diameter_max'))} {unit}"),
        _card("生成频率", f"{_fmt(s.get('frequency'))} {freq_unit}", f"平均间距 {_fmt(s.get('spacing_mean'))} {unit}"),
    ]
    if s.get("volume_mean_pl") is not None:
        model = {"sphere": "按球形", "flattened": f"按压扁形状，流道高 {_fmt(s.get('channel_height_um'))} µm"}.get(s.get("volume_model"), "")
        body.append(_card("平均体积", f"{_fmt(s['volume_mean_pl'], 3)} pL", f"CV {_fmt(s.get('volume_cv_percent'), 3)} % · {model}"))
    body.append("</div>")
    if enc is not None:
        body.append('<h3>包裹</h3><div class="cards">')
        body += [
            _card("平均每滴细胞数 λ", _fmt(enc.get("lambda"), 3), f"95% 区间 {_fmt(enc.get('lambda_ci_low'), 3)} – {_fmt(enc.get('lambda_ci_high'), 3)}"),
            _card("空滴", _pct(enc.get("frac_empty")), f"泊松预期 {_pct(enc.get('poisson_empty'))}"),
            _card("单包裹", _pct(enc.get("frac_single")), f"泊松预期 {_pct(enc.get('poisson_single'))} · 有细胞液滴中单包裹 {_pct(enc.get('single_among_occupied'))}"),
            _card("多包裹", _pct(enc.get("frac_multi")), f"泊松预期 {_pct(enc.get('poisson_multi'))}"),
            _card("离散指数 D", _fmt(enc.get("dispersion_index"), 3), f"卡方检验 p = {_fmt(enc.get('chi2_p'), 3)}"),
        ]
        if enc.get("inclusions_per_ul") is not None:
            body.append(_card("细胞浓度（个/µL）", _fmt(enc["inclusions_per_ul"]), f"95% 区间 {_fmt(enc.get('inclusions_per_ul_ci_low'))} – {_fmt(enc.get('inclusions_per_ul_ci_high'))}"))
        body.append("</div>")
    if fluo is not None:
        method = {"mixture": "自动分群", "outlier": "离群值规则", "fixed": "固定阈值", "otsu": "大津法"}.get(fluo.get("threshold_method"), "")
        body.append('<h3>荧光</h3><div class="cards">')
        body += [
            _card("阳性液滴", _pct(fluo.get("positive_fraction")), f"{_fmt(fluo.get('n_positive'))} / {_fmt(fluo.get('n_droplets'))} · 95% 区间 {_pct(fluo.get('positive_fraction_ci_low'))} – {_pct(fluo.get('positive_fraction_ci_high'))}"),
            _card("每滴平均拷贝数", _fmt(fluo.get("copies_per_droplet"), 3), f"95% 区间 {_fmt(fluo.get('copies_ci_low'), 3)} – {_fmt(fluo.get('copies_ci_high'), 3)}"),
            _card("浓度（拷贝/µL）", _fmt(fluo.get("copies_per_ul")), f"95% 区间 {_fmt(fluo.get('copies_per_ul_ci_low'))} – {_fmt(fluo.get('copies_per_ul_ci_high'))}"),
            _card("阴阳分离度 D", _fmt(fluo.get("separation_d"), 3), f"{method} · 阈值 {_fmt(fluo.get('threshold'))}"),
        ]
        body.append("</div>")

    # charts
    charts = []
    droplets = _table_with(run_dir, {"diameter", "crossing_frame"})
    charts.append(("粒径分布", svg_histogram(droplets["diameter"].to_numpy() if droplets is not None else np.empty(0), unit)))
    if enc is not None:
        charts.append(("包裹数分布", svg_counts(_table_with(run_dir, {"k", "observed_fraction", "poisson_fraction"}))))
    if fluo is not None:
        t = _table_with(run_dir, {"positive", "diameter", fluo.get("column", "fluo_mean")})
        if t is not None:
            pos = t["positive"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
            charts.append(("粒径与荧光", svg_scatter(t["diameter"].to_numpy(), t[fluo.get("column", "fluo_mean")].to_numpy(), pos, fluo.get("threshold"), unit)))
    body.append('<h2>图表</h2><div class="charts">' + "".join(f"<figure><h3>{_e(t)}</h3>{svg}</figure>" for t, svg in charts) + "</div>")

    # review
    review = sorted((run_dir / "review").glob("*.png"))
    if review:
        legend = "红线为液滴轮廓" + ("，黄圈为检出的细胞" if enc is not None else "")
        body.append(f'<h2>复核图</h2><p class="meta">均匀抽取的 {len(review)} 帧。{_e(legend)}。</p><div class="review">')
        for p in review:
            frame = p.stem.split("_")[-1].lstrip("0") or "0"
            body.append(f'<img src="{_png(p)}" alt="第 {_e(frame)} 帧"><div class="cap">第 {_e(frame)} 帧</div>')
        body.append("</div>")
    flags_table = _table_with(run_dir, {"reason_text", "track_id"})
    crops = run_dir / "review" / "flags"
    if flg is not None:
        body.append(f'<h2>可疑液滴</h2><p class="meta">{_fmt(flg.get("n_flagged"))} / {_fmt(flg.get("n_droplets"))} 个液滴需要人工确认。')
        if flags_table is not None and not flags_table.empty and crops.is_dir():
            shown = [r for r in flags_table.itertuples(index=False) if (crops / f"track{int(r.track_id):05d}_frame{int(r.frame):06d}.png").exists()]
            if len(shown) < len(flags_table):
                body.append(f"下面显示前 {len(shown)} 个。")
            body.append('</p><div class="gallery">')
            for r in shown:
                img = crops / f"track{int(r.track_id):05d}_frame{int(r.frame):06d}.png"
                short = "/".join(_REASON_SHORT.get(c, c) for c in str(r.reasons).split(";"))
                body.append(
                    f'<figure><img src="{_png(img)}" alt="{_e(r.reason_text)}" title="{_e(r.reason_text)}">'
                    f"<figcaption>轨迹 {int(r.track_id)} · 第 {int(r.frame)} 帧<br>{_e(short)}</figcaption></figure>"
                )
            body.append("</div>")
        else:
            body.append("</p>")

    if prov.get("mode") == "stream":
        body.extend(stream_section(run_dir, prov, unit))
    if prov.get("corrections"):
        body.extend(corrections_section(prov["corrections"]))

    # methods
    body.append('<h2>方法</h2><div class="methods">' + "".join(f"<p>{_e(t)}</p>" for t in methods_text(recipe, prov)) + "</div>")

    # provenance
    sp = inp.get("spacing", {})
    rows = [
        ("输入文件", inp.get("source") or "内存中的数组（无文件）"),
        ("SHA-256", inp.get("sha256", "未计算")),
        ("大小", f"{_fmt(inp.get('bytes'))} 字节" if inp.get("bytes") else "–"),
        ("尺寸 (TCZYX)", " × ".join(str(x) for x in inp.get("shape", []))),
        ("分析帧数", inp.get("frames_used")),
        ("像素尺寸", f"{_fmt(sp.get('x'))} {inp.get('space_unit', '')}"),
        ("帧间隔", f"{_fmt(sp.get('t'))} {inp.get('time_unit', '')}"),
        ("通道", f"分割通道 {inp.get('channel')}" + "".join(f"，{k} = {v}" for k, v in (inp.get("channels") or {}).items())),
        ("配方", f"{recipe.get('name')}（SHA-256 {prov.get('recipe', {}).get('sha256', '')}）"),
        ("结果摘要", prov.get("result_digest")),
        ("运行时长", f"{_fmt(prov.get('duration_s'))} 秒"),
        ("设备", prov.get("device")),
    ]
    body.append("<h2>溯源</h2><table>" + "".join(f"<tr><th>{_e(k)}</th><td><code>{_e(v)}</code></td></tr>" for k, v in rows) + "</table>")
    body.append("<h3>各步骤</h3><table><tr><th>步骤</th><th>算子</th><th>耗时</th><th>质控指标</th></tr>")
    for st in prov.get("steps", []):
        qc = "；".join(f"{k} = {_fmt(v)}" for k, v in st.get("qc", {}).items() if not k.endswith((".min", ".max")))
        bad = "".join(f"<br><strong>超限：{_e(v['metric'])}</strong>" for v in st.get("qc_violations", []))
        body.append(f"<tr><td>{_e(st['id'])}</td><td><code>{_e(st['op'])}</code></td><td class=\"num\">{_fmt(st.get('duration_s'))} s</td><td>{_e(qc)}{bad}</td></tr>")
    body.append("</table>")
    env = prov.get("environment", {})
    pk = env.get("packages", {})
    body.append(
        "<h3>软件</h3><table>"
        + f"<tr><th>Vexel</th><td>{_e(env.get('vexel'))}</td></tr><tr><th>Python</th><td>{_e(env.get('python'))}（{_e(env.get('platform'))}）</td></tr>"
        + "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in pk.items())
        + "</table>"
    )
    body.append(f'<footer>由 Vexel 生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}。完整数据见同一文件夹下的 tables/、summary.json 与 provenance.json。</footer>')

    title = f"液滴分析报告 · {source_name}"
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>" + "".join(body) + "</main></body></html>\n"
    )


def write_report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    path = Path(out) if out else run_dir / REPORT_NAME
    path.write_text(build_report(run_dir), encoding="utf-8")
    return path
