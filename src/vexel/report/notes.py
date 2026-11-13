"""Plain-language notes about a run, shared by the desktop UI and the HTML report."""

from __future__ import annotations

import math
from typing import Any


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:.{digits}g}"


def pick_summaries(summaries: dict[str, Any]) -> dict[str, dict | None]:
    """Find each kind of summary in ``summary.json`` by its fields, not by step id."""

    def find(key: str) -> dict | None:
        return next((v for v in summaries.values() if isinstance(v, dict) and key in v), None)

    return {
        "droplets": find("count_line_px"),
        "encapsulation": find("frac_empty"),
        "fluorescence": find("positive_fraction"),
        "flags": find("n_flagged"),
    }


def run_notes(summaries: dict[str, Any], provenance: dict[str, Any], preview: bool = False) -> list[str]:
    parts = pick_summaries(summaries)
    s, enc, fluo = parts["droplets"] or {}, parts["encapsulation"], parts["fluorescence"]
    notes: list[str] = []
    if preview:
        notes.append(f"这是预览：只分析了前 {provenance['input'].get('frames_used')} 帧，结果不代表全部数据，也不会保存。")
    if provenance.get("status") == "stopped_by_qc":
        notes.append("运行被质控规则中止，结果不完整。")
    if s and not s.get("calibrated", True):
        notes.append("未标定：长度单位为像素，没有体积与浓度。")
    if s and s.get("volume_model") == "sphere":
        notes.append("液滴体积按球形估计；液滴被流道压扁时，请填写流道高度，否则体积和浓度偏大。")
    if s.get("n_droplets") == 0:
        notes.append("没有检测到液滴。先看复核图，再调整分割参数（最低对比度、最小面积）。")
    if enc is not None and enc.get("n_droplets"):
        if enc.get("chi2_p") is not None and enc["chi2_p"] < 0.05:
            d = enc.get("dispersion_index")
            trend = "，D > 1 提示细胞聚集或上样不均" if d and d > 1 else ("，D < 1 提示排布比随机更规则" if d else "")
            notes.append(f"包裹分布与泊松分布差异显著（p = {enc['chi2_p']:.3g}，离散指数 D = {_fmt(d, 3)}{trend}）。")
        if enc.get("n_low_agreement"):
            notes.append(
                f"有 {enc['n_low_agreement']} 个液滴的细胞数在各帧之间不一致，计数可能不准，"
                "请在复核图中检查黄圈，必要时调整细胞直径或细胞最低对比度。"
            )
        if enc.get("n_excluded"):
            notes.append(f"有 {enc['n_excluded']} 个液滴无法计数（内部区域太小），未计入包裹统计。")
    if fluo is not None and fluo.get("n_droplets"):
        if fluo.get("populations") == 1:
            notes.append(
                f"荧光只发现一群液滴（未见明显的阴阳两群），已按离群值规则判读（阈值 {_fmt(fluo.get('threshold'))}）。"
                "若样本应含阳性，或全部液滴都是阳性，请用对照样本确定阈值并改用固定阈值。"
            )
        elif fluo.get("saturated"):
            notes.append("所有液滴均为阳性，无法估计每滴拷贝数（已饱和），建议稀释后重测。")
        d = fluo.get("separation_d")
        if fluo.get("threshold_method") == "mixture" and d is not None and d < 4:
            notes.append(f"阴阳两群有重叠（分离度 D = {d:.2f}），靠近阈值的液滴可能判错，请对照散点图确认。")
    for step in provenance.get("steps", []):
        for v in step.get("qc_violations", []):
            bounds = "，".join(f"{k} {v[k]:g}" for k in ("min", "max") if v.get(k) is not None)
            value = "缺失" if v.get("value") is None else f"{v['value']:.4g}"
            notes.append(f"质控提示：{step['id']}.{v['metric']} = {value}（要求 {bounds}），请对照复核图确认。")
    return notes
