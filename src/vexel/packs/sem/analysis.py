"""Automatic measurements (M4): recipes run on a region of the image, kept in the
measurement record and recomputed like manual measurements.

A recipe (``RECIPES``) turns a region of the corrected image into calibrated values:

==========  ===============================================================================
recipe      reports
==========  ===============================================================================
lines       line/space: CD (line width), space, pitch, LER and LWR (3σ, with and without
            the noise bias), correlation length (``metrology.lines_spaces``)
holes       contact holes and vias: diameter, ellipse, roundness, CD uniformity, the
            array's pitch and missing holes (``objects.holes``)
particles   particles and defects: equivalent and Feret diameters, circularity, aspect
            ratio, size distribution by number and volume (``objects.particles``)
layers      cross-section layer stack: each interface and each layer's thickness and
            uniformity (``metrology.layers``)
profile     one line or trench in cross-section: height, top/middle/bottom width,
            sidewall angles (``metrology.feature_profile``)
tip         apex radius of curvature (with its uncertainty) and cone angle (``tips.tip``)
==========  ===============================================================================

An ``AnalysisSpec`` (recipe, region ``[x0, y0, x1, y1]`` in image pixels, settings) is
all that is stored; ``run`` recomputes the ``Analysis``: a summary, one row per object
(line, hole, particle, layer...), shapes to draw on the image, warnings. Lengths are in
metres when the image is calibrated (else pixels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vexel.packs.sem.process import Param

COLOURS = {"edge_l": "#22d3ee", "edge_r": "#f472b6", "object": "#a3e635", "missing": "#ef4444", "fit": "#f59e0b", "region": "#facc15"}


class AnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    recipe: str
    region: list[float] | None = None  # [x0, y0, x1, y1] image pixels; None: the whole micrograph
    params: dict[str, Any] = Field(default_factory=dict)
    name: str = ""


@dataclass
class Recipe:
    name: str
    label: str
    params: list[Param]
    help: str
    run: Callable[..., "Analysis"]

    def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        from vexel.packs.sem.process import Op

        return Op(self.name, self.label, "correct", lambda a: a, self.params).resolve(params)


@dataclass
class Analysis:
    spec: AnalysisSpec
    summary: list[tuple[str, str, Any, str]] = field(default_factory=list)  # (key, label, value, kind: length|area|angle|count|ratio|text)
    objects: pd.DataFrame = field(default_factory=pd.DataFrame)
    shapes: list[tuple[str, Any, str, tuple[float, float], str]] = field(default_factory=list)  # as measure_io._shapes
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def value(self, key: str) -> Any:
        for k, _, v, _ in self.summary:
            if k == key:
                return v
        return None


# -- helpers ------------------------------------------------------------------------------------


def region_of(spec: AnalysisSpec, shape: tuple[int, int], crop_rows: int | None) -> tuple[int, int, int, int]:
    rows = crop_rows if crop_rows and crop_rows < shape[0] else shape[0]
    if spec.region is None:
        return 0, 0, shape[1], rows
    x0, y0, x1, y1 = spec.region
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    X0, Y0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    X1, Y1 = min(int(math.ceil(x1)), shape[1]), min(int(math.ceil(y1)), rows)
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        raise ValueError("the region is too small (at least 8 × 8 pixels)")
    return X0, Y0, X1, Y1


def _poly(xs: np.ndarray, ys: np.ndarray, step: int = 1) -> list[list[float]]:
    ok = np.isfinite(xs) & np.isfinite(ys)
    return [[float(x), float(y)] for x, y in zip(xs[ok][::step], ys[ok][::step])]


def _circle(cx: float, cy: float, r: float, n: int = 48) -> list[list[float]]:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[float(cx + r * math.cos(a)), float(cy + r * math.sin(a))] for a in t]


def _orientation(a: np.ndarray) -> str:
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return "vertical" if gx >= gy else "horizontal"


# -- recipes ------------------------------------------------------------------------------------


def _lines(a: np.ndarray, mask: np.ndarray | None, X0: int, Y0: int, sx: float, sy: float, p: dict[str, Any], spec: AnalysisSpec) -> Analysis:
    from vexel.packs.sem.metrology import lines_spaces

    orient = p["orientation"] if p["orientation"] != "auto" else _orientation(a)
    t = orient == "horizontal"
    arr, m = (a.T, None if mask is None else mask.T) if t else (a, mask)
    fx, fy = (sy, sx) if t else (sx, sy)  # across the lines, along the lines
    res = lines_spaces(arr, feature=p["feature"], method=p["method"], level=p["level"], smooth=p["smooth"], average=p["average"], mask=m)
    an = Analysis(spec=spec, warnings=list(res.warnings))
    s = res.summary
    if not s:
        return an

    def L(v: float | None) -> float | None:
        return None if v is None else v * fx

    an.summary = [("orientation", "线条方向", "竖直" if not t else "水平", "text"), ("lines", "线条数", s["lines"], "count"),
                  ("cd_mean", "线宽 CD（平均）", L(s["cd_mean"]), "length"), ("cd_sd", "线宽（线间标准差）", L(s["cd_sd"]), "length"),
                  ("space_mean", "间隔", L(s["space_mean"]), "length"), ("pitch", "周期（线中心）", L(s["pitch"]), "length"),
                  ("pitch_autocorrelation", "周期（自相关）", L(s["pitch_autocorrelation"]), "length"),
                  ("ler_3sigma", "LER 3σ（含噪声）", L(s["ler_3sigma"]), "length"), ("ler_3sigma_unbiased", "LER 3σ（扣除噪声）", L(s["ler_3sigma_unbiased"]), "length"),
                  ("lwr_3sigma", "LWR 3σ（含噪声）", L(s["lwr_3sigma"]), "length"), ("lwr_3sigma_unbiased", "LWR 3σ（扣除噪声）", L(s["lwr_3sigma_unbiased"]), "length"),
                  ("ler_noise_3sigma", "噪声造成的边缘抖动 3σ", L(s["ler_noise_3sigma"]), "length"),
                  ("corr_length", "相关长度", None if s["corr_length_lines"] is None else s["corr_length_lines"] * fy, "length"),
                  ("sampled_lines", "采样行数", s["sampled_lines"], "count"), ("line_spacing", "采样间距", fy, "length"),
                  ("edge_definition", "边缘定义", {"threshold": f"阈值 {p['level']:.0%}", "gradient": "最大梯度", "peak": "亮边峰值"}[p["method"]] + (f"，{p['average']} 行平均" if p["average"] > 1 else ""), "text"),
                  ("slant", "线条倾斜", s["slant_deg"], "angle")]  # fmt: skip
    rows = []
    n = arr.shape[0]
    yy = np.arange(n, dtype=float)
    step = max(1, n // 200)
    for i, ln in enumerate(res.lines, start=1):
        rows.append({"line": i, "position": L(ln["x"]), "cd": L(ln["cd"]), "cd_sd_along": L(ln["cd_sd"]),
                     "ler_left_3sigma": L(3 * ln["ler_left"]["sigma"]) if ln["ler_left"].get("sigma") is not None else None,
                     "ler_left_3sigma_unbiased": L(3 * ln["ler_left"]["sigma_unbiased"]) if ln["ler_left"].get("sigma_unbiased") is not None else None,
                     "ler_right_3sigma": L(3 * ln["ler_right"]["sigma"]) if ln["ler_right"].get("sigma") is not None else None,
                     "ler_right_3sigma_unbiased": L(3 * ln["ler_right"]["sigma_unbiased"]) if ln["ler_right"].get("sigma_unbiased") is not None else None,
                     "lwr_3sigma": L(3 * ln["lwr"]["sigma"]) if ln["lwr"].get("sigma") is not None else None,
                     "lwr_3sigma_unbiased": L(3 * ln["lwr"]["sigma_unbiased"]) if ln["lwr"].get("sigma_unbiased") is not None else None,
                     "missing": ln["missing"]})  # fmt: skip
        across = math.cos(math.atan(res.info["slant_px_per_line"]))
        for key, colour in (("left", COLOURS["edge_l"]), ("right", COLOURS["edge_r"])):
            pos = res.edges[ln[key]].position / across  # back to region pixels
            if t:
                xs, ys = yy + X0, pos + Y0
            else:
                xs, ys = pos + X0, yy + Y0
            label = f"{i}" if key == "left" else ""
            pts = _poly(xs, ys, step)
            if len(pts) >= 2:
                an.shapes.append(("polyline", pts, colour, tuple(pts[0]), label))
    an.objects = pd.DataFrame(rows)
    an.extra = {"edge_psd": [ln["ler_left"].get("psd") for ln in res.lines][:1], "freq": [ln["ler_left"].get("freq") for ln in res.lines][:1]}
    return an


def _holes(a: np.ndarray, mask: np.ndarray | None, X0: int, Y0: int, sx: float, sy: float, p: dict[str, Any], spec: AnalysisSpec) -> Analysis:
    from vexel.packs.sem.objects import holes

    s = math.sqrt(sx * sy)
    res = holes(a, feature=p["feature"], method=p["method"], level=p["level"], smooth=p["smooth"], min_diameter=p["min_diameter"], mask=mask)
    an = Analysis(spec=spec, warnings=list(res["warnings"]))
    sm = res["summary"]
    if abs(sx / sy - 1) > 0.01:
        an.warnings.append("像素在 x、y 方向尺寸不同：孔径按几何平均换算")
    an.summary = [("holes", "孔数", sm["holes"], "count"), ("diameter_mean", "孔径（圆拟合，平均）", sm["diameter_mean"] and sm["diameter_mean"] * s, "length"),
                  ("diameter_3sigma", "孔径均匀性 3σ", sm["diameter_3sigma"] and sm["diameter_3sigma"] * s, "length"),
                  ("roundness_mean", "圆度（短轴/长轴）", sm["roundness_mean"], "ratio"), ("merged", "疑似相连", sm["merged"], "count"),
                  ("at_border", "被边界截断（未计入）", sm["at_border"], "count"),
                  ("edge_definition", "边缘定义", f"阈值 {p['level']:.0%}" if p["method"] == "threshold" else "最大梯度", "text")]  # fmt: skip
    if "pitch_a" in sm:
        an.summary += [("pitch_a", "阵列周期 a", sm["pitch_a"] * s, "length"), ("pitch_b", "阵列周期 b", sm["pitch_b"] * s, "length"),
                       ("lattice_angle", "阵列夹角", sm["lattice_angle"], "angle"), ("missing", "缺失的孔", sm["missing"], "count")]  # fmt: skip
    rows = []
    for i, h in enumerate(res["holes"], start=1):
        rows.append({"hole": i, "x": (h["x"] + X0) * sx, "y": (h["y"] + Y0) * sy, "diameter": h["diameter"] * s, "eq_diameter": h["eq_diameter"] * s,
                     "major": h.get("major", np.nan) * s, "minor": h.get("minor", np.nan) * s, "orientation": h.get("orientation"),
                     "roundness": h.get("roundness"), "fit_residual": h["fit_residual"] * s, "merged": h["merged"]})  # fmt: skip
        an.shapes.append(("polygon", _circle(h["x"] + X0, h["y"] + Y0, h["diameter"] / 2), COLOURS["missing"] if h["merged"] else COLOURS["object"],
                          (h["x"] + X0, h["y"] + Y0), ""))  # fmt: skip
    if res["lattice"]:
        miss = [[x + X0, y + Y0] for x, y in res["lattice"]["missing"]]
        if miss:
            an.shapes.append(("points", miss, COLOURS["missing"], tuple(miss[0]), f"missing {len(miss)}"))
        an.extra["missing"] = [[x * sx, y * sy] for x, y in miss]
    an.objects = pd.DataFrame(rows)
    return an


def _particles(a: np.ndarray, mask: np.ndarray | None, X0: int, Y0: int, sx: float, sy: float, p: dict[str, Any], spec: AnalysisSpec) -> Analysis:
    from vexel.packs.sem.objects import particles

    s = math.sqrt(sx * sy)
    res = particles(a, feature=p["feature"], threshold=p["threshold"], split=p["split"], min_diameter=p["min_diameter"], smooth=p["smooth"], mask=mask)
    an = Analysis(spec=spec, warnings=list(res["warnings"]))
    sm = res["summary"]
    an.summary = [("particles", "颗粒数", sm["particles"], "count"), ("at_border", "被边界截断（未计入）", sm["at_border"], "count")]
    if sm.get("diameter_mean") is not None:
        L = lambda v: v * s  # noqa: E731
        an.summary += [("diameter_mean", "等效圆直径（平均）", L(sm["diameter_mean"]), "length"), ("diameter_sd", "等效圆直径（标准差）", L(sm["diameter_sd"]), "length"),
                       ("d10_n", "d10（数量）", L(sm["d10_n"]), "length"), ("d50_n", "d50（数量）", L(sm["d50_n"]), "length"), ("d90_n", "d90（数量）", L(sm["d90_n"]), "length"),
                       ("d10_v", "d10（体积）", L(sm["d10_v"]), "length"), ("d50_v", "d50（体积）", L(sm["d50_v"]), "length"), ("d90_v", "d90（体积）", L(sm["d90_v"]), "length"),
                       ("agglomerated", "相互接触的比例", sm["agglomerated"], "ratio"), ("area_fraction", "面积分数", sm["area_fraction"], "ratio")]  # fmt: skip
    rows = []
    for i, q in enumerate(res["particles"], start=1):
        rows.append({"particle": i, "x": (q["x"] + X0) * sx, "y": (q["y"] + Y0) * sy, "eq_diameter": q["eq_diameter"] * s, "area": q["area"] * sx * sy,
                     "feret_max": q["feret_max"] * s, "feret_min": q["feret_min"] * s, "circularity": q["circularity"],
                     "aspect_ratio": q["aspect_ratio"], "touching": q["touching"]})  # fmt: skip
        if len(q["outline"]) > 2:
            pts = (np.asarray(q["outline"]) + [X0, Y0])[:: max(1, len(q["outline"]) // 60)].tolist()
            an.shapes.append(("polygon", pts, COLOURS["object"], (q["x"] + X0, q["y"] + Y0), ""))
    an.objects = pd.DataFrame(rows)
    return an


def _layers(a: np.ndarray, mask: np.ndarray | None, X0: int, Y0: int, sx: float, sy: float, p: dict[str, Any], spec: AnalysisSpec) -> Analysis:
    from vexel.packs.sem.metrology import layers

    res = layers(a, method=p["method"], level=p["level"], smooth=p["smooth"], average=p["average"], mask=mask)
    an = Analysis(spec=spec, warnings=list(res["warnings"]))
    an.summary = [("interfaces", "界面数", len(res["interfaces"]), "count"), ("slant", "层面倾斜", res["slant_deg"], "angle"),
                  ("edge_definition", "边缘定义", f"阈值 {p['level']:.0%}" if p["method"] == "threshold" else "最大梯度", "text")]  # fmt: skip
    for ly in res["layers"]:
        if ly["thickness"] is not None:
            an.summary.append((f"layer{ly['layer']}", f"第 {ly['layer']} 层厚度", ly["thickness"] * sy, "length"))
    rows = [{"layer": ly["layer"], "top": (ly["top"] + Y0) * sy, "bottom": (ly["bottom"] + Y0) * sy,
             "thickness": ly["thickness"] and ly["thickness"] * sy, "sd": ly["sd"] and ly["sd"] * sy,
             "min": ly["min"] and ly["min"] * sy, "max": ly["max"] and ly["max"] * sy} for ly in res["layers"]]  # fmt: skip
    rows += [{"interface": it["index"], "y": (it["y"] + Y0) * sy, "roughness_3sigma": it["roughness_sigma"] and 3 * it["roughness_sigma"] * sy,
              "roughness_3sigma_unbiased": it["roughness_sigma_unbiased"] and 3 * it["roughness_sigma_unbiased"] * sy} for it in res["interfaces"]]  # fmt: skip
    xs = np.arange(a.shape[1], dtype=float) + X0
    step = max(1, a.shape[1] // 200)
    for it in res["interfaces"]:
        pts = _poly(xs, it["position"] + Y0, step)
        if len(pts) >= 2:
            an.shapes.append(("polyline", pts, COLOURS["edge_l"], tuple(pts[-1]), f"I{it['index']}"))
    an.objects = pd.DataFrame(rows)
    return an


def _profile(a: np.ndarray, mask: np.ndarray | None, X0: int, Y0: int, sx: float, sy: float, p: dict[str, Any], spec: AnalysisSpec) -> Analysis:
    from vexel.packs.sem.metrology import feature_profile

    res = feature_profile(a, feature=p["feature"], level=p["level"], smooth=p["smooth"], fit_range=(p["fit_from"], p["fit_to"]), mask=mask)
    an = Analysis(spec=spec, warnings=list(res["warnings"]))
    if "height" not in res:
        return an
    an.summary = [("height", "高度（深度）", res["height"] * sy, "length"), ("width_top", "顶部宽度（10%）", res["width_top"] * sx, "length"),
                  ("width_middle", "中部宽度（50%）", res["width_middle"] * sx, "length"), ("width_bottom", "底部宽度（90%）", res["width_bottom"] * sx, "length"),
                  ("angle_left", "左侧壁角", _true_angle(res["angle_left"], sx, sy), "angle"), ("angle_right", "右侧壁角", _true_angle(res["angle_right"], sx, sy), "angle"),
                  ("fit_range", "侧壁拟合范围", f"{p['fit_from']:.0%}–{p['fit_to']:.0%} 高度", "text")]  # fmt: skip
    ys = np.arange(a.shape[0], dtype=float) + Y0
    rows = res["rows"]
    for key, colour in (("left", COLOURS["edge_l"]), ("right", COLOURS["edge_r"])):
        pts = _poly(res[key][rows] + X0, ys[rows], max(1, len(rows) // 100))
        if len(pts) >= 2:
            ang = _true_angle(res[f"angle_{key}"], sx, sy)
            an.shapes.append(("polyline", pts, colour, tuple(pts[len(pts) // 2]), f"{ang:.1f}°"))
    an.objects = pd.DataFrame([{"row": int(r + Y0), "depth": (r - res["top"]) * sy, "width": (res["right"][r] - res["left"][r]) * sx} for r in rows])
    return an


def _true_angle(px_angle: float, sx: float, sy: float) -> float:
    """A wall angle measured in pixels, for pixels of size sx × sy."""
    if abs(sx - sy) <= 1e-12 * max(sx, sy):
        return px_angle
    t = math.tan(math.radians(px_angle))
    return math.degrees(math.atan2(t * sy, sx)) % 180


def _tip(a: np.ndarray, mask: np.ndarray | None, X0: int, Y0: int, sx: float, sy: float, p: dict[str, Any], spec: AnalysisSpec) -> Analysis:
    from vexel.packs.sem.tips import tip

    s = math.sqrt(sx * sy)
    res = tip(a, feature=p["feature"], smooth=p["smooth"], cone_from=p["cone_from"] or None, cone_to=p["cone_to"] or None, mask=mask)
    an = Analysis(spec=spec, warnings=list(res["warnings"]))
    if "radius" not in res:
        return an
    an.summary = [("radius", "尖端曲率半径", res["radius"] * s, "length"), ("radius_uncertainty", "半径不确定度（拟合范围变化）", res["radius_uncertainty"] * s, "length"),
                  ("fit_distance", "拟合范围（距尖端）", res["fit_distance"] * s, "length")]  # fmt: skip
    if "cone_angle" in res:
        an.summary += [("cone_angle", "锥角（全角）", res["cone_angle"], "angle"), ("half_angle_left", "左半角", res["half_angle_left"], "angle"),
                       ("half_angle_right", "右半角", res["half_angle_right"], "angle"),
                       ("cone_range", "锥角拟合范围", f"{res['cone_from'] * s * 1e9:.0f}–{res['cone_to'] * s * 1e9:.0f} nm" if s != 1 else f"{res['cone_from']:.0f}–{res['cone_to']:.0f} px", "text")]  # fmt: skip
    an.summary += [("axis_angle", "轴向（与水平）", res["axis_angle"], "angle"), ("length_visible", "可见长度", res["length_visible"] * s, "length")]
    ax, ay = res["apex"]
    outline = (np.asarray(res["outline"]) + [X0, Y0])
    an.shapes.append(("polyline", outline[:: max(1, len(outline) // 300)].tolist(), COLOURS["object"], (ax + X0, ay + Y0), ""))
    th = math.radians(res["axis_angle"])
    cx, cy = ax + X0 - res["radius"] * math.cos(th), ay + Y0 + res["radius"] * math.sin(th)
    an.shapes.append(("polygon", _circle(cx, cy, res["radius"]), COLOURS["fit"], (ax + X0, ay + Y0), f"R {res['radius'] * s * (1e9 if s != 1 else 1):.3g}{' nm' if s != 1 else ' px'}"))
    an.extra["apex"] = [(ax + X0) * sx, (ay + Y0) * sy]
    return an


FEATURE = Param("feature", "目标", "choice", "bright", (("bright", "亮"), ("dark", "暗")))
METHOD = Param("method", "边缘定义", "choice", "threshold", (("threshold", "阈值"), ("gradient", "最大梯度")))
LEVEL = Param("level", "阈值", "float", 0.5, lo=0.05, hi=0.95, step=0.05)
SMOOTH = Param("smooth", "平滑（像素）", "float", 1.0, lo=0, hi=10, step=0.5)

RECIPES: dict[str, Recipe] = {
    r.name: r
    for r in [
        Recipe("lines", "线条与间隔（CD、LER/LWR）", [
            Param("orientation", "线条方向", "choice", "auto", (("auto", "自动"), ("vertical", "竖直"), ("horizontal", "水平"))),
            Param("feature", "线条是", "choice", "bright", (("bright", "亮条"), ("dark", "暗条"))),
            Param("method", "边缘定义", "choice", "threshold", (("threshold", "阈值"), ("gradient", "最大梯度"), ("peak", "亮边峰值"))), LEVEL, SMOOTH,
            Param("average", "行平均", "int", 1, lo=1, hi=32, help="每个位置平均几行再找边缘：降噪，但会抹掉高频粗糙度"),
        ], "框选一组完整的线条（两侧边缘都在框内），沿线方向尽量长：线宽、间隔、周期、线边缘与线宽粗糙度", _lines),  # fmt: skip
        Recipe("holes", "接触孔与通孔", [
            Param("feature", "孔是", "choice", "dark", (("dark", "暗"), ("bright", "亮"))), METHOD, LEVEL, SMOOTH,
            Param("min_diameter", "最小直径（像素）", "float", 4.0, lo=2, hi=1000, step=1),
        ], "孔径（圆与椭圆拟合）、圆度、孔径均匀性、阵列周期与缺失的孔", _holes),
        Recipe("particles", "颗粒与缺陷", [
            FEATURE, Param("threshold", "阈值", "choice", "mid", (("mid", "全图（两类灰度的中点）"), ("local", "局部（亮度不均时）"))),
            Param("split", "分开相互接触的颗粒", "bool", True), Param("min_diameter", "最小直径（像素）", "float", 3.0, lo=1, hi=1000, step=1),
            Param("smooth", "平滑（像素）", "float", 0.5, lo=0, hi=10, step=0.5, help="平滑会让小颗粒的轮廓向内收：颗粒小于 15 像素时不要超过 0.5"),
        ], "等效圆直径、Feret 直径、圆度、长宽比、粒度分布（数量与体积加权）；被区域边界截断的不计入", _particles),  # fmt: skip
        Recipe("layers", "截面层厚", [
            Param("method", "边缘定义", "choice", "gradient", (("gradient", "最大梯度"), ("threshold", "阈值"))), LEVEL,
            Param("smooth", "平滑（像素）", "float", 1.5, lo=0, hi=10, step=0.5), Param("average", "列平均", "int", 1, lo=1, hi=32),
        ], "水平层叠的截面：逐列找各界面，给出每层厚度、均匀性与界面粗糙度（先把层面校正到水平）", _layers),  # fmt: skip
        Recipe("profile", "截面轮廓（侧壁角）", [
            Param("feature", "特征是", "choice", "bright", (("bright", "亮（线条）"), ("dark", "暗（沟槽）"))), LEVEL, SMOOTH,
            Param("fit_from", "侧壁拟合起点（高度比例）", "float", 0.1, lo=0, hi=0.9, step=0.05),
            Param("fit_to", "侧壁拟合终点（高度比例）", "float", 0.9, lo=0.1, hi=1, step=0.05),
        ], "框选一个线条或沟槽的截面（四周留出背景）：高度、顶/中/底宽度、两侧侧壁角（在特征内部、底部量，90° 为垂直）", _profile),  # fmt: skip
        Recipe("tip", "尖端", [
            FEATURE, SMOOTH,
            Param("cone_from", "锥角起点（距尖端，像素）", "float", 0.0, lo=0, hi=100000, step=5, help="0：2 倍曲率半径"),
            Param("cone_to", "锥角终点（距尖端，像素）", "float", 0.0, lo=0, hi=100000, step=5, help="0：10 倍曲率半径"),
        ], "框选一个尖端：曲率半径（自动选择拟合范围并给出不确定度）、锥角、轴向", _tip),  # fmt: skip
    ]
}


def run(spec: AnalysisSpec, image: np.ndarray, scale: Any, mask: np.ndarray | None = None, crop_rows: int | None = None) -> Analysis:
    """Recompute one analysis on the (corrected) image with the record's scale."""
    if spec.recipe not in RECIPES:
        return Analysis(spec=spec, error=f"unknown recipe {spec.recipe!r}")
    rec = RECIPES[spec.recipe]
    try:
        p = rec.resolve(spec.params)
        X0, Y0, X1, Y1 = region_of(spec, image.shape, crop_rows)
        a = np.asarray(image[Y0:Y1, X0:X1], np.float64)
        m = None if mask is None else mask[Y0:Y1, X0:X1]
        an = rec.run(a, m, X0, Y0, scale.sx, scale.sy, p, spec)
    except ValueError as e:
        return Analysis(spec=spec, error=str(e))
    an.shapes.insert(0, ("polygon", [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], COLOURS["region"], (X0, Y0), f"A{spec.id} {spec.recipe}"))
    an.extra["region"] = [X0, Y0, X1, Y1]
    return an


def summary_table(analyses: list[Analysis], unit: str, f: float) -> pd.DataFrame:
    """One row per summary value of every analysis, lengths in ``unit`` (size ``f`` metres)."""
    rows = []
    for an in analyses:
        if an.error:
            rows.append({"analysis": an.spec.id, "recipe": an.spec.recipe, "quantity": "error", "value": an.error})
            continue
        for key, label, v, kind in an.summary:
            u = unit if kind == "length" else f"{unit}²" if kind == "area" else "°" if kind == "angle" else ""
            val = v / f if kind == "length" and v is not None else v / f**2 if kind == "area" and v is not None else v
            rows.append({"analysis": an.spec.id, "recipe": an.spec.recipe, "name": an.spec.name, "key": key, "quantity": label, "value": val, "unit": u})
        for w in an.warnings:
            rows.append({"analysis": an.spec.id, "recipe": an.spec.recipe, "quantity": "warning", "value": w})
    return pd.DataFrame(rows)


LENGTH_COLUMNS = {"position", "cd", "cd_sd_along", "ler_left_3sigma", "ler_left_3sigma_unbiased", "ler_right_3sigma", "ler_right_3sigma_unbiased",
                  "lwr_3sigma", "lwr_3sigma_unbiased", "x", "y", "diameter", "eq_diameter", "major", "minor", "fit_residual", "feret_max",
                  "feret_min", "top", "bottom", "thickness", "sd", "min", "max", "roughness_3sigma", "roughness_3sigma_unbiased", "depth",
                  "width"}  # fmt: skip


def objects_table(analyses: list[Analysis], unit: str, f: float) -> pd.DataFrame:
    frames = []
    for an in analyses:
        if an.error or an.objects.empty:
            continue
        df = an.objects.copy()
        df.insert(0, "analysis", an.spec.id)
        df.insert(1, "recipe", an.spec.recipe)
        ren = {}
        for c in df.columns:
            if c in LENGTH_COLUMNS:
                df[c] = df[c].astype(float) / f
                ren[c] = f"{c} ({unit})"
            elif c == "area":
                df[c] = df[c].astype(float) / f**2
                ren[c] = f"area ({unit}²)"
        frames.append(df.rename(columns=ren))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = ["RECIPES", "Analysis", "AnalysisSpec", "Recipe", "objects_table", "region_of", "run", "summary_table"]
