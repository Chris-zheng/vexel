"""Experiment-level statistics with the replicate as the unit.

Droplets from one recording are not independent samples of an experimental condition:
they share a chip, a preparation and a day. Comparing thousands of droplets between two
recordings would give tiny p-values that say nothing about the conditions. So:

1. each file gives one value per metric (its summary: mean diameter, CV, λ, positive
   fraction, ...);
2. files with the same group and replicate are averaged into one value per replicate
   (without a ``replicate`` column every file is its own replicate);
3. groups are described (n, mean, SD, 95% CI of the mean, median) and compared over
   their replicate values:

   - two groups: Welch's t-test (unequal variances) with the mean difference and its 95% CI,
     Hedges' g, and the Mann–Whitney U test as a check that does not assume normality;
   - three or more: Welch's one-way ANOVA and the Kruskal–Wallis test, eta squared, then
     every pair by Welch's t-test with Holm's correction across the pairs.

Groups with fewer than two replicates are described but not tested. Counts that depend
on how long a recording is (number of droplets) and QC shares are described only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class Metric:
    key: str  # column in results.csv, e.g. "droplets.diameter_mean"
    zh: str
    en: str
    unit_key: str | None = None  # results column holding the unit, if any
    unit: str = ""
    test: bool = True
    digits: int = 3


METRICS = [
    Metric("droplets.diameter_mean", "平均直径", "Mean diameter", unit_key="droplets.length_unit"),
    Metric("droplets.diameter_cv_percent", "粒径 CV", "Diameter CV", unit="%"),
    Metric("droplets.volume_mean_pl", "平均体积", "Mean volume", unit="pL"),
    Metric("droplets.frequency", "生成频率", "Generation frequency", unit_key="droplets.frequency_unit"),
    Metric("droplets.velocity_mean", "平均速度", "Mean velocity", unit_key="droplets.velocity_unit"),
    Metric("droplets.spacing_mean", "平均间距", "Mean spacing", unit_key="droplets.length_unit"),
    Metric("encapsulation.lambda", "每滴平均细胞数 λ", "Mean cells per droplet (λ)"),
    Metric("encapsulation.frac_empty", "空滴比例", "Fraction empty"),
    Metric("encapsulation.frac_single", "单包裹比例", "Fraction with one cell"),
    Metric("encapsulation.frac_multi", "多包裹比例", "Fraction with two or more cells"),
    Metric("positivity.positive_fraction", "阳性比例", "Positive fraction"),
    Metric("positivity.copies_per_droplet", "每滴平均拷贝数", "Mean copies per droplet"),
    Metric("positivity.copies_per_ul", "浓度", "Concentration", unit="copies/µL"),
    Metric("droplets.n_droplets", "液滴数", "Droplets counted", test=False, digits=0),
    Metric("flags.flagged_fraction", "可疑液滴比例", "Fraction flagged for review", test=False),
]


@dataclass
class Analysis:
    metrics: list[Metric]
    units: pd.DataFrame  # one row per replicate (unit)
    groups: pd.DataFrame  # metric x group description
    comparisons: pd.DataFrame  # tests
    notes: list[tuple[str, str]] = field(default_factory=list)  # (zh, en)
    unit_column: str = "file"  # "replicate" when the design names replicates

    def unit_of(self, m: Metric) -> str:
        return m.unit


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _units(ok: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    ok = ok.copy()
    ok["replicate"] = ok["replicate"].fillna("").astype(str)
    ok["group"] = ok["group"].fillna("").astype(str)
    ok["unit"] = np.where(ok["replicate"] != "", ok["group"] + "/" + ok["replicate"], ok["group"] + "/" + ok["id"].astype(str))
    cols = [m.key for m in metrics]
    for c in cols:
        ok[c] = _num(ok[c])
    agg = ok.groupby(["group", "unit"], sort=False).agg(n_files=("id", "count"), **{c: (c, "mean") for c in cols}).reset_index()
    first = ok.groupby("unit", sort=False)["replicate"].first()
    agg.insert(2, "replicate", agg["unit"].map(first))
    return agg


def describe(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    n = len(v)
    out = {"n": n, "mean": math.nan, "sd": math.nan, "sem": math.nan, "ci_low": math.nan, "ci_high": math.nan,
           "median": math.nan, "min": math.nan, "max": math.nan}  # fmt: skip
    if n == 0:
        return out
    out.update(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    if n >= 2:
        sd = float(v.std(ddof=1))
        sem = sd / math.sqrt(n)
        t = float(sps.t.ppf(0.975, n - 1))
        out.update(sd=sd, sem=sem, ci_low=out["mean"] - t * sem, ci_high=out["mean"] + t * sem)
    return out


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return math.nan
    pooled = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return math.nan
    return float((a.mean() - b.mean()) / pooled * (1 - 3 / (4 * (n1 + n2) - 9)))


def welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    out = {"diff": float(a.mean() - b.mean()), "diff_ci_low": math.nan, "diff_ci_high": math.nan, "t": math.nan, "df": math.nan, "p": math.nan}
    if len(a) < 2 or len(b) < 2 or (a.var(ddof=1) == 0 and b.var(ddof=1) == 0):
        return out
    res = sps.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval(0.95)
    out.update(t=float(res.statistic), df=float(res.df), p=float(res.pvalue), diff_ci_low=float(ci.low), diff_ci_high=float(ci.high))
    return out


def mann_whitney(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2 or np.ptp(np.concatenate([a, b])) == 0:
        return math.nan
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def holm(p: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (NaN stays NaN)."""
    idx = [i for i, x in enumerate(p) if np.isfinite(x)]
    order = sorted(idx, key=lambda i: p[i])
    m = len(order)
    adj = [math.nan] * len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def _compare(metric: Metric, data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    testable = {g: v for g, v in data.items() if len(v) >= 2}
    rows: list[dict[str, Any]] = []
    if len(testable) < 2:
        return rows
    names = list(testable)
    if len(names) >= 3:
        arrays = [testable[g] for g in names]
        welch_p = kw_p = eta2 = math.nan
        if np.ptp(np.concatenate(arrays)) > 0:
            try:
                welch_p = float(sps.f_oneway(*arrays, equal_var=False).pvalue)
            except (TypeError, ValueError):
                welch_p = math.nan
            kw_p = float(sps.kruskal(*arrays).pvalue)
            allv = np.concatenate(arrays)
            ss_between = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in arrays)
            ss_total = float(((allv - allv.mean()) ** 2).sum())
            eta2 = ss_between / ss_total if ss_total > 0 else math.nan
        rows.append({"metric": metric.key, "comparison": "all groups", "a": "", "b": "", "n_a": None, "n_b": None,
                     "test": "Welch ANOVA", "p": welch_p, "p_holm": math.nan, "p_nonparametric": kw_p,
                     "nonparametric_test": "Kruskal-Wallis", "effect": eta2, "effect_name": "eta squared",
                     "diff": math.nan, "diff_ci_low": math.nan, "diff_ci_high": math.nan})  # fmt: skip
    pairs = []
    for a, b in combinations(names, 2):
        w = welch(testable[a], testable[b])
        pairs.append({"metric": metric.key, "comparison": f"{a} vs {b}", "a": a, "b": b, "n_a": len(testable[a]), "n_b": len(testable[b]),
                      "test": "Welch t", "p": w["p"], "p_holm": math.nan, "p_nonparametric": mann_whitney(testable[a], testable[b]),
                      "nonparametric_test": "Mann-Whitney U", "effect": hedges_g(testable[a], testable[b]), "effect_name": "Hedges g",
                      "diff": w["diff"], "diff_ci_low": w["diff_ci_low"], "diff_ci_high": w["diff_ci_high"]})  # fmt: skip
    adj = holm([r["p"] for r in pairs]) if len(pairs) > 1 else [r["p"] for r in pairs]
    for r, q in zip(pairs, adj):
        r["p_holm"] = q
    return rows + pairs


def analyse(results: pd.DataFrame) -> Analysis:
    notes: list[tuple[str, str]] = []
    empty = Analysis([], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notes)
    if results.empty or "status" not in results:
        return empty
    ok = results[results["status"].isin(["completed", "resumed"])]
    if ok.empty:
        notes.append(("没有成功完成的文件，无法统计。", "No file completed; nothing to summarise."))
        return empty
    metrics = []
    for m in METRICS:
        if m.key in ok.columns and _num(ok[m.key]).notna().any():
            unit = m.unit
            if m.unit_key and m.unit_key in ok.columns:
                units = ok[m.unit_key].dropna().astype(str).unique()
                unit = units[0] if len(units) == 1 else "mixed"
                if len(units) > 1:
                    notes.append((f"{m.zh}的单位不一致（{', '.join(units)}）：有的文件标定了像素尺寸或帧间隔，有的没有。",
                                  f"{m.en} has mixed units ({', '.join(units)}): some files are calibrated and some are not."))  # fmt: skip
            metrics.append(Metric(m.key, m.zh, m.en, m.unit_key, unit, m.test and unit != "mixed", m.digits))
    units = _units(ok, metrics)
    has_rep = bool((ok["replicate"].fillna("").astype(str) != "").any())
    if not has_rep and ok.groupby(ok["group"].fillna("")).size().max() > 1:
        notes.append(("设计表没有填写 replicate 列：每个文件被当作一个独立重复。如果几个文件来自同一芯片、同一次制备或同一天，请填写 replicate，否则检验会高估证据。",
                      "No replicate column: every file is treated as an independent replicate. If several files come from the same chip, preparation or day, give them the same replicate, otherwise the tests overstate the evidence."))  # fmt: skip
    group_rows, comp_rows = [], []
    group_names = list(dict.fromkeys(units["group"]))
    for m in metrics:
        data = {}
        for g in group_names:
            v = units.loc[units["group"] == g, m.key].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            data[g] = v
            group_rows.append({"metric": m.key, "group": g, **describe(v)})
        if m.test and len(group_names) >= 2:
            comp_rows.extend(_compare(m, data))
    small = [g for g in group_names if (units["group"] == g).sum() < 2]
    if len(group_names) >= 2 and small:
        notes.append((f"这些组只有 1 个重复，只作描述、不参与检验：{', '.join(g or '（未分组）' for g in small)}。",
                      f"Groups with a single replicate are described but not tested: {', '.join(g or '(no group)' for g in small)}."))  # fmt: skip
    few = [g for g in group_names if 2 <= (units["group"] == g).sum() < 3]
    if few:
        notes.append(("有的组只有 2 个重复：检验的把握很低，p 值仅供参考，请以效应量和置信区间为主。",
                      "Some groups have only two replicates: tests have little power; read the effect sizes and confidence intervals first."))  # fmt: skip
    if len(group_names) < 2:
        notes.append(("只有一个组：只作描述统计。要比较条件，请在设计表的 group 列填写不同的组名。",
                      "Only one group: descriptive statistics only. Fill in the group column to compare conditions."))  # fmt: skip
    if comp_rows and len({r["metric"] for r in comp_rows}) > 1:
        notes.append(("同时比较了多个指标；Holm 校正只在同一指标的两两比较之间进行，跨指标请谨慎解读。",
                      "Several metrics were compared; Holm's correction is applied across pairs within a metric only, not across metrics."))  # fmt: skip
    return Analysis(metrics, units, pd.DataFrame(group_rows), pd.DataFrame(comp_rows), notes, "replicate" if has_rep else "file")


def write_tables(a: Analysis, out_dir: str | Path) -> None:
    out = Path(out_dir)
    a.units.to_csv(out / "replicates.csv", index=False, float_format="%.10g")
    a.groups.to_csv(out / "groups.csv", index=False, float_format="%.10g")
    a.comparisons.to_csv(out / "comparisons.csv", index=False, float_format="%.10g")
    meta = {"unit": a.unit_column, "metrics": [{"key": m.key, "zh": m.zh, "en": m.en, "unit": m.unit, "tested": m.test} for m in a.metrics],
            "notes": [{"zh": z, "en": e} for z, e in a.notes]}  # fmt: skip
    (out / "stats.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
