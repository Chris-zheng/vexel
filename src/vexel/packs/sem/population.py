"""Statistics over many images: every object of every image pooled into one population
(particles, pores, holes, fibres), per folder and recipe - the way to reach a statistically
meaningful number of objects when one image holds too few.

The spread of per-image results (repeatability) is computed by ``vexel sem analyze``
itself; this module pools the objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# the object size each recipe reports (metres, or pixels when the image is not calibrated)
SIZE = {"particles": "eq_diameter", "pores": "eq_diameter", "holes": "diameter", "fibres": "diameter"}


def sizes(an: Any) -> tuple[np.ndarray, np.ndarray]:
    """The objects' sizes and their weights for a number distribution: fibres use their
    centre-line samples (each about one pixel of fibre length), others one per object."""
    if an.error:
        return np.empty(0), np.empty(0)
    if an.spec.recipe == "fibres" and an.extra.get("samples"):
        s = an.extra["samples"]
        return np.asarray(s["diameter"], float), np.asarray(s["length"], float)
    col = SIZE.get(an.spec.recipe)
    if col is None or an.objects.empty or col not in an.objects:
        return np.empty(0), np.empty(0)
    v = an.objects[col].to_numpy(float)
    v = v[np.isfinite(v)]
    return v, np.ones_like(v)


def _wq(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = (np.cumsum(w) - w / 2) / w.sum()
    return float(np.interp(q, c, v))


def stats(v: np.ndarray, w: np.ndarray, recipe: str) -> dict[str, float]:
    mean = float(np.average(v, weights=w))
    out = {"n": int(len(v)), "mean": mean, "sd": float(np.sqrt(np.average((v - mean) ** 2, weights=w))) if len(v) > 1 else 0.0,
           "min": float(v.min()), "max": float(v.max()), "d10": _wq(v, w, 0.1), "d50": _wq(v, w, 0.5), "d90": _wq(v, w, 0.9)}  # fmt: skip
    if recipe == "particles":  # by volume (weights d³), as for single images
        wv = w * v**3
        out.update(d10_v=_wq(v, wv, 0.1), d50_v=_wq(v, wv, 0.5), d90_v=_wq(v, wv, 0.9))
    return out


def pooled(entries: list[tuple[str, Any, bool]], unit: str, f: float) -> pd.DataFrame:
    """One row per (folder, recipe): images, objects and the pooled size distribution in
    ``unit`` (``f`` metres). Entries: (group, Analysis, calibrated). Uncalibrated images are
    left out when others are calibrated (pixels and metres do not mix)."""
    any_cal = any(ok for _, _, ok in entries)
    groups: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for grp, an, ok in entries:
        if an.error or an.spec.recipe not in SIZE or (any_cal and not ok):
            continue
        groups.setdefault((grp, an.spec.recipe), []).append(sizes(an))
    rows = []
    for (grp, rec), parts in groups.items():
        v = np.concatenate([p[0] for p in parts])
        w = np.concatenate([p[1] for p in parts])
        if not len(v):
            continue
        k = f if any_cal else 1.0
        st = stats(v, w, rec)
        row: dict[str, Any] = {"group": grp, "recipe": rec, "images": len(parts), "objects" if rec != "fibres" else "samples": st.pop("n"),
                               "unit": unit if any_cal else "px"}  # fmt: skip
        row.update({key: val / k for key, val in st.items()})
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["SIZE", "pooled", "sizes", "stats"]
