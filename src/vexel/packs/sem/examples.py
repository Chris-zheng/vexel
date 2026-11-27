"""Correct a measurement by example: the person clicks objects that should be found ("this
one") and spots that should not ("not this"), and Vexel searches the method's settings for
the results that agree with them best - instead of the person adjusting numbers.

Works for the object measurements (particles, pores, holes): a click counts as inside an
object when it falls within one of the outlines the measurement draws. Every candidate is a
complete run of the same deterministic method with other settings, so what is picked is as
reproducible and checkable as anything else in the record.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

OBJECT_RECIPES = ("particles", "pores", "holes")


@dataclass
class Candidate:
    params: dict[str, Any]
    analysis: Any
    hits: int  # "this one" clicks inside an object
    misses: int  # "this one" clicks outside every object
    false: int  # "not this" clicks inside an object
    changes: list[str] = field(default_factory=list)  # settings that differ from the current ones
    size_fit: float = 0.0  # |log| of (median object size / median size of the "this one" objects)

    negatives: int = 0  # "not this" clicks given

    @property
    def satisfied(self) -> int:
        """Clicks the result agrees with: "this one" inside an object, "not this" outside all."""
        return self.hits + self.negatives - self.false

    @property
    def score(self) -> float:
        # agreement with the clicks first; among equals, objects like the examples, then fewer changes
        return self.satisfied - 0.3 * min(self.size_fit, 1.0) - 0.05 * len(self.changes)

    @property
    def agrees(self) -> bool:
        return self.misses == 0 and self.false == 0


def _outlines(an: Any) -> list[np.ndarray]:
    """Object outlines of an analysis (polygons and ellipses), image pixels; the region box is skipped."""
    out = []
    for kind, geom, _colour, _anchor, label in an.shapes:
        if kind == "polygon" and not str(label).startswith("A"):  # "A1 particles": the region box
            out.append(np.asarray(geom, float))
        elif kind == "ellipse":
            cx, cy, a, b, deg = geom
            t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            th = math.radians(deg)
            out.append(np.column_stack([cx + a * np.cos(t) * math.cos(th) + b * np.sin(t) * math.sin(th),
                                        cy - a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)]))  # fmt: skip
    return [o for o in out if len(o) >= 3]


def inside(an: Any, points: list[tuple[float, float]], margin: float = 1.5) -> list[bool]:
    """For each point: is it inside (or within ``margin`` pixels of) one of the measured objects?"""
    from skimage.measure import points_in_poly

    polys = _outlines(an)
    res = []
    for x, y in points:
        hit = False
        for poly in polys:
            if not (poly[:, 0].min() - margin <= x <= poly[:, 0].max() + margin and poly[:, 1].min() - margin <= y <= poly[:, 1].max() + margin):
                continue
            if points_in_poly([[x, y]], poly)[0]:
                hit = True
                break
            d = np.min(np.hypot(poly[:, 0] - x, poly[:, 1] - y))
            if d <= margin:
                hit = True
                break
        res.append(hit)
    return res


def _variants(recipe: str, base: dict[str, Any]) -> list[dict[str, Any]]:
    """Settings to try: the base and combinations of the settings that decide what an object is."""
    from vexel.packs.sem.analysis import RECIPES

    names = {p.name: p for p in RECIPES[recipe].params}
    axes: dict[str, list[Any]] = {}
    if "feature" in names:
        axes["feature"] = [c[0] for c in names["feature"].choices]
    if "threshold" in names:
        axes["threshold"] = [c[0] for c in names["threshold"].choices]
    if "method" in names and recipe == "holes":
        axes["method"] = [c[0] for c in names["method"].choices]
    if "smooth" in names:
        s0 = float(base.get("smooth", names["smooth"].default))
        axes["smooth"] = sorted({s0, 0.5, 3.0})
    if "split" in names:
        axes["split"] = [True, False]
    keys = list(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        p = dict(base)
        p.update(dict(zip(keys, combo)))
        out.append(p)
    return out


def tune(image: np.ndarray, spec: Any, scale: Any, positives: list[tuple[float, float]], negatives: list[tuple[float, float]], *,
         mask: np.ndarray | None = None, crop_rows: int | None = None, keep: int = 3, budget: int = 48) -> list[Candidate]:  # fmt: skip
    """The ``keep`` best settings of ``spec``'s method for the clicked examples, best first.

    Each candidate is scored by the clicks: "this one" inside an object, "not this" outside
    every object. A minimum size is raised when "not this" clicks land on objects smaller
    than every "this one" object. Candidates that give the same objects are kept once."""
    from vexel.packs.sem.analysis import RECIPES, AnalysisSpec, run

    if spec.recipe not in OBJECT_RECIPES:
        raise ValueError(f"correcting by example works for {', '.join(OBJECT_RECIPES)}")
    if not positives and not negatives:
        raise ValueError("click at least one example first")
    rec = RECIPES[spec.recipe]
    base = rec.resolve(spec.params)
    tried: list[Candidate] = []
    seen: set[tuple] = set()

    def evaluate(params: dict[str, Any]) -> Candidate | None:
        an = run(AnalysisSpec(id=spec.id, recipe=spec.recipe, region=spec.region, params=params), image, scale, mask, crop_rows)
        if an.error:
            return None
        pos = inside(an, positives) if positives else []
        neg = inside(an, negatives) if negatives else []
        changes = [k for k in params if params[k] != base.get(k)]
        c = Candidate(params, an, sum(pos), len(pos) - sum(pos), sum(neg), changes, negatives=len(neg))
        want = _sizes_at(an, positives, scale)
        col = "eq_diameter" if "eq_diameter" in an.objects else "diameter" if "diameter" in an.objects else None
        if want and col and len(an.objects):
            c.size_fit = abs(math.log(float(np.median(an.objects[col])) / float(np.median(want))))
        return c

    def attempt_all(param_sets: list[dict[str, Any]]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        todo = []
        for params in param_sets:
            key = tuple(sorted((k, str(v)) for k, v in params.items()))
            if key not in seen:
                seen.add(key)
                todo.append(params)
        with ThreadPoolExecutor(max_workers=4) as pool:  # numpy / scipy release the GIL for most of the work
            tried.extend(c for c in pool.map(evaluate, todo) if c is not None)

    def attempt(params: dict[str, Any]) -> None:
        attempt_all([params])

    attempt_all(_variants(spec.recipe, base)[:budget])
    # "not this" on small objects: raise the minimum size above them, below the "this one" objects
    if negatives and "min_diameter" in base:
        for c in sorted(tried, key=lambda c: -c.score):
            if not c.false or c.misses:  # every "this one" found, some "not this" too
                continue
            sizes = _sizes_at(c.analysis, positives, scale), _sizes_at(c.analysis, negatives, scale)
            px = math.sqrt(scale.sx * scale.sy)
            if sizes[1] and (not sizes[0] or max(sizes[1]) < min(sizes[0])):
                lo = max(sizes[1]) / px
                hi = min(sizes[0]) / px if sizes[0] else lo * 3
                # just above the rejected objects, between them and the examples (geometric), and half the examples
                for m in sorted({lo * 1.15, math.sqrt(lo * hi), 0.5 * hi}):
                    if lo < m < hi:
                        p = dict(c.params)
                        p["min_diameter"] = round(max(m, 1.0), 1)
                        attempt(p)
            break  # the best agreeing candidate is enough to refine
    best = sorted(tried, key=lambda c: (-c.score, len(c.changes)))
    out: list[Candidate] = []
    signatures: set[tuple] = set()
    for c in best:
        sig = (len(c.analysis.objects), round(float(c.analysis.value(_main_key(spec.recipe)) or 0), 12))
        if sig in signatures:
            continue
        signatures.add(sig)
        out.append(c)
        if len(out) == keep:
            break
    return out


def _main_key(recipe: str) -> str:
    return {"particles": "diameter_mean", "pores": "porosity", "holes": "diameter_mean"}[recipe]


def _sizes_at(an: Any, points: list[tuple[float, float]], scale: Any) -> list[float]:
    """Equivalent diameters (as in the object table) of the objects under the points: the
    table row whose centre is nearest the centre of the outline the point falls in."""
    from skimage.measure import points_in_poly

    if an.objects.empty or not {"x", "y"} <= set(an.objects.columns):
        return []
    col = "eq_diameter" if "eq_diameter" in an.objects else "diameter" if "diameter" in an.objects else None
    if col is None:
        return []
    xs, ys = an.objects["x"].to_numpy(float), an.objects["y"].to_numpy(float)
    out = []
    for x, y in points:
        for poly in _outlines(an):
            if points_in_poly([[x, y]], poly)[0]:
                cx, cy = poly[:, 0].mean() * scale.sx, poly[:, 1].mean() * scale.sy
                out.append(float(an.objects.iloc[int(np.argmin(np.hypot(xs - cx, ys - cy)))][col]))
                break
    return out


__all__ = ["OBJECT_RECIPES", "Candidate", "inside", "tune"]
