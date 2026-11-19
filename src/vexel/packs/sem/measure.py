"""Manual SEM measurements: geometry a person draws on a calibrated image, kept as a record.

A measurement record (``MeasurementSet``) lists items. Each item stores only what the
person drew, in image pixels (``points`` are ``[x, y]``, origin top-left), and the
settings of its tool. Every value (length, area, angle, diameter, edge positions...) is
recomputed from the geometry, the calibration and, for profiles and the magic wand, the
image itself. So a record saved, reloaded and recomputed gives identical numbers, and a
corrected calibration updates every value.

Kinds and what they report:

============  ============================  =================================================
kind          points                        values
============  ============================  =================================================
line          2                             length, angle (to horizontal), dx, dy
polyline      >= 2                          length
polygon       >= 3 (also freehand)          area, perimeter, equivalent diameter, centroid
wand          seed (1) + ``params.tolerance``  as polygon; the region grown from the seed
angle         3 (arm, vertex, arm)          angle
point         >= 1, with a ``category``     count
circle        >= 3 on the edge              diameter, centre, fit residual
ellipse       >= 5 on the edge              major and minor diameter, orientation
profile       2 (+ ``params.width``)        grey levels along the line; edges, widths, spaces,
                                            pitch (see ``profile_edges``)
============  ============================  =================================================

The record also keeps the image's processing chain (``processing``, see ``process.py``):
grey-level tools (magic wand, profiles) measure on the corrected image, and leave out
pixels marked as saturated.

Physical sizes use the calibration's pixel size; with ``tilt`` set (the stage tilt in
degrees, tilt axis horizontal in the image), vertical distances are divided by
cos(tilt): the foreshortening of a tilted, flat sample.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vexel.packs.sem.process import Processing

SCHEMA_VERSION = 1
Kind = Literal["line", "polyline", "polygon", "wand", "angle", "point", "circle", "ellipse", "profile"]
MIN_POINTS = {"line": 2, "polyline": 2, "polygon": 3, "wand": 1, "angle": 3, "point": 1, "circle": 3, "ellipse": 5, "profile": 2}
KIND_NAMES = {"line": "长度", "polyline": "折线", "polygon": "面积", "wand": "魔棒", "angle": "角度", "point": "计数",
              "circle": "圆", "ellipse": "椭圆", "profile": "线剖面"}  # fmt: skip
EDGE_METHODS = ("threshold", "gradient", "peak")


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    kind: Kind
    points: list[tuple[float, float]] = Field(description="[x, y] image pixels")
    name: str = ""
    category: str = ""  # point counts: the class of the marked objects
    params: dict[str, Any] = Field(default_factory=dict)  # tool settings (profile width, edge method, wand tolerance...)
    note: str = ""

    def model_post_init(self, _ctx: Any) -> None:
        if len(self.points) < MIN_POINTS[self.kind]:
            raise ValueError(f"{self.kind} needs at least {MIN_POINTS[self.kind]} points")
        if self.kind == "line" and len(self.points) != 2 or self.kind == "angle" and len(self.points) != 3:
            raise ValueError(f"{self.kind} needs exactly {MIN_POINTS[self.kind]} points")


class Scale(BaseModel):
    """What one pixel is, as used for these measurements (a snapshot of the calibration)."""

    model_config = ConfigDict(extra="forbid")

    pixel_size: float | None = None  # metres per pixel along x; None: pixels only
    pixel_size_y: float | None = None
    source: str = "none"
    confirmed: bool = False
    tilt: float = 0.0  # degrees; vertical distances divided by cos(tilt)

    @property
    def sx(self) -> float:
        return self.pixel_size or 1.0

    @property
    def sy(self) -> float:
        base = self.pixel_size_y or self.pixel_size or 1.0
        return base / math.cos(math.radians(self.tilt)) if self.tilt else base

    @property
    def unit(self) -> str:
        return "m" if self.pixel_size else "px"


class MeasurementSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    image: dict[str, Any] = Field(default_factory=dict)  # file, sha256, shape [rows, cols], crop_rows
    scale: Scale = Field(default_factory=Scale)
    items: list[Item] = Field(default_factory=list)
    processing: Processing = Field(default_factory=Processing)
    author: str = ""
    modified: str = ""

    def next_id(self) -> int:
        return max((i.id for i in self.items), default=0) + 1

    def add(self, kind: str, points: list[tuple[float, float]] | list[list[float]], **kw: Any) -> Item:
        item = Item(id=self.next_id(), kind=kind, points=[(float(x), float(y)) for x, y in points], **kw)
        self.items.append(item)
        return item

    def get(self, item_id: int) -> Item:
        for i in self.items:
            if i.id == item_id:
                return i
        raise KeyError(f"no measurement #{item_id}")

    def remove(self, item_id: int) -> None:
        self.items = [i for i in self.items if i.id != item_id]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self.modified = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MeasurementSet":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def digest(self) -> str:
        data = self.model_dump(exclude={"modified", "author"})
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def scale_from(cal: Any, tilt: float = 0.0) -> Scale:
    """A ``Scale`` from a ``vexel.packs.sem.calibrate.Calibration`` (or None: pixels)."""
    if cal is None or not getattr(cal, "ok", False):
        return Scale(tilt=tilt)
    return Scale(pixel_size=cal.pixel_size, pixel_size_y=cal.pixel_size_y, source=cal.source, confirmed=cal.confirmed, tilt=tilt)


# -- geometry ---------------------------------------------------------------------------------


def _phys(points: Any, s: Scale) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    return p * np.array([s.sx, s.sy])


def _path_length(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def polygon_area_perimeter(p: np.ndarray) -> tuple[float, float, np.ndarray]:
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = cross.sum()
    area = abs(area2) / 2
    closed = np.vstack([p, p[:1]])
    perimeter = _path_length(closed)
    if abs(area2) > 0:
        cx = ((x + np.roll(x, -1)) * cross).sum() / (3 * area2)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (3 * area2)
    else:
        cx, cy = x.mean(), y.mean()
    return float(area), float(perimeter), np.array([cx, cy])


def fit_circle(p: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares circle (algebraic fit, then geometric refinement): cx, cy, r, rms residual."""
    x, y = p[:, 0], p[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx**2 + cy**2, 0.0))
    if len(p) >= 4:
        from scipy.optimize import least_squares

        res = least_squares(lambda q: np.hypot(x - q[0], y - q[1]) - q[2], [cx, cy, r])
        cx, cy, r = res.x
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), float(abs(r)), rms


def fit_ellipse(p: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Direct least-squares ellipse (Fitzgibbon; Halir & Flusser's stable form).

    Returns cx, cy, major diameter, minor diameter, orientation of the major axis
    (degrees from x, counter-clockwise as seen on screen, 0-180); None if not an ellipse."""
    x, y = p[:, 0], p[:, 1]
    mx, my = x.mean(), y.mean()
    sc = max(np.ptp(x), np.ptp(y), 1e-12)
    x, y = (x - mx) / sc, (y - my) / sc  # condition the problem
    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones_like(x)])
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t
    m = np.array([m[2] / 2, -m[1], m[0] / 2])
    w, v = np.linalg.eig(m)
    v = np.real(v)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    ok = np.flatnonzero(cond > 0)
    if not len(ok):
        return None
    a1 = v[:, ok[0]]
    a, b, c, d, e, f = np.concatenate([a1, t @ a1])
    den = b * b - 4 * a * c
    if den >= 0:
        return None
    x0 = (2 * c * d - b * e) / den
    y0 = (2 * a * e - b * d) / den
    num = 2 * (a * e * e + c * d * d - b * d * e + den * f)
    root = math.sqrt((a - c) ** 2 + b * b)
    r1 = -math.sqrt(max(num * (a + c + root), 0)) / den
    r2 = -math.sqrt(max(num * (a + c - root), 0)) / den
    theta = 0.5 * math.atan2(-b, c - a)  # angle of the axis with radius r1
    major, minor = (r1, r2) if r1 >= r2 else (r2, r1)
    if r2 > r1:
        theta += math.pi / 2
    deg = (-math.degrees(theta)) % 180  # image y points down: report counter-clockwise on screen
    return float(x0 * sc + mx), float(y0 * sc + my), float(2 * major * sc), float(2 * minor * sc), float(deg)


def angle_at(p: np.ndarray) -> float:
    v1, v2 = p[0] - p[1], p[2] - p[1]
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-300)
    return float(math.degrees(math.acos(float(np.clip(c, -1, 1)))))


# -- magic wand -------------------------------------------------------------------------------


def wand_region(image: np.ndarray, seed: tuple[float, float], tolerance: float = 0.5, smooth: float = 1.0) -> np.ndarray:
    """Pixels connected to ``seed`` ([x, y]) whose grey level is within ``tolerance`` (a
    fraction of the seed's contrast against the image) of the seed's, after light smoothing.
    0.5 puts the boundary halfway between the object and the background: unbiased for a
    uniform object on a uniform background."""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float64)
    if smooth:
        img = ndi.gaussian_filter(img, smooth)
    x, y = int(round(seed[0])), int(round(seed[1]))
    if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
        raise ValueError("the seed is outside the image")
    lo, hi = np.percentile(img, (1, 99))
    v = img[y, x]
    span = max(hi - lo, abs(v - lo), abs(v - hi), 1e-12)  # contrast of the seed against the image (a small particle is outside 1-99 %)
    within = np.abs(img - v) <= tolerance * span
    lab, _ = ndi.label(within)
    region = lab == lab[y, x]
    return ndi.binary_fill_holes(region)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """The outline of a region as an [x, y] polygon (largest contour, at the pixel boundary)."""
    from skimage.measure import find_contours

    contours = find_contours(np.pad(mask.astype(float), 1), 0.5)
    if not contours:
        return np.zeros((0, 2))
    c = max(contours, key=len) - 1  # undo the padding
    return c[:, ::-1]  # (row, col) -> (x, y)


# -- line profiles ----------------------------------------------------------------------------


def sample_profile(image: np.ndarray, a: tuple[float, float], b: tuple[float, float], width: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Grey levels along a->b ([x, y]) at 1-pixel steps, averaged over ``width`` pixels across.

    Returns (distance along the line in pixels, values)."""
    from scipy import ndimage as ndi

    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    length = float(np.hypot(*(b_ - a_)))
    n = max(int(math.floor(length)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)
    xs, ys = a_[0] + t * (b_[0] - a_[0]), a_[1] + t * (b_[1] - a_[1])
    d = (b_ - a_) / max(length, 1e-12)
    normal = np.array([-d[1], d[0]])
    offsets = np.arange(width) - (width - 1) / 2
    img = np.asarray(image, dtype=np.float64)
    rows = [ndi.map_coordinates(img, [ys + o * normal[1], xs + o * normal[0]], order=1, mode="nearest") for o in offsets]
    return t * length, np.mean(rows, axis=0)


def _crossings(values: np.ndarray, level: float) -> list[tuple[float, int]]:
    out = []
    s = values - level
    for i in range(len(s) - 1):
        if s[i] == 0:
            continue
        if s[i] * s[i + 1] < 0 or s[i + 1] == 0:
            frac = s[i] / (s[i] - s[i + 1])
            out.append((i + frac, 1 if s[i + 1] > s[i] else -1))
    return out


def _peaks(values: np.ndarray, min_height: float, min_distance: int, prominence: float = 0.0) -> list[float]:
    from scipy.signal import find_peaks

    idx, _ = find_peaks(values, height=min_height, distance=max(1, min_distance), prominence=prominence or None)
    out = []
    for i in idx:
        if 0 < i < len(values) - 1:
            y0, y1, y2 = values[i - 1], values[i], values[i + 1]
            den = y0 - 2 * y1 + y2
            out.append(float(i + (0.5 * (y0 - y2) / den if den else 0.0)))
        else:
            out.append(float(i))
    return out


def profile_edges(values: np.ndarray, method: str = "threshold", level: float = 0.5, smooth: float = 1.0,
                  min_distance: int = 3) -> list[tuple[float, int]]:  # fmt: skip
    """Edge positions (samples, sub-pixel) and polarity (+1 rising, -1 falling).

    - ``threshold``: where the profile crosses ``level`` between its low and high grey
      levels (5th and 95th percentiles); level 0.5 is the common "50 %" definition.
    - ``gradient``: the maxima of the slope magnitude (the steepest point of each edge).
    - ``peak``: the bright maxima (the edge-brightening peaks typical of SEM); polarity +1.

    Different definitions give different widths: the definition is part of every result.
    """
    from scipy import ndimage as ndi

    v = ndi.gaussian_filter1d(np.asarray(values, float), smooth) if smooth else np.asarray(values, float)
    lo, hi = np.percentile(v, (5, 95))
    if hi - lo <= 1e-12:
        return []
    if method == "threshold":
        return _crossings(v, lo + level * (hi - lo))
    if method == "gradient":
        g = np.gradient(v)
        mag = np.abs(g)
        pos = _peaks(mag, 0.3 * mag.max(), min_distance, prominence=0.3 * mag.max())
        return [(p, 1 if g[int(round(p))] > 0 else -1) for p in pos]
    if method == "peak":
        return [(p, 1) for p in _peaks(v, lo + level * (hi - lo), min_distance, prominence=0.3 * (hi - lo))]
    raise ValueError(f"edge method must be one of {EDGE_METHODS}")


def period(values: np.ndarray) -> float | None:
    """The dominant period of the profile (samples), from the autocorrelation's first peak, sub-pixel."""
    v = np.asarray(values, float) - np.mean(values)
    n = len(v)
    if n < 8 or not np.any(v):
        return None
    ac = np.correlate(v, v, mode="full")[n - 1 :]
    ac /= ac[0]
    # first local maximum after the first zero crossing
    neg = np.flatnonzero(ac < 0)
    if not len(neg):
        return None
    start = int(neg[0])
    if start >= n - 2:
        return None
    i = start + int(np.argmax(ac[start : n * 3 // 4] if n * 3 // 4 > start + 1 else ac[start:]))
    if ac[i] < 0.2 or not 0 < i < n - 1:
        return None
    y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
    den = y0 - 2 * y1 + y2
    return float(i + (0.5 * (y0 - y2) / den if den else 0.0))


def _stats(a: list[float]) -> dict[str, float]:
    if not a:
        return {}
    arr = np.asarray(a, float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": len(arr)}


# -- values -----------------------------------------------------------------------------------


def compute(item: Item, scale: Scale, image: np.ndarray | None = None, mask: np.ndarray | None = None) -> dict[str, Any]:
    """All values of one item: physical sizes in metres (or pixels when uncalibrated), angles in degrees.

    ``mask``: pixels not to use (saturated); the magic wand leaves them out of its region
    and a profile drops edges inside them and widths or spaces across them."""
    p = _phys(item.points, scale)
    px = np.asarray(item.points, float)
    out: dict[str, Any] = {"id": item.id, "kind": item.kind, "name": item.name, "unit": scale.unit}
    k = item.kind
    if k == "line":
        d = p[1] - p[0]
        out.update(length=float(np.hypot(*d)), dx=float(abs(d[0])), dy=float(abs(d[1])),
                   angle=float((math.degrees(math.atan2(-d[1], d[0])) + 180) % 180), length_px=float(np.hypot(*(px[1] - px[0]))))  # fmt: skip
    elif k == "polyline":
        out.update(length=_path_length(p), length_px=_path_length(px))
    elif k in ("polygon", "wand"):
        poly_px = px
        if k == "wand":
            if image is None:
                raise ValueError("the magic wand needs the image")
            x0, y0 = (int(round(v)) for v in item.points[0])
            if mask is not None and 0 <= y0 < mask.shape[0] and 0 <= x0 < mask.shape[1] and mask[y0, x0]:
                raise ValueError("the seed is in a saturated area")
            region = wand_region(image, item.points[0], float(item.params.get("tolerance", 0.5)), float(item.params.get("smooth", 1.0)))
            if mask is not None and mask.shape == region.shape and (region & mask).any():
                from scipy import ndimage as ndi

                region &= ~mask
                lab, _ = ndi.label(region)
                region = lab == lab[y0, x0]
                out["note"] = "区域与饱和区相接，饱和像素未计入"
            mask = region  # noqa: PLW2901 - from here on: the region
            poly_px = mask_outline(mask)
            out["pixels"] = int(mask.sum())
            out["outline"] = poly_px.tolist()
            area_px_count = float(mask.sum())
        if len(poly_px) < 3:
            raise ValueError("the region is empty")
        area, perim, c = polygon_area_perimeter(poly_px * np.array([scale.sx, scale.sy]))
        if k == "wand":
            area = area_px_count * scale.sx * scale.sy  # count pixels: the region's exact area
        out.update(area=area, perimeter=perim, eq_diameter=float(2 * math.sqrt(area / math.pi)), cx=float(c[0]), cy=float(c[1]))
    elif k == "angle":
        out.update(angle=angle_at(p))
    elif k == "point":
        out.update(count=len(item.points), category=item.category)
    elif k == "circle":
        cx, cy, r, rms = fit_circle(p)
        out.update(diameter=2 * r, cx=cx, cy=cy, residual=rms, n_points=len(p))
        if len(p) == 3:
            out["note"] = "3 点确定的圆，没有拟合误差估计"
    elif k == "ellipse":
        e = fit_ellipse(p)
        if e is None:
            raise ValueError("these points do not fit an ellipse")
        out.update(cx=e[0], cy=e[1], major=e[2], minor=e[3], orientation=e[4], n_points=len(p))
    elif k == "profile":
        if image is None:
            raise ValueError("a line profile needs the image")
        width = int(item.params.get("width", 5))
        method = str(item.params.get("method", "threshold"))
        level = float(item.params.get("level", 0.5))
        smooth = float(item.params.get("smooth", 1.0))
        dist_px, values = sample_profile(image, item.points[0], item.points[1], width)
        step = float(np.hypot(*(p[1] - p[0]))) / max(dist_px[-1], 1e-12)  # metres (or px) per sample
        edges = profile_edges(values, method, level, smooth)
        ds = dist_px[1] - dist_px[0]
        bad = None
        if mask is not None and mask.shape == np.shape(image) and mask.any():
            bad = sample_profile(mask.astype(float), item.points[0], item.points[1], width)[1] > 0
            if bad.any():
                edges = [e for e in edges if not bad[min(int(round(e[0])), len(bad) - 1)]]
                out["note"] = f"剖面经过饱和区（{bad.mean():.0%}）：其中的边缘和跨越它的线宽、间隔不计"
            else:
                bad = None
        pos = [e[0] * ds for e in edges]  # samples -> px along the line
        polar = [e[1] for e in edges]
        to_phys = step

        def clear(a: float, b: float) -> bool:  # no saturated sample between two edges
            return bad is None or not bad[int(math.floor(a / ds)) : int(math.ceil(b / ds)) + 1].any()

        widths, spaces = [], []
        for (a, pa), (b, pb) in zip(zip(pos, polar), zip(pos[1:], polar[1:])):
            if method == "peak" or not clear(a, b):
                continue
            if pa > 0 and pb < 0:
                widths.append((b - a) * to_phys)  # bright feature: rising then falling
            elif pa < 0 and pb > 0:
                spaces.append((b - a) * to_phys)
        same = [b - a for (a, pa), (b, pb) in _pairs_same_polarity(pos, polar) if clear(a, b)]
        per = period(values) if bad is None else None  # the autocorrelation would include the saturated stretch
        out.update(method=method, level=level if method != "gradient" else None, width_px=width,
                   length=float(np.hypot(*(p[1] - p[0]))), edges=[x * to_phys for x in pos], polarity=polar,
                   widths=widths, spaces=spaces, width_mean=_stats(widths).get("mean"), width_sd=_stats(widths).get("sd"),
                   space_mean=_stats(spaces).get("mean"), pitch_edges=float(np.mean(same)) * to_phys if same else None,
                   pitch=per * (dist_px[1] - dist_px[0]) * to_phys if per else None,
                   peak_spacing=float(np.mean(np.diff(pos))) * to_phys if method == "peak" and len(pos) > 1 else None,
                   profile=values.tolist(), distance=(dist_px * to_phys).tolist())  # fmt: skip
    return out


def _pairs_same_polarity(pos: list[float], polar: list[int]) -> list[tuple[tuple[float, int], tuple[float, int]]]:
    out = []
    for sign in (1, -1):
        sel = [(x, s) for x, s in zip(pos, polar) if s == sign]
        out += list(zip(sel, sel[1:]))
    return out


def compute_all(ms: MeasurementSet, image: np.ndarray | None = None, mask: np.ndarray | None = None) -> list[dict[str, Any]]:
    out = []
    for item in ms.items:
        try:
            out.append(compute(item, ms.scale, image, mask))
        except ValueError as e:
            out.append({"id": item.id, "kind": item.kind, "name": item.name, "error": str(e)})
    return out
