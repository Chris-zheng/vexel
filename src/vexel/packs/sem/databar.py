"""Find the data bar (the instrument's information strip under the micrograph) and the scale bar in it.

Calibration level 2 works from pixels alone, so it must never guess silently:

- **Data bar**: a full-width horizontal boundary in the lower part of the image, below
  which the picture is mostly one flat background colour (black or white) with text.
  Natural micrograph content almost never has a straight edge across the whole width.
- **Scale bar**: the longest straight horizontal stroke inside the bar that is not a
  table border (< half the width). A label drawn across the middle of the stroke (as on
  Thermo Fisher bars) is bridged; end ticks are part of the stroke. Its length is
  measured from the first to the last lit pixel, inclusive.

Both return None when nothing convincing is found; the caller then reports "无法识别".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class DataBar:
    top: int  # first row of the bar; rows [0, top) are the micrograph
    height: int
    background: str  # "dark" or "light"
    flatness: float  # fraction of bar pixels on the background colour
    edge: float  # fraction of columns that change across the boundary
    source: str = "detected"  # or "metadata"


@dataclass(frozen=True)
class ScaleBar:
    x0: float  # left end, columns (tick or cap centre when the bar has them, else first lit pixel)
    x1: float  # right end (tick or cap centre, else last lit pixel)
    y: int  # row, in full-image coordinates
    thickness: int
    ends: str  # "ticks": measured between tick/cap centres; "edges": first to last lit pixel; "dots": first to last dot

    @property
    def length_px(self) -> float:
        return self.x1 - self.x0 + (1.0 if self.ends == "edges" else 0.0)

    def to_dict(self) -> dict:
        return {"x0": self.x0, "x1": self.x1, "y": self.y, "thickness": self.thickness, "ends": self.ends, "length_px": self.length_px}


def normalise(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(-1) if img.shape[-1] in (3, 4) else img[0]
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def _flat_fraction(region: np.ndarray) -> tuple[float, float]:
    """Fraction of pixels on the dominant grey level (±0.06), and that level."""
    hist, edges = np.histogram(region, bins=64, range=(0.0, 1.0))
    i = int(np.argmax(hist))
    level = (edges[i] + edges[i + 1]) / 2
    return float(np.mean(np.abs(region - level) <= 0.06)), float(level)


def find_databar(image: np.ndarray, *, min_height: int = 8, max_fraction: float = 0.35) -> DataBar | None:
    img = normalise(image)
    h, w = img.shape
    if h < 64 or w < 64:
        return None
    lowest = int(h * (1 - max_fraction))
    diff = np.abs(np.diff(img, axis=0))  # diff[r-1]: change between rows r-1 and r
    change = (diff > 0.15).mean(axis=1)
    best: DataBar | None = None
    for r in range(max(lowest, 1), h - min_height + 1):
        edge = float(change[r - 1])
        if edge < 0.75:
            continue
        flat, level = _flat_fraction(img[r:])
        if flat < 0.55:
            continue
        best = DataBar(top=r, height=h - r, background="dark" if level < 0.5 else "light", flatness=flat, edge=edge)
        break  # the topmost such boundary: everything below belongs to the bar
    return best


def databar_from_rows(image: np.ndarray, scan_rows: int, source: str = "metadata") -> DataBar | None:
    img = normalise(image)
    h = img.shape[0]
    if not 0 < scan_rows < h:
        return None
    flat, level = _flat_fraction(img[scan_rows:])
    edge = float((np.abs(img[scan_rows] - img[scan_rows - 1]) > 0.15).mean())
    return DataBar(top=scan_rows, height=h - scan_rows, background="dark" if level < 0.5 else "light", flatness=flat, edge=edge, source=source)


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], row, [False]])
    d = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _strokes(fg: np.ndarray, min_len: int, max_len: int) -> list[tuple[int, int, int, int]]:
    """Horizontal strokes: (x0, x1, first row, last row). A label across the middle of a stroke is bridged."""
    w = fg.shape[1]
    pieces: list[tuple[int, int, int]] = []
    for y in range(fg.shape[0]):
        merged: list[list[int]] = []  # [x0, x1, length of the last piece]
        for a, b in _runs(fg[y]):
            if b - a + 1 < 8:
                continue
            # bridge a gap only between two pieces that are each stroke-long (a label across a bar),
            # never between the short strokes of letters
            if merged and b - a + 1 >= min_len and merged[-1][2] >= min_len and a - merged[-1][1] - 1 <= max(6, int(0.12 * w)):
                merged[-1][1], merged[-1][2] = b, b - a + 1
            else:
                merged.append([a, b, b - a + 1])
        for a, b, _ in merged:
            if min_len <= b - a + 1 < max_len and fg[y, a : b + 1].mean() >= 0.55:
                pieces.append((a, b, y))
    strokes: list[list[int]] = []  # [x0, x1, y_first, y_last]
    for a, b, y in sorted(pieces, key=lambda p: (p[2], p[0])):
        for s in strokes:
            if abs(a - s[0]) <= 2 and abs(b - s[1]) <= 2 and y - s[3] == 1:
                s[0], s[1], s[3] = min(s[0], a), max(s[1], b), y
                break
        else:
            strokes.append([a, b, y, y])
    return [tuple(s) for s in strokes]  # type: ignore[misc]


def _vertical_extent(fg: np.ndarray, x: int, y0: int, y1: int) -> tuple[int, int]:
    """Rows above y0 and below y1 that are lit in column x, contiguous with the stroke."""
    up = 0
    while y0 - up - 1 >= 0 and fg[y0 - up - 1, x]:
        up += 1
    down = 0
    while y1 + down + 1 < fg.shape[0] and fg[y1 + down + 1, x]:
        down += 1
    return up, down


def _end(fg: np.ndarray, xs: range, y0: int, y1: int, halo: int = 0) -> tuple[float | None, int]:
    """Centre column of a tick or rounded cap at one end, and how far it reaches beyond the stroke.

    ``halo``: how far the stroke's own body reaches beyond its rows (a blurred or JPEG-ringing
    stroke is lit a pixel above or below along its whole length); only columns reaching
    clearly further belong to a tick."""
    cols, reach = [], 0
    for x in xs:
        if 0 <= x < fg.shape[1]:
            up, down = _vertical_extent(fg, x, y0, y1)
            if up + down > halo:
                cols.append(x)
                reach = max(reach, up + down)
    return (float(np.mean(cols)) if cols else None), reach


def _halo(fg: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> int:
    """Typical reach beyond the stroke's rows along its body (median over its columns), plus a margin."""
    xs = range(x0 + (x1 - x0) // 10, x1 - (x1 - x0) // 10 + 1, max(1, (x1 - x0) // 60))
    reach = [sum(_vertical_extent(fg, x, y0, y1)) for x in xs]
    base = int(np.median(reach)) if reach else 0
    return base + 1 if base else 0


def _dotted(fg: np.ndarray) -> list[ScaleBar]:
    """A row of evenly spaced small marks (a dotted scale bar): first to last mark centre."""
    lab, n = ndi.label(fg)
    if n < 6:
        return []
    found = []
    objs = ndi.find_objects(lab)
    blobs = []
    for i, sl in enumerate(objs, start=1):
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh <= 6 and ww <= 6:
            cy, cx = ndi.center_of_mass(lab[sl] == i)
            blobs.append((sl[0].start + cy, sl[1].start + cx, int((lab[sl] == i).sum())))
    blobs.sort(key=lambda b: (round(b[0]), b[1]))
    rows: dict[int, list[tuple[float, float, int]]] = {}
    for b in blobs:
        rows.setdefault(round(b[0]), []).append(b)
    for y, row in rows.items():
        row = sorted(row + rows.get(y + 1, []), key=lambda b: b[1])
        if len(row) < 6:
            continue
        xs = np.array([b[1] for b in row])
        gaps = np.diff(xs)
        areas = np.array([b[2] for b in row], dtype=float)
        if gaps.min() > 1 and gaps.std() / gaps.mean() < 0.08 and areas.std() / areas.mean() < 0.25:
            found.append(ScaleBar(x0=float(xs[0]), x1=float(xs[-1]), y=int(y), thickness=1, ends="dots"))
    return found


def find_scalebar(image: np.ndarray, bar: DataBar | None, *, near: tuple[float, float] | None = None,
                  text_boxes: list[tuple[float, float, float, float]] = ()) -> ScaleBar | None:  # fmt: skip
    """The scale bar inside the data bar (or, without a bar, in the bottom quarter of the image).

    ``near``: (x, y) centre of the recognised scale label, full-image coordinates; when
    given, the bar closest to it wins, else the longest. ``text_boxes``: recognised text
    (x0, y0, x1, y1); strokes lying within a text box are letters, not bars.
    """
    img = normalise(image)
    h, w = img.shape
    top = bar.top if bar is not None else int(h * 0.75)
    region = img[top:]
    _, level = _flat_fraction(region)
    fg = np.abs(region - level) > 0.35
    rh = fg.shape[0]
    found: list[ScaleBar] = []
    for x0, x1, y0, y1 in _strokes(fg, max(12, int(0.03 * w)), int(0.5 * w)):
        thickness = y1 - y0 + 1
        yc = top + (y0 + y1) / 2
        if any(bx0 <= x0 + 0.5 * (x1 - x0) <= bx1 and by0 - 1 <= yc <= by1 + 1 and (min(x1, bx1) - max(x0, bx0)) > 0.5 * (x1 - x0) for bx0, by0, bx1, by1 in text_boxes):
            continue  # a stroke of a letter (e.g. the bar of an "E")
        if thickness > max(3, (x1 - x0 + 1) / 4):
            continue  # a filled box or logo, not a stroke
        span = max(3, int(0.06 * (x1 - x0 + 1)))
        halo = _halo(fg, x0, x1, y0, y1)
        left, left_reach = _end(fg, range(x0 - 1, x0 + span), y0, y1, halo)
        right, right_reach = _end(fg, range(x1 - span + 1, x1 + 2), y0, y1, halo)
        if max(left_reach, right_reach) + thickness >= 0.6 * rh:
            continue  # ends on cell borders that run through the whole bar: a table line
        if left is not None and right is not None:
            found.append(ScaleBar(x0=left, x1=right, y=top + (y0 + y1) // 2, thickness=thickness, ends="ticks"))
        else:
            found.append(ScaleBar(x0=float(x0), x1=float(x1), y=top + (y0 + y1) // 2, thickness=thickness, ends="edges"))
    if not found:
        found = [ScaleBar(x0=s.x0, x1=s.x1, y=top + s.y, thickness=s.thickness, ends=s.ends) for s in _dotted(fg)]
    if not found:
        return None
    if near is None:
        return max(found, key=lambda s: s.length_px)
    return min(found, key=lambda s: np.hypot((s.x0 + s.x1) / 2 - near[0], s.y - near[1]) - 0.25 * s.length_px)


__all__ = ["DataBar", "ScaleBar", "databar_from_rows", "find_databar", "find_scalebar", "normalise", "ndi"]
