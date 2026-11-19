"""Measurement tables, exports and records.

- ``tables(ms, image)``: the measurement table (one row per item, in a readable unit),
  the profile edges, a summary per kind and category, and the calibration.
- ``export_csv`` / ``export_xlsx``: those tables as files (Excel: one sheet each).
- ``render`` / ``export_png`` / ``export_svg``: the micrograph with every measurement drawn,
  numbered and labelled, and a scale bar drawn by Vexel (the instrument's data bar is
  cropped off).
- ``record_path`` / ``autosave`` / ``load_record``: the record of an image is kept by its
  SHA-256 in ``$VEXEL_HOME/sem/measurements``, so reopening the image restores it.
"""

from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vexel.packs.sem.measure import KIND_NAMES, MeasurementSet, compute_all

COLOURS = ["#f59e0b", "#22d3ee", "#a3e635", "#f472b6", "#a78bfa", "#fb7185", "#34d399", "#60a5fa"]
LENGTH_KEYS = ("length", "dx", "dy", "perimeter", "eq_diameter", "diameter", "major", "minor", "width_mean", "width_sd",
               "space_mean", "pitch", "pitch_edges", "peak_spacing", "residual")  # fmt: skip


def display_unit(ms: MeasurementSet) -> tuple[str, float]:
    """A readable length unit for the table and its size in metres (px when uncalibrated)."""
    px = ms.scale.pixel_size
    if not px:
        return "px", 1.0
    field = px * max(ms.image.get("shape", [0, 1000])[1], 1)
    for unit, f in (("nm", 1e-9), ("µm", 1e-6), ("mm", 1e-3)):
        if field < f * 1e4:
            return unit, f
    return "mm", 1e-3


def tables(ms: MeasurementSet, image: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    unit, f = display_unit(ms)
    values = compute_all(ms, image)
    rows, edges = [], []
    for v in values:
        row: dict[str, Any] = {"id": v["id"], "tool": KIND_NAMES.get(v["kind"], v["kind"]), "kind": v["kind"], "name": v.get("name", "")}
        if "error" in v:
            row["error"] = v["error"]
            rows.append(row)
            continue
        for key in LENGTH_KEYS:
            if v.get(key) is not None:
                row[f"{key} ({unit})"] = v[key] / f
        if v.get("area") is not None:
            row[f"area ({unit}²)"] = v["area"] / f**2
        for key in ("angle", "orientation"):
            if v.get(key) is not None:
                row[f"{key} (°)"] = v[key]
        if v["kind"] == "point":
            row["category"], row["count"] = v.get("category", ""), v["count"]
        if v["kind"] == "profile":
            row["edge definition"] = v["method"] + (f" {v['level']:.0%}" if v.get("level") is not None and v["method"] != "gradient" else "")
            row["profile width (px)"] = v["width_px"]
            row["edges"] = len(v["edges"])
            for i, (pos, pol) in enumerate(zip(v["edges"], v["polarity"]), start=1):
                edges.append({"id": v["id"], "edge": i, f"position ({unit})": pos / f, "polarity": "rising" if pol > 0 else "falling"})
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = []
    if len(table):
        main = {"line": f"length ({unit})", "polyline": f"length ({unit})", "polygon": f"area ({unit}²)", "wand": f"area ({unit}²)",
                "angle": "angle (°)", "circle": f"diameter ({unit})", "ellipse": f"major ({unit})", "profile": f"width_mean ({unit})"}  # fmt: skip
        for kind, col in main.items():
            sel = table[(table["kind"] == kind)] if "kind" in table else table.iloc[0:0]
            if col in sel and sel[col].notna().any():
                s = sel[col].dropna()
                summary.append({"tool": KIND_NAMES[kind], "quantity": col, "n": len(s), "mean": s.mean(), "sd": s.std(ddof=1) if len(s) > 1 else 0.0,
                                "min": s.min(), "max": s.max()})  # fmt: skip
        if "count" in table:
            for cat, s in table[table["kind"] == "point"].groupby("category"):
                summary.append({"tool": KIND_NAMES["point"], "quantity": f"count ({cat or '未分类'})", "n": len(s), "mean": float(s["count"].sum()),
                                "sd": None, "min": None, "max": None})  # fmt: skip
    sc = ms.scale
    calib = pd.DataFrame([{"file": ms.image.get("file", ""), "sha256": ms.image.get("sha256", ""),
                           "pixel size (m)": sc.pixel_size, "pixel size y (m)": sc.pixel_size_y, "source": sc.source,
                           "confirmed": sc.confirmed, "tilt correction (°)": sc.tilt, "record digest": ms.digest()}])  # fmt: skip
    return {"measurements": table, "profile_edges": pd.DataFrame(edges), "summary": pd.DataFrame(summary), "calibration": calib}


def export_csv(ms: MeasurementSet, image: np.ndarray | None, path: str | Path) -> list[Path]:
    """The measurement table at ``path``; profile edges and summary beside it (``*_edges.csv``, ``*_summary.csv``)."""
    path = Path(path)
    t = tables(ms, image)
    out = [path]
    t["measurements"].to_csv(path, index=False, encoding="utf-8-sig")  # Excel opens UTF-8 with a BOM correctly
    for key, suffix in (("profile_edges", "_edges"), ("summary", "_summary")):
        if len(t[key]):
            p = path.with_name(path.stem + suffix + path.suffix)
            t[key].to_csv(p, index=False, encoding="utf-8-sig")
            out.append(p)
    return out


def export_xlsx(ms: MeasurementSet, image: np.ndarray | None, path: str | Path) -> Path:
    path = Path(path)
    t = tables(ms, image)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, sheet in (("measurements", "测量"), ("summary", "汇总"), ("profile_edges", "剖面边缘"), ("calibration", "标定")):
            df = t[name]
            (df if len(df) else pd.DataFrame({"": []})).to_excel(xw, sheet_name=sheet, index=False)
    return path


# -- annotated image --------------------------------------------------------------------------


_FONTS = ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
          "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")  # fmt: skip
_ASCII = str.maketrans({"²": "2", "×": "x", "Ø": "D", "°": " deg", "µ": "u"})


def _font(size: int) -> tuple[Any, bool]:
    """A font with "µ ² × Ø °" when the system has one (True), else Pillow's built-in (False: use ASCII)."""
    from PIL import ImageFont

    for name in _FONTS:
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    return ImageFont.load_default(size=size), False


def _nice(length: float) -> float:
    exp = math.floor(math.log10(length))
    return max((n * 10**exp for n in (1, 2, 5) if n * 10**exp <= length), default=10**exp)


def _fmt_len(metres: float) -> str:
    for unit, f in (("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9)):
        if metres >= f * 0.999:
            v = metres / f
            return f"{v:.3g} {unit}"
    return f"{metres * 1e12:.3g} pm"


def num(value: float) -> str:
    """4 significant digits, never in exponent form (6.125e+04 -> 61,250)."""
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def label_of(v: dict[str, Any], unit: str, f: float) -> str:
    if "error" in v:
        return f"#{v['id']} !"
    k = v["kind"]
    if k in ("line", "polyline"):
        text = f"{num(v['length'] / f)} {unit}"
    elif k in ("polygon", "wand"):
        text = f"{num(v['area'] / f**2)} {unit}²"
    elif k == "angle":
        text = f"{v['angle']:.2f}°"
    elif k == "point":
        text = f"{v['count']}" + (f" {v['category']}" if v.get("category") else "")
    elif k == "circle":
        text = f"Ø {num(v['diameter'] / f)} {unit}"
    elif k == "ellipse":
        text = f"{num(v['major'] / f)} × {num(v['minor'] / f)} {unit}"
    elif k == "profile":
        parts = []
        if v.get("width_mean") is not None:
            parts.append(f"w {num(v['width_mean'] / f)}")
        if v.get("pitch") is not None:
            parts.append(f"p {num(v['pitch'] / f)}")
        text = (" ".join(parts) + f" {unit}") if parts else f"{len(v['edges'])} edges"
    else:
        text = ""
    return f"#{v['id']} {text}"


def _shapes(ms: MeasurementSet, values: list[dict[str, Any]]) -> list[tuple[str, Any, str, tuple[float, float], str]]:
    """(kind of drawing, geometry, colour, label anchor, label) per item, in pixels."""
    unit, f = display_unit(ms)
    by_id = {v["id"]: v for v in values}
    out = []
    for n, item in enumerate(ms.items):
        v = by_id[item.id]
        colour = COLOURS[n % len(COLOURS)]
        pts = [tuple(p) for p in item.points]
        label = label_of(v, unit, f)
        k = item.kind
        if k == "point":
            out.append(("points", pts, colour, pts[-1], label))
        elif k in ("polygon",):
            out.append(("polygon", pts, colour, pts[0], label))
        elif k == "wand":
            outline = [tuple(p) for p in v.get("outline", [])] or pts
            out.append(("polygon", outline, colour, pts[0], label))
        elif k == "circle" and "diameter" in v:
            sx, sy = ms.scale.sx, ms.scale.sy
            cx, cy, r = v["cx"] / sx, v["cy"] / sy, v["diameter"] / 2
            out.append(("ellipse", (cx, cy, r / sx, r / sy, 0.0), colour, (cx + r / sx, cy), label))
        elif k == "ellipse" and "major" in v:
            sx = ms.scale.sx  # drawn in pixels (tilt-corrected ellipses are drawn uncorrected)
            out.append(("ellipse", (v["cx"] / sx, v["cy"] / ms.scale.sy, v["major"] / 2 / sx, v["minor"] / 2 / sx, v["orientation"]), colour, pts[0], label))
        elif k == "angle":
            out.append(("polyline", pts, colour, pts[1], label))
        else:  # line, polyline, profile
            out.append(("polyline", pts, colour, pts[-1], label))
    return out


def render(image: np.ndarray, ms: MeasurementSet, *, crop_rows: int | None = None, scale_bar: bool = True) -> Any:
    """A PIL image: the micrograph (without the data bar), measurements, labels and a scale bar."""
    from PIL import Image, ImageDraw

    img = np.asarray(image, dtype=np.float64)
    if crop_rows:
        img = img[:crop_rows]
    lo, hi = np.percentile(img, (0.5, 99.5))
    gray = np.clip((img - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(gray).convert("RGB")
    dr = ImageDraw.Draw(canvas)
    size = max(12, canvas.width // 70)
    font, unicode_ok = _font(size)

    def say(text: str) -> str:
        return text if unicode_ok else text.translate(_ASCII)

    lw = max(1, canvas.width // 600)
    values = compute_all(ms, image)
    for kind, geom, colour, anchor, label in _shapes(ms, values):
        if kind == "points":
            for x, y in geom:
                r = 3 * lw + 2
                dr.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=lw)
        elif kind == "polygon":
            dr.polygon([tuple(p) for p in geom], outline=colour, width=lw)
        elif kind == "polyline":
            dr.line([tuple(p) for p in geom], fill=colour, width=lw)
            for x, y in (geom[0], geom[-1]):
                dr.line([x, y - 3 * lw, x, y + 3 * lw], fill=colour, width=lw)
        elif kind == "ellipse":
            cx, cy, a, b, deg = geom
            t = np.linspace(0, 2 * np.pi, 97)
            th = math.radians(deg)
            xs = cx + a * np.cos(t) * math.cos(th) + b * np.sin(t) * math.sin(th)
            ys = cy - a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)
            dr.line(list(zip(xs, ys)), fill=colour, width=lw)
        x, y = anchor
        tw = dr.textlength(say(label), font=font)
        tx, ty = min(x + 6, canvas.width - tw - 4), max(y - size - 4, 2)
        box = dr.textbbox((tx, ty), say(label), font=font)
        dr.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(0, 0, 0))
        dr.text((tx, ty), say(label), fill=colour, font=font)
    if scale_bar and ms.scale.pixel_size:
        length = _nice(ms.scale.pixel_size * canvas.width * 0.2)
        n = length / ms.scale.pixel_size
        x0, y0 = int(canvas.width * 0.04), canvas.height - int(size * 2.2)
        text = say(_fmt_len(length))
        box = dr.textbbox((x0, y0 - size - 6), text, font=font)
        dr.rectangle([x0 - 6, box[1] - 4, x0 + n + 6, y0 + 3 * lw + 6], fill=(0, 0, 0))
        dr.rectangle([x0, y0, x0 + n, y0 + 3 * lw], fill=(255, 255, 255))
        dr.text((x0 + n / 2, y0 - 6), text, fill=(255, 255, 255), font=font, anchor="ms")
    return canvas


def export_png(image: np.ndarray, ms: MeasurementSet, path: str | Path, *, crop_rows: int | None = None) -> Path:
    path = Path(path)
    render(image, ms, crop_rows=crop_rows).save(path)
    return path


def export_svg(image: np.ndarray, ms: MeasurementSet, path: str | Path, *, crop_rows: int | None = None) -> Path:
    """The micrograph as an embedded PNG under vector measurements (editable in Illustrator, Inkscape)."""
    from xml.sax.saxutils import escape

    path = Path(path)
    base = render(image, MeasurementSet(image=ms.image, scale=ms.scale), crop_rows=crop_rows)  # picture + scale bar only
    buf = io.BytesIO()
    base.save(buf, format="PNG")
    w, h = base.size
    lw = max(1, w // 600)
    fs = max(12, w // 70)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             f'<image width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>']  # fmt: skip
    for kind, geom, colour, anchor, label in _shapes(ms, compute_all(ms, image)):
        style = f'fill="none" stroke="{colour}" stroke-width="{lw}"'
        if kind == "points":
            parts += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3 * lw + 2}" {style}/>' for x, y in geom]
        elif kind in ("polygon", "polyline"):
            tag = "polygon" if kind == "polygon" else "polyline"
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in geom)
            parts.append(f'<{tag} points="{pts}" {style}/>')
        elif kind == "ellipse":
            cx, cy, a, b, deg = geom
            parts.append(f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{a:.2f}" ry="{b:.2f}" transform="rotate({-deg:.2f} {cx:.2f} {cy:.2f})" {style}/>')
        x, y = anchor
        parts.append(f'<text x="{min(x + 6, w - 10):.1f}" y="{max(y - 4, fs):.1f}" fill="{colour}" font-size="{fs}" font-family="sans-serif" '
                     f'paint-order="stroke" stroke="black" stroke-width="3">{escape(label)}</text>')  # fmt: skip
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# -- records ----------------------------------------------------------------------------------


def records_dir() -> Path:
    return Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel")) / "sem" / "measurements"


def record_path(sha256: str) -> Path:
    return records_dir() / f"{sha256}.json"


def autosave(ms: MeasurementSet) -> Path | None:
    sha = ms.image.get("sha256")
    if not sha:
        return None
    records_dir().mkdir(parents=True, exist_ok=True)
    return ms.save(record_path(sha))


def load_record(sha256: str) -> MeasurementSet | None:
    p = record_path(sha256)
    if not p.is_file():
        return None
    try:
        return MeasurementSet.load(p)
    except (OSError, ValueError):
        return None


def new_set(path: str | Path, cal: Any, tilt: float = 0.0) -> MeasurementSet:
    from vexel.packs.sem.measure import scale_from

    return MeasurementSet(image={"file": Path(path).name, "sha256": cal.sha256, "shape": list(cal.image_shape), "crop_rows": cal.crop_rows},
                          scale=scale_from(cal, tilt))  # fmt: skip
