"""SEM measurements over HTTP, for the web workbench: the browser sends the image it shows (its
pixels, or the original file so the instrument metadata and data bar can calibrate it) and
gets the validated Python measurement back - numbers, units, warnings and the outlines to draw.

- ``GET  /api/v1/sem/recipes?lang=zh|en``: every SEM measurement with its settings (names,
  kinds, choices, ranges, defaults), from the same definitions the SEM window uses.
- ``POST /api/v1/sem/measure``: one measurement of one image (see ``MeasureRequest``).

Images are held in memory or a temporary file for the request and deleted after it; nothing
is kept and nothing leaves this computer.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from vexel import __version__

MAX_BYTES = 96 * 1024 * 1024  # one request's image
UNITS = {"m": 1.0, "mm": 1e-3, "µm": 1e-6, "um": 1e-6, "nm": 1e-9, "pm": 1e-12, "px": None}


class MeasureRequest(BaseModel):
    recipe: str
    params: dict[str, Any] = Field(default_factory=dict)
    region: list[float] | None = None  # [x0, y0, x1, y1] image pixels
    # the image: either the pixels the browser shows (grey, float32 little-endian, row-major)...
    pixels: str | None = None
    width: int | None = None
    height: int | None = None
    # ...or the original file (TIFF keeps the instrument metadata and data bar for calibration)
    file: str | None = None
    filename: str = "image.tif"
    pixel_size: float | None = None  # in ``unit`` per pixel; given: used as is (a calibration the person set)
    unit: str = "nm"
    crop_rows: int | None = None  # rows of the micrograph above the data bar
    lang: str = "zh"


def _decode(text: str) -> bytes:
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("the image is not valid base64") from e
    if len(raw) > MAX_BYTES:
        raise ValueError(f"the image is larger than {MAX_BYTES // 2**20} MB")
    return raw


def _unit_for(px: float | None, cols: int) -> tuple[str, float]:
    if not px:
        return "px", 1.0
    field = px * max(cols, 1)
    for unit, f in (("nm", 1e-9), ("µm", 1e-6), ("mm", 1e-3)):
        if field < f * 1e4:
            return unit, f
    return "mm", 1e-3


def recipes(lang: str = "zh") -> list[dict[str, Any]]:
    from vexel.i18n import tr_in
    from vexel.packs.sem.analysis import RECIPES

    out = []
    for rec in RECIPES.values():
        params = []
        for p in rec.params:
            d: dict[str, Any] = {"name": p.name, "label": tr_in(p.label, lang), "kind": p.kind, "default": p.default, "help": tr_in(p.help, lang)}
            if p.kind == "choice":
                d["choices"] = [{"value": v, "label": tr_in(lab, lang)} for v, lab in p.choices]
            elif p.kind in ("int", "float"):
                d.update(min=p.lo, max=p.hi, step=p.step)
            params.append(d)
        out.append({"name": rec.name, "label": tr_in(rec.label, lang), "help": tr_in(rec.help, lang), "params": params})
    return out


def measure(req: MeasureRequest) -> dict[str, Any]:
    """One SEM measurement, as the SEM window makes it: the record's default processing
    (saturation marked), the recipe's settings, the pixel size from the person, the file's
    metadata or its data bar (in that order)."""
    from vexel.i18n import tr_in
    from vexel.packs.sem import calibrate as semcal
    from vexel.packs.sem import process
    from vexel.packs.sem.analysis import RECIPES, AnalysisSpec, objects_table, run
    from vexel.packs.sem.measure import Scale, scale_from

    if req.recipe not in RECIPES:
        raise ValueError(f"unknown SEM measurement {req.recipe!r}; known: {', '.join(RECIPES)}")
    lang = "en" if req.lang == "en" else "zh"
    cal = None
    if req.file:
        raw = _decode(req.file)
        suffix = Path(req.filename).suffix.lower() if Path(req.filename).suffix else ".tif"
        with tempfile.TemporaryDirectory(prefix="vexel-sem-") as d:
            path = Path(d) / f"image{suffix}"
            path.write_bytes(raw)
            image = semcal.load_image(path)
            if req.pixel_size is None:
                cal = semcal.calibrate(path)
    elif req.pixels and req.width and req.height:
        raw = _decode(req.pixels)
        n = int(req.width) * int(req.height)
        if len(raw) != 4 * n:
            raise ValueError(f"expected {n} float32 pixels ({4 * n} bytes), got {len(raw)} bytes")
        if n > semcal.MAX_PIXELS:
            raise ValueError(f"the image has {n / 1e6:.0f} MP, over the limit of {semcal.MAX_PIXELS / 1e6:.0f} MP")
        image = np.frombuffer(raw, dtype="<f4").reshape(int(req.height), int(req.width)).astype(np.float64)
    else:
        raise ValueError("send the image: 'pixels' with 'width' and 'height', or 'file'")

    crop_rows = req.crop_rows
    if req.pixel_size is not None:
        factor = UNITS.get(req.unit)
        scale = Scale(pixel_size=req.pixel_size * factor, source="manual", confirmed=True) if factor else Scale()
        calibration = {"source": "person", "confirmed": True, "text": f"{req.pixel_size:g} {req.unit}/px"}
    elif cal is not None and cal.ok:
        scale = scale_from(cal, 0.0)
        crop_rows = crop_rows or cal.crop_rows
        calibration = {"source": cal.source, "confirmed": bool(cal.confirmed), "text": tr_in(cal.describe(), lang),
                       "warnings": [tr_in(w, lang) for w in cal.warnings]}  # fmt: skip
    else:
        scale = Scale()
        calibration = {"source": "none", "confirmed": False, "text": tr_in("未标定：长度以像素为单位", lang)}

    res = process.Processor(image, crop_rows).run(process.Processing())
    spec = AnalysisSpec(id=1, recipe=req.recipe, region=req.region, params=dict(req.params))
    an = run(spec, res.analysis, scale, res.mask, crop_rows)
    unit, f = _unit_for(scale.pixel_size, image.shape[1])
    summary = []
    for key, label, v, kind in an.summary:
        if kind == "length" and v is not None and scale.pixel_size:
            v, u = v / f, unit
        elif kind == "length":
            u = "px"
        elif kind == "area" and v is not None and scale.pixel_size:
            v, u = v / f**2, f"{unit}²"
        else:
            u = "°" if kind == "angle" else ""
        summary.append({"key": key, "label": tr_in(label, lang), "value": tr_in(v, lang) if isinstance(v, str) else v, "unit": u, "kind": kind})
    table = objects_table([an], unit, f) if not an.error else None
    objects = [] if table is None or table.empty else table.drop(columns=["analysis", "recipe"], errors="ignore").head(1000).to_dict("records")
    shapes = [{"kind": k, "points": g if k != "ellipse" else list(g), "color": c, "anchor": list(a), "label": lab} for k, g, c, a, lab in an.shapes]
    digest = hashlib.sha256(np.ascontiguousarray(image, dtype=np.float32).tobytes()).hexdigest()
    return {"engine": "vexel-python", "version": __version__, "validated": True, "recipe": req.recipe,
            "recipe_label": tr_in(RECIPES[req.recipe].label, lang), "calibration": calibration, "unit": unit if scale.pixel_size else "px",
            "summary": summary, "objects": objects, "shapes": shapes, "warnings": [tr_in(w, lang) for w in an.warnings],
            "error": tr_in(an.error, lang) if an.error else None, "region": an.extra.get("region"), "crop_rows": crop_rows,
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0]), "sha256": digest},
            "settings": RECIPES[req.recipe].resolve(dict(req.params))}  # fmt: skip


def register(app: Any) -> None:
    from fastapi import HTTPException

    @app.get("/api/v1/sem/recipes")
    def sem_recipes(lang: str = "zh") -> dict[str, Any]:
        return {"recipes": recipes("en" if lang == "en" else "zh")}

    @app.post("/api/v1/sem/measure")
    def sem_measure(req: MeasureRequest) -> dict[str, Any]:
        try:
            return measure(req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e


__all__ = ["MeasureRequest", "measure", "recipes", "register"]
