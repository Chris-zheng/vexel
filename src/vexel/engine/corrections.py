"""Human corrections: edits to a run's segmentation and tables, applied on re-execution.

A correction file (``corrections.json``) lists edits. Each edit is applied by the batch
executor right after the step that produces its target, so every later step (measuring,
tracking, statistics) sees the corrected data, and the run stays reproducible:
recipe + input + corrections always give the same results.

Objects are named by a point (``frame`` and ``at = [y, x]`` in pixels), never by label or
track numbers, which can change when segmentation or tracking is re-run.

Edit kinds (generic, independent of the analysis):

``labels.delete``   remove the object under ``at`` from a labels output (false detection)
``labels.merge``    join the objects under the points in ``points`` (one object split in two)
``labels.paint``    add ``shape`` as a new object (missed detection), or add it to the object
                    under ``at`` if given (extend an outline)
``labels.erase``    clear ``shape`` from a labels output (trim, or cut one object in two)
``table.set``       set ``values`` on the row the point resolves to; the row is marked
                    ``verified`` (a person confirmed it)
``table.exclude``   drop the row(s) the point resolves to; on a tracking table this removes
                    the whole track (a droplet that is debris, a bubble, ...)

Shapes are ``{"circle": [y, x, r]}`` or ``{"polygon": [[y, x], ...]}``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
LABEL_KINDS = ("labels.delete", "labels.merge", "labels.paint", "labels.erase")
TABLE_KINDS = ("table.set", "table.exclude")


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["labels.delete", "labels.merge", "labels.paint", "labels.erase", "table.set", "table.exclude"]
    target: str = Field(description="value reference the edit applies to, e.g. 'seg.labels' or 'droplets.droplets'")
    frame: int = Field(ge=0)
    at: tuple[float, float] | None = Field(None, description="[y, x] of a point on the object, pixels")
    points: list[tuple[float, float]] | None = Field(None, description="labels.merge: one point per object")
    shape: dict[str, Any] | None = Field(None, description="labels.paint / labels.erase: {'circle': [y, x, r]} or {'polygon': [[y, x], ...]}")
    values: dict[str, Any] | None = Field(None, description="table.set: column -> value")
    note: str = ""
    author: str = ""
    time: str = ""

    def model_post_init(self, _ctx: Any) -> None:
        need = {
            "labels.delete": ("at",), "labels.merge": ("points",), "labels.paint": ("shape",), "labels.erase": ("shape",),
            "table.set": ("at", "values"), "table.exclude": ("at",),
        }[self.kind]  # fmt: skip
        missing = [k for k in need if getattr(self, k) is None]
        if missing:
            raise ValueError(f"{self.kind} needs {missing}")
        if self.shape is not None and not ({"circle", "polygon"} & set(self.shape)):
            raise ValueError("shape must be {'circle': [y, x, r]} or {'polygon': [[y, x], ...]}")


class Corrections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    base: dict[str, Any] = Field(default_factory=dict, description="the run the edits were made on (recipe and result digests)")
    edits: list[Edit] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Corrections":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def digest(self) -> str:
        canonical = json.dumps([e.model_dump(mode="json", exclude={"author", "time", "note"}) for e in self.edits], sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def add(self, edit: Edit) -> Edit:
        if not edit.time:
            edit = edit.model_copy(update={"time": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        self.edits.append(edit)
        return edit

    def label_edits(self) -> dict[str, dict[int, list[Edit]]]:
        """target -> frame -> edits, in the order they were made."""
        out: dict[str, dict[int, list[Edit]]] = {}
        for e in self.edits:
            if e.kind in LABEL_KINDS:
                out.setdefault(e.target, {}).setdefault(e.frame, []).append(e)
        return out

    def table_edits(self) -> dict[str, list[Edit]]:
        out: dict[str, list[Edit]] = {}
        for e in self.edits:
            if e.kind in TABLE_KINDS:
                out.setdefault(e.target, []).append(e)
        return out


# -- labels -----------------------------------------------------------------------------------


def shape_mask(shape: dict[str, Any], frame_shape: tuple[int, int]) -> np.ndarray:
    from skimage.draw import disk, polygon

    mask = np.zeros(frame_shape, dtype=bool)
    if "circle" in shape:
        y, x, r = (float(v) for v in shape["circle"])
        rr, cc = disk((y, x), r, shape=frame_shape)
    else:
        pts = np.asarray(shape["polygon"], dtype=float)
        rr, cc = polygon(pts[:, 0], pts[:, 1], shape=frame_shape)
    mask[rr, cc] = True
    return mask


def _label_at(labels: np.ndarray, at: tuple[float, float]) -> int:
    y, x = int(round(at[0])), int(round(at[1]))
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return 0
    return int(labels[y, x])


def apply_label_edits(labels: np.ndarray, edits: list[Edit]) -> tuple[np.ndarray, list[str]]:
    """Apply edits in order to one frame's labels. Returns the new labels and, per edit,
    'applied' or why it was not (an edit pointing at background, for instance)."""
    out = np.array(labels, dtype=np.int32, copy=True)
    status = []
    for e in edits:
        if e.kind == "labels.delete":
            k = _label_at(out, e.at)
            if k:
                out[out == k] = 0
            status.append("applied" if k else "no object at the point")
        elif e.kind == "labels.merge":
            ks = sorted({_label_at(out, p) for p in e.points} - {0})
            for k in ks[1:]:
                out[out == k] = ks[0]
            status.append("applied" if len(ks) >= 2 else "fewer than two objects at the points")
        elif e.kind == "labels.paint":
            mask = shape_mask(e.shape, out.shape)
            k = _label_at(out, e.at) if e.at is not None else 0
            out[mask] = k if k else int(out.max()) + 1
            status.append("applied" if mask.any() else "empty shape")
        elif e.kind == "labels.erase":
            mask = shape_mask(e.shape, out.shape)
            out[mask] = 0
            status.append("applied" if mask.any() else "empty shape")
    return out, status


# -- tables -----------------------------------------------------------------------------------


def _detections(tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """The table that places tracked objects in frames: frame, centroid and track_id."""
    need = {"frame", "centroid_y_px", "centroid_x_px", "track_id"}
    for t in tables.values():
        if isinstance(t, pd.DataFrame) and need <= set(t.columns):
            return t
    return None


def _nearest(rows: pd.DataFrame, at: tuple[float, float]) -> int | None:
    """Index of the row whose centroid is nearest ``at``, if the point lies within that
    object (diameter known) or within 10 px."""
    if rows.empty:
        return None
    d = np.hypot(rows["centroid_y_px"].to_numpy(float) - at[0], rows["centroid_x_px"].to_numpy(float) - at[1])
    i = int(np.argmin(d))
    reach = 0.6 * float(rows["diameter_px"].iloc[i]) if "diameter_px" in rows.columns else 10.0
    return int(rows.index[i]) if d[i] <= max(reach, 5.0) else None


def resolve_rows(table: pd.DataFrame, edit: Edit, tables: dict[str, pd.DataFrame]) -> pd.Index:
    """Rows of ``table`` the edit points at."""
    if {"frame", "centroid_y_px", "centroid_x_px"} <= set(table.columns):  # per-frame objects
        i = _nearest(table[table["frame"] == edit.frame], edit.at)
        if i is None:
            return table.index[:0]
        if "track_id" in table.columns and edit.kind == "table.exclude":
            return table.index[table["track_id"] == table.loc[i, "track_id"]]
        return pd.Index([i])
    if "track_id" in table.columns:  # one row per tracked object
        det = _detections(tables)
        if det is None:
            return table.index[:0]
        i = _nearest(det[det["frame"] == edit.frame], edit.at)
        if i is None:
            return table.index[:0]
        return table.index[table["track_id"] == det.loc[i, "track_id"]]
    return table.index[:0]


def apply_table_edits(table: pd.DataFrame, edits: list[Edit], tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Apply edits to one table (``tables`` holds the run's other tables, to resolve points
    to tracked objects). Edited rows get ``verified = True``."""
    out = table.copy()
    status = []
    for e in edits:
        rows = resolve_rows(out, e, {**tables, "__self__": out})
        if not len(rows):
            status.append("no object at the point")
            continue
        if e.kind == "table.exclude":
            out = out.drop(index=rows)
        else:
            for col, value in e.values.items():
                if col not in out.columns:
                    out[col] = pd.Series([None] * len(out), index=out.index, dtype=object)
                elif isinstance(value, (bool, str)) and out[col].dtype != object:
                    out[col] = out[col].astype(object)
                out.loc[rows, col] = value
            if "verified" not in out.columns:
                out["verified"] = False
            out.loc[rows, "verified"] = True
        status.append("applied")
    return out.reset_index(drop=True), status
