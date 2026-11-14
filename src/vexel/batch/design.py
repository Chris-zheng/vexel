"""Experiment design table.

A CSV file (UTF-8; files saved by Excel in GBK also work), one row per recording:

=================  ===========================================================================
column             meaning
=================  ===========================================================================
``file``           path to the recording (video, TIFF, image-sequence folder, OME-Zarr),
                   relative to the table's folder or absolute. Required.
``group``          experimental condition, e.g. ``ratio_1:2``. Needed for group statistics.
``replicate``      independent replicate (chip, day, preparation). Files sharing a group and a
                   replicate are averaged into one value before statistics. When left empty,
                   every file counts as its own replicate.
``pixel_size_um``  calibration for this file (overrides file metadata). Optional.
``frame_interval_s`` time between frames in seconds. Optional.
``settings``       recipe settings for this file only, ``key=value; key=value``. Optional.
``include``        ``no`` / ``0`` / ``false`` leaves the row out. Optional.
anything else      kept as metadata and copied into the result tables.
=================  ===========================================================================
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

KNOWN = ["file", "group", "replicate", "pixel_size_um", "frame_interval_s", "settings", "include"]
VIDEO = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
STACK = {".tif", ".tiff", ".npy"}
IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {"truth", "runs", "review", "tables", "__pycache__"}


class DesignError(ValueError):
    pass


@dataclass
class Row:
    index: int  # 1-based row number in the table
    file: Path
    group: str = ""
    replicate: str = ""
    pixel_size_um: float | None = None
    frame_interval_s: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    label: str = ""  # readable part of the run folder name; set by Design when file names repeat

    @property
    def id(self) -> str:
        """Folder name of this row's run: row number plus a readable stem."""
        stem = self.label or (self.file.name.split(".")[0] if self.file.name else "file")
        stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")[:40] or "file"
        return f"{self.index:03d}_{stem}"

    @property
    def unit(self) -> str:
        """Statistical unit: the replicate within its group, or the file itself."""
        return f"{self.group}/{self.replicate}" if self.replicate else f"{self.group}/{self.id}"


@dataclass
class Design:
    rows: list[Row]
    path: Path | None = None
    sha256: str = ""
    columns: list[str] = field(default_factory=list)

    @property
    def groups(self) -> list[str]:
        out: list[str] = []
        for r in self.rows:
            if r.group not in out:
                out.append(r.group)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Design":
        path = Path(path)
        raw = path.read_bytes()
        text = None
        for enc in ("utf-8-sig", "gb18030"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise DesignError(f"{path}: not readable as UTF-8 or GBK text")
        try:
            table = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            raise DesignError(f"{path}: not a CSV table ({e})") from e
        design = cls.from_frame(table, base=path.parent)
        design.path = path
        design.sha256 = hashlib.sha256(raw).hexdigest()
        return design

    @classmethod
    def from_frame(cls, table: pd.DataFrame, base: Path | None = None) -> "Design":
        table = table.rename(columns={c: str(c).strip().lower() for c in table.columns})
        if "file" not in table.columns:
            raise DesignError("the design table needs a 'file' column")
        rows, problems = [], []
        for i, rec in enumerate(table.to_dict("records"), start=1):
            rec = {k: (str(v).strip() if v is not None else "") for k, v in rec.items()}
            if not rec.get("file"):
                continue
            if rec.get("include", "").lower() in ("no", "n", "0", "false", "否", "不"):
                continue
            f = Path(rec["file"]).expanduser()
            if not f.is_absolute() and base is not None:
                f = base / f
            try:
                row = Row(
                    index=i,
                    file=f,
                    group=rec.get("group", ""),
                    replicate=rec.get("replicate", ""),
                    pixel_size_um=_number(rec.get("pixel_size_um", ""), "pixel_size_um"),
                    frame_interval_s=_number(rec.get("frame_interval_s", ""), "frame_interval_s"),
                    settings=parse_settings(rec.get("settings", "")),
                    meta={k: v for k, v in rec.items() if k not in KNOWN},
                )
            except DesignError as e:
                problems.append(f"row {i}: {e}")
                continue
            rows.append(row)
        if problems:
            raise DesignError("; ".join(problems))
        if not rows:
            raise DesignError("the design table lists no files")
        seen: dict[str, int] = {}
        for r in rows:
            key = str(r.file)
            if key in seen:
                raise DesignError(f"row {r.index}: {r.file} is already listed in row {seen[key]}")
            seen[key] = r.index
        stems = [r.file.name.split(".")[0] for r in rows]
        for r, stem in zip(rows, stems):
            if stems.count(stem) > 1:  # e.g. every folder holds droplets.ome.tif: name the run after the folder
                r.label = f"{r.file.parent.name}_{stem}"
        return cls(rows=rows, columns=list(table.columns))

    def missing(self) -> list[Row]:
        return [r for r in self.rows if not r.file.exists()]

    def to_frame(self) -> pd.DataFrame:
        recs = []
        for r in self.rows:
            rec = {"file": str(r.file), "group": r.group, "replicate": r.replicate,
                   "pixel_size_um": r.pixel_size_um, "frame_interval_s": r.frame_interval_s,
                   "settings": format_settings(r.settings), **r.meta}  # fmt: skip
            recs.append(rec)
        return pd.DataFrame(recs)


def _number(text: str, name: str) -> float | None:
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise DesignError(f"{name} must be a number, got {text!r}") from None
    if value <= 0:
        raise DesignError(f"{name} must be positive, got {text!r}")
    return value


def parse_settings(text: str) -> dict[str, Any]:
    """``key=value; key=value`` (also ``,`` or newlines between items); values as JSON when possible."""
    out: dict[str, Any] = {}
    for item in re.split(r"[;\n；]+", text or ""):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise DesignError(f"setting {item!r} is not key=value")
        key, value = (s.strip() for s in item.split("=", 1))
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def format_settings(settings: dict[str, Any]) -> str:
    return "; ".join(f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}" for k, v in settings.items())


def _is_sequence(folder: Path) -> bool:
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGES and not p.name.startswith(".")]
    if len(images) < 5:
        return False
    tifs = [p for p in images if p.suffix.lower() in (".tif", ".tiff")]
    if tifs and len(tifs) == len(images):
        import tifffile

        try:
            with tifffile.TiffFile(tifs[0]) as tf:
                return len(tf.pages) == 1  # one image per file: a sequence; stacks are separate recordings
        except Exception:  # noqa: BLE001
            return False
    return True


def find_recordings(root: str | Path) -> list[Path]:
    """Recordings under ``root``: videos, TIFF stacks, .npy, OME-Zarr folders, image-sequence folders."""
    root = Path(root)
    found: list[Path] = []

    def walk(folder: Path) -> None:
        if folder != root and _is_sequence(folder):
            found.append(folder)
            return
        for p in sorted(folder.iterdir()):
            if p.name.startswith("."):
                continue
            if p.is_dir():
                if p.suffix.lower() == ".zarr":
                    found.append(p)
                elif p.name.lower() not in SKIP_DIRS:
                    walk(p)
            elif p.suffix.lower() in VIDEO or p.suffix.lower() in STACK:
                if p.suffix.lower() in (".tif", ".tiff") and _is_sequence(p.parent) and p.parent != root:
                    continue
                found.append(p)

    walk(root)
    return found


def scan_folder(root: str | Path) -> pd.DataFrame:
    """Draft a design table for every recording under ``root``. The group is the first
    sub-folder name and the replicate the second, when the files are nested that deep;
    edit the table before running."""
    root = Path(root)
    recs = []
    for p in find_recordings(root):
        parts = p.relative_to(root).parts[:-1]
        recs.append({"file": str(p.relative_to(root)), "group": parts[0] if parts else "",
                     "replicate": parts[1] if len(parts) >= 2 else "", "pixel_size_um": "", "frame_interval_s": "",
                     "settings": "", "note": ""})  # fmt: skip
    return pd.DataFrame(recs, columns=["file", "group", "replicate", "pixel_size_um", "frame_interval_s", "settings", "note"])
