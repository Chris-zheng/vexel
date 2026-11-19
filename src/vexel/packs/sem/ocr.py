"""Text in the data bar: the scale label, field width, magnification (optional OCR).

The engine is RapidOCR (Apache-2.0) on ONNX Runtime with the PP-OCR models that ship
inside its wheel (Apache-2.0, from PaddleOCR). It is an optional dependency:

    uv tool install "vexel[sem]"      # or: pip install "vexel[sem]"

Recognition runs entirely on this computer. RapidOCR can download models it does not
find; Vexel points it at the bundled files and blocks that download path, so OCR never
makes a network request (see ``vexel.policy``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from vexel.packs.sem.metadata import UNITS, _mag, parse_length


@dataclass(frozen=True)
class Text:
    text: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in full-image pixels

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "score": round(self.score, 3), "box": [round(v, 1) for v in self.box]}


class OcrUnavailable(RuntimeError):
    pass


def _models_dir() -> Path | None:
    spec = importlib.util.find_spec("rapidocr")  # locate without importing (importing loads OpenCV)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent / "models"


def available() -> tuple[bool, str]:
    models = _models_dir()
    if models is None or importlib.util.find_spec("onnxruntime") is None:
        return False, "未安装文字识别组件（安装 vexel[sem] 后可用）"
    det, rec = sorted(models.glob("*det*.onnx")), sorted(models.glob("*rec*.onnx"))
    if not det or not rec:
        return False, f"文字识别模型文件缺失（{models}）"
    return True, ""


@lru_cache(maxsize=1)
def _engine() -> Any:
    import logging

    ok, why = available()
    if not ok:
        raise OcrUnavailable(why)
    from rapidocr import RapidOCR
    from rapidocr.utils import download_file

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise OcrUnavailable("文字识别组件试图下载文件，已阻止（Vexel 只使用本机的模型文件）")

    download_file.DownloadFile._make_http_request = classmethod(lambda cls, *a, **k: _no_network())  # type: ignore[method-assign]
    models = _models_dir()
    assert models is not None
    det, rec = sorted(models.glob("*det*.onnx"))[-1], sorted(models.glob("*rec*.onnx"))[-1]
    for name in ("RapidOCR", "rapidocr"):
        logging.getLogger(name).setLevel(logging.ERROR)
    params = {"Global.log_level": "error", "Global.use_cls": False, "Det.model_path": str(det), "Rec.model_path": str(rec)}
    return RapidOCR(params=params)


def _recognise(gray: np.ndarray) -> list[tuple[str, float, list[float]]]:
    """Run the engine on an 8-bit grey image: (text, score, [x0, y0, x1, y1])."""
    result = _engine()(np.ascontiguousarray(np.dstack([gray] * 3)))
    boxes = result.boxes if result.boxes is not None else []
    out = []
    for text, score, box in zip(result.txts or (), result.scores or (), boxes):
        b = np.asarray(box, dtype=float)
        out.append((str(text).strip(), float(score), [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())]))
    return out


class _Worker:
    """OCR in a helper process.

    RapidOCR loads OpenCV, whose macOS build bundles the same FFmpeg/AVFoundation classes
    as PyAV (used by the droplet module's video readers); both in one process can crash.
    The helper keeps OpenCV out of the application process and is reused for every image.
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()

    def _start(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.poll() is not None:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen([sys.executable, "-m", "vexel.packs.sem.ocr", "--serve"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)  # fmt: skip
        return self.proc

    def __call__(self, gray: np.ndarray) -> list[tuple[str, float, list[float]]]:
        with self.lock:
            proc = self._start()
            assert proc.stdin is not None and proc.stdout is not None
            header = struct.pack("<II", *gray.shape)
            try:
                proc.stdin.write(header + np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (BrokenPipeError, OSError) as e:
                self.proc = None
                raise OcrUnavailable(f"文字识别进程意外退出：{e}") from e
            if not line:
                self.proc = None
                raise OcrUnavailable("文字识别进程意外退出")
            reply = json.loads(line)
            if not reply.get("ok"):
                raise OcrUnavailable(reply.get("error", "文字识别失败"))
            return [(t, sc, b) for t, sc, b in reply["items"]]


_worker = _Worker()


def _serve() -> None:
    """The helper process: read (rows, cols, pixels) frames from stdin, answer one JSON line each."""
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.read(8)
        if len(header) < 8:
            return
        rows, cols = struct.unpack("<II", header)
        data = stdin.read(rows * cols)
        try:
            gray = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
            reply = {"ok": True, "items": _recognise(gray)}
        except Exception as e:  # noqa: BLE001 - reported to the application
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        stdout.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))
        stdout.flush()


def read_text(image: np.ndarray, top: int = 0) -> list[Text]:
    """Recognise text in a (bar) region; boxes are returned in full-image coordinates (``top`` = region's first row).

    Runs in the helper process unless ``VEXEL_OCR_INPROCESS=1``.
    """
    ok, why = available()
    if not ok:
        raise OcrUnavailable(why)
    region = np.asarray(image, dtype=np.float64)
    lo, hi = float(region.min()), float(region.max())
    gray = ((region - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(region)).astype(np.uint8)
    scale = 1
    if gray.shape[0] < 120:  # small bar fonts read much better enlarged
        scale = 3 if gray.shape[0] < 50 else 2
        gray = np.kron(gray, np.ones((scale, scale), dtype=np.uint8))
    items = _recognise(gray) if os.environ.get("VEXEL_OCR_INPROCESS") == "1" else _worker(gray)
    out = [Text(text=t, score=sc, box=(b[0] / scale, b[1] / scale + top, b[2] / scale, b[3] / scale + top)) for t, sc, b in items]
    return _join_units(out)


_UNIT_ONLY = re.compile(r"\s*(pm|nm|um|µm|μm|mm|Å)\s*", re.IGNORECASE)


def _join_units(texts: list[Text]) -> list[Text]:
    """OCR sometimes splits ``"500 µm"`` into ``"500"`` and ``"µm"``: join a number with the unit right after it."""
    out: list[Text] = []
    for t in sorted(texts, key=lambda t: (round(t.centre[1] / 6), t.box[0])):
        if out and _UNIT_ONLY.fullmatch(t.text) and re.search(r"\d$", out[-1].text):
            prev = out[-1]
            height = prev.box[3] - prev.box[1]
            same_line = min(prev.box[3], t.box[3]) - max(prev.box[1], t.box[1]) > 0.5 * height
            if same_line and t.box[0] - prev.box[2] < height:
                out[-1] = Text(text=f"{prev.text} {t.text.strip()}", score=min(prev.score, t.score),
                               box=(prev.box[0], min(prev.box[1], t.box[1]), t.box[2], max(prev.box[3], t.box[3])))  # fmt: skip
                continue
        out.append(t)
    return out


# -- reading the values ----------------------------------------------------------------------

_LENGTH = re.compile(r"(\d+(?:[.,]\d+)?)\s*(pm|nm|um|µm|μm|mm|Å)(?![a-zA-Z])", re.IGNORECASE)
_UNIT_FIX = {"μm": "µm", "um": "µm", "uM": "µm", "UM": "µm"}


def lengths_in(text: str) -> list[float]:
    out = []
    for m in _LENGTH.finditer(text.replace(" ", " ")):
        unit = _UNIT_FIX.get(m.group(2), m.group(2).lower())
        value = parse_length(f"{m.group(1)} {unit}")
        if value is not None:
            out.append(value)
    return out


def _label_only(text: str) -> bool:
    """A text that is just a length, e.g. ``"500 µm"`` (a scale label), not ``"WD = 4.0 mm"``."""
    return bool(re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*(pm|nm|um|µm|μm|mm|Å)\s*", text, re.IGNORECASE))


def scale_labels(texts: list[Text]) -> list[Text]:
    return [t for t in texts if _label_only(t.text)]


def fields(texts: list[Text]) -> dict[str, Any]:
    """Field width, magnification, working distance and voltage, where the bar shows them.

    Handles ``"Width = 5.965 µm"`` / ``"HFW 1.73 mm"`` on one line and the Thermo Fisher table
    (header ``HFW`` above its value).
    """
    out: dict[str, Any] = {}

    def below(header: Text) -> Text | None:
        cx = header.centre[0]
        cands = [t for t in texts if t is not header and t.box[1] >= header.box[3] - 2 and abs(t.centre[0] - cx) < (header.box[2] - header.box[0]) / 2 + 20]
        return min(cands, key=lambda t: t.box[1], default=None)

    for t in texts:
        low = t.text.lower().replace(" ", "")
        for key, names in (("hfw", ("hfw", "width", "fov", "fieldofview")), ("wd", ("wd",)), ("mag", ("mag",)), ("hv", ("hv", "eht", "acc.v", "kv"))):
            if key in out:
                continue
            hit = next((n for n in names if low.startswith(n)), None)
            if hit is None:
                continue
            rest = t.text[t.text.lower().find(hit[0]) :]
            rest = re.sub(r"^[A-Za-z. ]+[=:\-]?", "", rest).strip()
            if not re.search(r"\d", rest):
                v = below(t)
                rest = v.text if v else ""
            if key in ("hfw", "wd"):
                vals = lengths_in(rest)
                if vals:
                    out[key] = vals[0]
            elif key == "mag":
                m = _mag(rest)
                if m and re.search(r"\d", rest):
                    out[key] = m
            elif key == "hv":
                m = re.search(r"(\d+(?:[.,]\d+)?)\s*(kv|v)", rest, re.IGNORECASE)
                if m:
                    out[key] = float(m.group(1).replace(",", ".")) * (1e3 if m.group(2).lower() == "kv" else 1.0)
    return out


__all__ = ["OcrUnavailable", "Text", "available", "fields", "lengths_in", "read_text", "scale_labels", "UNITS"]


if __name__ == "__main__" and "--serve" in sys.argv:
    _serve()
