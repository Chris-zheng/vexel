"""Provenance: what ran, on what, with which software, and a digest of the result."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import vexel

_CHUNK = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> tuple[str, int]:
    """Digest and total size of a file, or of a directory (relative names + contents, sorted).

    Sidecar calibration files are excluded: calibration is recorded separately.
    """
    path = Path(path)
    if path.is_file():
        return sha256_file(path), path.stat().st_size
    h = hashlib.sha256()
    total = 0
    for f in sorted(p for p in path.rglob("*") if p.is_file() and p.name != "vexel.json"):
        h.update(f.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha256_file(f)))
        total += f.stat().st_size
    return h.hexdigest(), total


def digest_files(files: dict[str, Path]) -> tuple[dict[str, str], str]:
    """Per-file digests and one combined digest over ``name -> file`` (order-independent)."""
    per_file = {name: sha256_file(path) for name, path in sorted(files.items())}
    combined = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode("utf-8")).hexdigest()
    return per_file, combined


def environment() -> dict[str, Any]:
    """Interpreter, platform and the installed version of every declared dependency."""
    packages: dict[str, str] = {}
    try:
        requires = metadata.requires("vexel") or []
    except metadata.PackageNotFoundError:
        requires = []
    for req in requires:
        if "extra ==" in req:
            continue
        name = re.split(r"[\s\[<>=!~;]", req, maxsplit=1)[0]
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "vexel": vexel.__version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items())),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
