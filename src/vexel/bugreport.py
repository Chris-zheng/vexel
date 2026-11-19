"""Diagnostics bundle for testers: what is needed to reproduce a problem, without the images.

``vexel bugreport [RUN ...] -o vexel-bugreport.zip`` writes a zip with:

- ``environment.json``: Vexel and Python versions, operating system, package versions,
  Qt, which planning model is configured and whether its API key variable is set (never
  the key), and whether offline mode is on;
- for each run folder: ``provenance.json``, ``recipe.json``, ``summary.json``, the plan
  and corrections when present, the table of flagged droplets and stream alarms;
- ``README.txt`` listing what is in the bundle and what is not.

Never included: images, videos, review pictures, the HTML report (it embeds pictures),
API keys. Absolute paths are shortened to file names unless ``keep_paths`` is set; the
per-droplet tables are left out unless ``tables`` is set.
"""

from __future__ import annotations

import json
import os
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_FILES = ["provenance.json", "recipe.json", "summary.json", "corrections.json", "plan/plan.json", "plan.json"]
FLAGS = "tables/flags.table.csv"


def environment() -> dict[str, Any]:
    from vexel.agent import models
    from vexel.engine.provenance import environment as env

    from importlib import metadata

    out = env()
    for name in ("anthropic", "mcp"):  # optional extra "agent"
        try:
            out["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out["packages"][name] = "not installed"
    out["system"] = {"os": platform.system(), "release": platform.release(), "version": platform.version()}
    try:
        import qtpy
        from qtpy import QtCore

        out["qt"] = {"binding": qtpy.API_NAME, "version": QtCore.qVersion()}
    except Exception as e:  # no GUI libraries: still a useful report
        out["qt"] = {"error": type(e).__name__}
    cfg = models.load_config()
    planning = {"order": cfg["order"], "offline": os.environ.get("VEXEL_OFFLINE", "") not in ("", "0", "false")}
    for name in cfg["order"]:
        tier = cfg.get(name)
        if isinstance(tier, dict):
            env_name = tier.get("api_key_env") or ("ANTHROPIC_API_KEY" if name == "claude" else "")
            planning[name] = {"model": tier.get("model"), "send_images": tier.get("send_images"),
                              "api_key_env": env_name or None, "api_key_set": bool(env_name and os.environ.get(env_name))}  # fmt: skip
            if name in models.COMPATIBLE and tier.get("base_url") != models.COMPATIBLE[name]["base_url"]:
                planning[name]["base_url"] = "custom"
    out["planning_models"] = planning
    out["vexel_home_set"] = "VEXEL_HOME" in os.environ
    from vexel.policy import load_policy

    policy = load_policy()
    out["policy"] = {"source": policy.source, "offline": policy.offline, "allowed_hosts": list(policy.allowed_hosts),
                     "allow_images": policy.allow_images, "error": policy.error}  # fmt: skip
    try:
        from vexel.ops import list_ops
        from vexel.plugins import file_status, load_plugins, model_cards

        list_ops()
        out["plugins"] = [{"name": p.name, "version": p.version, "status": p.status, "error": p.error} for p in load_plugins()]
        out["models"] = [{"ref": c.ref, "file": file_status(c)} for c in model_cards()]
    except Exception as e:  # noqa: BLE001
        out["plugins"] = {"error": type(e).__name__}
    return out


def _redact(value: Any, keep_paths: bool) -> Any:
    """Shorten absolute paths to their last component; anything that looks like a key is dropped."""
    if isinstance(value, dict):
        return {k: ("<removed>" if _secret(k) else _redact(v, keep_paths)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, keep_paths) for v in value]
    if isinstance(value, str) and not keep_paths:
        home = str(Path.home())
        if value.startswith(home):
            value = "~" + value[len(home) :]
        if ("/" in value or "\\" in value) and (value.startswith(("/", "~")) or (len(value) > 2 and value[1] == ":")):
            return ".../" + Path(value.replace("\\", "/")).name
    return value


def _secret(key: str) -> bool:
    k = key.lower()
    return k in ("api_key", "apikey", "token", "password", "authorization") or k.endswith("_secret")


def _json_text(path: Path, keep_paths: bool) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return json.dumps({"error": f"{path.name} could not be read as JSON"})
    return json.dumps(_redact(data, keep_paths), indent=2, ensure_ascii=False)


def write_bundle(out: str | Path, runs: list[str | Path] = (), *, note: str = "", tables: bool = False, keep_paths: bool = False) -> Path:
    out = Path(out)
    if out.suffix != ".zip":
        out = out.with_suffix(".zip")
    listing: list[str] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("environment.json", json.dumps(environment(), indent=2, ensure_ascii=False))
        listing.append("environment.json")
        if note:
            zf.writestr("note.txt", note)
            listing.append("note.txt")
        for i, run in enumerate(runs):
            run = Path(run)
            if not run.is_dir():
                raise FileNotFoundError(f"run folder not found: {run}")
            prefix = f"run{i + 1}_{run.name}/"
            names = [*RUN_FILES, FLAGS]
            if tables:
                names += [p.relative_to(run).as_posix() for p in sorted(run.glob("tables/*.csv")) if p.relative_to(run).as_posix() != FLAGS]
                names += [p.relative_to(run).as_posix() for p in sorted(run.glob("corrected/tables/*.csv"))]
            names += ["corrected/summary.json", "corrected/provenance.json", "stream/alarms.csv"]
            if tables:
                names.append("stream/rolling.csv")
            for name in names:
                src = run / name
                if not src.is_file():
                    continue
                text = _json_text(src, keep_paths) if src.suffix == ".json" else src.read_text(encoding="utf-8", errors="replace")
                zf.writestr(prefix + name, text)
                listing.append(prefix + name)
        readme = [
            "Vexel diagnostics bundle",
            f"created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "Included:",
            *[f"  {n}" for n in listing],
            "",
            "Not included: images, videos, review pictures, HTML reports, API keys.",
            "Per-droplet tables: " + ("included" if tables else "not included (only the table of flagged droplets)") + ".",
            "Absolute paths: " + ("kept" if keep_paths else "shortened to file names") + ".",
        ]
        zf.writestr("README.txt", "\n".join(readme) + "\n")
    return out
