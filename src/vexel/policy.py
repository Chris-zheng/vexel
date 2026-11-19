"""Organisation policy: settings an administrator fixes for every user of a computer.

For sites where data must not leave the network (semiconductor fabs, hospitals), IT
places a JSON file in a system location that ordinary users cannot edit:

==========  ==============================================================
Windows     ``%PROGRAMDATA%\\Vexel\\policy.json``
macOS       ``/Library/Application Support/Vexel/policy.json``
Linux       ``/etc/vexel/policy.json``
==========  ==============================================================

or points the environment variable ``VEXEL_POLICY`` at one. Example::

    {"offline": true, "allowed_hosts": ["llm.intranet.example"], "allow_images": false,
     "note": "Fab policy FP-12: no external network access"}

- ``offline``: no network request leaves the computer. Planning uses the offline rules
  engine; a model is used only if its server's host is in ``allowed_hosts`` (for example
  a local Ollama on ``localhost`` or a company-internal model gateway).
- ``allowed_hosts``: hosts that remain reachable when ``offline`` is set.
- ``allow_images``: false forbids sending the diagnostic picture to any model.

The user's own settings (``~/.vexel/config.json``, the settings dialog, ``VEXEL_OFFLINE``)
can make Vexel stricter, never less strict than the policy. Vexel makes no other network
requests than planning-model calls, so this covers everything it sends.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Policy:
    offline: bool = False
    allowed_hosts: tuple[str, ...] = ()
    allow_images: bool = True
    note: str = ""
    source: str | None = None  # file it came from; None: no policy
    error: str = ""  # a policy file that could not be read is treated as the strictest policy

    def host_allowed(self, url: str) -> bool:
        if not self.offline:
            return True
        host = (urlparse(url if "//" in url else f"//{url}").hostname or "").lower()
        return host in {h.lower() for h in self.allowed_hosts}

    def describe(self) -> str:
        if self.source is None:
            return "无组织策略"
        parts = []
        if self.error:
            parts.append(f"策略文件无法读取（{self.error}），按最严格的设置执行：禁止一切网络请求")
        elif self.offline:
            parts.append("禁止向外发送任何请求" + (f"；允许的内部服务：{', '.join(self.allowed_hosts)}" if self.allowed_hosts else ""))
        if not self.allow_images:
            parts.append("禁止发送图像")
        text = "管理员策略（" + self.source + "）：" + ("；".join(parts) or "无限制")
        return text + (f"。{self.note}" if self.note else "")


def default_path() -> Path:
    if sys.platform.startswith("win"):
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Vexel" / "policy.json"
    if sys.platform == "darwin":
        return Path("/Library/Application Support/Vexel/policy.json")
    return Path("/etc/vexel/policy.json")


def load_policy() -> Policy:
    env = os.environ.get("VEXEL_POLICY", "").strip()
    path = Path(env) if env else default_path()
    if not path.exists():
        if env:  # named but missing: refuse to fall back to "no policy"
            return Policy(offline=True, allow_images=False, source=str(path), error="file not found")
        return Policy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return Policy(offline=bool(data.get("offline", False)), allowed_hosts=tuple(str(h) for h in data.get("allowed_hosts", [])),
                      allow_images=bool(data.get("allow_images", True)), note=str(data.get("note", "")), source=str(path))  # fmt: skip
    except (OSError, ValueError, TypeError) as e:
        return Policy(offline=True, allow_images=False, source=str(path), error=f"{type(e).__name__}: {e}")


__all__ = ["Policy", "default_path", "load_policy", "field"]
