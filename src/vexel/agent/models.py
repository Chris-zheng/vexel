"""Model abstraction layer: cloud model, local model, offline rules, with fallback.

A language model's role in planning is narrow and checked:

* read the user's goal (what to measure);
* propose settings from a fixed allowlist (each with its type and range), from the
  data's fingerprint, the rule-based draft and, after a trial run, its QC results.

It never computes results and never edits recipes directly: every proposal is validated
against the allowlist and the recipe schema, tried on a trial run, and accepted by the
user. Tiers are tried in the configured order (default: Claude, a local model, rules);
a tier that is not configured, not reachable, refuses or answers out of schema hands
over to the next one, and the plan records which tier answered and why others did not.
The rules tier is always there, so everything works offline.

Configuration: ``$VEXEL_HOME/config.json`` (default ``~/.vexel``)::

    {"models": {"order": ["claude", "local", "rules"],
                "claude": {"model": "claude-opus-5-5", "effort": "medium", "send_images": false},
                "local": {"url": "http://localhost:11434", "model": "qwen2.5vl:7b", "send_images": false},
                "deepseek": {"model": "deepseek-chat"}}}

DeepSeek, Qwen (DashScope), Gemini and any other OpenAI-style chat-completions service are
also available (``order`` names them: ``deepseek``, ``qwen``, ``gemini``, ``custom``); their
API keys come from environment variables (``DEEPSEEK_API_KEY``, ``DASHSCOPE_API_KEY``,
``GEMINI_API_KEY``, or the ``api_key_env`` a configuration names).

``VEXEL_OFFLINE=1`` restricts planning to the rules tier. Images (the 2 x 2 diagnostic
montage) are sent only where ``send_images`` is true; by default a model sees numbers only.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vexel.agent.memory import home

# Hosted models reached through an OpenAI-style chat-completions endpoint. Model names
# change often: they are defaults to be checked against the provider's current list.
COMPATIBLE: dict[str, dict[str, Any]] = {
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
    "qwen": {"label": "通义千问（阿里云百炼）", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus",
             "api_key_env": "DASHSCOPE_API_KEY"},  # fmt: skip
    "gemini": {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash",
               "api_key_env": "GEMINI_API_KEY"},  # fmt: skip
    "custom": {"label": "其他兼容服务（自定义）", "base_url": "", "model": "", "api_key_env": ""},
}
PROVIDERS = ["claude", *COMPATIBLE, "local", "rules"]
LABELS = {"claude": "Anthropic Claude", **{k: v["label"] for k, v in COMPATIBLE.items()}, "local": "本地模型（Ollama）", "rules": "离线规则引擎"}

DEFAULT_CONFIG: dict[str, Any] = {
    "order": ["claude", "local", "rules"],
    "claude": {"model": "claude-opus-5-5", "effort": "medium", "send_images": False, "timeout_s": 120},
    "local": {"url": "http://localhost:11434", "model": "", "send_images": False, "timeout_s": 120},
    **{k: {**{x: y for x, y in v.items() if x != "label"}, "send_images": False, "timeout_s": 120} for k, v in COMPATIBLE.items()},
}

# Settings a model may propose: key -> (type, lower, upper) or (type, allowed values)
ALLOWED: dict[str, tuple] = {
    "droplets.flow_axis": ("enum", ["x", "y"]),
    "droplets.count_line": ("float", 0.1, 0.9),
    "droplets.channel_height_um": ("float", 1.0, 2000.0),
    "seg.min_diameter_px": ("float", 2.0, 2000.0),
    "seg.max_diameter_px": ("float", 2.0, 2000.0),
    "seg.min_score": ("float", 0.1, 0.9),
    "seg.motion_threshold": ("float", 0.0, 255.0),
    "track.max_distance_px": ("float", 5.0, 500.0),
    "inclusions.polarity": ("enum", ["dark", "bright"]),
    "inclusions.cell_diameter_px": ("float", 1.0, 100.0),
    "inclusions.noise_factor": ("float", 2.0, 30.0),
    "positivity.method": ("enum", ["auto", "otsu", "outlier", "fixed"]),
    "positivity.threshold": ("float", 0.0, 1e9),
}


class ModelUnavailable(RuntimeError):
    """The tier cannot answer now (not configured, unreachable, refused, out of schema)."""


@dataclass
class Proposal:
    cells: bool | None = None
    fluorescence: bool | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tier: str = "rules"
    model: str = ""


def save_config(cfg: dict[str, Any]) -> Path:
    """Write the model settings (merged into any other content of the config file)."""
    path = home() / "config.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["models"] = {k: v for k, v in cfg.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_config() -> dict[str, Any]:
    path = home() / "config.json"
    user = json.loads(path.read_text(encoding="utf-8")).get("models", {}) if path.exists() else {}
    cfg = {**DEFAULT_CONFIG, **{k: v for k, v in user.items() if k == "order"}}
    for tier in ("claude", "local", *COMPATIBLE):
        cfg[tier] = {**DEFAULT_CONFIG[tier], **user.get(tier, {})}
    if os.environ.get("VEXEL_OFFLINE", "").strip() not in ("", "0", "false"):
        cfg["order"] = ["rules"]
    return cfg


def check_settings(settings: dict[str, Any], valid_keys: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Keep only allowlisted settings, of the right type and in range, that the recipe has."""
    kept, dropped = {}, []
    for key, value in settings.items():
        rule = ALLOWED.get(key)
        if rule is None or key not in valid_keys:
            dropped.append(f"{key}（不在可调整的设置之列）")
            continue
        if rule[0] == "enum":
            if value in rule[1]:
                kept[key] = value
            else:
                dropped.append(f"{key} = {value!r}（只能是 {rule[1]}）")
        else:
            try:
                v = float(value)
            except (TypeError, ValueError):
                dropped.append(f"{key} = {value!r}（不是数值）")
                continue
            if rule[1] <= v <= rule[2]:
                kept[key] = v
            else:
                dropped.append(f"{key} = {v:g}（超出 {rule[1]:g}–{rule[2]:g}）")
    return kept, dropped


# -- prompt ---------------------------------------------------------------------------------------

SYSTEM = """You help plan image analyses of microfluidic droplet recordings in Vexel.
Vexel runs deterministic, validated operators; you only choose what to measure and propose
settings. You never see raw results, and the user confirms every plan before it runs.

Built-in analyses: droplet size and frequency (always); optional cell counting inside each
droplet (Poisson encapsulation statistics); optional fluorescence positive/negative calls
(digital assays such as ddPCR). Decide `cells` and `fluorescence` from the user's goal; use null
when the goal does not say and the data should decide.

Propose a setting only when the fingerprint or trial results give a concrete reason, and only
from the allowed list (type and range given). Leave `settings` empty when the draft is fine.
Write each note as one short sentence in the user's language (Chinese if the goal is Chinese)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {"type": ["boolean", "null"]},
        "fluorescence": {"type": ["boolean", "null"]},
        "settings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": ["string", "number"]}, "reason": {"type": "string"}},
                "required": ["key", "value", "reason"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cells", "fluorescence", "settings", "notes"],
    "additionalProperties": False,
}


def prompt_text(context: dict[str, Any]) -> str:
    allowed = {k: (list(v[1]) if v[0] == "enum" else f"{v[0]} {v[1]:g}..{v[2]:g}") for k, v in ALLOWED.items() if k in context.get("valid_keys", [])}
    body = {k: v for k, v in context.items() if k != "valid_keys"}
    return (
        f"Stage: {context.get('stage')}\n\nAllowed settings for this recipe:\n{json.dumps(allowed, ensure_ascii=False)}\n\n"
        f"Context:\n{json.dumps(body, ensure_ascii=False, default=str)}"
    )


def _parse(text: str, tier: str, model: str) -> Proposal:
    try:
        data = json.loads(text)
        settings = {item["key"]: item["value"] for item in data.get("settings", [])}
        notes = [str(n) for n in data.get("notes", [])] + [f"{item['key']}：{item['reason']}" for item in data.get("settings", [])]
        return Proposal(data.get("cells"), data.get("fluorescence"), settings, notes, tier, model)
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        raise ModelUnavailable(f"answer out of schema ({type(e).__name__})") from e


# -- tiers ----------------------------------------------------------------------------------------


class RulesTier:
    name = "rules"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def propose(self, context: dict[str, Any], image: bytes | None = None) -> Proposal:
        from vexel.agent.planner import Goal

        goal = Goal.parse(context.get("goal", ""))
        return Proposal(goal.cells, goal.fluorescence, {}, [], self.name, "rules")


class ClaudeTier:
    """Claude through the Anthropic API (credentials from the environment or ``ant auth login``)."""

    name = "claude"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic SDK not installed (pip install 'vexel[agent]')"
        return True, ""

    def propose(self, context: dict[str, Any], image: bytes | None = None) -> Proposal:
        import anthropic

        content: list[dict[str, Any]] = []
        if image is not None and self.cfg.get("send_images"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(image).decode()}})
        content.append({"type": "text", "text": prompt_text(context)})
        try:
            client = anthropic.Anthropic(timeout=float(self.cfg.get("timeout_s", 120)), max_retries=1)
            response = client.beta.messages.create(
                model=self.cfg["model"],
                max_tokens=4000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM,
                output_config={"effort": self.cfg.get("effort", "medium"), "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.AuthenticationError as e:
            raise ModelUnavailable("no valid Anthropic credentials") from e
        except anthropic.APIConnectionError as e:
            raise ModelUnavailable("Anthropic API not reachable (offline?)") from e
        except anthropic.APIStatusError as e:
            raise ModelUnavailable(f"Anthropic API error {e.status_code}") from e
        except anthropic.AnthropicError as e:
            raise ModelUnavailable(f"{type(e).__name__}: {e}") from e
        except TypeError as e:  # the SDK raises this when no credential source is configured
            if "authentication" in str(e):
                raise ModelUnavailable("no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)") from e
            raise
        if response.stop_reason == "refusal":
            raise ModelUnavailable("the model declined")
        if response.stop_reason == "max_tokens":
            raise ModelUnavailable("answer cut off")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return _parse(text, self.name, response.model)


class LocalTier:
    """A local model served by Ollama (``/api/chat``, JSON-schema constrained output)."""

    name = "local"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("model"):
            return False, "no local model configured"
        return True, ""

    def propose(self, context: dict[str, Any], image: bytes | None = None) -> Proposal:
        import urllib.error
        import urllib.request

        message: dict[str, Any] = {"role": "user", "content": prompt_text(context)}
        if image is not None and self.cfg.get("send_images"):
            message["images"] = [base64.standard_b64encode(image).decode()]
        body = {"model": self.cfg["model"], "stream": False, "format": SCHEMA, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM}, message]}  # fmt: skip
        request = urllib.request.Request(self.cfg["url"].rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})  # fmt: skip
        try:
            with urllib.request.urlopen(request, timeout=float(self.cfg.get("timeout_s", 120))) as resp:
                reply = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ModelUnavailable(f"local model not reachable at {self.cfg['url']}") from e
        return _parse(reply.get("message", {}).get("content", ""), self.name, self.cfg["model"])


class CompatibleTier:
    """A hosted model behind an OpenAI-style ``/chat/completions`` endpoint (DeepSeek,
    Qwen through DashScope, Gemini, or another service). The API key is read from the
    environment variable the configuration names; it is never stored by Vexel."""

    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name, self.cfg = name, cfg

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("base_url") or not self.cfg.get("model"):
            return False, "base_url / model not configured"
        env = self.cfg.get("api_key_env") or ""
        if env and not os.environ.get(env):
            return False, f"API key not set (environment variable {env})"
        return True, ""

    def propose(self, context: dict[str, Any], image: bytes | None = None) -> Proposal:
        import urllib.error
        import urllib.request

        text = prompt_text(context) + "\n\nAnswer with one JSON object that matches this JSON schema:\n" + json.dumps(SCHEMA)
        content: Any = text
        if image is not None and self.cfg.get("send_images"):
            url = "data:image/png;base64," + base64.standard_b64encode(image).decode()
            content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]
        body = {"model": self.cfg["model"], "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}  # fmt: skip
        headers = {"Content-Type": "application/json"}
        env = self.cfg.get("api_key_env") or ""
        if env:
            headers["Authorization"] = f"Bearer {os.environ[env]}"
        request = urllib.request.Request(self.cfg["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=float(self.cfg.get("timeout_s", 120))) as resp:
                reply = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise ModelUnavailable(f"{LABELS.get(self.name, self.name)} returned HTTP {e.code}") from e
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ModelUnavailable(f"{LABELS.get(self.name, self.name)} not reachable") from e
        try:
            choice = reply["choices"][0]
            answer = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelUnavailable("unexpected reply") from e
        if choice.get("finish_reason") == "length":
            raise ModelUnavailable("answer cut off")
        answer = answer.strip()
        if answer.startswith("```"):  # some services wrap JSON in a code fence
            answer = answer.strip("`").removeprefix("json").strip()
        return _parse(answer, self.name, reply.get("model") or self.cfg["model"])


def tiers(cfg: dict[str, Any] | None = None) -> list[Any]:
    cfg = cfg or load_config()
    make: dict[str, Any] = {"claude": lambda: ClaudeTier(cfg["claude"]), "local": lambda: LocalTier(cfg["local"]), "rules": RulesTier}
    for name in COMPATIBLE:
        make[name] = lambda name=name: CompatibleTier(name, cfg[name])
    out = [make[name]() for name in cfg["order"] if name in make]
    if not out or out[-1].name != "rules":
        out.append(RulesTier())  # always the last resort
    return out


def propose(context: dict[str, Any], image: bytes | None = None, chain: list[Any] | None = None) -> tuple[Proposal, list[str]]:
    """First tier that answers; also, for each tier passed over, why."""
    skipped = []
    for tier in chain or tiers():
        ok, why = tier.available()
        if not ok:
            skipped.append(f"{tier.name}: {why}")
            continue
        try:
            return tier.propose(context, image), skipped
        except ModelUnavailable as e:
            skipped.append(f"{tier.name}: {e}")
    return RulesTier().propose(context), skipped


def montage_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()
