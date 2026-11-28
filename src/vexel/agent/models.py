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
PROVIDERS = ["gateway", "claude", *COMPATIBLE, "local", "rules"]
LABELS = {"gateway": "Vexel AI 规划服务", "claude": "Anthropic Claude", **{k: v["label"] for k, v in COMPATIBLE.items()}, "local": "本地模型（Ollama）",
          "rules": "离线规则引擎"}  # fmt: skip
GATEWAY_URL = os.environ.get("VEXEL_GATEWAY_URL", "https://ai.vexel.example")  # the service's address (set when it is deployed)

DEFAULT_CONFIG: dict[str, Any] = {
    "order": ["claude", "local", "rules"],
    "claude": {"model": "claude-opus-5-5", "effort": "medium", "send_images": False, "timeout_s": 120},
    "local": {"url": "http://localhost:11434", "model": "", "send_images": False, "timeout_s": 120},
    "gateway": {"url": GATEWAY_URL, "token": "", "send_images": False, "timeout_s": 60},
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


MAX_REVISIONS = 3  # the agent loop: revisions after trial runs (with the draft, at most 4 model calls per plan)


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
    recipe: str | None = None  # SEM: the measurement chosen (None: the probe's)


def save_config(cfg: dict[str, Any]) -> Path:
    """Write the model settings (merged into any other content of the config file)."""
    path = home() / "config.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["models"] = {k: v for k, v in cfg.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # it may hold the Vexel AI service's sign-in token
    except OSError:
        pass
    return path


def load_config(policy: bool = True) -> dict[str, Any]:
    """The model settings; with ``policy`` (default) the ones in force: restricted by the
    organisation policy, and to the offline rules unless ``VEXEL_PREVIEW`` is set (0.8). Without
    it, the settings as configured."""
    path = home() / "config.json"
    user = json.loads(path.read_text(encoding="utf-8")).get("models", {}) if path.exists() else {}
    cfg = {**DEFAULT_CONFIG, **{k: v for k, v in user.items() if k == "order"}}
    for tier in ("gateway", "claude", "local", *COMPATIBLE):
        cfg[tier] = {**DEFAULT_CONFIG[tier], **user.get(tier, {})}
    if os.environ.get("VEXEL_OFFLINE", "").strip() not in ("", "0", "false"):
        cfg["order"] = ["rules"]
    elif policy and os.environ.get("VEXEL_PREVIEW", "").strip() in ("", "0", "false"):
        # 0.8: hosted models only through the Vexel gateway (signed in with an invite code); the
        # other tiers are a preview (VEXEL_PREVIEW=1, or an explicit `vexel plan --model ...`)
        cfg["order"] = ["gateway", "rules"] if cfg["gateway"].get("token") else ["rules"]
    return apply_policy(cfg) if policy else cfg


def tier_host(name: str, cfg: dict[str, Any]) -> str | None:
    """The server a tier sends requests to (None: the offline rules engine)."""
    if name == "rules":
        return None
    if name == "gateway":
        return cfg.get("gateway", {}).get("url", "")
    if name == "claude":
        return os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    tier = cfg.get(name, {})
    return tier.get("url") if name == "local" else tier.get("base_url", "")


def apply_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    """Restrict a model configuration to what the organisation policy allows."""
    from vexel.policy import load_policy

    policy = load_policy()
    if policy.offline:
        cfg["order"] = [n for n in cfg["order"] if n == "rules" or policy.host_allowed(tier_host(n, cfg) or "")] or ["rules"]
    if not policy.allow_images:
        for name in ("gateway", "claude", "local", *COMPATIBLE):
            cfg[name] = {**cfg[name], "send_images": False}
    return cfg


def check_settings(settings: dict[str, Any], valid_keys: set[str], allowed: dict[str, tuple] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Keep only allowlisted settings, of the right type and in range, that the recipe has."""
    allowed = ALLOWED if allowed is None else allowed
    kept, dropped = {}, []
    for key, value in settings.items():
        rule = allowed.get(key)
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


SEM_SYSTEM = """You help plan measurements of scanning electron microscope (SEM) images in Vexel,
mostly semiconductor metrology. Vexel runs deterministic, validated measurement recipes; you only
choose which recipe fits the user's goal and the image, and propose settings. You never see
measured values, and the user confirms every plan before it runs.

Recipes: lines (line/space: CD, pitch, LER/LWR), holes (contact holes and vias), particles
(particles and defects), pores (porosity, pore sizes), fibres (fibre diameter, orientation), layers (cross-section layer thickness), profile (one line or trench in
cross-section: height, widths, sidewall angles), tip (apex radius, cone angle). Choose `recipe`
from the goal; use null when the goal does not say and the image recognition should decide.

Propose a setting only when the fingerprint or the trial gives a concrete reason, and only from
the allowed list (type and range given); `process.<step>` = "on" adds that processing step.
Never propose denoising before roughness (LER/LWR) measurements. Leave `settings` empty when the
draft is fine. Write each note as one short sentence in the user's language."""

SEM_SCHEMA = {
    "type": "object",
    "properties": {
        "recipe": {"type": ["string", "null"], "enum": ["lines", "holes", "particles", "pores", "fibres", "layers", "profile", "tip", None]},
        "settings": SCHEMA["properties"]["settings"],
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["recipe", "settings", "notes"],
    "additionalProperties": False,
}


def sem_allowed() -> dict[str, tuple]:
    """SEM settings a model may propose: every recipe setting (``<recipe>.<name>``, from the
    recipe definitions themselves) and the processing steps (``process.<op>``)."""
    from vexel.packs.sem.analysis import RECIPES

    out: dict[str, tuple] = {}
    for name, rec in RECIPES.items():
        for prm in rec.params:
            key = f"{name}.{prm.name}"
            if prm.kind == "choice":
                out[key] = ("enum", [c[0] for c in prm.choices])
            elif prm.kind == "bool":
                out[key] = ("enum", [True, False])
            else:
                out[key] = ("float", prm.lo, prm.hi)
    for op in ("line_level", "destripe", "background", "align_rows", "denoise"):
        out[f"process.{op}"] = ("enum", ["on", "off"])
    return out


def allowed_for_recipe(recipe: Any) -> dict[str, tuple]:
    """Dynamically derive the agent allowlist from every step's Pydantic ``Params`` schema
    in ``vexel.ops.registry``. Works automatically for droplet, SEM, custom workflows, and
    trained-model plugins without hardcoded parameter tables."""
    from vexel.ops.registry import allowed_params_from_schema, get_op

    out: dict[str, tuple] = {}
    for step in recipe.steps:
        spec = get_op(step.op)
        out.update(allowed_params_from_schema(step.id, spec))
    return out


def task_of(context: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, tuple]]:
    """System prompt, answer schema and allowlist of the planning task (droplet by default)."""
    if context.get("task") == "sem":
        return SEM_SYSTEM, SEM_SCHEMA, sem_allowed()
    return SYSTEM, SCHEMA, ALLOWED


def prompt_text(context: dict[str, Any]) -> str:
    _, _, table = task_of(context)
    allowed = {k: (list(v[1]) if v[0] == "enum" else f"{v[0]} {v[1]:g}..{v[2]:g}") for k, v in table.items() if k in context.get("valid_keys", [])}
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
        recipe = data.get("recipe")
        if recipe is not None and recipe not in ("lines", "holes", "particles", "pores", "fibres", "layers", "profile", "tip"):
            recipe = None
        return Proposal(data.get("cells"), data.get("fluorescence"), settings, notes, tier, model, recipe)
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
                system=task_of(context)[0],
                output_config={"effort": self.cfg.get("effort", "medium"), "format": {"type": "json_schema", "schema": task_of(context)[1]}},
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


class GatewayTier:
    """The Vexel AI gateway: hosted models (DeepSeek, Gemini) behind Vexel's own service, which
    holds the keys, the limits and the rules (``vexel.gateway``). Signed in with an invite code;
    the token is kept in the configuration. A refusal, a used-up limit or no connection hands
    over to the next tier (the offline rules), with the reason."""

    name = "gateway"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("url") or not self.cfg.get("token"):
            return False, "not signed in to the Vexel AI service"
        return True, ""

    @staticmethod
    def plan_id(context: dict[str, Any]) -> str:
        """The same for every step of one plan (task, goal, the image's numbers, today), so the
        service can count a plan's revisions."""
        import hashlib
        import time as _t

        key = {k: context.get(k) for k in ("task", "goal", "image", "fingerprint")}
        return hashlib.sha256((json.dumps(key, sort_keys=True, default=str) + _t.strftime("%Y-%m-%d")).encode()).hexdigest()[:32]

    def propose(self, context: dict[str, Any], image: bytes | None = None) -> Proposal:
        import urllib.error
        import urllib.request

        body: dict[str, Any] = {"plan_id": self.plan_id(context), "context": context}
        if image is not None and self.cfg.get("send_images"):
            body["image"] = base64.standard_b64encode(image).decode()
        req = urllib.request.Request(self.cfg["url"].rstrip("/") + "/v1/plan", data=json.dumps(body, default=str).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.cfg['token']}"})  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 60))) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get("message") or f"HTTP {e.code}"
            except (ValueError, OSError):
                msg = f"HTTP {e.code}"
            raise ModelUnavailable({401: "sign-in expired or revoked"}.get(e.code, msg)) from e
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ModelUnavailable("Vexel AI service not reachable") from e
        a = data.get("answer") or {}
        settings = {s["key"]: s["value"] for s in a.get("settings", []) if isinstance(s, dict) and "key" in s}
        notes = [str(n) for n in a.get("notes", [])] + [f"{s['key']}：{s.get('reason', '')}" for s in a.get("settings", []) if isinstance(s, dict)]
        notes += [str(n) for n in data.get("notes", []) if not str(n).startswith("skipped")]
        model = f"{data.get('provider', '')}/{data.get('model', '')}" + ("（缓存）" if data.get("cached") else "")
        return Proposal(a.get("cells"), a.get("fluorescence"), settings, notes, "gateway", model, a.get("recipe"))


def gateway_request(cfg: dict[str, Any], path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """A call to the gateway outside planning (sign in, quota); errors as ``ModelUnavailable``."""
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if cfg.get("token"):
        headers["Authorization"] = f"Bearer {cfg['token']}"
    req = urllib.request.Request(cfg["url"].rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_s", 60))) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error") or f"HTTP {e.code}"
        except (ValueError, OSError):
            msg = f"HTTP {e.code}"
        raise ModelUnavailable(msg) from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise ModelUnavailable("Vexel AI service not reachable") from e


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
        system, schema, _ = task_of(context)
        body = {"model": self.cfg["model"], "stream": False, "format": schema, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, message]}  # fmt: skip
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

        system, schema, _ = task_of(context)
        text = prompt_text(context) + "\n\nAnswer with one JSON object that matches this JSON schema:\n" + json.dumps(schema)
        content: Any = text
        if image is not None and self.cfg.get("send_images"):
            url = "data:image/png;base64," + base64.standard_b64encode(image).decode()
            content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]
        body = {"model": self.cfg["model"], "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}  # fmt: skip
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


def _host_of(tier: Any) -> str | None:
    if tier.name == "rules":
        return None
    if tier.name == "claude":
        return os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    cfg = getattr(tier, "cfg", {})
    return cfg.get("url") or cfg.get("base_url") or ""


def tiers(cfg: dict[str, Any] | None = None) -> list[Any]:
    cfg = cfg or load_config()
    make: dict[str, Any] = {"gateway": lambda: GatewayTier(cfg["gateway"]), "claude": lambda: ClaudeTier(cfg["claude"]),
                            "local": lambda: LocalTier(cfg["local"]), "rules": RulesTier}  # fmt: skip
    for name in COMPATIBLE:
        make[name] = lambda name=name: CompatibleTier(name, cfg[name])
    out = [make[name]() for name in cfg["order"] if name in make]
    if not out or out[-1].name != "rules":
        out.append(RulesTier())  # always the last resort
    return out


def propose(context: dict[str, Any], image: bytes | None = None, chain: list[Any] | None = None) -> tuple[Proposal, list[str]]:
    """First tier that answers; also, for each tier passed over, why."""
    from vexel.policy import load_policy

    policy = load_policy()
    if not policy.allow_images:
        image = None
    skipped = []
    for tier in chain or tiers():
        host = _host_of(tier)
        if host is not None and not policy.host_allowed(host):
            skipped.append(f"{tier.name}: blocked by the organisation policy ({policy.source})")
            continue
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
