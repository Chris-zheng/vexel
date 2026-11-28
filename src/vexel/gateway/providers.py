"""Hosted language models behind the gateway: DeepSeek and Gemini, and a simulated one.

Each provider answers ``complete(system, user, schema, image, max_tokens)`` with the reply's
text and the token counts **the provider reports** (not an estimate). The output is capped by
``max_tokens`` in the request itself; a reply cut off at the cap is an error, not an answer.
API keys come from the gateway's environment (``DEEPSEEK_API_KEY``, ``GEMINI_API_KEY``) and
never leave it.

Prices (USD per million tokens, input / output) are settings, ``VEXEL_PRICE_<NAME>="in,out"``:
the defaults below are placeholders to be checked against each provider's current price list.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_PRICES = {"deepseek": (0.30, 1.20), "gemini": (0.30, 2.50), "mock": (0.0, 0.0)}  # placeholders: check before charging budgets
TIMEOUT_S = 60


class ProviderError(RuntimeError):
    """The provider did not give a usable answer (not configured, refused, cut off, unreachable)."""


@dataclass
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    model: str
    ms: int


def price(name: str) -> tuple[float, float]:
    raw = os.environ.get(f"VEXEL_PRICE_{name.upper()}")
    if raw:
        a, b = (float(x) for x in raw.split(","))
        return a, b
    return DEFAULT_PRICES.get(name, (0.0, 0.0))


def cost(name: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = price(name)
    return (tokens_in * pin + tokens_out * pout) / 1e6


class Provider:
    name = "base"
    model = ""

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, schema: dict[str, Any], image: bytes | None, max_tokens: int) -> Reply:  # pragma: no cover
        raise NotImplementedError


def _post(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    import httpx

    try:
        r = httpx.post(url, json=body, headers=headers, timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise ProviderError(f"not reachable ({type(e).__name__})") from e
    if r.status_code != 200:
        raise ProviderError(f"HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as e:
        raise ProviderError("reply is not JSON") from e


class DeepSeek(Provider):
    """DeepSeek's OpenAI-style chat completions, JSON mode. Text only (no pictures)."""

    name = "deepseek"

    def __init__(self) -> None:
        self.key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = os.environ.get("VEXEL_DEEPSEEK_MODEL", "deepseek-chat")
        self.url = os.environ.get("VEXEL_DEEPSEEK_URL", "https://api.deepseek.com") + "/chat/completions"

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, user: str, schema: dict[str, Any], image: bytes | None, max_tokens: int) -> Reply:
        if not self.key:
            raise ProviderError("DEEPSEEK_API_KEY not set")
        t0 = time.monotonic()
        text = user + "\n\nAnswer with one JSON object matching this JSON schema:\n" + json.dumps(schema)
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}  # fmt: skip
        data = _post(self.url, body, {"Authorization": f"Bearer {self.key}"})
        try:
            choice = data["choices"][0]
            answer = choice["message"]["content"] or ""
            usage = data.get("usage") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError("unexpected reply") from e
        if choice.get("finish_reason") == "length":
            raise ProviderError("answer cut off at the token cap")
        return Reply(answer, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), data.get("model") or self.model,
                     int((time.monotonic() - t0) * 1000))  # fmt: skip


class Gemini(Provider):
    """Google's Gemini ``generateContent`` with a JSON response schema; pictures when allowed."""

    name = "gemini"

    def __init__(self) -> None:
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.model = os.environ.get("VEXEL_GEMINI_MODEL", "gemini-2.5-flash")
        self.base = os.environ.get("VEXEL_GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta")

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, user: str, schema: dict[str, Any], image: bytes | None, max_tokens: int) -> Reply:
        if not self.key:
            raise ProviderError("GEMINI_API_KEY not set")
        t0 = time.monotonic()
        parts: list[dict[str, Any]] = [{"text": user + "\n\nAnswer with one JSON object matching this JSON schema:\n" + json.dumps(schema)}]
        if image is not None:
            mime = "image/png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(image).decode()}})
        body = {"system_instruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}}  # fmt: skip
        data = _post(f"{self.base}/models/{self.model}:generateContent", body, {"x-goog-api-key": self.key})
        try:
            cand = data["candidates"][0]
            answer = "".join(p.get("text", "") for p in cand["content"]["parts"])
            usage = data.get("usageMetadata") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError("unexpected reply (blocked or empty)") from e
        if cand.get("finishReason") == "MAX_TOKENS":
            raise ProviderError("answer cut off at the token cap")
        tin = int(usage.get("promptTokenCount", 0))
        tout = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))  # thinking tokens are billed too
        return Reply(answer, tin, tout, self.model, int((time.monotonic() - t0) * 1000))


class Mock(Provider):
    """A simulated provider for development and tests (``VEXEL_GATEWAY_MOCK=1``): answers with
    the draft's own recipe and no changes, unless the goal names a method; reports token counts
    like a real one (about 4 characters per token)."""

    name = "mock"
    model = "simulated"

    def complete(self, system: str, user: str, schema: dict[str, Any], image: bytes | None, max_tokens: int) -> Reply:
        ctx = json.loads(user.split("Context:\n", 1)[1]) if "Context:\n" in user else {}
        if "recipe" in schema.get("properties", {}):
            recipe = (ctx.get("draft") or {}).get("recipe")
            answer = {"recipe": recipe, "settings": [], "notes": ["simulated answer: the draft is kept"], "out_of_scope": False}
        else:
            answer = {"cells": None, "fluorescence": None, "settings": [], "notes": ["simulated answer: the draft is kept"], "out_of_scope": False}
        text = json.dumps(answer)
        return Reply(text, (len(system) + len(user)) // 4, len(text) // 4, self.model, 1)


def chain(region: str) -> list[Provider]:
    """Providers in the order to try: mainland China DeepSeek first, elsewhere Gemini first;
    the simulated one only when ``VEXEL_GATEWAY_MOCK=1``."""
    if os.environ.get("VEXEL_GATEWAY_MOCK", "").strip() in ("1", "true"):
        return [Mock()]
    order = [DeepSeek(), Gemini()] if region == "cn" else [Gemini(), DeepSeek()]
    return [p for p in order if p.available()]


__all__ = ["DeepSeek", "Gemini", "Mock", "Provider", "ProviderError", "Reply", "chain", "cost", "price"]
