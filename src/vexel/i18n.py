"""Interface language: Chinese (as written in the code) or English.

The text shown to people is written in Chinese where it is made; ``tr`` turns it into the
chosen language when it is shown. Translations live in ``vexel/locale/<lang>.json`` as
``{"中文": "English"}``. A key may hold ``{}`` placeholders, so messages built with values
translate too: ``"找到 {} 个孔"`` → ``"{} holes found"``; what fills a placeholder is
translated in turn (a Chinese warning inside ``"• {}"`` comes out in English).

Text without a translation is shown as it is (``missing()`` lists it, for tests).

The language: ``$VEXEL_LANG`` (``zh`` or ``en``), else ``"language"`` in
``$VEXEL_HOME/config.json`` (``set_language``), else Chinese. It is read once; a change
applies when the window is opened again.
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path

CJK = re.compile(r"[　-〿一-鿿＀-￯]")
LANGUAGES = {"zh": "中文", "en": "English"}
_missing: set[str] = set()


def _config_path() -> Path:
    return Path(os.environ.get("VEXEL_HOME", Path.home() / ".vexel")) / "config.json"


@functools.lru_cache(maxsize=1)
def language() -> str:
    env = os.environ.get("VEXEL_LANG", "").lower()[:2]
    if env in LANGUAGES:
        return env
    try:
        lang = json.loads(_config_path().read_text(encoding="utf-8")).get("language", "zh")
    except (OSError, ValueError, AttributeError):
        lang = "zh"
    return lang if lang in LANGUAGES else "zh"


def set_language(lang: str) -> None:
    """Remember the interface language (applies to windows opened afterwards)."""
    if lang not in LANGUAGES:
        raise ValueError(f"language must be one of {list(LANGUAGES)}")
    p = _config_path()
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    cfg["language"] = lang
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    language.cache_clear()
    _table.cache_clear()
    tr.cache_clear()


@functools.lru_cache(maxsize=4)
def _table(lang: str) -> tuple[dict[str, str], list[tuple[re.Pattern[str], str, int]]]:
    path = Path(__file__).parent / "locale" / f"{lang}.json"
    try:
        data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    exact = {k: v for k, v in data.items() if "{}" not in k}
    patterns = []
    for k, v in data.items():
        if "{}" in k:
            rx = re.compile("^" + "(.*?)".join(re.escape(part) for part in k.split("{}")) + "$", re.S)
            patterns.append((rx, v, len(k.replace("{}", ""))))
    patterns.sort(key=lambda p: -p[2])  # the most specific first
    return exact, patterns


@functools.lru_cache(maxsize=20000)
def tr(text: str) -> str:
    """``text`` in the interface language."""
    if not text or language() == "zh" or not CJK.search(text):
        return text
    exact, patterns = _table(language())
    if text in exact:
        return _finish(exact[text])
    for rx, target, _ in patterns:
        m = rx.match(text)
        if m:
            parts = target.split("{}")
            out = parts[0]
            for g, rest in zip(m.groups(), parts[1:]):
                out += tr(g) + rest
            return _finish(out)
    # several messages joined: translate each
    for sep in ("\n", "<br>", "；", "。"):
        if sep in text:
            pieces = text.split(sep)
            if len(pieces) > 1:
                joined = {"；": "; ", "。": ". "}.get(sep, sep)
                return joined.join(tr(p) for p in pieces).replace(". . ", ". ").rstrip()
    stripped = text.strip()
    if stripped != text:
        return text.replace(stripped, tr(stripped))
    if not HAN.search(text):  # only full-width punctuation left: its English forms
        return _punctuation(text)
    _missing.add(text)
    return text


HAN = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = {"（": " (", "）": ")", "：": ": ", "，": ", ", "；": "; ", "、": ", ", "。": ". ", "“": "\"", "”": "\"", "！": "!", "？": "?"}


def _punctuation(text: str) -> str:
    lead = text[:1].isspace()
    for a, b in _PUNCT.items():
        text = text.replace(a, b)
    text = re.sub(r" {2,}", " ", text).replace("( ", "(").replace(" )", ")")
    return text if lead else text.lstrip()


def _finish(text: str) -> str:
    return text if HAN.search(text) else _punctuation(text) if CJK.search(text) else text


def missing() -> set[str]:
    """Text shown without a translation so far (in this session)."""
    return set(_missing)


__all__ = ["CJK", "LANGUAGES", "language", "missing", "set_language", "tr"]
