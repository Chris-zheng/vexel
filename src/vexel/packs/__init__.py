"""Domain analysis packs. Each pack contributes operators and built-in recipes; packs
belong to application modules (:mod:`vexel.modules`)."""

from __future__ import annotations

from importlib import resources


def _packs() -> list[str]:
    from vexel.modules import packs

    return packs()


def builtin_recipes() -> dict[str, str]:
    """Recipe name -> JSON text, for every recipe shipped inside a pack."""
    found: dict[str, str] = {}
    for pack in _packs():
        folder = resources.files(f"vexel.packs.{pack}") / "recipes"
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".json"):
                found[entry.name[: -len(".json")]] = entry.read_text(encoding="utf-8")
    return found


def review_adapter(recipe):  # -> vexel.engine.review.ReviewAdapter | None
    """The review adapter of the module whose recipe this is, if it has one."""
    from vexel.modules import MODULES, resolve

    for m in MODULES:
        if m.review is None:
            continue
        found = resolve(m.review)(recipe)
        if found is not None:
            return found
    return None
