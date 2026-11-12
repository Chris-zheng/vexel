"""Domain analysis packs. Each pack contributes operators and built-in recipes."""

from __future__ import annotations

from importlib import resources

PACKS = ("droplet",)


def builtin_recipes() -> dict[str, str]:
    """Recipe name -> JSON text, for every recipe shipped inside a pack."""
    found: dict[str, str] = {}
    for pack in PACKS:
        folder = resources.files(f"vexel.packs.{pack}") / "recipes"
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".json"):
                found[entry.name[: -len(".json")]] = entry.read_text(encoding="utf-8")
    return found
