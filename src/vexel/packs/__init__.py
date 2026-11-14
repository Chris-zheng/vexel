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


def review_adapter(recipe):  # -> vexel.engine.review.ReviewAdapter | None
    """The review adapter of the pack that understands this recipe, if any."""
    from vexel.packs.droplet.review import adapter as droplet_adapter

    for make in (droplet_adapter,):
        found = make(recipe)
        if found is not None:
            return found
    return None
