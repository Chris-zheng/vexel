"""HTTP API of the Vexel engine (``vexel serve``, optional extra ``web``): operators,
recipes, models and recipe runs under ``/api/v1/*``, for the web workbench and scripts.

It reads files from this computer, so it is local by default: it listens on 127.0.0.1,
browsers may call it only from pages served by localhost, and with ``VEXEL_SERVE_ROOT``
set only files inside that folder can be run.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vexel.engine import Recipe, run_recipe
from vexel.modules import modules as list_modules
from vexel.ops import list_ops
from vexel.packs import builtin_recipes
from vexel.plugins import file_status, load_plugins, model_cards
from vexel.readers import list_readers, open_volume


class RunRequest(BaseModel):
    """A recipe run (module level: FastAPI must be able to resolve it as the request body)."""

    path: str
    recipe: str | dict[str, Any]
    settings: dict[str, Any] = Field(default_factory=dict)
    pixel_size: float | None = None
    space_unit: str | None = None
    frame_interval: float | None = None
    max_frames: int | None = None
    in_memory: bool | None = True


def create_app() -> Any:
    """Create the FastAPI application (optional dependency: ``fastapi`` + ``uvicorn``)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    from vexel import __version__

    app = FastAPI(title="Vexel Engine API", version=__version__)
    # only pages from this computer: any other web site could otherwise make it read local files
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
                       allow_methods=["GET", "POST"], allow_headers=["*"])  # fmt: skip

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "vexel-python",
            "readers": list_readers(),
            "modules": [{"id": m.id, "name": m.name, "name_en": m.name_en, "status": m.status} for m in list_modules()],
        }

    @app.get("/api/v1/operators")
    def operators() -> dict[str, Any]:
        return {"operators": [op.describe() for op in list_ops()]}

    @app.get("/api/v1/recipes")
    def recipes() -> dict[str, Any]:
        items = []
        for name, raw in builtin_recipes().items():
            rec = Recipe.from_json(raw).resolve()
            items.append({"name": name, "description": rec.description, "steps": [s.model_dump(mode="json") for s in rec.steps]})
        return {"recipes": items}

    @app.get("/api/v1/models")
    def models() -> dict[str, Any]:
        plugins = load_plugins()
        cards = [c.to_json() | {"file_status": file_status(c)} for c in model_cards()]
        return {"models": cards, "plugins": [p.__dict__ for p in plugins]}

    @app.post("/api/v1/run")
    def run(req: RunRequest) -> dict[str, Any]:
        root = os.environ.get("VEXEL_SERVE_ROOT", "").strip()
        target = Path(req.path).expanduser().resolve()
        if root and not target.is_relative_to(Path(root).expanduser().resolve()):
            raise HTTPException(status_code=403, detail="path is outside VEXEL_SERVE_ROOT")
        if not target.exists():
            raise HTTPException(status_code=404, detail="no such file")
        try:
            rec = Recipe.model_validate(req.recipe) if isinstance(req.recipe, dict) else Recipe.load(req.recipe)
            if req.settings:
                rec = rec.with_overrides(req.settings)
            vol = open_volume(str(target), pixel_size=req.pixel_size, space_unit=req.space_unit, frame_interval=req.frame_interval)
            out_dir = Path(tempfile.mkdtemp(prefix="vexel-api-run-")) / "run"
            result = run_recipe(rec, vol, out_dir, max_frames=req.max_frames, hash_input=False, report=False, in_memory=req.in_memory)
            return {
                "status": result.status,
                "result_digest": result.result_digest,
                "summary": result.summary,
                "provenance": result.provenance,
                "tables": {k: df.head(500).to_dict(orient="records") for k, df in result.tables.items()},
                "warnings": [v.message() for v in result.violations],
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}") from e

    from vexel.server.sem_api import register as register_sem

    register_sem(app)
    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn  # type: ignore[import-not-found]

    uvicorn.run(create_app(), host=host, port=port)
