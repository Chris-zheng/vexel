"""SEM measurements inside the recipe engine (``vexel.packs.sem.ops``).

Checks:
1. Every SEM processing step, the saturation mask and every measurement register on the
   operator registry, with the SEM window's parameter names, choices and defaults.
2. The built-in SEM recipes resolve, route to the ``sem`` module and give the same numbers
   as the same measurement in the SEM window.
3. Saturated pixels are left out (the mask marks them), also on 16-bit images.
4. The in-memory fast path and the Zarr chunked path give identical results.
5. The agent allowlist is derived from the operators' parameter schemas.
6. The HTTP API (optional ``web`` extra) is local only.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from vexel.agent.models import allowed_for_recipe
from vexel.core.volume import SpatialVolume
from vexel.engine import Recipe, run_recipe
from vexel.modules import batch_metrics, for_recipe, get
from vexel.ops import get_op, list_ops
from vexel.packs.sem import analysis as A
from vexel.packs.sem import process as P
from vexel.synthetic import sem as S

SEM_RECIPES = ("sem_lines", "sem_holes", "sem_particles", "sem_pores", "sem_fibres", "sem_layers", "sem_profile", "sem_tip")


def _volume(img: np.ndarray, pixel_nm: float = 2.0) -> SpatialVolume:
    return SpatialVolume(data=da.from_array(img[None, None, None]), spacing={"y": pixel_nm, "x": pixel_nm}, space_unit="nm")


def _u16(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 60000).astype(np.uint16)


def test_sem_operators_mirror_the_window_definitions() -> None:
    names = {op.name for op in list_ops()}
    for op in P.OPS.values():
        assert (f"sem.{op.name}" in names) == (op.role == "correct")
    for rec in A.RECIPES.values():
        spec = get_op(f"sem.{rec.name}")
        defaults = spec.params().model_dump()
        for prm in rec.params:  # same names and defaults as the window
            assert defaults[prm.name] == prm.default, (rec.name, prm.name)
    assert "sem.saturation_mask" in names
    assert get("sem").status == "preview"  # until the acceptance on real images


@pytest.mark.parametrize("name", SEM_RECIPES)
def test_builtin_sem_recipes_resolve_and_route(name: str) -> None:
    rec = Recipe.load(name).resolve()
    assert for_recipe(rec).id == "sem" and len(batch_metrics(rec)) > 0


@pytest.mark.parametrize(("name", "img"), [("sem_lines", S.rough_lines(rows=200, cols=300, seed=3)[0]),
                                           ("sem_particles", S.particle_field(rows=250, cols=300, n=20, seed=2)[0]),
                                           ("sem_fibres", S.fibre_mat((300, 300), n=6, seed=1)[0]),
                                           ("sem_holes", S.hole_array(rows=250, cols=300, seed=1)[0])])  # fmt: skip
def test_engine_and_window_give_the_same_numbers(tmp_path: Path, name: str, img: np.ndarray) -> None:
    from vexel.packs.sem.measure import Scale

    raw = _u16(img)
    r = run_recipe(Recipe.load(name), _volume(raw), tmp_path / "run", report=False)
    engine = r.summary["measure.summary"]
    # the window: the record's default processing (saturation marked), the same pixel size
    res = P.Processor(raw, None).run(P.Processing())
    an = A.run(A.AnalysisSpec(id=1, recipe=name.removeprefix("sem_")), res.analysis, Scale(pixel_size=2e-9), res.mask)
    assert an.error is None and engine.get("error") is None
    for key, _label, value, kind in an.summary:
        if isinstance(value, (int, float)) and value is not None:
            expect = value / 1e-9 if kind == "length" else value / 1e-18 if kind == "area" else value
            assert engine[key] == pytest.approx(expect, rel=1e-9, abs=1e-12), key


def test_saturated_pixels_are_left_out_on_16_bit_images(tmp_path: Path) -> None:
    img, _ = S.particle_field(rows=250, cols=300, n=20, seed=4, touching=0.0)
    raw = _u16(img * 0.8)  # nothing near the limit
    clean = run_recipe(Recipe.load("sem_particles"), _volume(raw), tmp_path / "a", report=False)
    assert clean.summary["measure.summary"]["particles"] > 5
    sat = raw.copy()
    sat[100:140, 100:140] = np.iinfo(np.uint16).max  # one clipped square
    out = run_recipe(Recipe.load("sem_particles"), _volume(sat), tmp_path / "b", report=False)
    mask = get_op("sem.saturation_mask").fn(get_op("sem.saturation_mask").params(), None, image=sat).outputs["mask"]
    assert 0.02 < mask.mean() < 0.03  # only the square (and its 1-pixel rim), not the whole 16-bit image
    assert out.summary["measure.summary"]["particles"] <= clean.summary["measure.summary"]["particles"]


@pytest.mark.parametrize("op", [o for o in P.OPS if P.OPS[o].role == "correct"])
def test_processing_steps_run_in_the_engine(tmp_path: Path, op: str) -> None:
    rec = Recipe.model_validate({"schema_version": 1, "name": "t", "seed": 0, "input": {"channel": 0, "z": 0, "channels": {}},
                                 "steps": [{"id": "p", "op": f"sem.{op}", "inputs": {"image": "source"}, "params": {}},
                                           {"id": "measure", "op": "sem.lines", "inputs": {"image": "p.image"}, "params": {}}]})  # fmt: skip
    r = run_recipe(rec, _volume(_u16(S.rough_lines(rows=160, cols=240, seed=1)[0])), tmp_path / "r", report=False)
    assert r.status == "completed" and r.summary["measure.summary"].get("error") is None


def test_in_memory_fast_path_matches_zarr_path_bit_for_bit(tmp_path: Path) -> None:
    vol = _volume(_u16(S.rough_lines(rows=128, cols=256, seed=2)[0]))
    r_mem = run_recipe(Recipe.load("sem_lines"), vol, tmp_path / "mem", report=False, in_memory=True)
    r_zarr = run_recipe(Recipe.load("sem_lines"), vol, tmp_path / "zarr", report=False, in_memory=False)
    assert r_mem.status == "completed" and r_mem.result_digest == r_zarr.result_digest
    assert r_mem.summary["measure.summary"]["lines"] == r_zarr.summary["measure.summary"]["lines"] >= 3
    assert (tmp_path / "mem" / "geometries.json").exists()


def test_dynamic_agent_allowlist_covers_sem_and_droplet_recipes() -> None:
    sem_allow = allowed_for_recipe(Recipe.load("sem_lines").resolve())
    assert sem_allow["measure.method"] == ("enum", ["threshold", "gradient", "peak"]) and sem_allow["measure.level"][0] == "float"
    drop_allow = allowed_for_recipe(Recipe.load("droplet_basic").resolve())
    assert "seg.min_score" in drop_allow and "track.max_distance_px" in drop_allow


def test_http_api_is_local_only(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from vexel.server.api import create_app

    c = TestClient(create_app())
    assert c.get("/api/v1/health").json()["status"] == "ok"
    ok = c.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    bad = c.get("/api/v1/health", headers={"Origin": "https://example.com"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000" and "access-control-allow-origin" not in bad.headers
    monkeypatch.setenv("VEXEL_SERVE_ROOT", str(tmp_path / "allowed"))
    assert c.post("/api/v1/run", json={"path": str(tmp_path / "elsewhere.tif"), "recipe": "sem_lines"}).status_code == 403


def test_png_and_jpeg_open_as_one_frame(tmp_path: Path) -> None:
    import imageio.v3 as iio

    from vexel.readers import open_volume

    img = (np.linspace(0, 255, 64 * 80).reshape(64, 80)).astype(np.uint8)
    for name in ("a.png", "b.jpg"):
        iio.imwrite(tmp_path / name, img)
        vol = open_volume(str(tmp_path / name))
        assert vol.data.shape == (1, 1, 1, 64, 80) and vol.space_unit == "px"
    rgb = np.stack([img] * 3 + [np.full_like(img, 255)], axis=-1)  # RGBA: alpha dropped
    iio.imwrite(tmp_path / "c.png", rgb)
    assert open_volume(str(tmp_path / "c.png")).data.shape == (1, 3, 1, 64, 80)
