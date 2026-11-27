"""The SEM measurements over HTTP (``vexel serve``, used by the web workbench): the same
numbers as the SEM window for the same image and settings, the person's calibration or the
file's, both languages, and clear refusals."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from vexel.synthetic import sem as S


@pytest.fixture(scope="module")
def client() -> TestClient:
    from vexel.server.api import create_app

    return TestClient(create_app())


def _pixels(img: np.ndarray) -> dict:
    a = np.ascontiguousarray(img, dtype="<f4")
    return {"pixels": base64.b64encode(a.tobytes()).decode(), "width": int(a.shape[1]), "height": int(a.shape[0])}


def _value(res: dict, key: str) -> float:
    return next(s["value"] for s in res["summary"] if s["key"] == key)


def test_recipes_in_both_languages(client: TestClient) -> None:
    zh = client.get("/api/v1/sem/recipes").json()["recipes"]
    en = client.get("/api/v1/sem/recipes", params={"lang": "en"}).json()["recipes"]
    names = [r["name"] for r in zh]
    assert {"lines", "holes", "particles", "fibres", "layers"} <= set(names) and names == [r["name"] for r in en]
    layers = next(r for r in en if r["name"] == "layers")
    geometry = next(p for p in layers["params"] if p["name"] == "geometry")
    assert geometry["kind"] == "choice" and {c["value"] for c in geometry["choices"]} == {"auto", "straight", "curved"}
    assert geometry["label"] == "Layer shape" and not any("一" <= ch <= "鿿" for ch in layers["label"])
    smooth = next(p for p in layers["params"] if p["name"] == "smooth")
    assert smooth["min"] == 0 and smooth["max"] == 10


def test_lines_from_pixels_with_the_persons_calibration(client: TestClient) -> None:
    img = S.line_space(seed=1)
    res = client.post("/api/v1/sem/measure", json={"recipe": "lines", "pixel_size": 2.0, "unit": "nm", **_pixels(img)}).json()
    assert res["engine"] == "vexel-python" and res["validated"] and res["error"] is None
    assert res["calibration"]["source"] == "person" and res["unit"] == "nm"
    cd = _value(res, "cd_mean")
    assert res["image"]["width"] == img.shape[1] and len(res["image"]["sha256"]) == 64
    # the same pixels without calibration: the same measurement in pixels
    px = client.post("/api/v1/sem/measure", json={"recipe": "lines", **_pixels(img)}).json()
    assert px["unit"] == "px" and _value(px, "cd_mean") == pytest.approx(cd / 2.0, rel=1e-6)
    assert any(s["unit"] == "px" for s in px["summary"] if s["key"] == "cd_mean")


def test_same_numbers_as_the_sem_window(client: TestClient) -> None:
    """Parity: the HTTP result is the in-process measurement the SEM window makes."""
    from vexel.packs.sem import process
    from vexel.packs.sem.analysis import AnalysisSpec, run
    from vexel.packs.sem.measure import Scale

    img, tr = S.bent_layer_stack(seed=3, bow=40)
    params = {"geometry": "auto"}
    res = client.post("/api/v1/sem/measure", json={"recipe": "layers", "params": params, "pixel_size": 1.5, "unit": "nm", **_pixels(img)}).json()
    image = np.ascontiguousarray(img, dtype="<f4").astype(np.float64)
    pr = process.Processor(image, None).run(process.Processing())
    an = run(AnalysisSpec(id=1, recipe="layers", params=params), pr.analysis, Scale(pixel_size=1.5e-9, source="manual", confirmed=True), pr.mask, None)
    window = {k: v for k, _, v, kind in an.summary if kind == "length"}
    http = {s["key"]: s["value"] for s in res["summary"] if s["kind"] == "length"}
    assert window.keys() == http.keys() and len(window) == 3
    for k, v in window.items():
        assert http[k] == pytest.approx(v * 1e9, rel=1e-9)
    # and the bent stack is measured along the normal, against the truth
    for i, t in enumerate(tr["thicknesses"], start=1):
        assert http[f"layer{i}"] == pytest.approx(t * 1.5, abs=0.5 * 1.5)
    assert any(s["key"] == "bend" for s in res["summary"])
    assert sum(s["kind"] == "polyline" for s in res["shapes"]) == 4  # the four interfaces (and the region's outline)


def test_file_is_calibrated_from_its_metadata(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "thermo.tif"
    truth = S.write_sem(path, "thermo", pixel_size=4e-9, rows=300, cols=400, bar_height=40, seed=2)
    body = {"recipe": "particles", "file": base64.b64encode(path.read_bytes()).decode(), "filename": path.name, "lang": "en"}
    res = client.post("/api/v1/sem/measure", json=body).json()
    assert res["calibration"]["source"] != "none" and res["unit"] == "nm"
    assert res["image"]["height"] == 340 and res["crop_rows"] == 300  # the data bar is in the file; only the micrograph is measured
    assert not any("一" <= ch <= "鿿" for ch in res["recipe_label"] + res["calibration"]["text"])
    assert truth is not None


@pytest.mark.parametrize(
    "body, needle",
    [
        ({"recipe": "nonsense"}, "unknown SEM measurement"),
        ({"recipe": "lines"}, "send the image"),
        ({"recipe": "lines", "pixels": "%%%", "width": 2, "height": 2}, "base64"),
        ({"recipe": "lines", "pixels": base64.b64encode(b"\0" * 12).decode(), "width": 2, "height": 2}, "expected 4 float32 pixels"),
    ],
)
def test_refusals_are_clear(client: TestClient, body: dict, needle: str) -> None:
    r = client.post("/api/v1/sem/measure", json=body)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_noise_is_refused_not_measured(client: TestClient) -> None:
    rng = np.random.default_rng(0)
    res = client.post("/api/v1/sem/measure", json={"recipe": "layers", **_pixels(rng.normal(0.5, 0.05, (200, 300)))}).json()
    assert res["error"] and not [s for s in res["summary"] if s["key"].startswith("layer")]


def test_tr_in_does_not_depend_on_the_interface_language(tmp_path: Path, monkeypatch) -> None:
    from vexel import i18n

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path))  # set_language writes the config: keep the person's own untouched
    monkeypatch.delenv("VEXEL_LANG", raising=False)
    i18n.language.cache_clear()
    try:
        i18n.set_language("zh")
        assert i18n.tr_in("层面形状", "en") == "Layer shape"
        assert i18n.tr("层面形状") == "层面形状"
        i18n.set_language("en")
        assert i18n.tr_in("层面形状", "zh") == "层面形状"
        assert i18n.tr("层面形状") == "Layer shape"
        assert i18n.tr_in("层面倾斜 3.5°：厚度按垂直方向测量，倾斜大时偏大", "en").startswith("Layers slant by 3.5°")
    finally:
        monkeypatch.undo()
        i18n.language.cache_clear()
        i18n.tr_in.cache_clear()
