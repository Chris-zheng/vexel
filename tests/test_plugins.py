"""The model plugin contract: registration, pinned versions, weight checks, provenance, licences."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from vexel import plugins
from vexel.ops.registry import Context, Kind, OpResult, Params, Scope, register_op


class ThresholdModel:
    """A 'trained model' whose weights are one number: the threshold."""

    def __init__(self, card):
        self.card = card
        self.threshold = float(np.load(card.path()))

    def predict(self, image):
        from skimage.measure import label

        return label(image < self.threshold)


class ModelParams(Params):
    model: str


@pytest.fixture
def toy_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("VEXEL_HOME", str(tmp_path))
    monkeypatch.setattr(plugins, "_MODELS", {})
    monkeypatch.setattr(plugins, "_LOADED", {})
    weights = tmp_path / "models" / "toy-droplets" / "1.0" / "threshold.npy"
    weights.parent.mkdir(parents=True)
    np.save(weights, np.array(-20.0))
    card = plugins.ModelCard(
        id="toy-droplets", version="1.0", task="segment.instances", framework="numpy", license="MIT", file="threshold.npy",
        sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        training_data=(plugins.Dataset("synthetic droplets", "MIT"), plugins.Dataset("lab set", "CC BY-NC 4.0")),
    )  # fmt: skip
    plugins.register_model(card, ThresholdModel)
    from vexel.ops import registry

    if "test.model_segment" not in registry._REGISTRY:

        @register_op("test.model_segment", scope=Scope.FRAME, inputs={"image": Kind.IMAGE}, outputs={"labels": Kind.LABELS}, params=ModelParams)
        def model_segment(params: ModelParams, ctx: Context, *, image: np.ndarray) -> OpResult:
            """Droplets from a trained model."""
            return OpResult({"labels": plugins.load_model(params.model).predict(image).astype(np.int32)})

    return card, weights


def test_model_card_rules(toy_plugin):
    card, _ = toy_plugin
    assert card.ref == "toy-droplets@1.0" and card.commercial_use is False  # one dataset is non-commercial
    assert plugins.ModelCard("m", "1", "classify.objects", "onnx", "Apache-2.0", (plugins.Dataset("d", "CC BY 4.0"),)).commercial_use is True
    assert plugins.ModelCard("m", "1", "classify.objects", "onnx", "Apache-2.0", (plugins.Dataset("d", "待核实"),)).commercial_use is None
    with pytest.raises(ValueError, match="unknown task"):
        plugins.ModelCard("m", "1", "magic", "onnx", "MIT")
    with pytest.raises(ValueError, match="name the version"):
        plugins.get_card("toy-droplets")
    with pytest.raises(KeyError, match="unknown model"):
        plugins.get_card("toy-droplets@2.0")


def test_weights_are_verified(toy_plugin):
    _, weights = toy_plugin
    assert plugins.file_status(plugins.get_card("toy-droplets@1.0")) == "ok"
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        plugins.load_model("toy-droplets@1.0")
    weights.unlink()
    with pytest.raises(FileNotFoundError, match="put the weights at"):
        plugins.load_model("toy-droplets@1.0")


def test_model_step_is_recorded_in_provenance_and_methods(toy_plugin, droplet_dataset, tmp_path):
    from vexel.engine import Recipe, run_recipe
    from vexel.readers import open_volume
    from vexel.report.html_reporter import methods_text, methods_text_en

    recipe = Recipe.load("droplet_basic").model_dump(mode="json")
    seg = next(s for s in recipe["steps"] if s["id"] == "seg")
    seg.update(op="test.model_segment", inputs={"image": "bg.image"}, params={"model": "toy-droplets@1.0"}, qc=[])
    recipe = Recipe.model_validate(recipe)
    result = run_recipe(recipe, open_volume(droplet_dataset / "droplets.ome.tif"), tmp_path / "run", max_frames=40, hash_input=False, report=False)
    prov = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    step = next(s for s in prov["steps"] if s["id"] == "seg")
    assert step["model"]["ref"] == "toy-droplets@1.0" and step["model"]["commercial_use"] is False
    assert step["model"]["sha256"] == toy_plugin[0].sha256
    zh = " ".join(methods_text(recipe.resolve().model_dump(mode="json"), prov))
    en = " ".join(methods_text_en(recipe.resolve().model_dump(mode="json"), prov))
    assert "toy-droplets@1.0" in zh and "CC BY-NC 4.0" in zh and "trained model toy-droplets@1.0" in en
    assert result.status == "completed"


def test_task_contract_is_checked(toy_plugin):
    from vexel.ops.registry import get_op

    card = plugins.ModelCard("counter", "1", "count.inclusions", "numpy", "MIT")
    plugins.register_model(card, lambda c: None)
    with pytest.raises(ValueError, match="needs inputs"):
        plugins.model_record(ModelParams(model="counter@1"), get_op("test.model_segment"))


def test_a_broken_plugin_does_not_break_vexel(monkeypatch):
    class EP:
        name, value, dist = "broken", "nowhere:register", None

        def load(self):
            raise ImportError("no module named nowhere")

    monkeypatch.setattr(plugins.md, "entry_points", lambda group: [EP()])
    infos = plugins.load_plugins(force=True)
    try:
        assert infos[0].status == "failed" and "nowhere" in infos[0].error
        from vexel.cli import main

        assert main(["plugins"]) == 1  # reported, and listed operators still work
        from vexel.ops import list_ops

        assert any(o.name == "segment.circles" for o in list_ops())
    finally:
        monkeypatch.undo()
        plugins.load_plugins(force=True)
