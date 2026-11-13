"""Multi-channel recipes, fluorescence intensity and positive/negative calls."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from vexel.cli import main
from vexel.core.volume import SpatialVolume
from vexel.engine import Recipe, RecipeError, run_recipe
from vexel.ops import Context, get_op
from vexel.packs import builtin_recipes
from vexel.packs.droplet.ops import _copies, _wilson, otsu_1d
from vexel.packs.droplet.validate import validate_run
from vexel.readers import open_volume
from vexel.synthetic.droplets import DropletSynthConfig, generate

SEG = {"id": "seg", "op": "segment.threshold_watershed", "inputs": {"image": "source"}}


# -- channels in recipes ---------------------------------------------------------------------


def _two_channel_recipe(**input_) -> Recipe:
    return Recipe.model_validate(
        {
            "name": "t",
            "input": {"channels": {"fluo": 1}, **input_},
            "steps": [SEG, {"id": "m", "op": "measure.regionprops", "inputs": {"labels": "seg.labels", "intensity": "source.fluo"}}],
        }
    )


def test_channel_aliases_resolve_and_override():
    r = _two_channel_recipe()
    assert r.resolve().input.channels == {"fluo": 1}
    assert r.with_overrides({"input.channels.fluo": 2, "input.channel": 1}).input.model_dump() == {
        "channel": 1, "z": 0, "channels": {"fluo": 2}
    }  # fmt: skip
    assert r.digest() != r.with_overrides({"input.channels.fluo": 2}).digest()
    with pytest.raises(RecipeError, match="unknown input setting"):
        r.with_overrides({"input.channels.cells": 1})
    with pytest.raises(RecipeError, match="invalid input override"):
        r.with_overrides({"input.channel": -1})
    with pytest.raises(RecipeError, match="lower-case"):
        Recipe.from_json(json.dumps({"name": "t", "input": {"channels": {"Bad Name": 1}}, "steps": [SEG]}))
    with pytest.raises(RecipeError, match="no earlier step produces"):
        Recipe.model_validate(
            {"name": "t", "steps": [SEG, {"id": "m", "op": "measure.regionprops", "inputs": {"labels": "seg.labels", "intensity": "source.fluo"}}]}
        ).resolve()


def test_executor_reads_each_channel_and_rejects_missing_ones(tmp_path):
    bright = np.zeros((3, 40, 40), np.uint8)
    bright[:, 10:30, 10:30] = 200
    fluo = np.full((3, 40, 40), 50, np.uint8)
    vol = SpatialVolume.from_array(np.stack([bright, fluo], axis=1), "TCYX")
    result = run_recipe(_two_channel_recipe(), vol, tmp_path / "ok")
    assert result.tables["m.table"]["mean_intensity"].eq(50).all()  # measured in channel 1, segmented in channel 0
    assert result.provenance["input"]["channels"] == {"fluo": 1}
    with pytest.raises(ValueError, match="data has 2 channel"):
        run_recipe(_two_channel_recipe().with_overrides({"input.channels.fluo": 5}), vol, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_builtin_recipes_share_their_common_steps():
    """The four droplet recipes must not drift apart: a step present in several of them is identical."""
    recipes = {name: json.loads(text) for name, text in builtin_recipes().items()}
    assert set(recipes) == {"droplet_basic", "droplet_encapsulation", "droplet_fluorescence", "droplet_fluorescence_encapsulation"}
    by_id: dict[str, list] = {}
    for r in recipes.values():
        for step in r["steps"]:
            by_id.setdefault(step["id"], []).append({k: step.get(k) for k in ("op", "params", "qc")})
    for step_id, variants in by_id.items():
        assert all(v == variants[0] for v in variants), step_id
    for name in recipes:
        Recipe.load(name).resolve()


# -- operators ---------------------------------------------------------------------------------


def test_intensity_subtracts_background_and_ignores_the_edge():
    yy, xx = np.mgrid[:80, :160]
    labels = np.zeros((80, 160), np.int32)
    labels[np.hypot(yy - 40, xx - 40) <= 20] = 1
    labels[np.hypot(yy - 40, xx - 120) <= 20] = 2
    img = np.full((80, 160), 10.0)
    img[labels == 1] = 30.0
    img[labels == 2] = 130.0
    img[(labels > 0) & (np.hypot(yy - 40, xx - 40) > 19) & (np.hypot(yy - 40, xx - 120) > 19)] = 500.0  # bright rim
    op = get_op("droplet.intensity")
    table = op.fn(op.params(), Context(frame_shape=img.shape, n_frames=1), image=img, labels=labels).outputs["table"]
    assert table["fluo_background"].iloc[0] == 10
    assert table["fluo_mean"].tolist() == pytest.approx([20.0, 120.0])  # rim excluded by edge_margin_px
    assert table["fluo_integrated"].iloc[1] > table["fluo_integrated"].iloc[0]
    raw = op.fn(op.params(background="none"), Context(frame_shape=img.shape, n_frames=1), image=img, labels=labels)
    assert raw.outputs["table"]["fluo_mean"].tolist() == pytest.approx([30.0, 130.0])


def _call(values, **params):
    op = get_op("droplet.positivity")
    table = pd.DataFrame({"fluo_mean": values})
    return op.fn(op.params(**params), Context(frame_shape=(10, 10), n_frames=1), table=table)


def test_positivity_splits_two_clear_populations():
    rng = np.random.default_rng(0)
    values = np.r_[rng.normal(17, 1.7, 150), rng.normal(110, 11, 50)]
    out = _call(values)
    s = out.outputs["summary"]
    assert s["threshold_method"] == "mixture" and s["populations"] == 2
    assert s["n_positive"] == 50 and values[:150].max() < s["threshold"] < values[150:].min()
    assert s["positive_fraction"] == 0.25 and s["copies_per_droplet"] == pytest.approx(-math.log(0.75))
    assert s["positive_fraction_ci_low"] < 0.25 < s["positive_fraction_ci_high"]
    assert out.outputs["droplets"]["positive"].sum() == 50


@pytest.mark.parametrize("sample", ["normal", "lognormal", "gamma"])
def test_positivity_does_not_split_a_single_population(sample):
    rng = np.random.default_rng(1)
    values = {"normal": rng.normal(17, 2.5, 600), "lognormal": rng.lognormal(3, 0.3, 600), "gamma": rng.gamma(4, 5, 600)}[sample]
    s = _call(values).outputs["summary"]
    assert s["populations"] == 1 and s["threshold_method"] == "outlier"
    assert s["n_positive"] <= 3  # only far outliers


def test_positivity_fixed_otsu_saturation_and_errors():
    s = _call([1, 2, 3, 10, 11], method="fixed", threshold=5).outputs["summary"]
    assert s["n_positive"] == 2 and s["threshold"] == 5
    assert _call([1, 2, 3, 10, 11], method="otsu").outputs["summary"]["threshold"] == 6.5
    s = _call([100, 101, 102], method="fixed", threshold=5).outputs["summary"]
    assert s["saturated"] and s["copies_per_droplet"] is None
    assert _call([]).outputs["summary"]["positive_fraction"] is None
    with pytest.raises(ValueError):
        _call([1, 2], method="fixed")


def test_statistics_helpers():
    assert otsu_1d(np.array([1, 2, 3, 10, 11, 12])) == 6.5
    assert otsu_1d(np.array([5, 5, 5])) is None
    lo, hi = _wilson(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3) and hi == pytest.approx(0.1744, abs=1e-3)
    assert _copies(0.0) == 0 and _copies(1.0) is None


def test_synthetic_fluorescence_channel():
    plain, _, _ = generate(DropletSynthConfig(n_frames=10, seed=2))
    video, droplets, _ = generate(DropletSynthConfig(n_frames=10, seed=2, fluorescence=True))
    assert video.shape == (10, 2, 256, 768) and np.array_equal(video[:, 0], plain)
    assert {"copies", "positive", "fluo_level"} <= set(droplets.columns)
    assert (droplets["positive"] == (droplets["copies"] >= 1)).all()
    with pytest.raises(ValueError):
        generate(DropletSynthConfig(n_frames=2, cell_lambda=1.0, cells_in="fluorescence"))


def test_ome_channel_names_are_read(fluo_dataset):
    vol = open_volume(fluo_dataset / "droplets.ome.tif")
    assert vol.sizes["C"] == 2 and vol.channel_names == ["brightfield", "fluorescence"]


# -- end to end -----------------------------------------------------------------------------


def test_fluorescence_pipeline_matches_truth(fluo_dataset, tmp_path):
    r = run_recipe(Recipe.load("droplet_fluorescence"), open_volume(fluo_dataset / "droplets.ome.tif"), tmp_path / "r")
    checks = validate_run(tmp_path / "r", fluo_dataset / "truth")
    assert [c.name for c in checks][-2:] == ["positive_mismatch", "positive_fraction"]
    assert all(c.passed for c in checks), checks
    s = r.summary["positivity.summary"]
    assert s["threshold_method"] == "mixture" and s["separation_d"] > 4
    droplets = r.tables["positivity.droplets"]
    assert {"fluo_mean", "fluo_median", "fluo_p95", "fluo_integrated", "positive"} <= set(droplets.columns)
    assert "fluo_background" not in droplets.columns


def test_fluorescent_cells_and_calls_together(tmp_path):
    from vexel.synthetic.droplets import write_dataset

    data = tmp_path / "d"
    write_dataset(data, DropletSynthConfig(n_frames=300, seed=21, fluorescence=True, cell_lambda=1.2, cells_in="fluorescence"))
    recipe = Recipe.load("droplet_fluorescence_encapsulation").with_overrides(
        {"input.channels.cells": 1, "inclusions.polarity": "bright"}
    )
    run_recipe(recipe, open_volume(data / "droplets.ome.tif"), tmp_path / "r", hash_input=False)
    checks = validate_run(tmp_path / "r", data / "truth")
    assert {"inclusion_mismatch", "positive_mismatch"} <= {c.name for c in checks}
    assert all(c.passed for c in checks), checks


def test_cli_fluorescence_roundtrip(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["synth", str(data), "--frames", "200", "--fluo", "0.7", "--seed", "2"]) == 0
    run = str(tmp_path / "run")
    assert main(["run", str(data / "droplets.ome.tif"), "-r", "droplet_fluorescence", "-o", run, "--no-input-hash"]) == 0
    capsys.readouterr()
    assert main(["validate", run, "--truth", str(data / "truth")]) == 0
    assert "positive_mismatch" in capsys.readouterr().out
    bad = str(tmp_path / "bad")
    assert main(["run", str(data / "droplets.ome.tif"), "-r", "droplet_fluorescence", "-o", bad, "--set", "input.channels.fluo=3"]) == 2
    assert "data has 2 channel" in capsys.readouterr().err
