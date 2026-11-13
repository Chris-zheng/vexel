"""Operator registry, recipe resolution, batch execution, QC gate and provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vexel.core.volume import SpatialVolume
from vexel.engine import QCGateError, Recipe, RecipeError, run_recipe
from vexel.ops import Context, Scope, get_op, list_ops
from vexel.packs.droplet.validate import validate_run
from vexel.readers import open_volume


def _recipe(steps: list[dict], **extra) -> Recipe:
    return Recipe.model_validate({"name": "t", "steps": steps, **extra})


SEG = {"id": "seg", "op": "segment.threshold_watershed", "inputs": {"image": "source"}}
MEAS = {"id": "m", "op": "measure.regionprops", "inputs": {"labels": "seg.labels"}}


# -- registry ---------------------------------------------------------------------------


def test_every_operator_declares_a_scope_and_typed_io():
    ops = list_ops()
    assert {"preprocess.static_background", "segment.threshold_watershed", "measure.regionprops", "track.link", "droplet.summary"} <= {
        o.name for o in ops
    }
    for op in ops:
        assert isinstance(op.scope, Scope) and op.outputs and op.doc
    assert get_op("segment.threshold_watershed").scope is Scope.FRAME
    assert get_op("track.link").scope is Scope.WINDOW


# -- recipe resolution --------------------------------------------------------------------


def test_resolve_fills_defaults_and_digest_is_stable():
    resolved = _recipe([SEG, MEAS]).resolve()
    assert resolved.steps[0].params["method"] == "otsu"
    assert resolved.digest() == resolved.resolve().digest()
    assert resolved.digest() != _recipe([SEG, MEAS]).with_overrides({"seg.min_area": 7}).resolve().digest()


@pytest.mark.parametrize(
    "steps, message",
    [
        ([{"id": "a", "op": "nope.nope"}], "unknown operator"),
        ([{**SEG, "params": {"bogus": 1}}], "invalid parameters"),
        ([{**SEG, "params": {"min_distance": 0}}], "invalid parameters"),
        ([{"id": "seg", "op": "segment.threshold_watershed"}], "missing required input"),
        ([MEAS], "no earlier step produces"),
        ([SEG, {"id": "m", "op": "measure.regionprops", "inputs": {"labels": "source"}}], "expects labels"),
        ([SEG, {**SEG}], "duplicate"),
        ([SEG, {"id": "t", "op": "track.link", "inputs": {"table": "seg.labels"}}], "expects table"),
    ],
)
def test_resolve_rejects_broken_recipes(steps, message):
    with pytest.raises(RecipeError, match=message):
        _recipe(steps).resolve()


def test_bad_override_is_rejected():
    with pytest.raises(RecipeError):
        _recipe([SEG]).with_overrides({"nostep.x": 1})
    with pytest.raises(RecipeError):
        _recipe([SEG]).with_overrides({"seg.min_area": "big"}).resolve()


# -- operators ------------------------------------------------------------------------------


def _ctx(shape=(64, 64), n=1, **kw) -> Context:
    return Context(frame_shape=shape, n_frames=n, **kw)


def test_watershed_splits_touching_disks_and_measures_in_physical_units():
    yy, xx = np.mgrid[:64, :96]
    img = (((yy - 32) ** 2 + (xx - 34) ** 2 < 14**2) | ((yy - 32) ** 2 + (xx - 60) ** 2 < 14**2)).astype(np.float32) * 100
    seg = get_op("segment.threshold_watershed")
    labels = seg.fn(seg.params(smooth_sigma=0), _ctx((64, 96)), image=img).outputs["labels"]
    assert labels.max() == 2
    meas = get_op("measure.regionprops")
    table = meas.fn(meas.params(), _ctx((64, 96), spacing_yx=(2.0, 2.0), space_unit="um"), labels=labels).outputs["table"]
    assert len(table) == 2
    np.testing.assert_allclose(table["diameter"], 2 * table["diameter_px"])
    np.testing.assert_allclose(table["area"], 4 * table["area_px"])
    assert not table["touches_border"].any()


def test_empty_frame_is_not_segmented_as_noise():
    noise = np.abs(np.random.default_rng(0).normal(0, 3, (64, 64))).astype(np.float32)
    seg = get_op("segment.threshold_watershed")
    result = seg.fn(seg.params(min_threshold=15.0), _ctx(), image=noise)
    assert result.outputs["labels"].max() == 0 and result.qc["n_objects"] == 0


def test_tracking_follows_flow_and_ids_are_ordered_by_first_appearance():
    rows = []
    for frame in range(6):
        for k, x0 in enumerate((10.0, 60.0)):
            rows.append({"frame": frame, "centroid_y_px": 20.0 + k, "centroid_x_px": x0 + 12.0 * frame})
    table = pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # row order must not matter
    op = get_op("track.link")
    out = op.fn(op.params(max_distance_px=20), _ctx(n=6), table=table).outputs["table"]
    assert out["track_id"].nunique() == 2
    assert out.groupby("track_id")["centroid_y_px"].nunique().eq(1).all()
    assert out.sort_values(["frame", "centroid_x_px"]).iloc[0]["track_id"] == 0
    # a jump larger than the cutoff is only linkable once the expected flow is removed
    out = op.fn(op.params(max_distance_px=5), _ctx(n=6), table=table).outputs["table"]
    assert out["track_id"].nunique() == 12
    out = op.fn(op.params(max_distance_px=5, expected_shift_px=(0, 12)), _ctx(n=6), table=table).outputs["table"]
    assert out["track_id"].nunique() == 2


def test_tracking_empty_table():
    op = get_op("track.link")
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "centroid_y_px": [], "centroid_x_px": []})
    out = op.fn(op.params(), _ctx(), table=empty).outputs["table"]
    assert "track_id" in out.columns and out.empty


# -- end to end ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def droplet_run(droplet_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "run"
    volume = open_volume(droplet_dataset / "droplets.ome.tif")
    return run_recipe(Recipe.load("droplet_basic"), volume, out)


def test_droplet_pipeline_matches_ground_truth(droplet_dataset, droplet_run):
    checks = validate_run(droplet_run.out_dir, droplet_dataset / "truth")
    assert [c.name for c in checks] == ["n_droplets", "diameter_mean", "diameter_cv_percent", "frequency", "volume_mean"]
    assert all(c.passed for c in checks), checks
    assert not droplet_run.violations


def test_each_droplet_is_counted_once(droplet_run):
    droplets = droplet_run.tables["droplets.droplets"]
    assert droplets["track_id"].is_unique
    assert len(droplet_run.tables["track.table"]) > 5 * len(droplets)  # many detections per droplet
    assert droplet_run.summary["droplets.summary"]["n_droplets"] == len(droplets)


def test_run_directory_and_provenance(droplet_dataset, droplet_run):
    out = droplet_run.out_dir
    assert not (out / "scratch").exists()
    assert len(list((out / "review").glob("*.png"))) == 4
    p = json.loads((out / "provenance.json").read_text())
    assert p["status"] == "completed"
    assert p["recipe"]["sha256"] == Recipe.model_validate_json((out / "recipe.json").read_text()).digest()
    assert len(p["input"]["sha256"]) == 64 and p["input"]["frames_used"] == 200
    assert p["input"]["space_unit"] == "um"
    assert {"numpy", "scikit-image", "laptrack"} <= set(p["environment"]["packages"])
    assert [s["id"] for s in p["steps"]] == ["bg", "seg", "measure", "track", "droplets", "flags"]
    assert p["steps"][1]["scope"] == "frame" and "threshold.mean" in p["steps"][1]["qc"]
    assert set(p["outputs"]) == {
        "summary.json", "tables/measure.table.csv", "tables/track.table.csv", "tables/droplets.droplets.csv", "tables/flags.table.csv"
    }  # fmt: skip
    assert (out / "report.html").exists() and "report.html" not in p["outputs"]  # the report is a view, not a result


def test_same_recipe_and_data_give_identical_results(droplet_dataset, droplet_run, tmp_path):
    volume = open_volume(droplet_dataset / "droplets.ome.tif")
    again = run_recipe(Recipe.load("droplet_basic"), volume, tmp_path / "again", workers=1, hash_input=False)
    assert again.result_digest == droplet_run.result_digest
    changed = Recipe.load("droplet_basic").with_overrides({"droplets.count_line": 0.4})
    other = run_recipe(changed, volume, tmp_path / "other", hash_input=False)
    assert other.result_digest != droplet_run.result_digest


def test_uncalibrated_input_reports_pixels(tmp_path):
    from vexel.synthetic.droplets import DropletSynthConfig, generate

    video, _, _ = generate(DropletSynthConfig(n_frames=60, seed=3))
    result = run_recipe(Recipe.load("droplet_basic"), SpatialVolume.from_array(video, "TYX"), tmp_path / "r")
    s = result.summary["droplets.summary"]
    assert s["length_unit"] == "px" and s["calibrated"] is False and s["frequency_unit"] == "1/frame"
    assert s["diameter_mean"] == pytest.approx(30.0, rel=0.03)


def test_max_frames_and_kept_intermediates(droplet_dataset, tmp_path):
    volume = open_volume(droplet_dataset / "droplets.ome.tif")
    result = run_recipe(Recipe.load("droplet_basic"), volume, tmp_path / "r", max_frames=50, keep_intermediates=True, hash_input=False)
    assert result.provenance["input"]["frames_used"] == 50
    assert result.tables["measure.table"]["frame"].max() == 49
    assert {"bg.image.zarr", "seg.labels.zarr"} <= set(result.provenance["intermediates"])
    assert (tmp_path / "r" / "scratch" / "seg.labels.zarr").is_dir()


def test_qc_gate_stops_run_but_keeps_provenance(tmp_path):
    blank = SpatialVolume.from_array(np.full((5, 32, 32), 100, np.uint8), "TYX")
    recipe = _recipe([{**SEG, "qc": [{"metric": "n_objects.max", "min": 1, "on_fail": "stop"}]}, MEAS])
    with pytest.raises(QCGateError, match="n_objects.max"):
        run_recipe(recipe, blank, tmp_path / "r")
    p = json.loads((tmp_path / "r" / "provenance.json").read_text())
    assert p["status"] == "stopped_by_qc" and [s["id"] for s in p["steps"]] == ["seg"]
    assert p["steps"][0]["qc_violations"][0]["metric"] == "n_objects.max"


def test_qc_warning_does_not_stop(tmp_path):
    blank = SpatialVolume.from_array(np.full((5, 32, 32), 100, np.uint8), "TYX")
    recipe = _recipe([{**SEG, "qc": [{"metric": "n_objects.max", "min": 1}]}, MEAS])
    result = run_recipe(recipe, blank, tmp_path / "r")
    assert result.status == "completed" and len(result.violations) == 1
    assert result.tables["m.table"].empty and "frame" in result.tables["m.table"].columns


def test_output_directory_is_protected(tmp_path):
    blank = SpatialVolume.from_array(np.zeros((3, 16, 16), np.uint8), "TYX")
    recipe = _recipe([SEG])
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "thesis.txt").write_text("precious")
    with pytest.raises(FileExistsError):
        run_recipe(recipe, blank, foreign, overwrite=True)
    assert (foreign / "thesis.txt").exists()
    run_recipe(recipe, blank, tmp_path / "r")
    with pytest.raises(FileExistsError):
        run_recipe(recipe, blank, tmp_path / "r")
    run_recipe(recipe, blank, tmp_path / "r", overwrite=True)
