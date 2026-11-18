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
    p = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert p["status"] == "completed"
    assert p["recipe"]["sha256"] == Recipe.model_validate_json((out / "recipe.json").read_text(encoding="utf-8")).digest()
    assert len(p["input"]["sha256"]) == 64 and p["input"]["frames_used"] == 200
    assert p["input"]["space_unit"] == "um"
    assert {"numpy", "scikit-image", "laptrack"} <= set(p["environment"]["packages"])
    assert [s["id"] for s in p["steps"]] == ["bg", "seg", "measure", "track", "droplets", "flags"]
    assert p["steps"][1]["scope"] == "frame" and "radius_median_px.mean" in p["steps"][1]["qc"]
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
    p = json.loads((tmp_path / "r" / "provenance.json").read_text(encoding="utf-8"))
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
    (foreign / "thesis.txt").write_text("precious", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_recipe(recipe, blank, foreign, overwrite=True)
    assert (foreign / "thesis.txt").exists()
    run_recipe(recipe, blank, tmp_path / "r")
    with pytest.raises(FileExistsError):
        run_recipe(recipe, blank, tmp_path / "r")
    run_recipe(recipe, blank, tmp_path / "r", overwrite=True)


# -- real-looking droplets: rings, bright centres, trains ----------------------------------


def _ring_droplets(centres, r=15.0, shape=(80, 240), bright_centre=False, seed=0):
    """Droplets drawn as a thin dark outline on a flat background, optionally with a bright
    core, plus a static wall line; returns (frame, background)."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(float)
    bg = np.full(shape, 120.0)
    bg[8:10, :] = 40.0  # channel wall: static, must be ignored
    img = bg.copy()
    for cy, cx in centres:
        d = np.hypot(yy - cy, xx - cx)
        img -= 60.0 * np.exp(-0.5 * ((d - (r - 1.5)) / 1.0) ** 2)
        if bright_centre:
            img += 30.0 * np.exp(-0.5 * (d / (0.4 * r)) ** 2)
    img += np.random.default_rng(seed).normal(0, 2.0, shape)
    return img.astype(np.float32), bg.astype(np.float32)


@pytest.mark.parametrize("bright_centre", [False, True])
def test_circles_find_ring_droplets_a_threshold_cannot_fill(bright_centre):
    centres = [(40.0, 40.5), (40.0, 71.0), (41.0, 101.5), (39.5, 160.0)]  # three touching, one apart
    img, bg = _ring_droplets(centres, bright_centre=bright_centre)
    op = get_op("segment.circles")
    res = op.fn(op.params(), _ctx(img.shape), image=img, contrast=np.abs(img - bg))
    labels = res.outputs["labels"]
    assert labels.max() == 4
    for k, (cy, cx) in enumerate(sorted(centres, key=lambda c: c[1]), 1):
        ys, xs = np.nonzero(labels == labels[int(cy), int(cx)])
        assert abs(ys.mean() - cy) < 0.7 and abs(xs.mean() - cx) < 0.7
        assert np.sqrt(len(ys) / np.pi) == pytest.approx(15.0, abs=1.2)
    assert res.qc["radius_median_px"] == pytest.approx(15.0, abs=1.0)


def test_circles_ignore_static_scene_and_noise():
    img, bg = _ring_droplets([])
    op = get_op("segment.circles")
    res = op.fn(op.params(), _ctx(img.shape), image=img, contrast=np.abs(img - bg))
    assert res.outputs["labels"].max() == 0


def test_flow_estimate_resolves_a_dense_train():
    # Objects 35 px apart moving 16-19 px per frame: a neighbour's next position is about
    # as close as the object's own, so plain nearest-neighbour linking swaps identities.
    rng = np.random.default_rng(1)
    rows, offset = [], 0.0
    for frame in range(30):
        for k in range(12):
            rows.append({"frame": frame, "centroid_y_px": 50.0 + rng.normal(0, 0.3), "centroid_x_px": offset + 35.0 * k})
        offset += rng.uniform(16.0, 19.0)
    table = pd.DataFrame(rows)
    table["true_id"] = np.tile(np.arange(12), 30)
    op = get_op("track.link")
    res = op.fn(op.params(max_distance_px=30, estimate_shift=True), _ctx(n=30), table=table)
    out = res.outputs["table"]
    assert out.groupby("track_id")["true_id"].nunique().eq(1).all()
    assert out["track_id"].nunique() == 12
    assert res.qc["flow_dx_px"] == pytest.approx(17.5, abs=1.0)


def test_cells_counted_against_the_droplets_own_outline():
    # A droplet with a ring thicker on one side, shading, one cell near the edge and one
    # in the centre; a second droplet is empty. Faint shading must not count as cells.
    yy, xx = np.mgrid[:80, :160].astype(float)
    img = np.full((80, 160), 120.0)
    labels = np.zeros((80, 160), np.int32)
    for k, cx in enumerate((40.0, 110.0), 1):
        d = np.hypot(yy - 40, xx - cx)
        side = 1.0 + 0.8 * np.clip((xx - cx) / 15.0, 0, 1)  # thicker on the right
        img -= 70.0 * np.exp(-0.5 * ((d - 13.0) / side) ** 2)
        img -= 6.0 * np.clip((yy - 40) / 15.0, 0, 1) * (d < 13)  # shading towards the bottom
        labels[d <= 15.0] = k
    for cy, cx in ((40, 40), (40, 30)):  # centre and near the left edge of droplet 1
        img -= 50.0 * np.exp(-0.5 * (np.hypot(yy - cy, xx - cx) / 1.5) ** 2)
    img += np.random.default_rng(3).normal(0, 1.0, img.shape)
    op = get_op("droplet.inclusions")
    params = op.params(cell_diameter_px=5.0, edge_margin_px=3, background="radial", relative_floor=0.3)
    counts = op.fn(params, _ctx(img.shape), image=img.astype(np.float32), labels=labels).outputs["table"]
    assert counts["n_inclusions"].tolist() == [2.0, 0.0]


def test_cells_in_droplets_cut_by_the_border_are_not_counted():
    labels = np.zeros((40, 60), np.int32)
    yy, xx = np.mgrid[:40, :60]
    labels[np.hypot(yy - 20, xx - 5) <= 12] = 1
    labels[np.hypot(yy - 20, xx - 35) <= 12] = 2
    img = np.full(labels.shape, 100.0, np.float32)
    op = get_op("droplet.inclusions")
    out = op.fn(op.params(edge_margin_px=2), _ctx(labels.shape), image=img, labels=labels).outputs["table"]
    assert np.isnan(out["n_inclusions"].iloc[0]) and out["n_inclusions"].iloc[1] == 0
