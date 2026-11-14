"""Human corrections: label and table edits, re-execution, frame inspection, droplet adapter."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vexel.engine import Recipe, run_recipe
from vexel.engine.corrections import Corrections, Edit, apply_label_edits, apply_table_edits
from vexel.engine.review import FrameInspector, rerun_with_corrections, review_adapter
from vexel.readers import open_volume

# -- label edits ----------------------------------------------------------------------------


def _two_disks() -> np.ndarray:
    yy, xx = np.mgrid[:40, :80]
    labels = np.zeros((40, 80), np.int32)
    labels[np.hypot(yy - 20, xx - 20) <= 8] = 1
    labels[np.hypot(yy - 20, xx - 40) <= 8] = 2
    return labels


def test_label_edits_delete_merge_paint_erase():
    labels = _two_disks()
    out, st = apply_label_edits(labels, [Edit(kind="labels.delete", target="seg.labels", frame=0, at=(20, 20))])
    assert set(np.unique(out)) == {0, 2} and st == ["applied"]
    out, st = apply_label_edits(labels, [Edit(kind="labels.merge", target="seg.labels", frame=0, points=[(20, 20), (20, 40)])])
    assert set(np.unique(out)) == {0, 1} and (out > 0).sum() == (labels > 0).sum()
    out, _ = apply_label_edits(labels, [Edit(kind="labels.paint", target="seg.labels", frame=0, shape={"circle": [20, 65, 6]})])
    assert out.max() == 3 and out[20, 65] == 3
    out, _ = apply_label_edits(labels, [Edit(kind="labels.paint", target="seg.labels", frame=0, at=(20, 40), shape={"circle": [20, 52, 4]})])
    assert out[20, 52] == 2  # extends the object under the point
    out, _ = apply_label_edits(labels, [Edit(kind="labels.erase", target="seg.labels", frame=0, shape={"polygon": [[0, 18], [0, 22], [40, 22], [40, 18]]})])
    assert out[20, 20] == 0 and out[20, 40] == 2
    _, st = apply_label_edits(labels, [Edit(kind="labels.delete", target="seg.labels", frame=0, at=(2, 2))])
    assert st == ["no object at the point"]


def test_edits_are_validated_and_round_trip(tmp_path):
    with pytest.raises(ValueError, match="needs"):
        Edit(kind="table.set", target="t", frame=0, at=(1, 1))
    with pytest.raises(ValueError):
        Edit(kind="labels.paint", target="t", frame=0, shape={"square": [1]})
    c = Corrections()
    c.add(Edit(kind="labels.delete", target="seg.labels", frame=3, at=(5, 6), author="a"))
    assert c.edits[0].time
    path = c.save(tmp_path / "c.json")
    back = Corrections.load(path)
    assert back == c
    other = Corrections(edits=[c.edits[0].model_copy(update={"author": "b", "time": "x", "note": "n"})])
    assert other.digest() == c.digest()  # who and when do not change what the edit does


def test_table_edits_resolve_points_to_tracked_objects():
    tracks = pd.DataFrame({"frame": [5, 5, 6, 6], "label": [1, 2, 1, 2], "centroid_y_px": [10.0, 10.0, 10.0, 10.0],
                           "centroid_x_px": [20.0, 60.0, 30.0, 70.0], "diameter_px": [16.0] * 4, "track_id": [0, 1, 0, 1]})  # fmt: skip
    droplets = pd.DataFrame({"track_id": [0, 1], "n_inclusions": [1.0, 2.0]})
    tables = {"track.table": tracks}
    out, st = apply_table_edits(droplets, [Edit(kind="table.set", target="d", frame=6, at=(11, 69), values={"n_inclusions": 4})], tables)
    assert out["n_inclusions"].tolist() == [1, 4] and out["verified"].tolist() == [False, True] and st == ["applied"]
    out, st = apply_table_edits(tracks, [Edit(kind="table.exclude", target="track.table", frame=5, at=(10, 21))], tables)
    assert out["track_id"].tolist() == [1, 1]  # the whole track
    _, st = apply_table_edits(droplets, [Edit(kind="table.set", target="d", frame=6, at=(10, 200), values={"n_inclusions": 1})], tables)
    assert st == ["no object at the point"]


# -- re-execution ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cells_run(cells_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("corr") / "run"
    run_recipe(Recipe.load("droplet_encapsulation"), open_volume(cells_dataset / "droplets.ome.tif"), out, hash_input=False,
               keep_intermediates=True, report=False)  # fmt: skip
    return out


def _droplet_point(run_dir, i):
    """A frame and point on the i-th counted droplet."""
    d = pd.read_csv(run_dir / "tables" / "droplets.droplets.csv").iloc[i]
    tr = pd.read_csv(run_dir / "tables" / "track.table.csv")
    det = tr[tr["track_id"] == d["track_id"]].iloc[2]
    return int(det["frame"]), (float(det["centroid_y_px"]), float(det["centroid_x_px"])), d


def test_label_corrections_change_measurements(cells_run, tmp_path):
    frame, at, _ = _droplet_point(cells_run, 0)
    before = pd.read_csv(cells_run / "tables" / "measure.table.csv")
    c = Corrections(edits=[
        Edit(kind="labels.delete", target="seg.labels", frame=frame, at=at),
        Edit(kind="labels.paint", target="seg.labels", frame=frame, shape={"circle": [40, 700, 12]}),
    ])  # fmt: skip
    result = rerun_with_corrections(cells_run, c, tmp_path / "c")
    after = result.tables["measure.table"]
    b, a = before[before["frame"] == frame], after[after["frame"] == frame]
    assert len(a) == len(b)  # one removed, one added
    added = a[(a["centroid_x_px"] - 700).abs() < 2]
    assert len(added) == 1 and added["diameter_px"].iloc[0] == pytest.approx(24, abs=1.5)
    assert not ((a["centroid_x_px"] - at[1]).abs() < 2).any()
    assert result.provenance["corrections"]["n_applied"] == 2
    assert result.provenance["status"] == "completed" and (tmp_path / "c" / "report.html").exists()


def test_cell_count_and_exclusion_reach_the_statistics(cells_run, tmp_path):
    f0, at0, d0 = _droplet_point(cells_run, 0)
    f1, at1, _ = _droplet_point(cells_run, 1)
    base = json.loads((cells_run / "summary.json").read_text())
    c = Corrections(edits=[
        Edit(kind="table.set", target="droplets.droplets", frame=f0, at=at0, values={"n_inclusions": int(d0["n_inclusions"]) + 3}),
        Edit(kind="table.exclude", target="track.table", frame=f1, at=at1),
    ])  # fmt: skip
    r = rerun_with_corrections(cells_run, c, tmp_path / "c")
    n0, n1 = base["droplets.summary"]["n_droplets"], r.summary["droplets.summary"]["n_droplets"]
    assert n1 == n0 - 1
    table = r.tables["droplets.droplets"]
    row = table[table["verified"] == True]  # noqa: E712
    assert len(row) == 1 and row["n_inclusions"].iloc[0] == d0["n_inclusions"] + 3
    assert r.summary["encapsulation.summary"]["n_inclusions_total"] != base["encapsulation.summary"]["n_inclusions_total"]
    flags = r.tables["flags.table"]
    assert row["track_id"].iloc[0] not in set(flags.get("track_id", []))  # confirmed by a person: not flagged
    again = rerun_with_corrections(cells_run, c, tmp_path / "again")
    assert again.result_digest == r.result_digest  # recipe + input + corrections reproduce the result
    assert json.loads((cells_run / "corrections.json").read_text())["edits"]  # kept with the run


def test_manual_positive_call_changes_the_positive_fraction(fluo_dataset, tmp_path):
    run = tmp_path / "run"
    base = run_recipe(Recipe.load("droplet_fluorescence"), open_volume(fluo_dataset / "droplets.ome.tif"), run, hash_input=False, report=False)
    table = base.tables["positivity.droplets"]
    neg = table[~table["positive"].astype(bool)].iloc[0]
    tr = base.tables["track.table"]
    det = tr[tr["track_id"] == neg["track_id"]].iloc[1]
    c = Corrections(edits=[Edit(kind="table.set", target="droplets.droplets", frame=int(det["frame"]),
                                at=(det["centroid_y_px"], det["centroid_x_px"]), values={"positive_manual": True})])  # fmt: skip
    r = rerun_with_corrections(run, c, tmp_path / "c")
    s0, s1 = base.summary["positivity.summary"], r.summary["positivity.summary"]
    assert s1["n_positive"] == s0["n_positive"] + 1 and s1["n_manual"] == 1


def test_corrections_must_name_values_the_recipe_produces(cells_run, tmp_path):
    c = Corrections(edits=[Edit(kind="labels.delete", target="nope.labels", frame=0, at=(1, 1))])
    with pytest.raises(ValueError, match="does not produce"):
        rerun_with_corrections(cells_run, c, tmp_path / "c")


# -- frame inspector and adapters -------------------------------------------------------------


def test_inspector_reproduces_the_run_segmentation(cells_run):
    import zarr

    inspector = FrameInspector(cells_run)
    stored = zarr.open_array(str(cells_run / "scratch" / "seg.labels.zarr"), mode="r")
    for t in (0, 77, 199):
        _, labels, marks = inspector.view(t)
        np.testing.assert_array_equal(labels, stored[t])
        assert marks is not None
    frame, at, _ = _droplet_point(cells_run, 0)
    inspector.corrections = Corrections(edits=[Edit(kind="labels.delete", target="seg.labels", frame=frame, at=at)])
    inspector.invalidate(frame)
    _, labels, _ = inspector.view(frame)
    assert labels[int(at[0]), int(at[1])] == 0


def test_review_adapters():
    a = review_adapter(Recipe.load("droplet_fluorescence_encapsulation").resolve())
    assert (a.objects, a.labels, a.exclude, a.flags) == ("positivity.droplets", "seg.labels", "track.table", "flags.table")
    assert [f.key for f in a.fields] == ["n_inclusions", "positive_manual"]
    assert review_adapter(Recipe.load("droplet_basic").resolve()).fields == []
    generic = review_adapter(Recipe.model_validate({
        "name": "g", "steps": [{"id": "seg", "op": "segment.threshold_watershed", "inputs": {"image": "source"}},
                               {"id": "m", "op": "measure.regionprops", "inputs": {"labels": "seg.labels"}}],
        "review": {"labels": "seg.labels"}}).resolve())  # fmt: skip
    assert (generic.objects, generic.labels, generic.fields) == ("m.table", "seg.labels", [])


def test_watched_runs_are_corrected_from_their_stored_frames(tmp_path):
    import tifffile

    from vexel.engine import run_stream
    from vexel.sources.watch import WatchSource
    from vexel.synthetic.droplets import DropletSynthConfig, generate

    folder = tmp_path / "acq"
    folder.mkdir()
    video, _, _ = generate(DropletSynthConfig(n_frames=120, seed=2))
    for i, f in enumerate(video):
        tifffile.imwrite(folder / f"f{i}.tif", f)
    recipe = Recipe.load("droplet_basic")
    out = tmp_path / "w"
    src = WatchSource(folder, recipe.resolve().input.sources(), pixel_size=2.0, frame_interval_s=0.001, idle_timeout_s=0.3,
                      poll_s=0.02, zarr_path=out / "frames.zarr")  # fmt: skip
    run_stream(recipe, src, out, report=False)
    for p in folder.iterdir():  # the acquisition folder may be gone; the stored frames are enough
        p.unlink()
    frame, at, _ = _droplet_point(out, 0)
    r = rerun_with_corrections(out, Corrections(edits=[Edit(kind="table.exclude", target="track.table", frame=frame, at=at)]))
    assert r.summary["droplets.summary"]["n_droplets"] == json.loads((out / "summary.json").read_text())["droplets.summary"]["n_droplets"] - 1


def test_correct_command(cells_run, tmp_path, capsys):
    from vexel.cli import main

    frame, at, _ = _droplet_point(cells_run, 2)
    c = Corrections(edits=[Edit(kind="table.exclude", target="track.table", frame=frame, at=at),
                           Edit(kind="labels.delete", target="seg.labels", frame=0, at=(1, 1))])  # fmt: skip
    path = c.save(tmp_path / "c.json")
    assert main(["correct", str(cells_run), "--corrections", str(path), "-o", str(tmp_path / "o")]) == 1  # one edit missed
    text = capsys.readouterr().out
    assert "applied 1 of 2" in text and "no object at the point" in text
