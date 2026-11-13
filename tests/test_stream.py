"""Stream executor: replay, online background and tracking, rolling statistics, alarms,
frame thinning, and agreement with the batch executor."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vexel.core.volume import SpatialVolume
from vexel.engine import Recipe, StreamSettings, run_recipe, run_stream
from vexel.packs.droplet.validate import validate_run
from vexel.readers import open_volume
from vexel.sources import ReplaySource


def _replay(path, recipe: Recipe, **kw) -> ReplaySource:
    return ReplaySource(open_volume(path), recipe.resolve().input.sources(), speed=kw.pop("speed", None), **kw)


def _by_crossing(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values("crossing_frame").reset_index(drop=True)


@pytest.fixture(scope="module")
def basic_pair(droplet_dataset, tmp_path_factory):
    recipe = Recipe.load("droplet_basic")
    root = tmp_path_factory.mktemp("stream")
    batch = run_recipe(recipe, open_volume(droplet_dataset / "droplets.ome.tif"), root / "batch", hash_input=False, report=False)
    stream = run_stream(recipe, _replay(droplet_dataset / "droplets.ome.tif", recipe), root / "stream", report=False)
    return batch, stream


def test_stream_matches_batch_droplet_by_droplet(droplet_dataset, basic_pair):
    batch, stream = basic_pair
    b, s = (_by_crossing(r.tables["droplets.droplets"]) for r in (batch, stream))
    assert len(s) == len(b) > 10
    np.testing.assert_allclose(s["crossing_frame"], b["crossing_frame"], atol=0.05)
    np.testing.assert_allclose(s["diameter"], b["diameter"], rtol=0.01)  # background estimated online
    assert all(c.passed for c in validate_run(stream.out_dir, droplet_dataset / "truth"))


def test_stream_run_directory(basic_pair):
    _, stream = basic_pair
    out = stream.out_dir
    p = json.loads((out / "provenance.json").read_text())
    assert p["mode"] == "stream" and p["status"] == "completed"
    st = p["stream"]
    assert st["frames_received"] == st["frames_processed"] == 200 and st["frames_dropped"] == 0
    assert st["processing_fps"] > 0 and st["latency_ms"] is None  # unpaced: no latency to report
    assert {s["id"]: s["streamed"] for s in p["steps"]} == {
        "bg": True, "seg": False, "measure": False, "track": True, "droplets": False, "flags": False
    }  # fmt: skip
    assert "flow_dx_px" in p["steps"][3]["qc"]
    assert len(list((out / "review").glob("*.png"))) == 4
    rolling = pd.read_csv(out / "stream" / "rolling.csv")
    assert rolling["frame"].tolist() == [49, 99, 149, 199] and rolling["final"].tolist() == [False] * 3 + [True]
    assert rolling["droplets.n_droplets"].is_monotonic_increasing  # window (500) longer than the video
    assert rolling["droplets.n_droplets"].iloc[-1] == stream.summary["droplets.summary"]["n_droplets"]
    assert pd.read_csv(out / "stream" / "alarms.csv").empty


def test_stream_counts_cells_like_batch(cells_dataset, tmp_path):
    recipe = Recipe.load("droplet_encapsulation")
    video = cells_dataset / "droplets.ome.tif"
    batch = run_recipe(recipe, open_volume(video), tmp_path / "b", hash_input=False, report=False)
    stream = run_stream(recipe, _replay(video, recipe), tmp_path / "s", report=False)
    b, s = (_by_crossing(r.tables["droplets.droplets"]) for r in (batch, stream))
    assert s["n_inclusions"].tolist() == b["n_inclusions"].tolist()
    assert stream.summary["encapsulation.summary"]["lambda"] == batch.summary["encapsulation.summary"]["lambda"]


def test_falling_behind_thins_frames_evenly_and_keeps_counting(droplet_dataset, tmp_path):
    # 1 ms frames replayed at 0.2x = 200 fps: faster than the operators, so frames are thinned.
    recipe = Recipe.load("droplet_basic")
    source = _replay(droplet_dataset / "droplets.ome.tif", recipe, speed=0.2)
    result = run_stream(recipe, source, tmp_path / "s", settings=StreamSettings(queue_size=16), report=False, workers=2)
    st = result.provenance["stream"]
    assert st["frames_dropped"] > 0 and st["frames_received"] == 200
    assert st["frames_dropped"] == st["frames_sampled_out"] + st["frames_lost_queue_full"]
    assert st["sampling_changes"] and st["sampling_changes"][0]["stride"] == 2
    assert st["latency_ms"]["median"] > 0
    # A 0.2 s clip outruns the processing; what was processed (frames_used = the span up to
    # the last processed frame) must still count every droplet that crossed in that span.
    assert result.provenance["input"]["frames_used"] <= 200
    check = {c.name: c for c in validate_run(result.out_dir, droplet_dataset / "truth")}
    assert check["n_droplets"].truth >= 5 and abs(check["n_droplets"].measured - check["n_droplets"].truth) <= 1


def test_drift_in_droplet_size_raises_an_alarm(tmp_path):
    from vexel.synthetic.droplets import DropletSynthConfig, generate

    common = dict(n_frames=300, seed=4, frequency_hz=150.0)
    before, _, _ = generate(DropletSynthConfig(diameter_mean_um=60.0, **common))
    after, _, _ = generate(DropletSynthConfig(diameter_mean_um=48.0, **{**common, "seed": 5}))
    volume = SpatialVolume.from_array(np.concatenate([before, after]), "TYX", spacing={"y": 2.0, "x": 2.0, "t": 0.001},
                                      space_unit="um", time_unit="s")  # fmt: skip
    recipe = Recipe.load("droplet_basic")
    snaps = []
    settings = StreamSettings(update_every=50, window=150, drift_min_droplets=10)
    result = run_stream(recipe, ReplaySource(volume, recipe.resolve().input.sources(), speed=None), tmp_path / "s",
                        settings=settings, on_update=snaps.append, report=False)  # fmt: skip
    drift = result.alarms[result.alarms["kind"] == "drift"]
    assert (drift["metric"] == "diameter_mean").any()
    first = drift[drift["metric"] == "diameter_mean"].iloc[0]
    assert first["frame"] >= 300 and first["value"] < 56  # raised after the change, on the smaller droplets
    assert (tmp_path / "s" / "stream" / first["image"]).exists()
    # every rolling update reached the callback; the last one, after all tracks closed, is final
    assert snaps[-1]["final"] and len(snaps) in (len(result.rolling), len(result.rolling) + 1)


def test_steady_stream_raises_no_drift(droplet_dataset, tmp_path):
    recipe = Recipe.load("droplet_basic")
    settings = StreamSettings(update_every=25, window=100, drift_min_droplets=5)
    result = run_stream(recipe, _replay(droplet_dataset / "droplets.ome.tif", recipe), tmp_path / "s", settings=settings, report=False)
    assert not (result.alarms["kind"] == "drift").any()


def test_stop_request_ends_the_run_with_partial_results(droplet_dataset, tmp_path):
    recipe = Recipe.load("droplet_basic")
    calls = {"n": 0}

    def stop_after_some() -> bool:
        calls["n"] += 1
        return calls["n"] > 100

    result = run_stream(recipe, _replay(droplet_dataset / "droplets.ome.tif", recipe), tmp_path / "s", should_stop=stop_after_some,
                        report=False, workers=1)  # fmt: skip
    assert result.status == "stopped"
    assert 0 < result.provenance["stream"]["frames_processed"] < 200
    assert (tmp_path / "s" / "summary.json").exists()


def test_recipes_that_cannot_stream_are_rejected(droplet_dataset, tmp_path):
    recipe = Recipe.model_validate(
        {
            "name": "t",
            "steps": [
                {"id": "seg", "op": "segment.threshold_watershed", "inputs": {"image": "source"}},
                {"id": "m", "op": "measure.regionprops", "inputs": {"labels": "seg.labels"}},
                {"id": "t1", "op": "track.link", "inputs": {"table": "m.table"}},
                {"id": "t2", "op": "track.link", "inputs": {"table": "m.table"}},
            ],
        }
    )
    with pytest.raises(NotImplementedError, match="one tracking step"):
        run_stream(recipe, _replay(droplet_dataset / "droplets.ome.tif", recipe), tmp_path / "s")
    assert not (tmp_path / "s").exists()


def test_paced_replay_needs_a_frame_interval():
    volume = SpatialVolume.from_array(np.zeros((3, 8, 8), np.uint8), "TYX")
    with pytest.raises(ValueError, match="frame interval"):
        ReplaySource(volume, {"source": 0}, speed=1.0)
    assert ReplaySource(volume, {"source": 0}, speed=None).n_frames == 3


def test_report_of_a_stream_run_shows_throughput_and_alarms(tmp_path):
    from vexel.report import write_report
    from vexel.synthetic.droplets import DropletSynthConfig, generate

    common = dict(n_frames=200, seed=4, frequency_hz=150.0)
    a, _, _ = generate(DropletSynthConfig(diameter_mean_um=60.0, **common))
    b, _, _ = generate(DropletSynthConfig(diameter_mean_um=48.0, **{**common, "seed": 5}))
    volume = SpatialVolume.from_array(np.concatenate([a, b]), "TYX", spacing={"y": 2.0, "x": 2.0, "t": 0.001}, space_unit="um", time_unit="s")
    recipe = Recipe.load("droplet_basic")
    settings = StreamSettings(update_every=40, window=120, drift_min_droplets=8)
    run_stream(recipe, ReplaySource(volume, recipe.resolve().input.sources(), speed=None), tmp_path / "s", settings=settings, report=False)
    html = write_report(tmp_path / "s").read_text()
    assert "在线回放" in html and "处理速度" in html and "滚动平均粒径" in html
    assert "漂移" in html and 'class="event"' in html


def test_thinned_frames_do_not_flag_droplets_as_seen_too_briefly(cells_dataset, tmp_path):
    recipe = Recipe.load("droplet_encapsulation")
    source = _replay(cells_dataset / "droplets.ome.tif", recipe, speed=0.1)  # 100 frames/s: thinned
    result = run_stream(recipe, source, tmp_path / "s", settings=StreamSettings(queue_size=16), report=False, workers=2)
    assert result.provenance["stream"]["frames_sampled_out"] > 0
    assert result.summary["flags.summary"]["n_few_frames"] == 0
