"""Regression on real recordings (skipped when the data are not on this machine).

The files live outside git (licences unconfirmed). Point ``VEXEL_REALDATA`` at the
folder that holds ``DropletMicrofluidics.mov`` and ``WSCNet2-master.zip``; by default
``<repo>/microfluidics dataset`` is used. Truth files are in ``tests/realdata``.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vexel.core.volume import SpatialVolume
from vexel.engine import Recipe, run_recipe

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("VEXEL_REALDATA", ROOT / "microfluidics dataset"))
TRUTH = Path(__file__).resolve().parent / "realdata"


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"real data not available: {path}")
    return path


def _crossings(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "tables" / "droplets.droplets.csv").sort_values("crossing_frame").reset_index(drop=True)


@pytest.fixture(scope="module")
def mov_run(tmp_path_factory):
    import imageio.v3 as iio

    truth = json.loads((TRUTH / "droplet_microfluidics_mov.json").read_text(encoding="utf-8"))
    frames = iio.imread(_need(DATA / "DropletMicrofluidics.mov"), plugin="pyav").mean(-1).astype(np.float32)
    px = truth["pixel_size_um"]
    volume = SpatialVolume.from_array(frames, "TYX", spacing={"y": px, "x": px}, space_unit="um")
    out = tmp_path_factory.mktemp("mov") / "run"
    run_recipe(Recipe.load("droplet_basic"), volume, out, hash_input=False, report=False)
    return out, truth


def test_mov_counts_every_droplet_once_at_the_right_time(mov_run):
    out, truth = mov_run
    table = _crossings(out)
    assert len(table) == len(truth["crossing_frames"])
    np.testing.assert_allclose(table["crossing_frame"], truth["crossing_frames"], atol=1.0)


def test_mov_droplet_size_matches_the_scale_bar(mov_run):
    out, truth = mov_run
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))["droplets.summary"]
    assert summary["diameter_mean"] == pytest.approx(truth["diameter_um"], rel=truth["diameter_tolerance"])
    assert summary["diameter_cv_percent"] < 3.0  # one generator, steady flow: monodisperse


@pytest.fixture(scope="module")
def wscnet_run(tmp_path_factory):
    import imageio.v3 as iio

    archive = _need(DATA / "WSCNet2-master.zip")
    images = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            m = re.search(r"examples/example-folder/(\d+)\.jpg$", name)
            if m:
                images[int(m.group(1))] = iio.imread(io.BytesIO(zf.read(name))).mean(-1)
    frames = np.stack([images[k] for k in sorted(images)]).astype(np.float32)
    out = tmp_path_factory.mktemp("wscnet") / "run"
    run_recipe(Recipe.load("droplet_encapsulation"), SpatialVolume.from_array(frames, "TYX"), out, hash_input=False, report=False)
    return out, json.loads((TRUTH / "wscnet_example_folder.json").read_text(encoding="utf-8"))


def test_wscnet_counts_every_droplet_once(wscnet_run):
    out, truth = wscnet_run
    table = _crossings(out)
    expected = np.array([d["crossing_frame"] for d in truth["droplets"]])
    assert len(table) == len(expected)
    np.testing.assert_allclose(table["crossing_frame"], expected, atol=1.0)
    gaps = np.diff(table["crossing_frame"])
    assert gaps.max() < 1.5 * np.median(gaps)  # a missed droplet would leave a double gap


def test_wscnet_cells_per_droplet(wscnet_run):
    out, truth = wscnet_run
    table = _crossings(out)
    sure = [(m, d["n_cells"]) for m, d in zip(table["n_inclusions"], truth["droplets"]) if "uncertain" not in d]
    measured, expected = np.array(sure).T
    wrong = measured != expected
    assert wrong.mean() <= 0.06, f"{wrong.sum()} of {len(sure)} droplets miscounted"  # 5 of 132 (3.8%) since touching cells are split
    assert not ((expected == 0) & (measured > 0)).any(), "an empty droplet was given cells"
    assert ((measured <= expected) | ~wrong).all(), "errors so far are cells hidden behind each other (undercounts)"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))["encapsulation.summary"]
    all_truth = np.array([d["n_cells"] for d in truth["droplets"]])
    assert summary["frac_empty"] == pytest.approx(np.mean(all_truth == 0), abs=0.03)
    # Measured 1.52 against 1.57: the remaining undercounts (cells hidden behind each other) bias lambda low.
    assert summary["lambda"] == pytest.approx(all_truth.mean(), rel=0.05)


# -- the same recordings replayed through the stream executor -------------------------------


def test_mov_replayed_online_counts_the_same_droplets(mov_run, tmp_path):
    import imageio.v3 as iio

    from vexel.engine import run_stream
    from vexel.sources import ReplaySource

    _, truth = mov_run
    frames = iio.imread(DATA / "DropletMicrofluidics.mov", plugin="pyav").mean(-1).astype(np.float32)
    px = truth["pixel_size_um"]
    volume = SpatialVolume.from_array(frames, "TYX", spacing={"y": px, "x": px}, space_unit="um")
    recipe = Recipe.load("droplet_basic")
    result = run_stream(recipe, ReplaySource(volume, recipe.resolve().input.sources(), speed=None), tmp_path / "s", report=False)
    table = _crossings(tmp_path / "s")
    np.testing.assert_allclose(table["crossing_frame"], truth["crossing_frames"], atol=1.0)
    assert result.summary["droplets.summary"]["diameter_mean"] == pytest.approx(truth["diameter_um"], rel=truth["diameter_tolerance"])


def test_wscnet_replayed_online_matches_the_batch_run(wscnet_run, tmp_path):
    from vexel.engine import run_stream
    from vexel.sources import ReplaySource

    out, truth = wscnet_run
    images = {}
    with zipfile.ZipFile(DATA / "WSCNet2-master.zip") as zf:
        import imageio.v3 as iio

        for name in zf.namelist():
            m = re.search(r"examples/example-folder/(\d+)\.jpg$", name)
            if m:
                images[int(m.group(1))] = iio.imread(io.BytesIO(zf.read(name))).mean(-1)
    volume = SpatialVolume.from_array(np.stack([images[k] for k in sorted(images)]).astype(np.float32), "TYX")
    recipe = Recipe.load("droplet_encapsulation")
    run_stream(recipe, ReplaySource(volume, recipe.resolve().input.sources(), speed=None), tmp_path / "s", report=False)
    stream, batch = _crossings(tmp_path / "s"), _crossings(out)
    assert len(stream) == len(batch) == len(truth["droplets"])
    np.testing.assert_allclose(stream["crossing_frame"], batch["crossing_frame"], atol=0.1)
    agree = (stream["n_inclusions"].to_numpy() == batch["n_inclusions"].to_numpy()).mean()
    assert agree >= 0.95  # the online background differs slightly from the batch one
