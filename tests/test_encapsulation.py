"""Encapsulation: counting cells inside droplets and Poisson statistics."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from vexel.cli import main
from vexel.engine import Recipe, run_recipe
from vexel.ops import Context, get_op
from vexel.packs.droplet.validate import match_droplets, validate_run
from vexel.readers import open_volume
from vexel.synthetic.droplets import DropletSynthConfig, generate


def _droplet_frame(cells: list[tuple[float, float]], *, polarity: str = "dark", contrast: float = 50.0, noise: float = 3.0):
    """One 120x120 frame with a single droplet (radius 30) holding cells at the given offsets."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :120]
    d = np.hypot(yy - 60, xx - 60)
    img = np.full((120, 120), 180.0)
    img[(d > 27) & (d <= 30)] -= 90
    sign = -1 if polarity == "dark" else 1
    for dy, dx in cells:
        img[np.hypot(yy - 60 - dy, xx - 60 - dx) <= 2.5] += sign * contrast
    img += rng.normal(0, noise, img.shape)
    labels = (d <= 30).astype(np.int32)
    return img.astype(np.float32), labels


def _count(img, labels, **params) -> pd.DataFrame:
    op = get_op("droplet.inclusions")
    return op.fn(op.params(**params), Context(frame_shape=img.shape, n_frames=1), image=img, labels=labels)


@pytest.mark.parametrize("cells", [[], [(0, 0)], [(-8, -6), (6, 9)], [(0, -3.1), (0, 3.1)], [(-10, 0), (0, 10), (10, 0)]])
def test_inclusions_counts_dark_cells_including_touching_ones(cells):
    img, labels = _droplet_frame(cells)
    result = _count(img, labels)
    assert int(result.outputs["table"]["n_inclusions"].iloc[0]) == len(cells)
    assert result.outputs["labels"].max() == len(cells)
    assert result.qc["contrast_threshold"] > 0


def test_inclusions_bright_polarity_and_wrong_polarity():
    img, labels = _droplet_frame([(5, 5), (-7, 3)], polarity="bright")
    assert int(_count(img, labels, polarity="bright").outputs["table"]["n_inclusions"].iloc[0]) == 2
    assert int(_count(img, labels, polarity="dark").outputs["table"]["n_inclusions"].iloc[0]) == 0


def test_inclusions_ignore_the_dark_droplet_outline():
    img, labels = _droplet_frame([], noise=0.5)
    assert int(_count(img, labels, edge_margin_px=0).outputs["table"]["n_inclusions"].iloc[0]) > 0  # outline leaks in
    assert int(_count(img, labels).outputs["table"]["n_inclusions"].iloc[0]) == 0


def test_inclusions_fixed_threshold_and_no_droplets():
    img, labels = _droplet_frame([(0, 0)], contrast=15)
    assert int(_count(img, labels, min_contrast=30).outputs["table"]["n_inclusions"].iloc[0]) == 0
    assert int(_count(img, labels, min_contrast=5).outputs["table"]["n_inclusions"].iloc[0]) == 1
    empty = _count(img, np.zeros_like(labels))
    assert empty.outputs["table"].empty and empty.outputs["labels"].max() == 0


def test_inclusions_scale_with_16bit_data():
    img, labels = _droplet_frame([(0, 0), (9, -9)], contrast=50, noise=3)
    img16 = (img * 200).astype(np.float32)  # same image on a 16-bit scale
    assert int(_count(img16, labels).outputs["table"]["n_inclusions"].iloc[0]) == 2


def _poisson(counts, **params):
    op = get_op("droplet.poisson")
    table = pd.DataFrame({"n_inclusions": counts, "inclusion_agreement": 1.0})
    return op.fn(op.params(**params), Context(frame_shape=(10, 10), n_frames=1), table=table)


def test_poisson_statistics_match_hand_calculation():
    counts = [0] * 37 + [1] * 37 + [2] * 18 + [3] * 6 + [4] * 2
    s = _poisson(counts).outputs["summary"]
    lam = (37 + 36 + 18 + 8) / 100
    assert s["lambda"] == pytest.approx(lam)
    assert s["frac_empty"] == 0.37 and s["frac_single"] == 0.37 and s["frac_multi"] == pytest.approx(0.26)
    assert s["poisson_empty"] == pytest.approx(math.exp(-lam))
    assert s["poisson_single"] == pytest.approx(lam * math.exp(-lam))
    assert s["single_among_occupied"] == pytest.approx(0.37 / 0.63)
    assert s["dispersion_index"] == pytest.approx(np.var(counts, ddof=1) / lam)
    assert s["lambda_ci_low"] < lam < s["lambda_ci_high"]
    assert s["chi2_p"] > 0.5  # these counts are close to Poisson(0.99)


def test_poisson_detects_overdispersion():
    counts = [0] * 70 + [3] * 20 + [5] * 10  # clumped
    s = _poisson(counts).outputs["summary"]
    assert s["dispersion_index"] > 2 and s["chi2_p"] < 1e-6


def test_poisson_edge_cases():
    out = _poisson([])
    assert out.outputs["summary"]["lambda"] is None and out.outputs["distribution"].empty
    s = _poisson([0, 0, 0]).outputs["summary"]
    assert s["lambda"] == 0 and s["frac_empty"] == 1 and s["chi2_p"] is None
    s = _poisson([-1, 1, 1], min_agreement=0.0).outputs["summary"]
    assert s["n_droplets"] == 2 and s["n_excluded"] == 1


def test_synthetic_cells_follow_poisson_and_leave_cell_free_videos_unchanged():
    plain, _, _ = generate(DropletSynthConfig(n_frames=30))
    with_zero, droplets, _ = generate(DropletSynthConfig(n_frames=30, cell_lambda=0.0))
    assert np.array_equal(plain, with_zero) and "n_cells" not in droplets
    _, droplets, _ = generate(DropletSynthConfig(n_frames=2000, cell_lambda=1.0, seed=3))
    n = len(droplets)
    assert abs(droplets["n_cells"].mean() - 1.0) < 4 * math.sqrt(1.0 / n)  # within 4 standard errors


@pytest.fixture(scope="module")
def cells_run(cells_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("enc") / "run"
    return run_recipe(Recipe.load("droplet_encapsulation"), open_volume(cells_dataset / "droplets.ome.tif"), out)


def test_encapsulation_pipeline_matches_truth(cells_dataset, cells_run):
    checks = validate_run(cells_run.out_dir, cells_dataset / "truth")
    names = [c.name for c in checks]
    assert names[-5:] == ["inclusion_mismatch", "lambda", "frac_empty", "frac_single", "frac_multi"]
    assert all(c.passed for c in checks), checks
    droplets = cells_run.tables["droplets.droplets"]
    assert droplets["n_inclusions"].dtype == np.int64 and (droplets["inclusion_agreement"] > 0.8).all()
    dist = cells_run.tables["encapsulation.distribution"]
    assert dist["observed"].sum() == len(droplets) and dist["poisson_fraction"].sum() == pytest.approx(1.0, abs=0.01)
    assert len(list((cells_run.out_dir / "review").glob("*.png"))) == 4


def test_basic_recipe_is_unchanged_by_the_encapsulation_feature(droplet_dataset, tmp_path):
    """droplet_basic must not grow encapsulation columns, and its result must not depend on them."""
    r = run_recipe(Recipe.load("droplet_basic"), open_volume(droplet_dataset / "droplets.ome.tif"), tmp_path / "r", hash_input=False)
    assert "n_inclusions" not in r.tables["droplets.droplets"].columns
    assert set(r.summary) == {"droplets.summary", "flags.summary"}


def test_match_droplets_pairs_closest_crossings_once():
    pairs = match_droplets(np.array([10.1, 20.0, 50.0]), {1: 10.0, 2: 10.4, 3: 20.3})
    assert pairs == {0: 1, 1: 3}


def test_cli_synth_cells_and_validate(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["synth", str(data), "--frames", "120", "--cells", "1.5", "--seed", "8"]) == 0
    assert "n_cells" in pd.read_csv(data / "truth" / "droplets.csv").columns
    run = str(tmp_path / "run")
    assert main(["run", str(data / "droplets.ome.tif"), "-r", "droplet_encapsulation", "-o", run, "--no-input-hash"]) == 0
    capsys.readouterr()
    assert main(["validate", run, "--truth", str(data / "truth")]) == 0
    out = capsys.readouterr().out
    assert "inclusion_mismatch" in out and "validation PASSED" in out
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["encapsulation.summary"]["n_inclusions_total"] > 0
