"""Pixel classifier trained from brush strokes: training file, reproducibility, recipes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk, line

from vexel.engine import Recipe, run_recipe
from vexel.ops import get_op
from vexel.ops.classify import file_digest, save_training, train, use_in_recipe
from vexel.ops.registry import Context
from vexel.packs.droplet.validate import validate_run
from vexel.readers import open_volume


def strokes_from_truth(dataset, frames=(20, 120)):
    """A few strokes a person might paint: short lines inside some droplets, lines along the
    background above and below them, and short ticks between neighbouring droplets."""
    volume = open_volume(dataset / "droplets.ome.tif")
    pos = pd.read_csv(dataset / "truth" / "positions.csv")
    images, strokes = {}, {}
    for t in frames:
        img = np.asarray(volume.frames(0, 0)[t]).astype(np.float32)
        s = np.zeros(img.shape, np.uint8)
        here = pos[(pos["frame"] == t) & pos["fully_visible"]]
        radius = pd.read_csv(dataset / "truth" / "droplets.csv").set_index("droplet_id")["diameter_um"] / 2.0 / 2.0  # px

        def radius_px(row) -> float:
            return float(radius[row["droplet_id"]])

        for _, r in here.head(4).iterrows():  # a cross through the whole droplet, rim included
            reach = 0.9 * float(radius[r["droplet_id"]])
            y, x = float(r["y_px"]), float(r["x_px"])
            for (y0, x0), (y1, x1) in (((y, x - reach), (y, x + reach)), ((y - reach, x), (y + reach, x))):
                rr, cc = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
                ok = (rr >= 0) & (rr < s.shape[0]) & (cc >= 0) & (cc < s.shape[1])
                s[rr[ok], cc[ok]] = 1
        for y in (8, 29, 30, 40, 220, 229, 230, 248):  # background: channel, its edges, the dark bands outside
            rr, cc = line(y, 5, y, img.shape[1] - 6)
            s[rr, cc] = 2
        for _, r in here.head(4).iterrows():  # a ring of background just outside each painted droplet
            from skimage.draw import circle_perimeter

            rr, cc = circle_perimeter(int(round(r["y_px"])), int(round(r["x_px"])), int(0.5 * radius_px(r) * 2 + 3), shape=s.shape)
            s[rr, cc] = 2
        for _, r in here.head(3).iterrows():
            x = int(r["x_px"] + 50)
            if x < img.shape[1] - 1:
                rr, cc = line(int(r["y_px"]) - 6, x, int(r["y_px"]) + 6, x)
                s[rr, cc] = 2
        images[t], strokes[t] = img, s
    return images, strokes


@pytest.fixture(scope="module")
def training(droplet_dataset, tmp_path_factory):
    images, strokes = strokes_from_truth(droplet_dataset)
    path = tmp_path_factory.mktemp("brush") / "brush.npz"
    sha = save_training(path, images, strokes, ["液滴", "背景"])
    return path, sha, images


def test_training_file_keeps_patches_and_is_deterministic(training):
    path, sha, images = training
    data = np.load(path)
    assert set(data["frames"].tolist()) == {20, 120} and list(data["classes"]) == ["液滴", "背景"]
    small = np.zeros(images[20].shape, np.uint8)
    small[100:110, 100:110], small[150:160, 300:310] = 1, 2
    save_training(path.parent / "small.npz", {20: images[20]}, {20: small}, ["液滴", "背景"])
    assert np.load(path.parent / "small.npz")["image_20"].size < images[20].size  # patches, not whole frames
    a, b = train(path), train(path)
    np.testing.assert_array_equal(a.probabilities(images[20]), b.probabilities(images[20]))
    assert file_digest(path) == sha


def test_operator_finds_droplets_and_refuses_a_changed_file(training, tmp_path):
    path, sha, images = training
    op = get_op("segment.pixel_classifier")
    ctx = Context(frame_shape=images[20].shape, n_frames=1)
    result = op.fn(op.params(training=str(path), training_sha256=sha), ctx, image=images[120])
    assert result.outputs["labels"].max() >= 3 and result.outputs["probability"].shape == images[120].shape
    copy = tmp_path / "b.npz"
    copy.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="has changed"):
        op.fn(op.params(training=str(copy), training_sha256=sha[:-1] + "0"), ctx, image=images[120])
    with pytest.raises(FileNotFoundError):
        op.fn(op.params(training=str(tmp_path / "none.npz"), training_sha256="0" * 64), ctx, image=images[120])


def test_one_class_is_not_enough(training, tmp_path):
    _, _, images = training
    s = np.zeros(images[20].shape, np.uint8)
    s[100:104, 100:104] = 1
    path = tmp_path / "one.npz"
    save_training(path, {20: images[20]}, {20: s}, ["液滴", "背景"])
    with pytest.raises(ValueError, match="two classes"):
        train(path)


def test_recipe_with_the_classifier_matches_truth(droplet_dataset, training, tmp_path):
    path, sha, _ = training
    recipe = use_in_recipe(Recipe.load("droplet_basic"), path, sha)
    seg = next(s for s in recipe.steps if s.id == "seg")
    assert seg.op == "segment.pixel_classifier" and seg.inputs == {"image": "source"} and seg.params["training_sha256"] == sha
    result = run_recipe(recipe, open_volume(droplet_dataset / "droplets.ome.tif"), tmp_path / "r", hash_input=False, report=False)
    checks = {c.name: c for c in validate_run(result.out_dir, droplet_dataset / "truth")}
    assert checks["n_droplets"].passed and checks["frequency"].passed
    assert abs(checks["diameter_mean"].error) < 0.06
    again = run_recipe(recipe, open_volume(droplet_dataset / "droplets.ome.tif"), tmp_path / "r2", hash_input=False, report=False)
    assert again.result_digest == result.result_digest


def test_classifier_recipes_also_stream(droplet_dataset, training, tmp_path):
    from vexel.engine import run_stream
    from vexel.sources import ReplaySource

    path, sha, _ = training
    recipe = use_in_recipe(Recipe.load("droplet_basic"), path, sha)
    src = ReplaySource(open_volume(droplet_dataset / "droplets.ome.tif"), recipe.resolve().input.sources(), speed=None)
    result = run_stream(recipe, src, tmp_path / "s", report=False)
    assert result.summary["droplets.summary"]["n_droplets"] >= 15
