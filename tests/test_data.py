"""Data foundation: SpatialVolume, readers, OME-Zarr cache."""

from __future__ import annotations

import json

import dask.array as da
import imageio.v3 as iio
import numpy as np
import pytest
import tifffile

from vexel.core.cache import read_ome_zarr, write_ome_zarr
from vexel.core.device import plan_block_frames
from vexel.core.volume import SpatialVolume
from vexel.readers import open_volume


def test_from_array_canonicalises_axes():
    arr = np.arange(2 * 3 * 4 * 5).reshape(4, 5, 3, 2)  # Y X C T
    vol = SpatialVolume.from_array(arr, "YXCT")
    assert vol.sizes == {"T": 2, "C": 3, "Z": 1, "Y": 4, "X": 5}
    np.testing.assert_array_equal(vol.frames(channel=1)[1].compute(), arr[:, :, 1, 1])
    assert vol.space_unit == "px" and vol.spacing["x"] == 1.0


def test_from_array_rejects_bad_axes():
    with pytest.raises(ValueError):
        SpatialVolume.from_array(np.zeros((2, 2)), "YXZ")
    with pytest.raises(ValueError):
        SpatialVolume.from_array(np.zeros((2, 2)), "YY")
    with pytest.raises(IndexError):
        SpatialVolume.from_array(np.zeros((2, 2)), "YX").frames(channel=1)


def test_ome_tiff_is_lazy_and_calibrated(droplet_dataset):
    vol = open_volume(droplet_dataset / "droplets.ome.tif")
    assert isinstance(vol.data, da.Array)
    assert vol.sizes["T"] == 200 and vol.sizes["C"] == 1
    assert vol.space_unit == "um" and vol.spacing["x"] == pytest.approx(2.0)
    assert vol.time_unit == "s" and vol.spacing["t"] == pytest.approx(0.001)
    expected = tifffile.imread(droplet_dataset / "droplets.ome.tif", key=5)
    np.testing.assert_array_equal(vol.frames()[5].compute(), expected)


def test_plain_tiff_is_uncalibrated_and_cli_calibration_wins(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((3, 8, 9), np.uint16), photometric="minisblack")
    vol = open_volume(path)
    assert vol.sizes["T"] == 3 and vol.space_unit == "px" and vol.time_unit == "frame"
    vol = open_volume(path, pixel_size=0.5, frame_interval=0.01)
    assert (vol.spacing["x"], vol.space_unit, vol.spacing["t"], vol.time_unit) == (0.5, "um", 0.01, "s")


def test_image_sequence_natural_order_and_sidecar(tmp_path):
    folder = tmp_path / "seq"
    folder.mkdir()
    for i in (10, 2, 1):
        iio.imwrite(folder / f"img_{i}.png", np.full((6, 7), i, np.uint8))
    (folder / "vexel.json").write_text(json.dumps({"pixel_size": 1.5, "frame_interval": 0.002}))
    vol = open_volume(folder)
    assert vol.sizes["T"] == 3
    assert [int(vol.frames()[t, 0, 0].compute()) for t in range(3)] == [1, 2, 10]
    assert vol.spacing["y"] == 1.5 and vol.space_unit == "um" and vol.spacing["t"] == 0.002


def test_npy_reader(tmp_path):
    arr = np.random.default_rng(0).integers(0, 255, (4, 5, 6), dtype=np.uint8)
    np.save(tmp_path / "a.npy", arr)
    np.testing.assert_array_equal(open_volume(tmp_path / "a.npy").frames().compute(), arr)


def test_video_reader(tmp_path):
    frames = np.zeros((12, 64, 96, 3), np.uint8)
    for t in range(12):
        frames[t, :, : 8 * (t + 1)] = 255
    path = tmp_path / "v.mp4"
    iio.imwrite(path, frames, plugin="pyav", codec="libx264", fps=25)
    vol = open_volume(path)
    assert vol.sizes == {"T": 12, "C": 3, "Z": 1, "Y": 64, "X": 96}
    assert vol.spacing["t"] == pytest.approx(1 / 25) and vol.time_unit == "s"
    got = vol.frames().compute()
    bright_columns = (got > 127).mean(axis=1).sum(axis=1)  # lossy codec: compare bar widths
    np.testing.assert_allclose(bright_columns, 8 * np.arange(1, 13), atol=1.5)


def test_unknown_format_and_missing_file(tmp_path):
    (tmp_path / "x.xyz").write_text("no")
    with pytest.raises(ValueError):
        open_volume(tmp_path / "x.xyz")
    with pytest.raises(FileNotFoundError):
        open_volume(tmp_path / "missing.tif")


def test_ome_zarr_roundtrip_with_pyramid(tmp_path):
    arr = np.random.default_rng(1).integers(0, 255, (3, 128, 256), dtype=np.uint8)
    vol = SpatialVolume.from_array(arr, "TYX", spacing={"t": 0.5, "y": 2.0, "x": 2.0}, space_unit="um", time_unit="s")
    path = write_ome_zarr(vol, tmp_path / "v.zarr", levels=3)
    back = open_volume(path)
    np.testing.assert_array_equal(back.frames().compute(), arr)
    assert back.spacing == vol.spacing and back.space_unit == "um" and back.time_unit == "s"
    assert back.metadata["n_levels"] == 2  # third level would be < 64 px
    level1 = read_ome_zarr(path, level=1)
    assert level1.sizes["Y"] == 64 and level1.sizes["X"] == 128 and level1.spacing["x"] == 4.0
    with pytest.raises(Exception):
        write_ome_zarr(vol, path)  # refuses to overwrite silently


def test_block_planning_respects_budget_and_chunk_multiple():
    assert plan_block_frames(1000, 4, budget=40_000) == 10
    assert plan_block_frames(1000, 4, multiple=4, budget=40_000) == 8
    assert plan_block_frames(10**9, 4, multiple=3, budget=1) == 3
