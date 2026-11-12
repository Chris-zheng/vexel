"""CLI: the same engine, driven from the command line."""

from __future__ import annotations

import json

from vexel.cli import main


def test_synth_run_validate_roundtrip(tmp_path, capsys):
    data = tmp_path / "data"
    assert main(["synth", str(data), "--frames", "150", "--seed", "4"]) == 0
    video = str(data / "droplets.ome.tif")
    run = str(tmp_path / "run")
    assert main(["run", video, "-r", "droplet_basic", "-o", run]) == 0
    out = capsys.readouterr().out
    assert "n_droplets" in out and "result digest" in out
    assert main(["validate", run, "--truth", str(data / "truth")]) == 0
    assert "validation PASSED" in capsys.readouterr().out
    # an impossible tolerance must fail with a non-zero exit code
    assert main(["validate", run, "--truth", str(data / "truth"), "--diameter-tol", "0"]) == 1


def test_run_with_override_and_errors(tmp_path, capsys):
    data = tmp_path / "data"
    main(["synth", str(data), "--frames", "40"])
    video = str(data / "droplets.ome.tif")
    assert main(["run", video, "-r", "droplet_basic", "-o", str(tmp_path / "a"), "--set", "seg.min_area=200", "--no-input-hash"]) == 0
    recipe = json.loads((tmp_path / "a" / "recipe.json").read_text())
    assert recipe["steps"][1]["params"]["min_area"] == 200
    capsys.readouterr()
    assert main(["run", video, "-r", "droplet_basic", "-o", str(tmp_path / "a")]) == 2  # exists
    assert main(["run", video, "-r", "no_such_recipe", "-o", str(tmp_path / "b")]) == 2
    assert main(["run", video, "-r", "droplet_basic", "-o", str(tmp_path / "c"), "--set", "seg.nope=1"]) == 2
    assert main(["run", str(tmp_path / "missing.tif"), "-r", "droplet_basic"]) == 2
    assert capsys.readouterr().err.count("error:") == 4


def test_info_ops_recipes(tmp_path, capsys):
    data = tmp_path / "data"
    main(["synth", str(data), "--frames", "10"])
    capsys.readouterr()
    assert main(["info", str(data / "droplets.ome.tif"), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["shape"] == [10, 1, 1, 256, 768] and info["space_unit"] == "um"
    assert main(["ops", "--json"]) == 0
    ops = {o["name"]: o for o in json.loads(capsys.readouterr().out)}
    assert ops["track.link"]["scope"] == "window" and "max_distance_px" in ops["track.link"]["params"]
    assert main(["recipes"]) == 0
    assert "droplet_basic" in capsys.readouterr().out
