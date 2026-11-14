"""CLI: the same engine, driven from the command line."""

from __future__ import annotations

import json

import pytest

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
    assert main(["run", video, "-r", "droplet_basic", "-o", str(tmp_path / "a"), "--set", "seg.min_score=0.4", "--no-input-hash"]) == 0
    recipe = json.loads((tmp_path / "a" / "recipe.json").read_text())
    assert recipe["steps"][1]["params"]["min_score"] == 0.4
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


def test_replay_command(tmp_path, capsys):
    data = tmp_path / "data"
    main(["synth", str(data), "--frames", "120"])
    out = tmp_path / "r"
    assert main(["replay", str(data / "droplets.ome.tif"), "-r", "droplet_basic", "-o", str(out), "--speed", "max", "--update-every", "40"]) == 0
    text = capsys.readouterr().out
    assert "fps" in text and "dropped 0" in text and "droplets.summary" in text
    assert (out / "stream" / "rolling.csv").exists() and (out / "report.html").exists()
    with pytest.raises(SystemExit):  # rejected by the argument parser
        main(["replay", str(data / "droplets.ome.tif"), "-r", "droplet_basic", "--speed", "-1"])


def test_bugreport_has_no_images_paths_or_keys(tmp_path, monkeypatch):
    import zipfile

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-value")
    run = tmp_path / "run"
    (run / "tables").mkdir(parents=True)
    (run / "review").mkdir()
    secret_path = str(tmp_path / "private" / "patient_42.avi")
    (run / "provenance.json").write_text(json.dumps({"input": {"source": secret_path}, "api_key": "sk-x", "status": "completed"}))
    (run / "summary.json").write_text(json.dumps({"droplets": {"n_droplets": 3}}))
    (run / "recipe.json").write_text(json.dumps({"name": "droplet_basic"}))
    (run / "tables" / "flags.table.csv").write_text("label,reason\n1,edge\n")
    (run / "tables" / "measure.table.csv").write_text("label,diameter\n1,50\n")
    (run / "review" / "frame_000000.png").write_bytes(b"\x89PNG")
    (run / "report.html").write_text("<img>")
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "config.json").write_text(json.dumps({"models": {"order": ["deepseek", "rules"]}}))

    out = tmp_path / "bug.zip"
    assert main(["bugreport", str(run), "-o", str(out), "--note", "count looks low"]) == 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        text = "".join(zf.read(n).decode() for n in names)
        env = json.loads(zf.read("environment.json"))
    assert "run1_run/tables/flags.table.csv" in names and "note.txt" in names
    assert not any(n.endswith((".png", ".html")) for n in names) and "run1_run/tables/measure.table.csv" not in names
    assert "sk-secret-value" not in text and "sk-x" not in text and str(tmp_path) not in text
    assert ".../patient_42.avi" in text
    assert env["planning_models"]["deepseek"]["api_key_set"] is True and env["vexel"]

    main(["bugreport", str(run), "-o", str(out), "--tables"])
    with zipfile.ZipFile(out) as zf:
        assert "run1_run/tables/measure.table.csv" in zf.namelist()
