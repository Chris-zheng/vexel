"""Phase 6: design tables, unattended batch runs, replicate-level statistics, reproduce script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from vexel.batch import Design, DesignError, run_batch, scan_folder
from vexel.batch.design import parse_settings
from vexel.batch.stats import analyse, holm
from vexel.engine import Recipe
from vexel.synthetic.droplets import DropletSynthConfig, write_dataset

# -- design table ---------------------------------------------------------------------------------


def test_design_table_reads_excel_style_csv(tmp_path):
    (tmp_path / "a.avi").write_bytes(b"x")
    text = 'file,group,replicate,pixel_size_um,settings,include,培养基\na.avi,对照,1,2.0,"seg.min_score=0.5; droplets.flow_axis=y",,LB\nb.avi,处理,1,,,否,LB\n'
    (tmp_path / "design.csv").write_bytes(text.encode("gb18030"))  # Excel on a Chinese system
    d = Design.load(tmp_path / "design.csv")
    assert len(d.rows) == 1  # the excluded row is left out
    r = d.rows[0]
    assert r.file == tmp_path / "a.avi" and r.group == "对照" and r.replicate == "1" and r.pixel_size_um == 2.0
    assert r.settings == {"seg.min_score": 0.5, "droplets.flow_axis": "y"} and r.meta == {"培养基": "LB"}
    assert d.sha256 and r.unit == "对照/1"


def test_design_table_errors(tmp_path):
    with pytest.raises(DesignError, match="'file' column"):
        Design.from_frame(pd.DataFrame({"path": ["a"]}))
    with pytest.raises(DesignError, match="already listed"):
        Design.from_frame(pd.DataFrame({"file": ["a.avi", "a.avi"]}), base=tmp_path)
    with pytest.raises(DesignError, match="row 1: pixel_size_um"):
        Design.from_frame(pd.DataFrame({"file": ["a.avi"], "pixel_size_um": ["two"]}))
    with pytest.raises(DesignError, match="key=value"):
        parse_settings("seg.min_score")


def test_scan_finds_videos_stacks_and_sequences(tmp_path):
    import imageio.v3 as iio

    (tmp_path / "A" / "chip1").mkdir(parents=True)
    (tmp_path / "A" / "chip1" / "run.avi").write_bytes(b"x")
    (tmp_path / "B").mkdir()
    (tmp_path / "B" / "stack.tif").write_bytes(b"x")
    seq = tmp_path / "B" / "seq"
    seq.mkdir()
    for i in range(6):
        iio.imwrite(seq / f"{i:03d}.png", np.zeros((8, 8), np.uint8))
    (tmp_path / "B" / "truth").mkdir()
    (tmp_path / "B" / "truth" / "x.tif").write_bytes(b"x")  # truth folders are skipped
    table = scan_folder(tmp_path)
    got = {(f, g, r) for f, g, r in table[["file", "group", "replicate"]].itertuples(index=False)}
    assert got == {("A/chip1/run.avi", "A", "chip1"), ("B/seq", "B", ""), ("B/stack.tif", "B", "")}


# -- statistics -----------------------------------------------------------------------------------


def _results(values: dict[str, list[float]], replicate: bool = True, key: str = "droplets.diameter_mean") -> pd.DataFrame:
    rows = []
    for g, vs in values.items():
        for i, v in enumerate(vs):
            rows.append({"id": f"{g}{i}", "group": g, "replicate": f"r{i}" if replicate else "", "status": "completed",
                         key: v, "droplets.n_droplets": 100, "droplets.length_unit": "um"})  # fmt: skip
    return pd.DataFrame(rows)


def test_two_groups_welch_matches_scipy():
    a, b = [50.0, 51.2, 49.5, 50.4], [55.1, 54.0, 56.3]
    an = analyse(_results({"A": a, "B": b}))
    c = an.comparisons
    row = c[(c.metric == "droplets.diameter_mean")].iloc[0]
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert row.p == pytest.approx(ref.pvalue) and row["diff"] == pytest.approx(np.mean(a) - np.mean(b))
    assert row.diff_ci_low < row["diff"] < row.diff_ci_high < 0
    assert row.p_nonparametric == pytest.approx(sps.mannwhitneyu(a, b).pvalue)
    assert row.effect < -2  # Hedges g: a large difference
    g = an.groups[(an.groups.metric == "droplets.diameter_mean") & (an.groups.group == "A")].iloc[0]
    assert g["n"] == 4 and g.ci_low < g["mean"] < g.ci_high
    assert "droplets.n_droplets" not in set(c.metric)  # counts are described, not tested
    assert an.unit_column == "replicate"


def test_files_of_one_replicate_are_averaged_first():
    df = _results({"A": [50, 52, 60], "B": [70, 71]})
    df.loc[df.id == "A1", "replicate"] = "r0"  # A0 and A1 are the same chip
    an = analyse(df)
    units = an.units[an.units.group == "A"]
    assert len(units) == 2 and sorted(units["droplets.diameter_mean"]) == [51, 60]
    assert int(units.loc[units.replicate == "r0", "n_files"].iloc[0]) == 2


def test_three_groups_anova_and_holm():
    rng = np.random.default_rng(0)
    vals = {"A": list(rng.normal(50, 1, 4)), "B": list(rng.normal(50, 1, 4)), "C": list(rng.normal(58, 1, 4))}
    an = analyse(_results(vals))
    c = an.comparisons[an.comparisons.metric == "droplets.diameter_mean"]
    overall = c[c.comparison == "all groups"].iloc[0]
    assert overall.p < 0.01 and overall.p_nonparametric < 0.05 and 0 < overall.effect <= 1
    pairs = c[c.comparison != "all groups"]
    assert len(pairs) == 3
    assert list(pairs.p_holm) == pytest.approx(holm(list(pairs.p)))
    assert pairs[pairs.comparison == "A vs B"].p_holm.iloc[0] > 0.05
    assert pairs[pairs.comparison == "A vs C"].p_holm.iloc[0] < 0.01


def test_notes_for_weak_designs():
    an = analyse(_results({"A": [50, 51], "B": [60]}, replicate=False))
    zh = " ".join(z for z, _ in an.notes)
    assert "replicate" in zh and "只有 1 个重复" in zh and "只有 2 个重复" in zh
    assert an.comparisons.empty  # B cannot be tested
    mixed = _results({"A": [50, 51, 52], "B": [60, 61, 62]})
    mixed.loc[0, "droplets.length_unit"] = "px"
    an = analyse(mixed)
    assert any("单位不一致" in z for z, _ in an.notes) and an.comparisons.empty


# -- running a batch ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batch_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("batch")
    for g, d in (("small", 50.0), ("large", 66.0)):
        for k in range(2):
            write_dataset(root / g / f"chip{k + 1}", DropletSynthConfig(n_frames=120, height=128, width=448, diameter_mean_um=d, seed=k + int(d)))
    (root / "small" / "broken.avi").write_bytes(b"not a video")
    table = scan_folder(root)
    table.loc[len(table)] = {"file": "large/gone.avi", "group": "large", "replicate": "", "pixel_size_um": "", "frame_interval_s": "", "settings": "", "note": ""}
    table.to_csv(root / "design.csv", index=False)
    return root


def test_batch_run_resume_and_reproduce(batch_data, tmp_path):
    design = Design.load(batch_data / "design.csv")
    out = tmp_path / "out"
    seen = []
    result = run_batch(design, Recipe.load("droplet_basic"), out, hash_input=False, on_file=lambda i, n, o: seen.append(o.status))
    status = {Path(o.file).name if o.file.endswith(".avi") else o.id: o.status for o in result.outcomes}
    assert status["broken.avi"] == "failed" and status["gone.avi"] == "missing" and result.n_ok == 4 and len(seen) == 6
    assert (out / "runs").is_dir() and sorted(p.name for p in (out / "runs").iterdir() if p.is_dir()) == [
        "001_chip1_droplets", "002_chip2_droplets", "004_chip1_droplets", "005_chip2_droplets"]  # fmt: skip
    for name in ("results.csv", "replicates.csv", "groups.csv", "comparisons.csv", "report.html", "methods_zh.txt", "methods_en.txt", "reproduce.py", "recipe.json", "design.csv"):
        assert (out / name).exists(), name
    record = json.loads((out / "batch.json").read_text())
    assert record["status"] == "completed" and record["n_completed"] == 4 and record["n_failed"] == 2
    comp = pd.read_csv(out / "comparisons.csv")
    size = comp[comp.metric == "droplets.diameter_mean"].iloc[0]
    assert size["diff"] == pytest.approx(16, abs=2)  # large - small
    html = (out / "report.html").read_text()
    assert "平均直径" in html and "Methods (English)" in html and "找不到文件" in html
    assert "Welch" in (out / "methods_en.txt").read_text()

    # resume: nothing is run again
    again = run_batch(Design.load(batch_data / "design.csv"), Recipe.load("droplet_basic"), out, hash_input=False)
    assert [o.status for o in again.outcomes if o.ok] == ["resumed"] * 4
    # a changed setting runs again
    changed = run_batch(Design.load(batch_data / "design.csv"), Recipe.load("droplet_basic"), out, hash_input=False,
                        settings={"droplets.count_line": 0.4})  # fmt: skip
    assert [o.status for o in changed.outcomes if o.ok] == ["completed"] * 4

    # the exported script reproduces every result on its own
    proc = subprocess.run([sys.executable, str(out / "reproduce.py"), "--out", str(tmp_path / "again")], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "4 / 4 files reproduced with identical results" in proc.stdout
    assert pd.read_csv(tmp_path / "again" / "groups.csv").equals(pd.read_csv(out / "groups.csv"))


def test_batch_refuses_a_folder_that_is_not_a_batch(tmp_path, batch_data):
    (tmp_path / "x.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        run_batch(Design.load(batch_data / "design.csv"), Recipe.load("droplet_basic"), tmp_path)


def test_cli_batch(batch_data, tmp_path):
    from vexel.cli import main

    design = tmp_path / "d.csv"
    assert main(["batch", "init", str(batch_data / "small"), "-o", str(design)]) == 0
    table = pd.read_csv(design, encoding="utf-8-sig")
    assert len(table) == 3 and all(Path(f).is_absolute() for f in table["file"])
    assert main(["batch", "run", str(design), "-r", "droplet_basic", "-o", str(tmp_path / "b"), "--no-input-hash"]) == 4  # broken.avi
    assert (tmp_path / "b" / "report.html").exists()
