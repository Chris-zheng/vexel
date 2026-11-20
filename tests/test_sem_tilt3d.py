"""SEM tilt 3D (M6): height maps from tilted views of synthetic surfaces with known heights,
confidence, step heights, roughness, files, the record and the command line."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from vexel.packs.sem import tilt3d as T
from vexel.synthetic import sem as S

ROWS, COLS = 200, 240


def views(z, tilts, noise=0.02, seed=0, flat=None):
    tex = S.surface_texture(*z.shape, seed=seed + 2)
    return [S.tilt_view(z, tex, t, noise=noise, flat=flat, seed=seed + i) for i, t in enumerate(tilts)]


def err(hm, z, border=12):
    c = slice(border, -border)
    d = (hm.z - z)[c, c]
    m = hm.valid[c, c]
    d = d - np.median(d[m])
    return float(np.sqrt(np.mean(d[m] ** 2)))


def test_the_geometry_without_noise():
    """z = (ya·cos φb − yb·cos φa) / sin(φa − φb): exact up to the matching's own floor."""
    z = S.height_map("pyramid", ROWS, COLS, height=20, seed=1)
    for tilts in ((-5, 5), (0, 10), (3, -4)):
        hm = T.reconstruct(views(z, tilts, noise=0.0), list(tilts), smooth=0)
        slope = np.polyfit(z[12:-12, 12:-12].ravel(), hm.z[12:-12, 12:-12].ravel(), 1)[0]
        assert abs(slope - 1) < 0.02 and err(hm, z) < 0.05 * 20


@pytest.mark.parametrize("kind", ["pyramid", "spheres", "sine", "rough"])
def test_height_maps_within_5_percent(kind):
    """Height noise is about 1 px at a 10° tilt difference whatever the height: 5 % of a 30 px relief."""
    z = S.height_map(kind, ROWS, COLS, height=30, seed=1)
    hm = T.reconstruct(views(z, (-5, 5)), [-5, 5])
    e = err(hm, z)
    assert e / np.ptp(z) < 0.05 if np.ptp(z) >= 25 else e < 1.2  # height noise ~1-1.5 px: 5 % of a relief of 25 px or more
    assert hm.valid.mean() > 0.9 and hm.pairs[0]["match"] > 0.8


def test_more_views_and_a_vertical_axis():
    z = S.height_map("pyramid", ROWS, COLS, height=20, seed=1)
    two = T.reconstruct(views(z, (-5, 5), noise=0.05), [-5, 5])
    five = T.reconstruct(views(z, (-10, -5, 0, 5, 10), noise=0.05), [-10, -5, 0, 5, 10], reference=2)
    assert err(five, z) < err(two, z) and len(five.pairs) == 4
    turned = T.reconstruct([np.rot90(v) for v in views(z, (-5, 5))], [-5, 5], axis=90)
    assert err(turned, np.rot90(z)) / 20 < 0.05


def test_textureless_areas_are_marked_and_uncertainty_is_reported():
    z = S.height_map("pyramid", ROWS, COLS, height=20, seed=1)
    flat = np.zeros_like(z, bool)
    flat[30:90, 30:110] = True
    hm = T.reconstruct(views(z, (-5, 5), flat=flat), [-5, 5])
    assert hm.valid[flat].mean() < 0.4 and hm.valid[40:-40, 120:-20].mean() > 0.95
    assert np.median(hm.uncertainty[flat]) > 3 * np.median(hm.uncertainty[~flat])
    actual = err(hm, z)
    assert 0.4 < actual / np.median(hm.uncertainty[hm.valid]) < 2.5


def test_step_heights_ignore_a_tilted_sample():
    zs = S.height_map("steps", ROWS, COLS, height=30, seed=1)
    yy, xx = np.mgrid[:ROWS, :COLS]
    true = zs[:, -8].mean() - zs[:, 8].mean()
    for zz in (zs, zs + 0.04 * xx + 0.02 * yy):
        hm = T.reconstruct(views(zz, (-5, 5)), [-5, 5])
        st = T.step_height(hm, hm.z, [4, 15, int(COLS * 0.15), ROWS - 15], [int(COLS * 0.75), 15, COLS - 4, ROWS - 15])
        assert abs(st["step"] / true - 1) < 0.05 and abs(st["step"] - true) <= 2 * st["uncertainty"]
    with pytest.raises(ValueError):
        T.step_height(hm, hm.z, [0, 0, 2, 2], [10, 10, 40, 40])


def test_levelling_and_roughness():
    z = S.height_map("rough", ROWS, COLS, height=12, seed=2)
    yy, xx = np.mgrid[:ROWS, :COLS]
    hm = T.HeightMap(z=z + 0.05 * xx - 0.03 * yy + 7, confidence=np.ones_like(z), uncertainty=np.zeros_like(z))
    flat = T.level(hm, 1)
    r = T.roughness(flat)
    truth = T.roughness(z - z.mean())
    assert r["Sq"] == pytest.approx(truth["Sq"], rel=0.05) and r["Sa"] == pytest.approx(truth["Sa"], rel=0.05)
    assert r["Sq"] >= r["Sa"] and r["Sz"] == pytest.approx(r["Sp"] + r["Sv"])
    gauss = T.roughness(np.random.default_rng(0).normal(0, 1, (300, 300)))
    assert gauss["Ssk"] == pytest.approx(0, abs=0.05) and gauss["Sku"] == pytest.approx(3, abs=0.1)
    assert gauss["Sa"] == pytest.approx(math.sqrt(2 / math.pi), rel=0.03)


def test_profiles_files_and_previews(tmp_path):
    import tifffile

    z = S.height_map("spheres", ROWS, COLS, height=20, seed=3)
    hm = T.reconstruct(views(z, (-5, 5)), [-5, 5], pixel_size=2e-9)
    rng = np.percentile(hm.z[hm.valid], 99.5) - np.percentile(hm.z[hm.valid], 0.5)
    assert hm.unit == "m" and rng == pytest.approx(np.ptp(z) * 2e-9, rel=0.15)
    pr = T.profile(hm, hm.z, (10, ROWS / 2), (COLS - 10, ROWS / 2))
    assert pr["distance"][-1] == pytest.approx((COLS - 20) * 2e-9) and len(pr["height"]) == len(pr["confidence"])
    tif = T.save_tiff(hm, tmp_path / "h.tif")
    with tifffile.TiffFile(tif) as tf:
        stack = tf.asarray()
        desc = json.loads(tf.pages[0].description)
    assert stack.shape == (3, ROWS, COLS) and desc["vexel_height_map"]["unit"] == "m"
    stl = T.save_mesh(hm, tmp_path / "h.stl")
    n = int.from_bytes(stl.read_bytes()[80:84], "little")
    assert n > 1000 and stl.stat().st_size == 84 + 50 * n
    obj = T.save_mesh(hm, tmp_path / "h.obj").read_text(encoding="utf-8")
    assert obj.count("\nv ") > 1000 and obj.count("\nf ") > 1000
    assert T.colour(hm.z, hm.valid).shape == (ROWS, COLS, 3)
    pic = T.render_3d(hm.z, mask=hm.valid, size=300)
    assert pic.ndim == 3 and pic.shape[1] == 301 and (pic < 250).any()


def test_wrong_input_is_refused():
    z = S.height_map("pyramid", ROWS, COLS, height=20, seed=1)
    v = views(z, (0, 0.5))
    with pytest.raises(ValueError, match="1°"):
        T.reconstruct(v, [0, 0.5])
    with pytest.raises(ValueError):
        T.reconstruct(v[:1], [0])
    with pytest.raises(ValueError):
        T.reconstruct([v[0], v[1][:-1]], [0, 5])
    small = T.reconstruct(views(z, (0, 2)), [0, 2])
    assert any("倾转角差" in n for n in small.notes)


def test_record_and_command_line(tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C
    from vexel.packs.sem import measure_io

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    zs = S.height_map("steps", ROWS, COLS, height=30, seed=1)
    paths = []
    for i, (t, v) in enumerate(zip((-5, 5), views(zs, (-5, 5)))):
        p = tmp_path / f"v{i}.tif"
        tifffile.imwrite(p, (np.clip(v, 0, 1) * 65535).astype(np.uint16))
        C.save(C.manual(p, pixel_size=2e-9).confirm())
        paths.append(p)
    spec = T.Tilt3DSpec(reference_tilt=-5, views=[T.TiltView(file="v1.tif", sha256=C.file_sha256(paths[1]), tilt=5)],
                        steps=[{"lower": [4, 15, 36, ROWS - 15], "upper": [180, 15, COLS - 4, ROWS - 15]}])  # fmt: skip
    res = T.run_spec(spec, paths[0])
    assert res.steps[0]["step"] == pytest.approx(20 * 2e-9, rel=0.05) and res.hm.pixel_size == pytest.approx(2e-9)
    tifffile.imwrite(paths[1], np.zeros((ROWS, COLS), np.uint16))  # the view changed after it was chosen
    C.save(C.manual(paths[1], pixel_size=2e-9).confirm())
    assert any("改动" in w for w in T.load_views(spec, paths[0])[3])
    tifffile.imwrite(paths[1], (np.clip(views(zs, (-5, 5))[1], 0, 1) * 65535).astype(np.uint16))
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-m", "vexel.cli", "sem", "tilt3d", str(paths[0]), str(paths[1]), "--tilts=-5,5",
                        "--step", f"4,15,36,{ROWS - 15},180,15,{COLS - 4},{ROWS - 15}", "--profile", f"5,{ROWS // 2},{COLS - 5},{ROWS // 2}",
                        "-o", str(tmp_path / "h.tif"), "--mesh", str(tmp_path / "h.stl"), "--save", "--json"],
                       capture_output=True, text=True, env=env)  # fmt: skip
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["unit"] == "m" and out["steps"][0]["step"] == pytest.approx(40e-9, rel=0.05) and out["reliable_fraction"] > 0.9
    assert (tmp_path / "h.tif").is_file() and (tmp_path / "h.stl").is_file()
    ms = measure_io.load_record(C.file_sha256(paths[0]))
    assert ms.tilt3d is not None and ms.tilt3d.views[0].tilt == 5 and len(ms.tilt3d.steps) == 1
    r = subprocess.run([sys.executable, "-m", "vexel.cli", "sem", "tilt3d", str(paths[0]), str(paths[1])], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--tilts" in r.stdout  # no tilt in these files' metadata
