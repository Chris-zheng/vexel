"""SEM image processing (M3): artefacts injected into clean synthetic pictures are reduced;
saturated pixels are all marked; frames realign; the chain is recorded and reproducible."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

from vexel.packs.sem import measure_io
from vexel.packs.sem import process as P
from vexel.packs.sem.measure import MeasurementSet, Scale, compute
from vexel.synthetic import sem as S

B = 8


def rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a[B:-B, B:-B] - b[B:-B, B:-B]
    return float(np.sqrt(((d - d.mean()) ** 2).mean()))


def fix(img: np.ndarray, *steps: P.Step) -> np.ndarray:
    return P.apply(img, P.Processing(steps=list(steps), saturation=P.Saturation(enabled=False))).analysis


@pytest.fixture(scope="module")
def particles() -> np.ndarray:
    return S.micrograph(256, 384, seed=1, noise=0)


@pytest.fixture(scope="module")
def lines() -> np.ndarray:
    return S.line_space(256, 384, noise=0)


# -- recovery of injected artefacts --------------------------------------------------------------


@pytest.mark.parametrize("pic", ["particles", "lines"])
def test_line_levelling_removes_scan_line_stripes(pic, request):
    clean = request.getfixturevalue(pic)
    bad = S.add_stripes(clean)
    assert rms(fix(bad, P.step("line_level")), clean) < 0.1 * rms(bad, clean)


def test_destriping_removes_stripes_that_fade_along_the_line(lines, particles):
    for clean, need in ((lines, 0.3), (particles, 0.5)):
        bad = S.add_stripes(clean, vary=0.9)
        assert rms(fix(bad, P.step("destripe")), clean) < need * rms(bad, clean)
    vertical = S.add_stripes(particles, direction="vertical")
    assert rms(fix(vertical, P.step("destripe", direction="vertical")), particles) < 0.6 * rms(vertical, particles)
    assert rms(fix(vertical, P.step("line_level", direction="columns")), particles) < 0.1 * rms(vertical, particles)


@pytest.mark.parametrize(("method", "need"), [("polynomial", 0.15), ("gaussian", 0.35), ("rolling_ball", 0.35)])
def test_background_levelling(particles, method, need):
    bad = S.add_gradient(particles)
    out = fix(bad, P.step("background", method=method, size=60 if method == "rolling_ball" else 0))
    assert rms(out, particles) < need * rms(bad, particles)
    divided = fix(bad + 0.5, P.step("background", method=method, mode="divide", size=60 if method == "rolling_ball" else 0))
    assert rms(divided, particles) < 0.5 * rms(bad, particles)


@pytest.mark.parametrize(("method", "need"), [("nlm", 0.4), ("bilateral", 0.5), ("median", 0.6), ("gaussian", 0.7)])
def test_denoising(particles, method, need):
    bad = S.add_noise(particles, 0.05)
    assert rms(fix(bad, P.step("denoise", method=method)), particles) < need * rms(bad, particles)
    assert P.noise_sigma(bad - particles) == pytest.approx(0.05, rel=0.1)


def test_scan_line_alignment(lines, particles):
    bad, shifts = S.add_row_jitter(lines)
    est = P.row_shifts(bad)
    err = est - shifts
    assert np.percentile(np.abs(err - np.median(err)), 95) < 0.4  # sub-pixel on a line/space pattern
    assert rms(fix(bad, P.step("align_rows")), lines) < 0.4 * rms(bad, lines)
    # few horizontal features: lines it cannot place are left alone, never made worse
    bad, _ = S.add_row_jitter(particles)
    assert rms(fix(bad, P.step("align_rows")), particles) <= rms(bad, particles)
    # slanted features are not "straightened": nothing to correct
    slant = ndi.rotate(np.tile(lines[:1], (512, 1)), 20, reshape=False, mode="nearest")[128:384, :]
    assert np.abs(P.row_shifts(slant)).max() < 0.3
    # alternate lines offset (interlace): both halves meet
    inter = P.shift_rows(lines, np.where(np.arange(lines.shape[0]) % 2, -1.5, 0.0))
    est = P.row_shifts(inter)
    assert abs((est[1::2] - est[::2]).mean() - 1.5) < 0.2


def test_everything_at_once(lines):
    j, _ = S.add_row_jitter(lines)
    bad = S.add_noise(S.add_gradient(S.add_stripes(j, 0.04)), 0.03)
    out = fix(bad, P.step("align_rows"), P.step("line_level"), P.step("background"), P.step("denoise"))
    assert rms(out, lines) < 0.4 * rms(bad, lines)


def test_line_widths_through_artefacts_are_recovered():
    width, pitch = 13.3, 31.7
    clean = S.line_space(200, 600, width=width, pitch=pitch, noise=0)
    bad = S.add_noise(S.add_gradient(S.add_stripes(clean, 0.05)), 0.04)
    ms = MeasurementSet()
    item = ms.add("profile", [(20, 100), (580, 100)], params={"width": 9})
    res = P.apply(bad, P.Processing(steps=[P.step("line_level"), P.step("background"), P.step("denoise")]))
    v = compute(item, ms.scale, res.analysis, res.mask)
    assert abs(v["width_mean"] - width) < 0.5 and v["width_sd"] < 0.3 and abs(v["pitch"] - pitch) < 0.2


# -- saturation ----------------------------------------------------------------------------------


def test_every_saturated_pixel_is_marked(particles):
    rng = np.random.default_rng(3)
    clean = particles * 0.8 + rng.normal(0, 0.01, particles.shape)  # nothing clipped
    sat, truth = S.add_saturation(clean)
    for img, t in ((sat, truth), (np.round(sat * 255).astype(np.uint8), None), (np.round(sat * 4095).astype(np.uint16), None)):
        t = t if t is not None else img == img.max()
        mask, info = P.saturation_mask(img)
        assert (mask & t).sum() == t.sum() and info["pixels"] == t.sum() and info["regions"] >= 1
        assert mask.sum() > t.sum()  # widened by a pixel: the edge of a clipped area is unreliable too
    for img in (clean, np.round(clean * 4095).astype(np.uint16)):
        assert not P.saturation_mask(img)[0].any()
    dark = clean.copy()
    dark[10:20, 10:20] = clean.min() - 0.1
    assert not P.saturation_mask(dark)[0][15, 15]
    assert P.saturation_mask(dark, P.Saturation(dark=True))[0][15, 15]
    assert P.saturation_mask(clean, P.Saturation(high=0.6, dilate=0))[0].sum() == (clean >= 0.6).sum()


def test_measurements_leave_saturated_pixels_out(lines):
    img = lines.copy()
    img[:, 150:200] = 1.0  # a clipped band across the lines
    res = P.apply(img, P.Processing())
    assert res.mask[:, 150:200].all() and res.notes[0].startswith("饱和区")
    ms = MeasurementSet()
    prof = ms.add("profile", [(10, 100), (370, 100)])
    free = compute(prof, ms.scale, res.analysis, None)
    v = compute(prof, ms.scale, res.analysis, res.mask)
    assert "饱和区" in v["note"] and v["pitch"] is None and len(v["edges"]) < len(free["edges"])
    assert all(not 148 <= e + 10 <= 202 for e in v["edges"])  # positions along the profile, which starts at x = 10
    assert len(v["widths"]) < len(free["widths"])
    with pytest.raises(ValueError, match="saturated"):
        compute(ms.add("wand", [(170, 100)]), ms.scale, res.analysis, res.mask)


# -- the chain -----------------------------------------------------------------------------------


def test_display_steps_leave_measurements_alone_and_the_data_bar_untouched(particles):
    img = np.vstack([particles, np.full((40, particles.shape[1]), 0.1)])
    corr = P.Processing(steps=[P.step("line_level")])
    shown = P.Processing(steps=[P.step("line_level"), P.step("contrast", gamma=0.5, clahe=True), P.step("sharpen")])
    a, b = P.apply(img, corr, crop_rows=256), P.apply(img, shown, crop_rows=256)
    assert np.array_equal(a.analysis, b.analysis) and not np.array_equal(a.display, b.display)
    assert np.array_equal(a.analysis[256:], img[256:].astype(a.analysis.dtype)) and np.array_equal(a.display[256:], a.original[256:])
    assert a.display.dtype == np.uint8 and a.display.shape == img.shape


def test_display_steps_run_after_corrections_whatever_the_order(particles):
    a = P.apply(particles, P.Processing(steps=[P.step("contrast", gamma=0.7), P.step("denoise", method="median")]))
    b = P.apply(particles, P.Processing(steps=[P.step("denoise", method="median"), P.step("contrast", gamma=0.7)]))
    assert np.array_equal(a.display, b.display) and np.array_equal(a.analysis, b.analysis)


def test_disabled_steps_are_skipped_and_prefixes_are_cached(particles, monkeypatch):
    calls = []
    real = P.OPS["denoise"].func
    monkeypatch.setattr(P.OPS["denoise"], "func", lambda a, **kw: calls.append(1) or real(a, **kw))
    pr = P.Processor(particles)
    chain = P.Processing(steps=[P.step("denoise", method="median"), P.step("contrast")])
    pr.run(chain)
    chain.steps[1].params = {"gamma": 0.5}
    pr.run(chain)  # only the display step changed
    assert len(calls) == 1
    chain.steps[0].enabled = False
    assert np.array_equal(pr.run(chain).analysis, particles.astype(np.float32))  # measurements work in single precision


def test_steps_are_validated_and_parsed():
    s = P.parse_step("denoise:method=median,strength=2")
    assert s.op == "denoise" and s.params == {"method": "median", "strength": 2.0}
    assert P.parse_step("contrast:clahe=true").params == {"clahe": True}
    for bad in ("nothing", "denoise:method=foo", "denoise:strength=99", "denoise:colour=1", "denoise:strength"):
        with pytest.raises(ValueError):
            P.parse_step(bad)
    text = P.describe(P.Processing(steps=[P.step("background", method="gaussian"), P.step("sharpen")]))
    assert text[0].startswith("背景校平（校正）：方法 高斯") and "仅显示" in text[1] and text[-1].startswith("饱和区标记")


def test_the_record_keeps_the_chain_and_old_records_still_load(tmp_path):
    ms = MeasurementSet(scale=Scale(pixel_size=1e-9))
    ms.processing.steps.append(P.step("line_level", keep=10))
    ms.processing.saturation.dark = True
    back = MeasurementSet.load(ms.save(tmp_path / "r.json"))
    assert back.processing == ms.processing and back.digest() == ms.digest()
    old = json.loads(ms.to_json())
    del old["processing"]
    assert MeasurementSet.model_validate(old).processing == P.Processing()


def test_exports_use_the_chain(tmp_path, lines):
    ms = MeasurementSet()
    ms.add("profile", [(10, 100), (370, 100)])
    bad = S.add_noise(S.add_stripes(lines, 0.06), 0.04)
    raw = measure_io.tables(ms, bad)["measurements"].loc[0, "width_sd (px)"]
    ms.processing.steps += [P.step("line_level"), P.step("denoise")]
    t = measure_io.tables(ms, bad)
    assert t["measurements"].loc[0, "width_sd (px)"] < raw
    assert list(t["processing"]["processing"])[:2] == P.describe(ms.processing)[:2]
    assert [p.name for p in measure_io.export_csv(ms, bad, tmp_path / "m.csv")][-1] == "m_processing.csv"


# -- frames and derived files --------------------------------------------------------------------


def test_frames_are_aligned_and_averaged():
    particles = S.micrograph(400, 600, seed=4, noise=0)
    true = [(0.0, 0.0), (0.7, -1.3), (-2.4, 0.45), (1.15, 2.6)]
    frames = [S.add_noise(ndi.shift(particles, s, order=3, mode="nearest"), 0.05, seed=i) for i, s in enumerate(true)]
    mean, shifts = P.average_frames(frames)
    e = np.asarray(shifts) + np.asarray(true)
    e -= e.mean(0)
    assert np.hypot(e[:, 0], e[:, 1]).max() < 0.1
    assert rms(mean, particles) < 0.6 * rms(frames[0], particles)
    with pytest.raises(ValueError):
        P.average_frames(frames[:1])
    with pytest.raises(ValueError):
        P.average_frames([particles, particles[:-1]])


def test_averaged_and_processed_files_keep_the_calibration(tmp_path, monkeypatch):
    import tifffile

    from vexel.packs.sem import calibrate as C

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    a = tmp_path / "a.tif"
    S.write_sem(a, "zeiss", pixel_size=3e-9, rows=240, cols=320)
    cal = C.for_file(a, use_ocr=False)
    img = C.load_image(a)
    b = tmp_path / "b.tif"
    frame = img.copy()
    frame[: cal.crop_rows] = np.roll(img[: cal.crop_rows], (1, -2), axis=(0, 1))
    tifffile.imwrite(b, frame)
    info = P.average_files([a, b], tmp_path / "avg.tif", use_ocr=False)
    assert info["frames"] == 2 and abs(info["shifts"][1][0] + 1) < 0.1 and abs(info["shifts"][1][1] - 2) < 0.1
    avg = C.for_file(tmp_path / "avg.tif", use_ocr=False)
    assert avg.ok and avg.pixel_size == pytest.approx(3e-9) and avg.crop_rows == cal.crop_rows and "多帧平均" in avg.warnings[-1]
    out = C.load_image(tmp_path / "avg.tif")
    assert out.dtype == np.float32 and np.allclose(out[cal.crop_rows :], img[cal.crop_rows :])
    desc = json.loads(tifffile.TiffFile(tmp_path / "avg.tif").pages[0].description)
    assert desc["vexel"]["frames"] == ["a.tif", "b.tif"] and desc["vexel"]["pixel_size"] == pytest.approx(3e-9)


def test_cli_process_and_average(tmp_path):
    env_home = tmp_path / "home"
    a = tmp_path / "a.tif"
    S.write_sem(a, "thermo", pixel_size=2.5e-9, rows=240, cols=320)
    run = lambda *args: subprocess.run([sys.executable, "-m", "vexel.cli", *args], capture_output=True, text=True,  # noqa: E731
                                       env={**__import__("os").environ, "VEXEL_HOME": str(env_home)})  # fmt: skip
    r = run("sem", "process", "--list")
    assert r.returncode == 0 and "line_level" in r.stdout and "denoise" in r.stdout
    r = run("sem", "process", str(a), "-s", "line_level", "-s", "denoise:method=median", "-o", str(tmp_path / "p.tif"),
            "--picture", str(tmp_path / "p.png"), "--save", "--json")  # fmt: skip
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    assert rep["noise_after"] < rep["noise_before"] and (tmp_path / "p.tif").is_file() and (tmp_path / "p.png").is_file()
    r = run("sem", "process", str(a), "--json")  # the saved chain is used next time
    assert json.loads(r.stdout)["steps"][:2] == rep["steps"][:2]
    r = run("sem", "average", str(a), str(a), "-o", str(tmp_path / "avg.tif"))
    assert r.returncode == 0 and "averaged 2 frames" in r.stdout
    r = run("sem", "process", str(a), "-s", "denoise:method=foo")
    assert r.returncode != 0 and "method must be one of" in r.stderr + r.stdout
