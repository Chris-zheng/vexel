"""Files that used to escape as errors (found by ``tests/benchmarks/robustness_eval.py``):
each is now read, or refused with a message, and never raises out of the window."""

from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_boolean_image_is_read_as_black_and_white(tmp_path):
    from vexel.packs.sem.calibrate import load_image
    from vexel.ui.sem_canvas import to_qimage

    p = tmp_path / "mask.tif"
    tifffile.imwrite(p, np.indices((40, 50)).sum(0) % 7 == 0)
    img = load_image(p)
    assert img.dtype == np.uint8 and set(np.unique(img)) == {0, 255}
    assert not to_qimage(np.zeros((20, 20), bool)).isNull()


@pytest.mark.parametrize("shape", [(1, 1), (1, 400), (400, 1), (5, 5), (15, 300)])
def test_tiny_images_are_refused_with_a_reason(tmp_path, shape):
    from vexel.packs.sem.calibrate import ImageTooSmall, load_image

    p = tmp_path / "tiny.tif"
    tifffile.imwrite(p, np.full(shape, 128, np.uint8))
    with pytest.raises(ImageTooSmall, match="太小"):
        load_image(p)


def test_planner_leaves_tiny_images_to_the_person():
    from vexel.packs.sem.plan import make_plan

    plan = make_plan(np.full((8, 300), 0.5), trial_run=False, models=False)
    assert plan.recipe is None and plan.status == "needs_user" and "太小" in plan.reasons[0]


def test_ocr_failure_is_a_warning_not_an_error(tmp_path, monkeypatch):
    from vexel.packs.sem import calibrate, ocr
    from vexel.synthetic.sem import write_sem

    p = tmp_path / "strip.tif"
    write_sem(p, "plain", pixel_size=2e-9, rows=200, cols=600)

    def broken(*_a, **_k):
        raise ocr.OcrUnavailable("ResizeImgError")

    monkeypatch.setattr(ocr, "available", lambda: (True, ""))
    monkeypatch.setattr(ocr, "read_text", broken)
    cal = calibrate.calibrate(p)
    assert any("文字识别失败" in w for w in cal.warnings)


def test_sem_window_says_why_a_file_cannot_be_opened(tmp_path, monkeypatch):
    from qtpy.QtWidgets import QApplication

    from vexel.ui.sem_window import SemWindow

    monkeypatch.setenv("VEXEL_HOME", str(tmp_path / "home"))
    app = QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr("vexel.ui.sem_window.QMessageBox.warning", lambda *a: shown.append(a[2]))
    p = tmp_path / "row.tif"
    tifffile.imwrite(p, np.zeros((1, 300), np.uint8))
    w = SemWindow()
    assert not w.open_path(p, wait=True)
    app.processEvents()
    assert shown and "太小" in shown[0] and "ImageTooSmall" not in shown[0]
    (tmp_path / "junk.tif").write_bytes(b"not a tiff")
    assert not w.open_path(tmp_path / "junk.tif", wait=True) and len(shown) == 2
    w.close()


@pytest.mark.parametrize("compression", ["tiff_lzw", "jpeg", "tiff_adobe_deflate", "packbits"])
def test_compressed_tiffs_open(tmp_path, compression):
    """LZW and JPEG inside TIFF need imagecodecs (a dependency since 0.8): microscope software
    often writes LZW. Both SEM images and droplet videos are read through tifffile."""
    from PIL import Image

    from vexel.packs.sem.calibrate import load_image

    a = (np.random.default_rng(0).random((64, 80)) * 255).astype(np.uint8)
    p = tmp_path / f"{compression}.tif"
    Image.fromarray(a).save(p, compression=compression)
    img = load_image(p)
    assert img.shape == (64, 80)
    if compression != "jpeg":
        assert np.array_equal(img, a)


def test_colour_frames_and_grey_alpha_are_read_whole(tmp_path):
    """An animated GIF (frames of RGB) and a grey + alpha PNG used to come out as a sliver of
    the image (the wrong axes were dropped)."""
    import imageio.v3 as iio

    from vexel.packs.sem.calibrate import load_image

    a = (np.random.default_rng(0).random((60, 80)) * 255).astype(np.uint8)
    iio.imwrite(tmp_path / "anim.gif", np.stack([a, a]))
    iio.imwrite(tmp_path / "ga.png", np.stack([a, np.full_like(a, 255)], -1))
    iio.imwrite(tmp_path / "rgba.png", np.stack([a, a, a, np.full_like(a, 255)], -1))
    tifffile.imwrite(tmp_path / "pages.tif", np.stack([a, 255 - a]))
    tifffile.imwrite(tmp_path / "rgb_stack.tif", np.stack([np.stack([a] * 3, -1)] * 2))
    for name in ("anim.gif", "ga.png", "rgba.png", "pages.tif", "rgb_stack.tif"):
        img = load_image(tmp_path / name)
        assert img.shape == (60, 80), name
        assert np.allclose(img, a, atol=1.5), name


def test_droplet_window_says_why_a_recording_cannot_be_opened(tmp_path, monkeypatch):
    from qtpy.QtWidgets import QApplication

    from vexel.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    shown = []
    w.warn = lambda title, text: shown.append(text)
    (tmp_path / "broken.avi").write_bytes(np.random.default_rng(0).integers(0, 256, 4000).astype(np.uint8).tobytes())
    (tmp_path / "empty").mkdir()
    assert not w.open_path(tmp_path / "broken.avi") and not w.open_path(tmp_path / "empty")
    assert "已损坏" in shown[0] and "Errno" not in shown[0]
    assert "没有可读的图像" in shown[1] and str(tmp_path) not in shown[1]
    w.close()
    app.processEvents()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_layer_stack_with_non_finite_pixels(bad):
    """Found by the corpus after the 0.8 memory work: flattening a bent stack with NaN or inf
    pixels raised. Every automatic measurement must take such images."""
    from vexel.packs.sem.analysis import RECIPES, AnalysisSpec, run
    from vexel.packs.sem.measure import Scale
    from vexel.synthetic import sem as S

    img, _ = S.bent_layer_stack(seed=1)
    img = img.astype(np.float32)
    img[40:60, 100:160] = bad
    for name in RECIPES:
        an = run(AnalysisSpec(id=1, recipe=name), img, Scale())
        assert an is not None  # a result or a stated error, never an exception
