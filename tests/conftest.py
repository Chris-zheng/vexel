from __future__ import annotations

from pathlib import Path

import pytest

from vexel.synthetic.droplets import DropletSynthConfig, write_dataset


@pytest.fixture(scope="session")
def droplet_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small synthetic droplet video with truth: ``<dir>/droplets.ome.tif`` and ``<dir>/truth``."""
    root = tmp_path_factory.mktemp("droplets")
    write_dataset(root, DropletSynthConfig(n_frames=200, seed=11))
    return root


@pytest.fixture(scope="session")
def cells_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic droplets carrying Poisson-distributed cells; truth has ``n_cells``."""
    root = tmp_path_factory.mktemp("cells")
    write_dataset(root, DropletSynthConfig(n_frames=200, seed=5, cell_lambda=1.2))
    return root


@pytest.fixture(scope="session")
def fluo_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two-channel (brightfield, fluorescence) digital-assay video; truth has ``positive``."""
    root = tmp_path_factory.mktemp("fluo")
    write_dataset(root, DropletSynthConfig(n_frames=300, seed=1, fluorescence=True, fluo_lambda=0.7))
    return root


@pytest.fixture(scope="session")
def hard_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run with flagged droplets: faint cells in noisy video (unstable counts)."""
    from vexel.engine import Recipe, run_recipe
    from vexel.readers import open_volume

    root = tmp_path_factory.mktemp("hard")
    video = write_dataset(root, DropletSynthConfig(n_frames=300, seed=26, cell_lambda=1.0, cell_contrast=15.0, noise_sigma=6.0))
    run_recipe(Recipe.load("droplet_encapsulation"), open_volume(video), root / "run", hash_input=False)
    return root / "run"


@pytest.fixture(autouse=True)
def _no_escaped_errors(monkeypatch):
    """An exception raised in a Qt signal handler is only printed (PySide calls sys.excepthook):
    it would pass unnoticed. Any such error during a test fails the test (0.8 phase 2)."""
    import sys
    import traceback

    escaped: list[str] = []
    monkeypatch.setattr(sys, "excepthook", lambda t, e, tb: escaped.append("".join(traceback.format_exception(t, e, tb))))
    yield
    try:  # what is still queued (deferred deletions, single-shot timers) belongs to this test too
        from qtpy.QtCore import QCoreApplication

        if QCoreApplication.instance() is not None:
            QCoreApplication.processEvents()
    except ImportError:
        pass
    if escaped:
        pytest.fail("an error escaped to sys.excepthook (a signal handler raised):\n" + escaped[0][-3000:], pytrace=False)
