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
