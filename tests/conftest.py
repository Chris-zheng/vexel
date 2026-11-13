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
