"""Batch processing and experiment-level statistics (roadmap Phase 6).

- :mod:`design`: the experiment design table (which files, which group, which replicate,
  per-file calibration and settings), and a scanner that drafts one from a folder;
- :mod:`runner`: runs one recipe over every row, unattended and resumable, isolating
  failures, and collects one row of results per file;
- :mod:`stats`: per-replicate values, group descriptions and comparisons, with the
  replicate (not the droplet) as the unit;
- :mod:`report`: the batch HTML report, methods text and the reproduce script.
"""

from vexel.batch.design import Design, DesignError, scan_folder
from vexel.batch.runner import BatchResult, run_batch

__all__ = ["BatchResult", "Design", "DesignError", "run_batch", "scan_folder"]
