"""Device routing and memory-watermark based block sizing."""

from __future__ import annotations

import os

GIB = 1024**3
_ASSUMED_TOTAL = 16 * GIB  # minimum supported spec, used when the OS cannot be queried


def total_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return _ASSUMED_TOTAL


def memory_budget_bytes(fraction: float = 0.10, cap: int = 2 * GIB) -> int:
    """Bytes one executor block may hold in memory."""
    return int(min(total_memory_bytes() * fraction, cap))


def plan_block_frames(frame_nbytes: int, n_arrays: int, *, multiple: int = 1, budget: int | None = None) -> int:
    """How many frames to process per block.

    ``n_arrays`` counts every per-frame array alive at once (inputs, float temporaries,
    outputs). The result is a positive multiple of ``multiple`` so that block writes
    stay aligned with on-disk chunks.
    """
    budget = memory_budget_bytes() if budget is None else budget
    frames = max(1, budget // max(1, frame_nbytes * max(1, n_arrays)))
    return max(multiple, (frames // multiple) * multiple)


def select_device(prefer: str | None = None) -> str:
    """CUDA / MPS / CPU, in that order. Phase 1 operators are CPU-only; this is recorded
    in provenance and used by later GPU operators."""
    if prefer:
        return prefer
    try:
        import torch  # optional
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
