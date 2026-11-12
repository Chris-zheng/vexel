"""SpatialVolume: a lazy N-D array with fixed TCZYX axis semantics and physical spacing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np

AXES = "TCZYX"


@dataclass
class SpatialVolume:
    """Lazy 5-D array in ``TCZYX`` order.

    ``spacing`` maps the axes ``t``, ``z``, ``y``, ``x`` to a physical step. When the
    pixel size is unknown the spatial spacing is 1.0 with ``space_unit == "px"``; when
    the frame interval is unknown ``spacing["t"]`` is 1.0 with ``time_unit == "frame"``.
    No resampling is ever done here: anisotropy is passed to operators through spacing.
    """

    data: da.Array
    spacing: dict[str, float] = field(default_factory=dict)
    space_unit: str = "px"
    time_unit: str = "frame"
    channel_names: list[str] | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 5:
            raise ValueError(f"SpatialVolume needs a 5-D TCZYX array, got {self.data.ndim}-D")
        self.spacing = {ax: float(self.spacing.get(ax, 1.0)) for ax in "tzyx"}

    @classmethod
    def from_array(
        cls,
        array: Any,
        axes: str,
        *,
        chunks: Any = None,
        **kwargs: Any,
    ) -> "SpatialVolume":
        """Wrap a numpy/dask/zarr array whose axes are a subset of ``TCZYX`` in any order."""
        axes = axes.upper()
        if len(axes) != array.ndim:
            raise ValueError(f"axes {axes!r} do not match array with {array.ndim} dimensions")
        if len(set(axes)) != len(axes) or not set(axes) <= set(AXES):
            raise ValueError(f"axes must be unique letters from {AXES}, got {axes!r}")
        if isinstance(array, da.Array):
            data = array
        else:
            if chunks is None:
                # One chunk per frame/plane: the natural unit for frame-scope operators.
                chunks = tuple(-1 if ax in "YX" else 1 for ax in axes)
            data = da.from_array(array, chunks=chunks)
        order = [axes.index(ax) for ax in AXES if ax in axes]
        data = data.transpose(order)
        for i, ax in enumerate(AXES):
            if ax not in axes:
                data = da.expand_dims(data, i)
        return cls(data=data, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.data.shape)

    @property
    def sizes(self) -> dict[str, int]:
        return dict(zip(AXES, self.shape))

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def frames(self, channel: int = 0, z: int = 0) -> da.Array:
        """Lazy ``(T, Y, X)`` view of one channel and one z-plane."""
        sizes = self.sizes
        if not 0 <= channel < sizes["C"]:
            raise IndexError(f"channel {channel} out of range (volume has {sizes['C']})")
        if not 0 <= z < sizes["Z"]:
            raise IndexError(f"z {z} out of range (volume has {sizes['Z']})")
        return self.data[:, channel, z]

    def describe(self) -> dict[str, Any]:
        """Metadata only; never touches pixels."""
        return {
            "source": self.source,
            "axes": AXES,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "nbytes": self.nbytes,
            "spacing": dict(self.spacing),
            "space_unit": self.space_unit,
            "time_unit": self.time_unit,
            "channel_names": self.channel_names,
            "metadata": self.metadata,
        }
