"""Shared helpers for analytical reconstruction algorithms (backend-neutral).

Ported from the MLX implementation; the array operations are routed through the
active backend so the same code runs on MLX and Torch/CUDA. The ramp filter uses
numpy FFT (as in the MLX original) so results are numerically identical across
backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import math
import numpy as np

from ..backend import active as _b


AnalyticalWeighting = Callable[[Any], Any]
AnalyticalFilter = Callable[[Any], Any]
AnalyticalBackProjector = Callable[[Any], Any]


@dataclass
class AnalyticalReconstructionParameters:
    """Common parameters for analytical reconstruction algorithms."""

    detector_spacing: float = 1.0
    voxel_spacing: float = 1.0
    enforce_positivity: bool = True
    normalization_scale: float | None = None
    dtype: Any = field(default_factory=lambda: _b.float32)


def _as_array(value, dtype=None):
    """Convert arbitrary array-like inputs to the active backend's array."""
    return _b.as_array(value, dtype=dtype)


def _positive_and_scale(reconstruction, *, enforce_positivity, normalization_scale):
    """Apply optional positivity and multiplicative normalization."""
    reconstruction = _b.as_array(reconstruction)
    if enforce_positivity:
        reconstruction = _b.clamp_min(reconstruction, 0.0)
    if normalization_scale is not None:
        reconstruction = reconstruction * float(normalization_scale)
    return reconstruction


def default_angular_normalization(num_views: int) -> float:
    """Standard circular-scan angular normalization used in the examples."""
    if int(num_views) <= 0:
        raise ValueError("num_views must be positive.")
    return math.pi / float(num_views)


def _ramp(sino_np, axis):
    detector_count = sino_np.shape[axis]
    freqs = np.fft.fftfreq(detector_count)
    ramp = (2.0 * np.abs(freqs)).astype(np.float32)
    shape = [1] * sino_np.ndim
    shape[axis] = detector_count
    sino_fft = np.fft.fft(sino_np, axis=axis)
    return np.real(np.fft.ifft(sino_fft * ramp.reshape(shape), axis=axis)).astype(np.float32)


def ramp_filter_2d(sinogram):
    """Apply a ramp filter along the detector axis of a 2D sinogram."""
    dev = _b.device_of(sinogram)
    sino_np = _b.to_numpy(sinogram).astype(np.float32)
    if sino_np.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram, got shape {sino_np.shape!r}.")
    return _b.as_array(_ramp(sino_np, 1), device=dev)


def ramp_filter_3d(sinogram):
    """Apply a ramp filter along the detector-u axis of a 3D cone sinogram."""
    dev = _b.device_of(sinogram)
    sino_np = _b.to_numpy(sinogram).astype(np.float32)
    if sino_np.ndim != 3:
        raise ValueError(f"Expected a 3D sinogram, got shape {sino_np.shape!r}.")
    return _b.as_array(_ramp(sino_np, 1), device=dev)


def ramp_filter(sinogram, axis=1):
    """Apply a ramp filter along an arbitrary detector axis."""
    dev = _b.device_of(sinogram)
    sino_np = _b.to_numpy(sinogram).astype(np.float32)
    if sino_np.ndim < 2:
        raise ValueError(f"Expected sinogram with at least 2 dimensions, got shape {sino_np.shape!r}.")
    axis = int(axis)
    if axis < 0:
        axis += sino_np.ndim
    if axis < 0 or axis >= sino_np.ndim:
        raise ValueError(f"Invalid filter axis {axis} for sinogram with ndim={sino_np.ndim}.")
    return _b.as_array(_ramp(sino_np, axis), device=dev)
