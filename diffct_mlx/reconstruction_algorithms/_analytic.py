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

xp = _b.xp


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


def _ramp_numpy(sino_np, axis):
    """CPU reference ramp filter (``2*|fftfreq|`` DFT convention)."""
    detector_count = sino_np.shape[axis]
    freqs = np.fft.fftfreq(detector_count)
    ramp = (2.0 * np.abs(freqs)).astype(np.float32)
    shape = [1] * sino_np.ndim
    shape[axis] = detector_count
    sino_fft = np.fft.fft(sino_np, axis=axis)
    return np.real(np.fft.ifft(sino_fft * ramp.reshape(shape), axis=axis)).astype(np.float32)


def _ramp_device(sinogram, axis):
    """On-device ramp filter reproducing the ``2*|fftfreq|`` DFT convention.

    Identical maths to :func:`_ramp_numpy` (same DFT-bin ramp — no ``2*pi`` and
    no sample-spacing rescale) but evaluated with the backend FFT, so the
    analytic FBP/FDK pipeline stays GPU-resident and every downstream
    normalization constant (see ``cases.py``) is unchanged.
    """
    n = int(sinogram.shape[axis])
    freqs = xp.fft.fftfreq(n)
    ramp = 2.0 * xp.abs(freqs)
    shape = [1] * sinogram.ndim
    shape[axis] = n
    ramp = xp.reshape(ramp, shape)
    spectrum = xp.fft.fft(sinogram, axis=axis)
    filtered = xp.fft.ifft(spectrum * ramp, axis=axis)
    return xp.real(filtered)


def _ramp(sinogram, axis):
    """Ramp filter along ``axis`` — on-device when the backend exposes an FFT.

    Falls back to the numpy reference (host round-trip) only if the backend has
    no FFT namespace or the device transform fails, so behaviour is preserved on
    any backend while GPU backends never leave the device.
    """
    if getattr(xp, "fft", None) is not None:
        try:
            return _ramp_device(_b.as_array(sinogram, dtype=_b.float32), axis)
        except Exception:  # pragma: no cover - defensive host fallback
            pass
    dev = _b.device_of(sinogram)
    sino_np = _b.to_numpy(sinogram).astype(np.float32)
    return _b.as_array(_ramp_numpy(sino_np, axis), device=dev)


def ramp_filter_2d(sinogram):
    """Apply a ramp filter along the detector axis of a 2D sinogram."""
    sinogram = _b.as_array(sinogram, dtype=_b.float32)
    if sinogram.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram, got shape {tuple(sinogram.shape)!r}.")
    return _ramp(sinogram, 1)


def ramp_filter_3d(sinogram):
    """Apply a ramp filter along the detector-u axis of a 3D cone sinogram."""
    sinogram = _b.as_array(sinogram, dtype=_b.float32)
    if sinogram.ndim != 3:
        raise ValueError(f"Expected a 3D sinogram, got shape {tuple(sinogram.shape)!r}.")
    return _ramp(sinogram, 1)


def ramp_filter(sinogram, axis=1):
    """Apply a ramp filter along an arbitrary detector axis."""
    sinogram = _b.as_array(sinogram, dtype=_b.float32)
    if sinogram.ndim < 2:
        raise ValueError(f"Expected sinogram with at least 2 dimensions, got shape {tuple(sinogram.shape)!r}.")
    axis = int(axis)
    if axis < 0:
        axis += sinogram.ndim
    if axis < 0 or axis >= sinogram.ndim:
        raise ValueError(f"Invalid filter axis {axis} for sinogram with ndim={sinogram.ndim}.")
    return _ramp(sinogram, axis)
