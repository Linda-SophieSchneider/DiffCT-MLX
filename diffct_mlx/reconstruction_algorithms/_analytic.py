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

    #: Physical detector pitch along the filter axis. Used as the ramp
    #: filter's ``sample_spacing`` (``|f|/du``): leave at 1.0 for the legacy
    #: DFT-bin convention that the case normalization constants assume; set to
    #: the real pitch for quantitative pipelines.
    detector_spacing: float = 1.0
    voxel_spacing: float = 1.0
    enforce_positivity: bool = True
    normalization_scale: float | None = None
    #: Zero-pad the detector axis to ``pad_factor * n`` before the ramp FFT.
    #: Padding suppresses the circular-convolution wrap-around that depresses
    #: large objects (object-size-DEPENDENT bias, up to 0.40x at 75 % FOV
    #: coverage) — keep the default 2 unless you must reproduce legacy
    #: unpadded results (set 1).
    pad_factor: int = 2
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


def _ramp_numpy(sino_np, axis, pad_factor=1, sample_spacing=1.0):
    """CPU reference ramp filter (``2*|fftfreq|/sample_spacing`` convention).

    ``pad_factor > 1`` zero-pads the detector axis to ``pad_factor * n`` before
    the FFT and crops afterwards. Without padding the ramp's negative side
    lobes wrap around (circular convolution) and depress the interior of
    objects that fill a large fraction of the detector — the bias grows with
    object size (measured: 0.94x/0.74x/0.40x of the true value for objects
    covering 25/50/75 % of the FOV; with 2x padding 0.99x/0.94x/0.90x), so no
    global constant can compensate it. ``sample_spacing`` rescales the DFT-bin
    frequencies to physical units (``|f|/du``) for quantitative pipelines.
    """
    n = sino_np.shape[axis]
    n_pad = max(int(pad_factor), 1) * n
    if n_pad > n:
        pad_width = [(0, 0)] * sino_np.ndim
        pad_width[axis] = (0, n_pad - n)
        sino_np = np.pad(sino_np, pad_width)
    freqs = np.fft.fftfreq(n_pad)
    ramp = (2.0 * np.abs(freqs) / float(sample_spacing)).astype(np.float32)
    shape = [1] * sino_np.ndim
    shape[axis] = n_pad
    sino_fft = np.fft.fft(sino_np, axis=axis)
    filtered = np.real(np.fft.ifft(sino_fft * ramp.reshape(shape), axis=axis)).astype(np.float32)
    if n_pad > n:
        index = [slice(None)] * filtered.ndim
        index[axis] = slice(0, n)
        filtered = filtered[tuple(index)]
    return filtered


def _ramp_device(sinogram, axis, pad_factor=1, sample_spacing=1.0):
    """On-device ramp filter reproducing the ``2*|fftfreq|`` DFT convention.

    Identical maths to :func:`_ramp_numpy` (same DFT-bin ramp — no ``2*pi`` and
    no sample-spacing rescale) but evaluated with the backend FFT, so the
    analytic FBP/FDK pipeline stays GPU-resident and every downstream
    normalization constant (see ``cases.py``) is unchanged.
    """
    n = int(sinogram.shape[axis])
    n_pad = max(int(pad_factor), 1) * n
    if n_pad > n:
        pad_shape = [int(d) for d in sinogram.shape]
        pad_shape[axis] = n_pad - n
        sinogram = xp.concatenate(
            [sinogram, xp.zeros(pad_shape, dtype=_b.float32)], axis=axis)
    freqs = xp.fft.fftfreq(n_pad)
    ramp = 2.0 * xp.abs(freqs) / float(sample_spacing)
    shape = [1] * sinogram.ndim
    shape[axis] = n_pad
    ramp = xp.reshape(ramp, shape)
    spectrum = xp.fft.fft(sinogram, axis=axis)
    filtered = xp.real(xp.fft.ifft(spectrum * ramp, axis=axis))
    if n_pad > n:
        index = [slice(None)] * filtered.ndim
        index[axis] = slice(0, n)
        filtered = filtered[tuple(index)]
    return filtered


def _ramp(sinogram, axis, pad_factor=1, sample_spacing=1.0):
    """Ramp filter along ``axis`` — on-device when the backend exposes an FFT.

    Falls back to the numpy reference (host round-trip) only if the backend has
    no FFT namespace or the device transform fails, so behaviour is preserved on
    any backend while GPU backends never leave the device.
    """
    if getattr(xp, "fft", None) is not None:
        try:
            return _ramp_device(_b.as_array(sinogram, dtype=_b.float32), axis,
                                pad_factor=pad_factor, sample_spacing=sample_spacing)
        except Exception:  # pragma: no cover - defensive host fallback
            pass
    dev = _b.device_of(sinogram)
    sino_np = _b.to_numpy(sinogram).astype(np.float32)
    return _b.as_array(_ramp_numpy(sino_np, axis, pad_factor=pad_factor,
                                   sample_spacing=sample_spacing), device=dev)


def ramp_filter_2d(sinogram, pad_factor=1, sample_spacing=1.0):
    """Apply a ramp filter along the detector axis of a 2D sinogram."""
    sinogram = _b.as_array(sinogram, dtype=_b.float32)
    if sinogram.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram, got shape {tuple(sinogram.shape)!r}.")
    return _ramp(sinogram, 1, pad_factor=pad_factor, sample_spacing=sample_spacing)


def ramp_filter_3d(sinogram, pad_factor=1, sample_spacing=1.0):
    """Apply a ramp filter along the detector-u axis of a 3D cone sinogram."""
    sinogram = _b.as_array(sinogram, dtype=_b.float32)
    if sinogram.ndim != 3:
        raise ValueError(f"Expected a 3D sinogram, got shape {tuple(sinogram.shape)!r}.")
    return _ramp(sinogram, 1, pad_factor=pad_factor, sample_spacing=sample_spacing)


def ramp_filter(sinogram, axis=1, pad_factor=1, sample_spacing=1.0):
    """Apply a ramp filter along an arbitrary detector axis."""
    sinogram = _b.as_array(sinogram, dtype=_b.float32)
    if sinogram.ndim < 2:
        raise ValueError(f"Expected sinogram with at least 2 dimensions, got shape {tuple(sinogram.shape)!r}.")
    axis = int(axis)
    if axis < 0:
        axis += sinogram.ndim
    if axis < 0 or axis >= sinogram.ndim:
        raise ValueError(f"Invalid filter axis {axis} for sinogram with ndim={sinogram.ndim}.")
    return _ramp(sinogram, axis, pad_factor=pad_factor, sample_spacing=sample_spacing)
