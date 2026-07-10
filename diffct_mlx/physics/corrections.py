"""GPU-native projection-domain corrections.

These operate on projection/sinogram arrays and stay on the active backend
(``xp``), so they compose with the differentiable pipeline and run where the
data already lives. They are the corrections LEAP applies before/around
reconstruction. Spectral quantities (spectra, cross-sections, beam-hardening /
dual-energy coefficients) are computed offline by the caller; LLNL XrayPhysics
is kept only as a reference for that physics and is not a dependency here.

Sinogram convention: ``(n_views, det_u)`` for 2D or ``(n_views, det_u, det_v)``
for 3D; view axis is 0, detector axes follow.
"""

from __future__ import annotations

import math

from ..backend import active as _b
from ..filters import box_filter, median_filter

xp = _b.xp

__all__ = [
    "flat_field",
    "ring_removal",
    "bad_pixel_correction",
    "beam_hardening_polynomial",
    "detector_deblur",
    "scatter_correction",
    "mar_inpaint",
]

_EPS = 1e-8


def _detector_axes(sino):
    return tuple(range(1, sino.ndim))


def flat_field(raw, flat, dark=None, *, clip_min: float = 0.0, log: bool = True):
    """Flat/dark-field (gain) correction, then optional negative-log transform.

    ``transmission = (raw - dark) / (flat - dark)``; with ``log`` the attenuation
    line integral ``p = -log(transmission)`` is returned (clamped ``>= clip_min``).
    ``flat`` / ``dark`` broadcast against a single projection.
    """
    raw = xp.array(raw, dtype=_b.float32)
    flat = xp.array(flat, dtype=_b.float32)
    if dark is None:
        transmission = raw / xp.maximum(flat, _EPS)
    else:
        dark = xp.array(dark, dtype=_b.float32)
        transmission = (raw - dark) / xp.maximum(flat - dark, _EPS)
    transmission = xp.maximum(transmission, _EPS)
    if not log:
        return transmission
    p = -xp.log(transmission)
    return xp.maximum(p, float(clip_min))


def ring_removal(sinogram, *, radius: int = 8, strength: float = 1.0):
    """Remove ring artefacts via the detector mean-curve method.

    A detector element with a persistent gain offset produces a ring; averaging
    the sinogram over views exposes it as a bump in the per-element mean curve.
    Subtracting the high-frequency part of that mean curve (mean minus its
    ``box``-smoothed version) removes rings while preserving real structure.
    """
    sinogram = xp.array(sinogram, dtype=_b.float32)
    mean_curve = xp.mean(sinogram, axis=0)                 # (det...) persistent per-element level
    smoothed = box_filter(mean_curve, radius)              # low-frequency (real) component
    correction = float(strength) * (mean_curve - smoothed)  # high-frequency stripe component
    return sinogram - correction[None]


def bad_pixel_correction(sinogram, *, threshold: float = 4.0, radius: int = 1):
    """Replace outlier detector pixels with the local spatial median.

    Pixels deviating from their local median by more than ``threshold`` scale
    units are treated as bad and replaced. The scale is the global mean absolute
    deviation from the local median (a robust, std-like spread estimate).
    """
    sinogram = xp.array(sinogram, dtype=_b.float32)
    med = median_filter(sinogram, radius)
    resid = xp.abs(sinogram - med)
    mad = xp.mean(resid) + _EPS
    is_bad = resid > (float(threshold) * mad)
    return xp.where(is_bad, med, sinogram)


def beam_hardening_polynomial(sinogram, coefficients):
    """Single-material (water) beam-hardening correction by a polynomial map.

    ``p_corr = c[0]*p + c[1]*p^2 + ...`` applied to the attenuation line
    integrals. ``coefficients`` are supplied by the caller (e.g. from an empirical
    water calibration); the values can be derived offline from a spectral
    reference, but no external physics library is used here.
    """
    sinogram = xp.array(sinogram, dtype=_b.float32)
    coefficients = [float(c) for c in coefficients]
    out = xp.zeros_like(sinogram)
    power = sinogram
    for c in coefficients:
        out = out + c * power
        power = power * sinogram
    return out


def _wiener_1d(p, axis, sigma, reg):
    n = int(p.shape[axis])
    freqs = xp.fft.fftfreq(n)
    # Gaussian PSF is real & even -> its transfer function H(f) is real.
    h = xp.exp(-2.0 * (math.pi ** 2) * (float(sigma) ** 2) * (freqs * freqs))
    wiener = h / (h * h + float(reg))
    # Normalize to unit DC gain: the raw Wiener gain at f=0 is 1/(1+reg), which
    # would uniformly shrink all attenuation values by ~reg (a bias in mu).
    wiener = wiener * (1.0 + float(reg))
    shape = [1] * p.ndim
    shape[axis] = n
    wiener = xp.reshape(wiener, shape)
    spectrum = xp.fft.fft(p, axis=axis)
    return xp.real(xp.fft.ifft(spectrum * wiener, axis=axis))


def detector_deblur(sinogram, *, sigma: float = 1.0, reg: float = 1e-2):
    """Deconvolve a separable Gaussian detector blur (Wiener filter, per detector axis).

    Counters detector cross-talk / focal-spot blur. ``sigma`` is the PSF width in
    pixels; ``reg`` is the Wiener regularization (larger = smoother / safer).
    """
    sinogram = xp.array(sinogram, dtype=_b.float32)
    out = sinogram
    for axis in _detector_axes(sinogram):
        out = _wiener_1d(out, axis, sigma, reg)
    return out


def scatter_correction(intensity, *, fraction: float = 0.05, radius: int = 12):
    """First-order convolutional scatter estimate & subtraction (intensity domain).

    Models scatter as a broad low-pass of the measured intensity: ``I_corr =
    I - fraction * boxblur(I)`` (clamped ``> 0``). The blur acts per view in the
    detector plane only (scatter is a detector-domain effect; views must not
    mix). Apply to intensity / transmission **before** the log transform. A
    simple, fast stand-in for kernel-based scatter.
    """
    intensity = xp.array(intensity, dtype=_b.float32)
    scatter = float(fraction) * box_filter(intensity, radius, axes=_detector_axes(intensity))
    return xp.maximum(intensity - scatter, _EPS)


def mar_inpaint(sinogram, mask, *, iterations: int = 40, radius: int = 1):
    """Metal-artefact reduction: inpaint masked sinogram traces by diffusion.

    ``mask`` is truthy where the metal trace corrupts the sinogram (e.g. the
    forward projection of a metal segmentation). Those samples are replaced by a
    normalized-convolution diffusion of the surrounding valid data, giving a
    smooth interpolation across the trace.
    """
    sinogram = xp.array(sinogram, dtype=_b.float32)
    mask = xp.array(mask, dtype=_b.float32)
    keep = xp.maximum(1.0 - mask, 0.0)          # 1 where valid, 0 in the trace
    filled = sinogram * keep
    known = keep                                 # grows inward as the hole fills
    for _ in range(int(iterations)):
        num = box_filter(filled, radius)
        den = box_filter(known, radius)
        estimate = num / xp.maximum(den, _EPS)
        has_neighbor = den > _EPS
        # original valid samples stay fixed; masked pixels adopt the estimate
        # once any known neighbour exists, and become "known" for the next pass.
        filled = xp.where(keep > 0.5, sinogram, xp.where(has_neighbor, estimate, filled))
        known = xp.where((keep > 0.5) | has_neighbor, xp.ones_like(known), known)
    return filled
