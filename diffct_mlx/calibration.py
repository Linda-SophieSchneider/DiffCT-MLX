"""Geometric self-calibration: estimate the center of rotation from the data.

A miscentered detector (center-of-rotation offset) is the most common geometry
error and produces ring / doubling artifacts. These estimators recover the
detector-column offset (in pixels) directly from a sinogram:

* :func:`center_of_mass_offset` — fast, from the projection center of mass,
* :func:`estimate_center_of_rotation` — center-of-mass or opposing-view
  cross-correlation (robust for ~360deg scans),
* :func:`refine_center_by_sharpness` — the practical gold standard: reconstruct
  at candidate offsets and pick the sharpest image.

Apply the recovered offset with :func:`apply_center_offset` (shifts the detector
center along ``det_u``), then reconstruct.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import active as _b

__all__ = [
    "center_of_mass_offset",
    "estimate_center_of_rotation",
    "refine_center_by_sharpness",
    "apply_center_offset",
]


def _sino_2d(sinogram) -> np.ndarray:
    """Sinogram as ``(n_views, n_det)`` (central detector row for 3D cone data)."""
    s = _b.to_numpy(sinogram).astype(np.float64)
    if s.ndim == 3:
        s = s[:, :, s.shape[2] // 2]
    if s.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D sinogram, got shape {s.shape!r}.")
    return s


def center_of_mass_offset(sinogram) -> float:
    """Center-of-rotation offset (detector pixels) from the projection center of mass."""
    s = np.maximum(_sino_2d(sinogram), 0.0)
    n_det = s.shape[1]
    u = np.arange(n_det, dtype=np.float64) - (n_det - 1) / 2.0
    weight = np.maximum(s.sum(axis=1), 1e-12)
    com = (s * u[None, :]).sum(axis=1) / weight
    return float(np.mean(com))


def _opposing_offset(sinogram, angular_range: float, search_pixels: int) -> float:
    s = _sino_2d(sinogram)
    n_views, n_det = s.shape
    half = int(round(n_views * (math.pi / float(angular_range))))
    if half <= 0 or half >= n_views:
        return center_of_mass_offset(sinogram)
    margin = max(1, int(search_pixels))
    step = max(1, (n_views - half) // 32)
    offsets = []
    for j in range(0, n_views - half, step):
        a = s[j]
        b = s[j + half][::-1]                         # opposing view, detector-flipped
        best = (float("inf"), 0)
        for lag in range(-margin, margin + 1):
            shifted = np.roll(b, lag)
            diff = a[margin:n_det - margin] - shifted[margin:n_det - margin]
            ssd = float(np.dot(diff, diff))
            if ssd < best[0]:
                best = (ssd, lag)
        offsets.append(best[1] / 2.0)                 # lag = 2 * center offset
    return float(np.median(offsets)) if offsets else 0.0


def estimate_center_of_rotation(sinogram, *, method: str = "com",
                                angular_range: float = 2.0 * math.pi,
                                search_pixels: int = 40) -> float:
    """Estimate the center-of-rotation offset (detector pixels).

    ``method="com"`` (center of mass; fast, good for a centered object) or
    ``"opposing"`` (cross-correlate views 180deg apart; robust for full scans).
    """
    key = str(method).strip().lower()
    if key == "com":
        return center_of_mass_offset(sinogram)
    if key == "opposing":
        return _opposing_offset(sinogram, angular_range, search_pixels)
    raise ValueError(f"Unknown method {method!r}; use 'com' or 'opposing'.")


def _sharpness(volume) -> float:
    """Image sharpness = mean squared gradient magnitude (higher = sharper)."""
    v = _b.to_numpy(volume).astype(np.float64)
    total = 0.0
    for axis in range(v.ndim):
        d = np.diff(v, axis=axis)
        total += float(np.mean(d * d))
    return total


def refine_center_by_sharpness(reconstruct_fn, offsets, *, metric=None):
    """Pick the detector offset whose reconstruction is sharpest.

    ``reconstruct_fn(offset)`` must reconstruct a volume for a candidate offset
    (pixels). Returns ``(best_offset, scores)`` where ``scores`` maps offset ->
    sharpness. ``metric`` overrides the default gradient-energy sharpness.
    """
    metric = metric or _sharpness
    scores = {}
    for offset in offsets:
        scores[float(offset)] = float(metric(reconstruct_fn(float(offset))))
    best = max(scores, key=scores.get)
    return best, scores


def apply_center_offset(det_center, det_u_vec, offset_pixels: float, detector_spacing: float):
    """Shift the detector center by ``offset_pixels`` along ``det_u`` (returns backend array).

    Use the offset from :func:`estimate_center_of_rotation` /
    :func:`refine_center_by_sharpness` to correct the geometry before recon.
    """
    dc = _b.to_numpy(det_center).astype(np.float64)
    uu = _b.to_numpy(det_u_vec).astype(np.float64)
    shift = float(offset_pixels) * float(detector_spacing)
    corrected = dc + shift * uu
    return _b.as_array(corrected.astype(np.float32), dtype=_b.float32)
