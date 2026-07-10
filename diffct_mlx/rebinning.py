"""Sinogram-domain geometry tools: redundancy weighting and rebinning.

These operate purely on the sinogram and enable analytic reconstruction for
geometries beyond the plain full-scan flat-detector case:

* **Parker weighting** for short scans (``[0, pi + 2*gamma_max]``),
* **offset / half-fan detector weighting** for extended field-of-view scans,
* **truncation extension** to suppress truncation cupping,
* **fan -> parallel rebinning** (use the simpler parallel FBP on fan data), and
* **curved <-> flat detector rebinning** (equiangular cylindrical detectors).

Both the weightings and the rebinning resamplers are backend-neutral: they are
built entirely from the ``xp`` namespace (including gather-based bilinear/linear
interpolation), so they run on any backend without a torch-specific path.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import active as _b

xp = _b.xp

__all__ = [
    "detector_fan_angles",
    "parker_weights",
    "apply_parker_weighting",
    "offset_detector_weights",
    "apply_offset_weighting",
    "extend_truncation",
    "view_angles",
    "fan_to_parallel",
    "curved_to_flat",
    "flat_to_curved",
]


# ---------------------------------------------------------------------------
# Short-scan (Parker) weighting  — backend-neutral
# ---------------------------------------------------------------------------
def detector_fan_angles(num_detectors: int, detector_spacing: float, sdd: float):
    """Per-detector fan angles ``gamma_k = atan(u_k / sdd)`` for a flat detector."""
    # (k - n/2)*spacing is the package-wide detector-grid convention (the same
    # one the projector kernels and diffct.analytical use).
    u = (xp.arange(num_detectors) - num_detectors / 2.0) * float(detector_spacing)
    return xp.arctan(u / float(sdd))


def parker_weights(num_views: int, gammas, angular_range: float, betas=None):
    """Parker short-scan redundancy weights, shape ``(num_views, num_detectors)``.

    ``gammas`` are the per-detector fan angles (see :func:`detector_fan_angles`);
    ``angular_range`` is the total source rotation (radians), which for a minimal
    short scan is ``pi + 2*gamma_max``. Pass the actual per-view source angles as
    ``betas`` (relative to the scan start); otherwise views are assumed uniformly
    spaced over ``[0, angular_range]``. Weighting normalizes conjugate-ray
    redundancy (``w(beta, gamma) + w(beta + pi - 2*gamma, -gamma) = 1``) so a
    ramp-filtered backprojection over the short scan is exact.
    """
    gamma_max = float(xp.max(xp.abs(gammas)))
    if betas is None:
        betas = xp.arange(num_views) * (float(angular_range) / max(num_views - 1, 1))
    else:
        betas = xp.array(_b.to_numpy(betas).astype(np.float32))
    beta = xp.reshape(betas, (num_views, 1))
    gamma = xp.reshape(gammas, (1, -1))

    def _smooth(x):
        # Parker feathering sin^2(pi/4 * beta/(gamma_max -+ gamma)); ``x`` is the
        # normalized coordinate beta / (2*(gamma_max -+ gamma)) in [0, 1], so the
        # ramp must reach 1 at x = 1: sin^2(pi/2 * x).
        s = xp.sin(0.5 * math.pi * xp.clip(x, 0.0, 1.0))
        return s * s

    denom_lo = xp.maximum(gamma_max - gamma, 1e-6)
    denom_hi = xp.maximum(gamma_max + gamma, 1e-6)
    w_lo = _smooth(beta / (2.0 * denom_lo))
    w_hi = _smooth((math.pi + 2.0 * gamma_max - beta) / (2.0 * denom_hi))

    in_lo = beta < (2.0 * (gamma_max - gamma))
    in_hi = beta > (math.pi - 2.0 * gamma)
    weights = xp.where(in_lo, w_lo, xp.where(in_hi, w_hi, xp.ones_like(beta) + 0.0 * gamma))
    return weights


def apply_parker_weighting(sinogram, sdd, detector_spacing, angular_range=None, betas=None):
    """Apply Parker short-scan weighting to a fan/cone sinogram ``(views, u[, v])``."""
    sinogram = xp.array(sinogram, dtype=_b.float32)
    num_views = int(sinogram.shape[0])
    num_u = int(sinogram.shape[1])
    gammas = detector_fan_angles(num_u, detector_spacing, sdd)
    if angular_range is None:
        angular_range = math.pi + 2.0 * float(xp.max(xp.abs(gammas)))
    w = parker_weights(num_views, gammas, angular_range, betas=betas)   # (views, u)
    if sinogram.ndim == 3:
        w = xp.reshape(w, (num_views, num_u, 1))
    return sinogram * w


# ---------------------------------------------------------------------------
# Offset / half-fan detector weighting  — backend-neutral
# ---------------------------------------------------------------------------
def offset_detector_weights(num_detectors: int, detector_spacing: float, offset: float,
                            overlap: float | None = None):
    """Smooth half-fan weights for a laterally offset detector (extended FOV).

    ``offset`` is the detector-local u coordinate of the isocenter-ray piercing
    point, i.e. the center of the redundant band (same units as
    ``detector_spacing``; this is the quantity
    :func:`diffct_mlx.calibration.estimate_center_of_rotation` returns — for a
    detector physically shifted by ``+s`` the piercing point is at ``-s``).
    ``overlap`` is the half-width of the redundant central band (defaults to
    ``|offset|``). Weights ramp 0->1 across the overlap so that a full 360deg
    weighted backprojection reconstructs the doubled FOV without a seam.
    """
    u = (xp.arange(num_detectors) - num_detectors / 2.0) * float(detector_spacing) - float(offset)
    if overlap is None or overlap <= 0:
        overlap = max(abs(float(offset)), float(detector_spacing))
    t = xp.clip(u / float(overlap), -1.0, 1.0)
    return 0.5 * (1.0 + xp.sin(0.5 * math.pi * t))


def apply_offset_weighting(sinogram, detector_spacing, offset, overlap=None):
    """Apply half-fan offset weighting (x2 for conjugate-ray sum) to ``(views, u[, v])``."""
    sinogram = xp.array(sinogram, dtype=_b.float32)
    num_u = int(sinogram.shape[1])
    w = offset_detector_weights(num_u, detector_spacing, offset, overlap)
    w = xp.reshape(w, (1, num_u, 1)) if sinogram.ndim == 3 else xp.reshape(w, (1, num_u))
    return sinogram * (2.0 * w)


# ---------------------------------------------------------------------------
# Truncation extension  — backend-neutral
# ---------------------------------------------------------------------------
def extend_truncation(sinogram, width: int, axis: int = 1):
    """Extend truncated projections with a cosine roll-off before filtering.

    Pads ``width`` samples on both ends of the detector ``axis``, decaying the
    edge value smoothly to zero (a cos^2 taper), which suppresses the bright-ring
    / cupping truncation artifact. Returns the padded sinogram; crop back to the
    original detector width after ramp filtering + backprojection if desired.
    """
    sinogram = xp.array(sinogram, dtype=_b.float32)
    width = int(width)
    if width <= 0:
        return sinogram

    def _cos2(dist):  # cos^2 roll-off, 1 at the data edge -> 0 far away
        c = xp.cos(0.5 * math.pi * dist / (width + 1.0))
        return c * c

    # Distances from the data edge: right pad runs 1..width (decaying away);
    # left pad runs width..1 so it rises toward the data join.
    right_taper = _cos2(xp.arange(width) + 1.0)
    left_taper = _cos2(width - xp.arange(width))

    def _edge(index):
        idx = [slice(None)] * sinogram.ndim
        idx[axis] = slice(index, index + 1)
        return sinogram[tuple(idx)]

    n = int(sinogram.shape[axis])
    shape_lead = [1] * sinogram.ndim
    shape_lead[axis] = width
    left = _edge(0) * xp.reshape(left_taper, shape_lead)
    right = _edge(n - 1) * xp.reshape(right_taper, shape_lead)
    return xp.concatenate([left, sinogram, right], axis=axis)


# ---------------------------------------------------------------------------
# Rebinning  — backend-neutral (xp) interpolation
# ---------------------------------------------------------------------------
def _mod(x, n):
    """Elementwise ``x mod n`` via floor (backend-neutral)."""
    n = float(n)
    return x - n * xp.floor(x / n)


def _bilinear(image, row, col):
    """Bilinear sample of a 2D ``image`` at fractional ``(row, col)`` (same shape)."""
    V, D = int(image.shape[0]), int(image.shape[1])
    r = xp.clip(row, 0.0, float(V - 1))
    c = xp.clip(col, 0.0, float(D - 1))
    r0, c0 = xp.floor(r), xp.floor(c)
    fr, fc = r - r0, c - c0
    r0i, r1i = xp.clip(r0, 0.0, float(V - 1)), xp.clip(r0 + 1.0, 0.0, float(V - 1))
    c0i, c1i = xp.clip(c0, 0.0, float(D - 1)), xp.clip(c0 + 1.0, 0.0, float(D - 1))
    flat = xp.reshape(image, (V * D,))
    out_shape = tuple(int(s) for s in row.shape)
    n = 1
    for s in out_shape:
        n *= s

    def g(ri, cj):
        # Flat indices must be built in int64: float32 index arithmetic silently
        # rounds above 2^24 elements (any realistic 3D sinogram).
        lin = xp.astype(ri, xp.int64) * D + xp.astype(cj, xp.int64)
        return xp.reshape(xp.take(flat, xp.reshape(lin, (n,))), out_shape)

    return ((1.0 - fr) * (1.0 - fc) * g(r0i, c0i) + (1.0 - fr) * fc * g(r0i, c1i)
            + fr * (1.0 - fc) * g(r1i, c0i) + fr * fc * g(r1i, c1i))


def _interp_detector(sino, col_index, axis: int = -1):
    """Linear resample along the detector ``axis`` at fractional ``col_index``."""
    sino = xp.array(sino, dtype=_b.float32)
    ndim = int(len(sino.shape))
    axis = axis % ndim
    if axis != ndim - 1:
        sino = xp.moveaxis(sino, axis, -1)
    lead = tuple(int(s) for s in sino.shape[:-1])
    D = int(sino.shape[-1])
    rows_total = 1
    for s in lead:
        rows_total *= s
    img = xp.reshape(sino, (rows_total, D))
    n_out = int(col_index.shape[0])
    c = xp.clip(col_index, 0.0, float(D - 1))
    c0 = xp.floor(c)
    frac = xp.reshape(c - c0, (1, n_out))
    c0i, c1i = xp.clip(c0, 0.0, float(D - 1)), xp.clip(c0 + 1.0, 0.0, float(D - 1))
    flat = xp.reshape(img, (rows_total * D,))
    # int64 index arithmetic: float32 rounds above 2^24 elements.
    row_base = xp.reshape(xp.arange(rows_total, dtype=xp.int64) * D, (rows_total, 1))

    def g(cidx):
        lin = xp.reshape(row_base + xp.reshape(xp.astype(cidx, xp.int64), (1, n_out)),
                         (rows_total * n_out,))
        return xp.reshape(xp.take(flat, lin), (rows_total, n_out))

    out = (1.0 - frac) * g(c0i) + frac * g(c1i)
    out = xp.reshape(out, lead + (n_out,))
    if axis != ndim - 1:
        out = xp.moveaxis(out, -1, axis)
    return out


def view_angles(vectors):
    """Per-view angle ``atan2(y, x)`` (radians) of a stack of 2D/3D vectors.

    Use it to get the fan ``source_angles`` (from ``src_pos``) and the parallel
    ``out_angles`` (from the parallel operator's ``ray_dir``) for
    :func:`fan_to_parallel`, so both are expressed in the same world frame.
    """
    v = _b.to_numpy(vectors)
    return np.arctan2(v[:, 1], v[:, 0])


def fan_to_parallel(sinogram, *, sid: float, sdd: float, detector_spacing: float,
                    source_angles, out_angles, out_positions=None,
                    num_out_positions: int | None = None, flip_s: bool = False):
    """Rebin an equispaced-detector fan-beam sinogram to parallel-beam geometry.

    Uses the fan<->parallel relation ``beta = theta - gamma`` with ``s = sid *
    sin(gamma)`` (``gamma`` the detector fan angle). Both angle sets must be in the
    same world frame — the fan trajectory need not start at 0:

    * ``source_angles`` — the fan source angle per view (``view_angles(src_pos)``),
    * ``out_angles``     — the desired parallel projection angles, i.e. the angles
      of the parallel operator you will reconstruct with
      (``view_angles(ray_dir)``).

    Returns the parallel sinogram ``(len(out_angles), num_positions)`` on the
    ``s`` grid (defaults to ``linspace(-s_max, s_max, num_positions)``). Build the
    matching parallel operator at ``out_angles`` and detector spacing
    ``2*s_max/(num_positions-1)``, then run a parallel FBP. Set ``flip_s`` if the
    parallel detector axis points opposite to the ``s`` convention.
    """
    sino = xp.array(sinogram, dtype=_b.float32)
    n_views, n_det = int(sino.shape[0]), int(sino.shape[1])
    center = n_det / 2.0

    beta_np = np.asarray(_b.to_numpy(source_angles), dtype=np.float64).ravel()
    if beta_np.size < 2:
        raise ValueError("fan_to_parallel needs at least 2 views to rebin.")
    beta_np = np.unwrap(beta_np)   # atan2 angles wrap at +-pi mid-scan
    steps = np.diff(beta_np)
    dbeta = float(np.mean(steps))
    if dbeta == 0.0 or np.max(np.abs(steps - dbeta)) > 1e-3 * abs(dbeta):
        raise ValueError(
            "fan_to_parallel requires uniformly spaced source_angles; "
            "resample the trajectory (or rebin per uniform segment) first."
        )
    beta0 = float(beta_np[0])
    theta = xp.reshape(xp.array(_b.to_numpy(out_angles).astype(np.float32)), (-1, 1))   # (n_ang, 1)
    n_ang = int(theta.shape[0])

    if out_positions is None:
        s_max = sid * math.sin(math.atan(center * detector_spacing / sdd))
        n_pos = int(num_out_positions or n_det)
        spacing = (2.0 * s_max / (n_pos - 1)) if n_pos > 1 else 0.0
        positions = (xp.arange(n_pos) - n_pos / 2.0) * spacing
    else:
        positions = xp.array(_b.to_numpy(out_positions).astype(np.float32))
        n_pos = int(positions.shape[0])
    if flip_s:
        positions = -positions
    S = xp.reshape(positions, (1, n_pos))                          # (1, n_pos)

    sin_g = xp.clip(S / float(sid), -1.0, 1.0)
    cos_g = xp.sqrt(xp.maximum(1.0 - sin_g * sin_g, 1e-12))
    tan_g = sin_g / cos_g
    gamma = xp.arctan(tan_g)                                       # (1, n_pos)
    det_idx = float(sdd) * tan_g / float(detector_spacing) + center
    beta_needed = theta - gamma                                    # (n_ang, n_pos)
    view_idx = _mod((beta_needed - beta0) / dbeta, n_views)
    col_idx = det_idx + 0.0 * beta_needed                          # broadcast to (n_ang, n_pos)
    return _bilinear(sino, view_idx, col_idx)


def curved_to_flat(sinogram, *, sdd: float, det_spacing_angle: float, flat_spacing: float | None = None,
                   num_out: int | None = None, axis: int = 1):
    """Rebin an equiangular curved (cylindrical) detector sinogram to a flat detector.

    ``det_spacing_angle`` is the angular pixel pitch (radians) of the curved
    detector. For each flat position ``u`` the fan angle ``gamma = atan(u/sdd)``
    picks the curved sample. ``axis`` is the fan (u) detector axis — axis 1 in
    the package's ``(views, u[, v])`` sinogram convention, which is also the
    last axis for 2D input.
    """
    sino = xp.array(sinogram, dtype=_b.float32)
    n_det = int(sino.shape[axis])
    n_out = int(num_out or n_det)
    center = n_det / 2.0
    if flat_spacing is None:
        flat_spacing = sdd * det_spacing_angle               # match central pixel size
    u = (xp.arange(n_out) - n_out / 2.0) * float(flat_spacing)
    src_index = xp.arctan(u / float(sdd)) / float(det_spacing_angle) + center
    return _interp_detector(sino, src_index, axis=axis)


def flat_to_curved(sinogram, *, sdd: float, flat_spacing: float, det_spacing_angle: float | None = None,
                   num_out: int | None = None, axis: int = 1):
    """Rebin a flat-detector sinogram to an equiangular curved detector (inverse of :func:`curved_to_flat`)."""
    sino = xp.array(sinogram, dtype=_b.float32)
    n_det = int(sino.shape[axis])
    n_out = int(num_out or n_det)
    center = n_det / 2.0
    if det_spacing_angle is None:
        det_spacing_angle = math.atan(flat_spacing / sdd)
    gamma = (xp.arange(n_out) - n_out / 2.0) * float(det_spacing_angle)
    u = float(sdd) * xp.sin(gamma) / xp.cos(gamma)           # sdd * tan(gamma)
    src_index = u / float(flat_spacing) + center
    return _interp_detector(sino, src_index, axis=axis)
