"""Separable-footprint CUDA kernels for 2D parallel-beam projection.

A more accurate forward model than line-based Siddon tracing: each pixel casts a
finite footprint onto the detector axis and its overlap with every detector cell
is integrated. Ported (numba-CUDA) from the DiffCT-MLX Metal footprint kernels,
using the same per-view arbitrary-trajectory convention as the Siddon kernels in
:mod:`diffct.kernels.parallel_beam` (so the two are interchangeable).
"""

import math

import numpy as np
from numba import cuda

from ..constants import _FASTMATH_DECORATOR, _EPSILON, _HALF, _ZERO, _ONE


@_FASTMATH_DECORATOR
def _parallel_2d_footprint_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec, cx, cy, voxel_spacing
):
    """Separable-footprint parallel-beam forward projection (thread = (iang, iy))."""
    iang, iy = cuda.grid(2)
    if iang >= n_ang or iy >= Ny:
        return

    eps = _EPSILON
    det_spacing_vox = det_spacing / voxel_spacing
    center_u = np.float32(n_det) * _HALF

    dir_x = d_ray_dir[iang, 0]
    dir_y = d_ray_dir[iang, 1]
    det_ox = d_det_origin[iang, 0] / voxel_spacing
    det_oy = d_det_origin[iang, 1] / voxel_spacing
    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]

    l_phi = _ONE / max(max(abs(dir_x), abs(dir_y)), eps)
    half_u = _HALF * (abs(u_vec_x) + abs(u_vec_y))
    half_u = max(half_u, _HALF * det_spacing_vox)
    width_u = max(half_u + half_u, det_spacing_vox)
    support_u = half_u + _HALF * det_spacing_vox
    py = (np.float32(iy) + _HALF) - cy

    for ix in range(Nx):
        val = d_image[iy, ix]
        if abs(val) <= eps:
            continue

        px = (np.float32(ix) + _HALF) - cx
        rel_x = px - det_ox
        rel_y = py - det_oy
        u0 = rel_x * u_vec_x + rel_y * u_vec_y

        idet_min = int(math.floor(u0 / det_spacing_vox + center_u - support_u / det_spacing_vox))
        idet_max = int(math.ceil(u0 / det_spacing_vox + center_u + support_u / det_spacing_vox))
        if idet_min < 0:
            idet_min = 0
        if idet_max > n_det - 1:
            idet_max = n_det - 1

        fp_lo = u0 - half_u
        fp_hi = u0 + half_u
        scaled_val = val * l_phi

        for idet in range(idet_min, idet_max + 1):
            pixel_u = (np.float32(idet) - center_u) * det_spacing_vox
            pixel_lo = pixel_u - _HALF * det_spacing_vox
            pixel_hi = pixel_u + _HALF * det_spacing_vox
            overlap = min(pixel_hi, fp_hi) - max(pixel_lo, fp_lo)
            if overlap <= eps:
                continue
            weight = overlap / width_u
            cuda.atomic.add(d_sino, (iang, idet), scaled_val * weight)


@_FASTMATH_DECORATOR
def _parallel_2d_footprint_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec, cx, cy, voxel_spacing
):
    """Adjoint of the footprint forward projection (thread = (ix, iy))."""
    ix, iy = cuda.grid(2)
    if ix >= Nx or iy >= Ny:
        return

    eps = _EPSILON
    det_spacing_vox = det_spacing / voxel_spacing
    center_u = np.float32(n_det) * _HALF
    px = (np.float32(ix) + _HALF) - cx
    py = (np.float32(iy) + _HALF) - cy

    accum = _ZERO
    for iang in range(n_ang):
        dir_x = d_ray_dir[iang, 0]
        dir_y = d_ray_dir[iang, 1]
        det_ox = d_det_origin[iang, 0] / voxel_spacing
        det_oy = d_det_origin[iang, 1] / voxel_spacing
        u_vec_x = d_det_u_vec[iang, 0]
        u_vec_y = d_det_u_vec[iang, 1]

        l_phi = _ONE / max(max(abs(dir_x), abs(dir_y)), eps)
        half_u = _HALF * (abs(u_vec_x) + abs(u_vec_y))
        half_u = max(half_u, _HALF * det_spacing_vox)
        width_u = max(half_u + half_u, det_spacing_vox)
        support_u = half_u + _HALF * det_spacing_vox

        rel_x = px - det_ox
        rel_y = py - det_oy
        u0 = rel_x * u_vec_x + rel_y * u_vec_y

        idet_min = int(math.floor(u0 / det_spacing_vox + center_u - support_u / det_spacing_vox))
        idet_max = int(math.ceil(u0 / det_spacing_vox + center_u + support_u / det_spacing_vox))
        if idet_min < 0:
            idet_min = 0
        if idet_max > n_det - 1:
            idet_max = n_det - 1

        fp_lo = u0 - half_u
        fp_hi = u0 + half_u

        for idet in range(idet_min, idet_max + 1):
            pixel_u = (np.float32(idet) - center_u) * det_spacing_vox
            pixel_lo = pixel_u - _HALF * det_spacing_vox
            pixel_hi = pixel_u + _HALF * det_spacing_vox
            overlap = min(pixel_hi, fp_hi) - max(pixel_lo, fp_lo)
            if overlap <= eps:
                continue
            weight = overlap / width_u
            accum += d_sino[iang, idet] * weight * l_phi

    d_image[iy, ix] = accum
