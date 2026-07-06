"""Separable-footprint CUDA kernels for 2D fan-beam projection.

Ported (numba-CUDA) from the DiffCT-MLX Metal footprint kernels, using the same
per-view arbitrary-trajectory convention (src_pos, det_center, det_u_vec) as the
Siddon kernels in :mod:`diffct.kernels.fan_beam`. The pixel footprint on the
detector is the magnified projection of the pixel through the source.
"""

import math

import numpy as np
from numba import cuda

from ..constants import _FASTMATH_DECORATOR, _EPSILON, _HALF, _ZERO, _ONE


@_FASTMATH_DECORATOR
def _fan_2d_footprint_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_src, d_det_center, d_det_u_vec, cx, cy, voxel_spacing
):
    """Separable-footprint fan-beam forward projection (thread = (iang, iy))."""
    iang, iy = cuda.grid(2)
    if iang >= n_ang or iy >= Ny:
        return

    eps = _EPSILON
    det_spacing_vox = det_spacing / voxel_spacing
    center_u = np.float32(n_det) * _HALF

    src_x = d_src[iang, 0] / voxel_spacing
    src_y = d_src[iang, 1] / voxel_spacing
    det_cx = d_det_center[iang, 0] / voxel_spacing
    det_cy = d_det_center[iang, 1] / voxel_spacing
    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]
    n_x = -u_vec_y
    n_y = u_vec_x

    plane_dist = (det_cx - src_x) * n_x + (det_cy - src_y) * n_y
    if abs(plane_dist) <= eps:
        return

    py = (np.float32(iy) + _HALF) - cy

    for ix in range(Nx):
        val = d_image[iy, ix]
        if abs(val) <= eps:
            continue

        px = (np.float32(ix) + _HALF) - cx
        rx = px - src_x
        ry = py - src_y
        denom = rx * n_x + ry * n_y
        if abs(denom) <= eps:
            continue

        t = plane_dist / denom
        if t <= _ZERO:
            continue

        qx = src_x + t * rx
        qy = src_y + t * ry
        u0 = (qx - det_cx) * u_vec_x + (qy - det_cy) * u_vec_y

        ru = rx * u_vec_x + ry * u_vec_y
        inv_denom = _ONE / denom
        grad_ux = t * u_vec_x - t * ru * inv_denom * n_x
        grad_uy = t * u_vec_y - t * ru * inv_denom * n_y
        half_u = _HALF * (abs(grad_ux) + abs(grad_uy))
        half_u = max(half_u, _HALF * det_spacing_vox)
        width_u = max(half_u + half_u, det_spacing_vox)
        support_u = half_u + _HALF * det_spacing_vox

        ray_len = math.sqrt(rx * rx + ry * ry)
        if ray_len <= eps:
            continue
        dir_x = rx / ray_len
        dir_y = ry / ray_len
        l_phi = _ONE / max(max(abs(dir_x), abs(dir_y)), eps)

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
def _fan_2d_footprint_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_src, d_det_center, d_det_u_vec, cx, cy, voxel_spacing
):
    """Adjoint of the fan-beam footprint forward projection (thread = (ix, iy))."""
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
        src_x = d_src[iang, 0] / voxel_spacing
        src_y = d_src[iang, 1] / voxel_spacing
        det_cx = d_det_center[iang, 0] / voxel_spacing
        det_cy = d_det_center[iang, 1] / voxel_spacing
        u_vec_x = d_det_u_vec[iang, 0]
        u_vec_y = d_det_u_vec[iang, 1]
        n_x = -u_vec_y
        n_y = u_vec_x

        plane_dist = (det_cx - src_x) * n_x + (det_cy - src_y) * n_y
        if abs(plane_dist) <= eps:
            continue

        rx = px - src_x
        ry = py - src_y
        denom = rx * n_x + ry * n_y
        if abs(denom) <= eps:
            continue

        t = plane_dist / denom
        if t <= _ZERO:
            continue

        qx = src_x + t * rx
        qy = src_y + t * ry
        u0 = (qx - det_cx) * u_vec_x + (qy - det_cy) * u_vec_y

        ru = rx * u_vec_x + ry * u_vec_y
        inv_denom = _ONE / denom
        grad_ux = t * u_vec_x - t * ru * inv_denom * n_x
        grad_uy = t * u_vec_y - t * ru * inv_denom * n_y
        half_u = _HALF * (abs(grad_ux) + abs(grad_uy))
        half_u = max(half_u, _HALF * det_spacing_vox)
        width_u = max(half_u + half_u, det_spacing_vox)
        support_u = half_u + _HALF * det_spacing_vox

        ray_len = math.sqrt(rx * rx + ry * ry)
        if ray_len <= eps:
            continue
        dir_x = rx / ray_len
        dir_y = ry / ray_len
        l_phi = _ONE / max(max(abs(dir_x), abs(dir_y)), eps)

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
