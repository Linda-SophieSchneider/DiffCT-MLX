"""Separable-footprint CUDA kernels for 3D cone-beam projection.

Ported (numba-CUDA) from the DiffCT-MLX Metal footprint kernels. Voxel-driven:
each voxel casts a magnified 2D footprint onto the (u, v) detector and its
overlap with every detector cell is integrated separably. Uses the same per-view
arbitrary-trajectory convention (src_pos, det_center, det_u_vec, det_v_vec) and
the permuted (Nx, Ny, Nz) = (W, H, D) volume layout as the Siddon cone kernels
in :mod:`diffct.kernels.cone_beam`.
"""

import math

import numpy as np
from numba import cuda

from ..constants import _FASTMATH_DECORATOR, _EPSILON, _HALF, _ZERO, _ONE


@_FASTMATH_DECORATOR
def _cone_3d_footprint_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_src, d_det_center, d_det_u, d_det_v, cx, cy, cz, voxel_spacing
):
    """Separable-footprint cone-beam forward projection (thread = (iview, iy, iz))."""
    iview, iy, iz = cuda.grid(3)
    if iview >= n_views or iy >= Ny or iz >= Nz:
        return

    eps = _EPSILON
    center_u = (np.float32(n_u) - _ONE) * _HALF
    center_v = (np.float32(n_v) - _ONE) * _HALF

    src_x = d_src[iview, 0]
    src_y = d_src[iview, 1]
    src_z = d_src[iview, 2]
    det_cx = d_det_center[iview, 0]
    det_cy = d_det_center[iview, 1]
    det_cz = d_det_center[iview, 2]
    u_vec_x = d_det_u[iview, 0]
    u_vec_y = d_det_u[iview, 1]
    u_vec_z = d_det_u[iview, 2]
    v_vec_x = d_det_v[iview, 0]
    v_vec_y = d_det_v[iview, 1]
    v_vec_z = d_det_v[iview, 2]

    n_x = u_vec_y * v_vec_z - u_vec_z * v_vec_y
    n_y = u_vec_z * v_vec_x - u_vec_x * v_vec_z
    n_z = u_vec_x * v_vec_y - u_vec_y * v_vec_x

    plane_dist = (det_cx - src_x) * n_x + (det_cy - src_y) * n_y + (det_cz - src_z) * n_z
    if abs(plane_dist) <= eps:
        return

    py = ((np.float32(iy) + _HALF) - cy) * voxel_spacing
    pz = ((np.float32(iz) + _HALF) - cz) * voxel_spacing

    for ix in range(Nx):
        val = d_vol[ix, iy, iz]
        if abs(val) <= eps:
            continue

        px = ((np.float32(ix) + _HALF) - cx) * voxel_spacing
        rx = px - src_x
        ry = py - src_y
        rz = pz - src_z
        denom = rx * n_x + ry * n_y + rz * n_z
        if abs(denom) <= eps:
            continue

        t = plane_dist / denom
        if t <= _ZERO:
            continue

        qx = src_x + t * rx
        qy = src_y + t * ry
        qz = src_z + t * rz
        rel_x = qx - det_cx
        rel_y = qy - det_cy
        rel_z = qz - det_cz
        u0 = rel_x * u_vec_x + rel_y * u_vec_y + rel_z * u_vec_z
        v0 = rel_x * v_vec_x + rel_y * v_vec_y + rel_z * v_vec_z

        ru = rx * u_vec_x + ry * u_vec_y + rz * u_vec_z
        rv = rx * v_vec_x + ry * v_vec_y + rz * v_vec_z
        inv_denom = _ONE / denom

        grad_ux = t * u_vec_x - t * ru * inv_denom * n_x
        grad_uy = t * u_vec_y - t * ru * inv_denom * n_y
        grad_uz = t * u_vec_z - t * ru * inv_denom * n_z
        grad_vx = t * v_vec_x - t * rv * inv_denom * n_x
        grad_vy = t * v_vec_y - t * rv * inv_denom * n_y
        grad_vz = t * v_vec_z - t * rv * inv_denom * n_z

        half_u = _HALF * voxel_spacing * (abs(grad_ux) + abs(grad_uy) + abs(grad_uz))
        half_v = _HALF * voxel_spacing * (abs(grad_vx) + abs(grad_vy) + abs(grad_vz))
        half_u = max(half_u, _HALF * du)
        half_v = max(half_v, _HALF * dv)
        width_u = max(half_u + half_u, du)
        width_v = max(half_v + half_v, dv)
        support_u = half_u + _HALF * du
        support_v = half_v + _HALF * dv

        iu_min = int(math.floor((u0 - support_u) / du + center_u))
        iu_max = int(math.ceil((u0 + support_u) / du + center_u))
        iv_min = int(math.floor((v0 - support_v) / dv + center_v))
        iv_max = int(math.ceil((v0 + support_v) / dv + center_v))
        if iu_min < 0:
            iu_min = 0
        if iv_min < 0:
            iv_min = 0
        if iu_max > n_u - 1:
            iu_max = n_u - 1
        if iv_max > n_v - 1:
            iv_max = n_v - 1

        fp_u_lo = u0 - half_u
        fp_u_hi = u0 + half_u
        fp_v_lo = v0 - half_v
        fp_v_hi = v0 + half_v
        scaled_val = val * voxel_spacing

        for iu in range(iu_min, iu_max + 1):
            pixel_u = (np.float32(iu) - center_u) * du
            overlap_u = min(pixel_u + _HALF * du, fp_u_hi) - max(pixel_u - _HALF * du, fp_u_lo)
            if overlap_u <= eps:
                continue
            wu = overlap_u / width_u
            for iv in range(iv_min, iv_max + 1):
                pixel_v = (np.float32(iv) - center_v) * dv
                overlap_v = min(pixel_v + _HALF * dv, fp_v_hi) - max(pixel_v - _HALF * dv, fp_v_lo)
                if overlap_v <= eps:
                    continue
                wv = overlap_v / width_v
                cuda.atomic.add(d_sino, (iview, iu, iv), scaled_val * wu * wv)


@_FASTMATH_DECORATOR
def _cone_3d_footprint_backward_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_src, d_det_center, d_det_u, d_det_v, cx, cy, cz, voxel_spacing
):
    """Adjoint of the cone-beam footprint forward projection (thread = (ix, iy, iz))."""
    ix, iy, iz = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    eps = _EPSILON
    center_u = (np.float32(n_u) - _ONE) * _HALF
    center_v = (np.float32(n_v) - _ONE) * _HALF

    px = ((np.float32(ix) + _HALF) - cx) * voxel_spacing
    py = ((np.float32(iy) + _HALF) - cy) * voxel_spacing
    pz = ((np.float32(iz) + _HALF) - cz) * voxel_spacing

    accum = _ZERO
    for iview in range(n_views):
        src_x = d_src[iview, 0]
        src_y = d_src[iview, 1]
        src_z = d_src[iview, 2]
        det_cx = d_det_center[iview, 0]
        det_cy = d_det_center[iview, 1]
        det_cz = d_det_center[iview, 2]
        u_vec_x = d_det_u[iview, 0]
        u_vec_y = d_det_u[iview, 1]
        u_vec_z = d_det_u[iview, 2]
        v_vec_x = d_det_v[iview, 0]
        v_vec_y = d_det_v[iview, 1]
        v_vec_z = d_det_v[iview, 2]

        n_x = u_vec_y * v_vec_z - u_vec_z * v_vec_y
        n_y = u_vec_z * v_vec_x - u_vec_x * v_vec_z
        n_z = u_vec_x * v_vec_y - u_vec_y * v_vec_x

        plane_dist = (det_cx - src_x) * n_x + (det_cy - src_y) * n_y + (det_cz - src_z) * n_z
        if abs(plane_dist) <= eps:
            continue

        rx = px - src_x
        ry = py - src_y
        rz = pz - src_z
        denom = rx * n_x + ry * n_y + rz * n_z
        if abs(denom) <= eps:
            continue

        t = plane_dist / denom
        if t <= _ZERO:
            continue

        qx = src_x + t * rx
        qy = src_y + t * ry
        qz = src_z + t * rz
        rel_x = qx - det_cx
        rel_y = qy - det_cy
        rel_z = qz - det_cz
        u0 = rel_x * u_vec_x + rel_y * u_vec_y + rel_z * u_vec_z
        v0 = rel_x * v_vec_x + rel_y * v_vec_y + rel_z * v_vec_z

        ru = rx * u_vec_x + ry * u_vec_y + rz * u_vec_z
        rv = rx * v_vec_x + ry * v_vec_y + rz * v_vec_z
        inv_denom = _ONE / denom

        grad_ux = t * u_vec_x - t * ru * inv_denom * n_x
        grad_uy = t * u_vec_y - t * ru * inv_denom * n_y
        grad_uz = t * u_vec_z - t * ru * inv_denom * n_z
        grad_vx = t * v_vec_x - t * rv * inv_denom * n_x
        grad_vy = t * v_vec_y - t * rv * inv_denom * n_y
        grad_vz = t * v_vec_z - t * rv * inv_denom * n_z

        half_u = _HALF * voxel_spacing * (abs(grad_ux) + abs(grad_uy) + abs(grad_uz))
        half_v = _HALF * voxel_spacing * (abs(grad_vx) + abs(grad_vy) + abs(grad_vz))
        half_u = max(half_u, _HALF * du)
        half_v = max(half_v, _HALF * dv)
        width_u = max(half_u + half_u, du)
        width_v = max(half_v + half_v, dv)
        support_u = half_u + _HALF * du
        support_v = half_v + _HALF * dv

        iu_min = int(math.floor((u0 - support_u) / du + center_u))
        iu_max = int(math.ceil((u0 + support_u) / du + center_u))
        iv_min = int(math.floor((v0 - support_v) / dv + center_v))
        iv_max = int(math.ceil((v0 + support_v) / dv + center_v))
        if iu_min < 0:
            iu_min = 0
        if iv_min < 0:
            iv_min = 0
        if iu_max > n_u - 1:
            iu_max = n_u - 1
        if iv_max > n_v - 1:
            iv_max = n_v - 1

        fp_u_lo = u0 - half_u
        fp_u_hi = u0 + half_u
        fp_v_lo = v0 - half_v
        fp_v_hi = v0 + half_v

        for iu in range(iu_min, iu_max + 1):
            pixel_u = (np.float32(iu) - center_u) * du
            overlap_u = min(pixel_u + _HALF * du, fp_u_hi) - max(pixel_u - _HALF * du, fp_u_lo)
            if overlap_u <= eps:
                continue
            wu = overlap_u / width_u
            for iv in range(iv_min, iv_max + 1):
                pixel_v = (np.float32(iv) - center_v) * dv
                overlap_v = min(pixel_v + _HALF * dv, fp_v_hi) - max(pixel_v - _HALF * dv, fp_v_lo)
                if overlap_v <= eps:
                    continue
                wv = overlap_v / width_v
                accum += d_sino[iview, iu, iv] * wu * wv * voxel_spacing

    d_vol[ix, iy, iz] = accum
