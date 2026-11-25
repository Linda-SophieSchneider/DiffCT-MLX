"""Differentiable projection along explicit rays on the Metal ray kernels.

:func:`project` and :func:`backproject` trace each ray in one Metal thread.
They cover per-axis voxel spacing, ``parallel3d`` beams and curved detectors
in float32 on the GPU, next to the frame-based kernels for flat parallel, fan
and cone geometries.

The ray endpoints are built from the trajectory (and the detector surface) in
ordinary MLX code. The kernels are wrapped in ``mx.custom_function`` with exact
VJPs: the volume gradient is the matched adjoint, the geometry gradient the
per-ray face-crossing derivative, which MLX autodiff carries back to the
trajectory and to surface parameters.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .kernels.ray_beam import (
    ray_backward_kernel,
    ray_forward_kernel,
    ray_geometry_grad_kernel,
)
from .detector import pixel_points

_FINITE = ("fan", "cone")


def _params(n_rays, shape_zyx, finite, pitch_xyz):
    params = mx.array([n_rays, shape_zyx[2], shape_zyx[1], shape_zyx[0], int(finite)],
                      dtype=mx.int32)
    fparams = mx.array(list(pitch_xyz), dtype=mx.float32)
    return params, fparams


def _grid(n):
    tg = 256
    return (n, 1, 1), (min(tg, max(n, 1)), 1, 1)


def _forward(vol, A, B, finite, pitch_xyz):
    n = A.shape[0]
    params, fparams = _params(n, vol.shape, finite, pitch_xyz)
    grid, tg = _grid(n)
    return ray_forward_kernel(
        inputs=[vol, A, B, params, fparams], output_shapes=[(n,)],
        output_dtypes=[mx.float32], grid=grid, threadgroup=tg)[0]


def _backward(y, A, B, finite, pitch_xyz, shape_zyx):
    n = A.shape[0]
    params, fparams = _params(n, shape_zyx, finite, pitch_xyz)
    grid, tg = _grid(n)
    return ray_backward_kernel(
        inputs=[y, A, B, params, fparams], output_shapes=[tuple(shape_zyx)],
        output_dtypes=[mx.float32], grid=grid, threadgroup=tg, init_value=0)[0]


def _geometry(vol, cot, A, B, finite, pitch_xyz):
    n = A.shape[0]
    params, fparams = _params(n, vol.shape, finite, pitch_xyz)
    grid, tg = _grid(n)
    out = ray_geometry_grad_kernel(
        inputs=[vol, cot, A, B, params, fparams], output_shapes=[(n, 3), (n, 3)],
        output_dtypes=[mx.float32, mx.float32], grid=grid, threadgroup=tg)
    return out[0], out[1]


@lru_cache(maxsize=None)
def _ops(finite: bool, pitch_xyz: tuple, shape_zyx: tuple):
    """Differentiable ray projector pair for one configuration."""

    @mx.custom_function
    def ray_project(vol, A, B):
        return _forward(vol, A, B, finite, pitch_xyz)

    @ray_project.vjp
    def ray_project_vjp(primals, cotangent, _):
        vol, A, B = primals
        grad_vol = _backward(cotangent, A, B, finite, pitch_xyz, shape_zyx)
        grad_A, grad_B = _geometry(vol, cotangent, A, B, finite, pitch_xyz)
        return grad_vol, grad_A, grad_B

    @mx.custom_function
    def ray_backproject(y, A, B):
        return _backward(y, A, B, finite, pitch_xyz, shape_zyx)

    @ray_backproject.vjp
    def ray_backproject_vjp(primals, cotangent, _):
        # <c, A^T y> = <A c, y>: the geometry derivative is the forward one
        # with the volume cotangent as image and y as ray weights.
        y, A, B = primals
        grad_y = _forward(cotangent, A, B, finite, pitch_xyz)
        grad_A, grad_B = _geometry(cotangent, y, A, B, finite, pitch_xyz)
        return grad_y, grad_A, grad_B

    return ray_project, ray_backproject


def _rays(beam, frames, points):
    """Per-ray ``A`` and ``B`` in 3D world coordinates (z = 0 for 2D beams)."""
    k = points.shape[-1]
    pts = points.reshape(points.shape[0], -1, k)
    if beam in _FINITE:
        A = mx.broadcast_to(frames[0][:, None, :], pts.shape)
        B = pts
    else:
        r = frames[0] / mx.sqrt((frames[0] * frames[0]).sum(-1, keepdims=True))
        r = mx.broadcast_to(r[:, None, :], pts.shape)
        # Move the line origin next to the volume centre for float32 accuracy.
        A = pts - (pts * r).sum(-1, keepdims=True) * r
        B = r
    A = A.reshape(-1, k)
    B = B.reshape(-1, k)
    if k == 2:
        zero = mx.zeros((A.shape[0], 1), dtype=A.dtype)
        A = mx.concatenate([A, zero], axis=1)
        B = mx.concatenate([B, zero], axis=1)
    return A, B


def _layout(volume_shape, pitch):
    """``(Nz, Ny, Nx)`` and ``(sx, sy, sz)`` for 2D and 3D volumes."""
    if len(volume_shape) == 2:
        return (1,) + tuple(volume_shape), (float(pitch[1]), float(pitch[0]), 1.0)
    return tuple(volume_shape), tuple(float(p) for p in pitch[::-1])


def project(volume, beam, frames, detector_shape, detector_spacing, voxel_spacing, *,
            pixels=None, surface=None, surface_parameters=None, view_ids=None,
            n_total_views=None):
    """Forward projection ``(views, *pixels)`` on the Metal ray kernels."""
    frames = tuple(f.astype(mx.float32) for f in frames)
    pts = pixel_points(beam, frames, detector_shape, detector_spacing, pixels=pixels,
                           surface=surface, surface_parameters=surface_parameters,
                           view_ids=view_ids, n_total_views=n_total_views, dtype=mx.float32)
    out_shape = pts.shape[:-1]
    shape_zyx, pitch_xyz = _layout(volume.shape, voxel_spacing)
    A, B = _rays(beam, frames, pts)
    fwd, _ = _ops(beam in _FINITE, pitch_xyz, shape_zyx)
    vol = volume.astype(mx.float32).reshape(shape_zyx)
    return fwd(vol, A, B).reshape(out_shape)


def backproject(sinogram, beam, frames, volume_shape, detector_shape, detector_spacing,
                voxel_spacing, *, pixels=None, surface=None, surface_parameters=None,
                view_ids=None, n_total_views=None):
    """Matched adjoint of :func:`project`."""
    frames = tuple(f.astype(mx.float32) for f in frames)
    pts = pixel_points(beam, frames, detector_shape, detector_spacing, pixels=pixels,
                           surface=surface, surface_parameters=surface_parameters,
                           view_ids=view_ids, n_total_views=n_total_views, dtype=mx.float32)
    shape_zyx, pitch_xyz = _layout(tuple(volume_shape), voxel_spacing)
    A, B = _rays(beam, frames, pts)
    _, bwd = _ops(beam in _FINITE, pitch_xyz, shape_zyx)
    out = bwd(sinogram.astype(mx.float32).reshape(-1), A, B)
    return out.reshape(tuple(volume_shape))
