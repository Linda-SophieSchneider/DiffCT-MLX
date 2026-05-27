"""MLX differentiable projector functions for CT reconstruction.

This module provides differentiable forward projection and backprojection
functions using custom Metal kernels on Apple Silicon, with VJP (vector-Jacobian
product) support for gradient computation via ``mx.custom_function``.
"""

import os
from functools import lru_cache

import numpy as np
import mlx.core as mx

from .constants import _MX_DTYPE
from .utils import _grid_2d, _grid_3d
from .kernels import (
    parallel_2d_forward_kernel,
    parallel_2d_backward_kernel,
    parallel_2d_footprint_forward_kernel,
    parallel_2d_footprint_backward_kernel,
    fan_2d_forward_kernel,
    fan_2d_backward_kernel,
    fan_2d_footprint_forward_kernel,
    fan_2d_footprint_backward_kernel,
    cone_3d_forward_kernel,
    cone_3d_backward_kernel,
    cone_3d_footprint_forward_kernel,
    cone_3d_footprint_backward_kernel,
    cone_3d_geometry_grad_kernel,
)

# Set DIFFCT_GEOMETRY_VJP=1 to use the analytical geometry gradient kernel
# instead of the finite-difference fallback for cone_forward.
_GEOMETRY_VJP_ENABLED: bool = os.getenv("DIFFCT_GEOMETRY_VJP", "0") == "1"


def _as_mx_float_array(value):
    """Accept NumPy or MLX arrays and return an MLX float array."""
    return mx.array(value, dtype=_MX_DTYPE)


@lru_cache(maxsize=32)
def _projector_params_2d(
    n_ang: int,
    n_det: int,
    nx: int,
    ny: int,
    detector_spacing: float,
    cx: float,
    cy: float,
    voxel_spacing: float,
):
    """Cache small 2D kernel parameter buffers across repeated projector calls."""
    return (
        mx.array([n_ang, n_det, nx, ny], dtype=mx.int32),
        mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32),
    )


@lru_cache(maxsize=32)
def _projector_params_3d(
    n_views: int,
    n_u: int,
    n_v: int,
    nx: int,
    ny: int,
    nz: int,
    du: float,
    dv: float,
    cx: float,
    cy: float,
    cz: float,
    voxel_spacing: float,
):
    """Cache small 3D kernel parameter buffers across repeated projector calls."""
    return (
        mx.array([n_views, n_u, n_v, nx, ny, nz], dtype=mx.int32),
        mx.array([du, dv, cx, cy, cz, voxel_spacing], dtype=mx.float32),
    )


# ============================================================================
# 2D Parallel Beam
# ============================================================================

def _parallel_forward_impl(image, ray_dir, det_origin, det_u_vec,
                           num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
    """Raw parallel beam forward projection (no VJP)."""
    image = _as_mx_float_array(image)
    ray_dir = _as_mx_float_array(ray_dir)
    det_origin = _as_mx_float_array(det_origin)
    det_u_vec = _as_mx_float_array(det_u_vec)

    Ny, Nx = image.shape
    n_ang = ray_dir.shape[0]
    n_det = int(num_detectors)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params, fparams = _projector_params_2d(
        n_ang,
        n_det,
        Nx,
        Ny,
        float(detector_spacing),
        cx,
        cy,
        float(voxel_spacing),
    )

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = parallel_2d_forward_kernel(
        inputs=[image, ray_dir, det_origin, det_u_vec, params, fparams],
        output_shapes=[(n_ang, n_det)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


def _parallel_backward_impl(sinogram, ray_dir, det_origin, det_u_vec,
                            detector_spacing, H, W, voxel_spacing=1.0):
    """Raw parallel beam backprojection (no VJP)."""
    sinogram = _as_mx_float_array(sinogram)
    ray_dir = _as_mx_float_array(ray_dir)
    det_origin = _as_mx_float_array(det_origin)
    det_u_vec = _as_mx_float_array(det_u_vec)

    n_ang, n_det = sinogram.shape
    Ny, Nx = int(H), int(W)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params, fparams = _projector_params_2d(
        n_ang,
        n_det,
        Nx,
        Ny,
        float(detector_spacing),
        cx,
        cy,
        float(voxel_spacing),
    )

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = parallel_2d_backward_kernel(
        inputs=[sinogram, ray_dir, det_origin, det_u_vec, params, fparams],
        output_shapes=[(Ny, Nx)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


def _parallel_forward_footprint_impl(image, ray_dir, det_origin, det_u_vec,
                                     num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
    """Approximate separable-footprint parallel-beam forward projector."""
    image = _as_mx_float_array(image)
    ray_dir = _as_mx_float_array(ray_dir)
    det_origin = _as_mx_float_array(det_origin)
    det_u_vec = _as_mx_float_array(det_u_vec)

    Ny, Nx = image.shape
    n_ang = ray_dir.shape[0]
    n_det = int(num_detectors)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    params, fparams = _projector_params_2d(
        n_ang,
        n_det,
        Nx,
        Ny,
        float(detector_spacing),
        cx,
        cy,
        float(voxel_spacing),
    )
    grid, tg = _grid_2d(n_ang, Ny)

    outputs = parallel_2d_footprint_forward_kernel(
        inputs=[image, ray_dir, det_origin, det_u_vec, params, fparams],
        output_shapes=[(n_ang, n_det)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


def _parallel_backward_footprint_impl(sinogram, ray_dir, det_origin, det_u_vec,
                                      detector_spacing, H, W, voxel_spacing=1.0):
    """Adjoint of the approximate separable-footprint parallel-beam projector."""
    sinogram = _as_mx_float_array(sinogram)
    ray_dir = _as_mx_float_array(ray_dir)
    det_origin = _as_mx_float_array(det_origin)
    det_u_vec = _as_mx_float_array(det_u_vec)

    n_ang, n_det = sinogram.shape
    Ny, Nx = int(H), int(W)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    params, fparams = _projector_params_2d(
        n_ang,
        n_det,
        Nx,
        Ny,
        float(detector_spacing),
        cx,
        cy,
        float(voxel_spacing),
    )
    grid, tg = _grid_2d(Nx, Ny)

    outputs = parallel_2d_footprint_backward_kernel(
        inputs=[sinogram, ray_dir, det_origin, det_u_vec, params, fparams],
        output_shapes=[(Ny, Nx)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


@mx.custom_function
def parallel_forward(image, ray_dir, det_origin, det_u_vec,
                     num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    """Differentiable 2D parallel beam forward projection.

    Parameters
    ----------
    image : mx.array
        2D input image, shape ``(H, W)``.
    ray_dir : mx.array
        Ray direction unit vectors, shape ``(n_views, 2)``.
    det_origin : mx.array
        Detector origin positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    num_detectors : int
        Number of detector elements.
    detector_spacing : float
        Physical spacing between detector elements.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    sinogram : mx.array
        Shape ``(n_views, num_detectors)``.
    """
    return _parallel_forward_impl(image, ray_dir, det_origin, det_u_vec,
                                  num_detectors, detector_spacing, voxel_spacing)


@parallel_forward.vjp
def parallel_forward_vjp(primals, cotangent, _):
    image, ray_dir, det_origin, det_u_vec = primals[:4]
    num_detectors = primals[4] if len(primals) > 4 else 128
    detector_spacing = primals[5] if len(primals) > 5 else 1.0
    voxel_spacing = primals[6] if len(primals) > 6 else 1.0

    Ny, Nx = image.shape
    grad_image = _parallel_backward_impl(
        cotangent, ray_dir, det_origin, det_u_vec,
        detector_spacing, Ny, Nx, voxel_spacing
    )
    return (grad_image,
            mx.zeros_like(ray_dir), mx.zeros_like(det_origin), mx.zeros_like(det_u_vec),
            None, None, None)


@mx.custom_function
def parallel_backward(sinogram, ray_dir, det_origin, det_u_vec,
                      detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    """Differentiable 2D parallel beam backprojection.

    Parameters
    ----------
    sinogram : mx.array
        2D sinogram, shape ``(n_views, num_detectors)``.
    ray_dir : mx.array
        Ray direction unit vectors, shape ``(n_views, 2)``.
    det_origin : mx.array
        Detector origin positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    detector_spacing : float
        Physical spacing between detector elements.
    H, W : int
        Output image dimensions.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    reco : mx.array
        Shape ``(H, W)``.
    """
    return _parallel_backward_impl(sinogram, ray_dir, det_origin, det_u_vec,
                                   detector_spacing, H, W, voxel_spacing)


@parallel_backward.vjp
def parallel_backward_vjp(primals, cotangent, _):
    sinogram, ray_dir, det_origin, det_u_vec = primals[:4]
    detector_spacing = primals[4] if len(primals) > 4 else 1.0
    n_ang, n_det = sinogram.shape

    grad_sinogram = _parallel_forward_impl(
        cotangent, ray_dir, det_origin, det_u_vec,
        n_det, detector_spacing, 1.0
    )
    return (grad_sinogram,
            mx.zeros_like(ray_dir), mx.zeros_like(det_origin), mx.zeros_like(det_u_vec),
            None, None, None, None)


@mx.custom_function
def parallel_forward_footprint(image, ray_dir, det_origin, det_u_vec,
                               num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    """Matched footprint-style 2D parallel-beam forward projection."""
    return _parallel_forward_footprint_impl(
        image,
        ray_dir,
        det_origin,
        det_u_vec,
        num_detectors,
        detector_spacing,
        voxel_spacing,
    )


@parallel_forward_footprint.vjp
def parallel_forward_footprint_vjp(primals, cotangent, _):
    image, ray_dir, det_origin, det_u_vec = primals[:4]
    detector_spacing = primals[5] if len(primals) > 5 else 1.0
    voxel_spacing = primals[6] if len(primals) > 6 else 1.0

    Ny, Nx = image.shape
    grad_image = _parallel_backward_footprint_impl(
        cotangent,
        ray_dir,
        det_origin,
        det_u_vec,
        detector_spacing,
        Ny,
        Nx,
        voxel_spacing,
    )
    return (grad_image,
            mx.zeros_like(ray_dir), mx.zeros_like(det_origin), mx.zeros_like(det_u_vec),
            None, None, None)


@mx.custom_function
def parallel_backward_footprint(sinogram, ray_dir, det_origin, det_u_vec,
                                detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    """Adjoint of the matched footprint-style 2D parallel-beam forward projector."""
    return _parallel_backward_footprint_impl(
        sinogram,
        ray_dir,
        det_origin,
        det_u_vec,
        detector_spacing,
        H,
        W,
        voxel_spacing,
    )


@parallel_backward_footprint.vjp
def parallel_backward_footprint_vjp(primals, cotangent, _):
    sinogram, ray_dir, det_origin, det_u_vec = primals[:4]
    detector_spacing = primals[4] if len(primals) > 4 else 1.0
    n_ang, n_det = sinogram.shape
    voxel_spacing = primals[7] if len(primals) > 7 else 1.0

    grad_sinogram = _parallel_forward_footprint_impl(
        cotangent,
        ray_dir,
        det_origin,
        det_u_vec,
        n_det,
        detector_spacing,
        voxel_spacing,
    )
    return (grad_sinogram,
            mx.zeros_like(ray_dir), mx.zeros_like(det_origin), mx.zeros_like(det_u_vec),
            None, None, None, None)


# ============================================================================
# 2D Fan Beam
# ============================================================================

def _fan_forward_impl(image, src_pos, det_center, det_u_vec,
                      num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
    """Raw fan beam forward projection."""
    image = _as_mx_float_array(image)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)

    Ny, Nx = image.shape
    n_ang = src_pos.shape[0]
    n_det = int(num_detectors)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params = mx.array([n_ang, n_det, Nx, Ny], dtype=mx.int32)
    fparams = mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = fan_2d_forward_kernel(
        inputs=[image, src_pos, det_center, det_u_vec, params, fparams],
        output_shapes=[(n_ang, n_det)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


def _fan_backward_impl(sinogram, src_pos, det_center, det_u_vec,
                       detector_spacing, H, W, voxel_spacing=1.0):
    """Raw fan beam backprojection."""
    sinogram = _as_mx_float_array(sinogram)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)

    n_ang, n_det = sinogram.shape
    Ny, Nx = int(H), int(W)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params = mx.array([n_ang, n_det, Nx, Ny], dtype=mx.int32)
    fparams = mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = fan_2d_backward_kernel(
        inputs=[sinogram, src_pos, det_center, det_u_vec, params, fparams],
        output_shapes=[(Ny, Nx)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


def _fan_forward_footprint_impl(image, src_pos, det_center, det_u_vec,
                                num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
    """Approximate separable-footprint fan-beam forward projector."""
    image = _as_mx_float_array(image)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)

    Ny, Nx = image.shape
    n_ang = src_pos.shape[0]
    n_det = int(num_detectors)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    params, fparams = _projector_params_2d(
        n_ang,
        n_det,
        Nx,
        Ny,
        float(detector_spacing),
        cx,
        cy,
        float(voxel_spacing),
    )
    grid, tg = _grid_2d(n_ang, Ny)

    outputs = fan_2d_footprint_forward_kernel(
        inputs=[image, src_pos, det_center, det_u_vec, params, fparams],
        output_shapes=[(n_ang, n_det)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


def _fan_backward_footprint_impl(sinogram, src_pos, det_center, det_u_vec,
                                 detector_spacing, H, W, voxel_spacing=1.0):
    """Adjoint of the approximate separable-footprint fan-beam projector."""
    sinogram = _as_mx_float_array(sinogram)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)

    n_ang, n_det = sinogram.shape
    Ny, Nx = int(H), int(W)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    params, fparams = _projector_params_2d(
        n_ang,
        n_det,
        Nx,
        Ny,
        float(detector_spacing),
        cx,
        cy,
        float(voxel_spacing),
    )
    grid, tg = _grid_2d(Nx, Ny)

    outputs = fan_2d_footprint_backward_kernel(
        inputs=[sinogram, src_pos, det_center, det_u_vec, params, fparams],
        output_shapes=[(Ny, Nx)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


@mx.custom_function
def fan_forward(image, src_pos, det_center, det_u_vec,
                num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    """Differentiable 2D fan beam forward projection.

    Parameters
    ----------
    image : mx.array
        2D input image, shape ``(H, W)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 2)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    num_detectors : int
        Number of detector elements.
    detector_spacing : float
        Physical spacing between detector elements.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    sinogram : mx.array
        Shape ``(n_views, num_detectors)``.
    """
    return _fan_forward_impl(image, src_pos, det_center, det_u_vec,
                             num_detectors, detector_spacing, voxel_spacing)


@fan_forward.vjp
def fan_forward_vjp(primals, cotangent, _):
    image, src_pos, det_center, det_u_vec = primals[:4]
    detector_spacing = primals[5] if len(primals) > 5 else 1.0
    voxel_spacing = primals[6] if len(primals) > 6 else 1.0

    Ny, Nx = image.shape
    grad_image = _fan_backward_impl(
        cotangent, src_pos, det_center, det_u_vec,
        detector_spacing, Ny, Nx, voxel_spacing
    )
    return (grad_image,
            mx.zeros_like(src_pos), mx.zeros_like(det_center), mx.zeros_like(det_u_vec),
            None, None, None)


@mx.custom_function
def fan_backward(sinogram, src_pos, det_center, det_u_vec,
                 detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    """Differentiable 2D fan beam backprojection.

    Parameters
    ----------
    sinogram : mx.array
        2D fan beam sinogram, shape ``(n_views, num_detectors)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 2)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    detector_spacing : float
        Physical spacing between detector elements.
    H, W : int
        Output image dimensions.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    reco : mx.array
        Shape ``(H, W)``.
    """
    return _fan_backward_impl(sinogram, src_pos, det_center, det_u_vec,
                              detector_spacing, H, W, voxel_spacing)


@fan_backward.vjp
def fan_backward_vjp(primals, cotangent, _):
    sinogram, src_pos, det_center, det_u_vec = primals[:4]
    detector_spacing = primals[4] if len(primals) > 4 else 1.0
    n_ang, n_det = sinogram.shape

    grad_sinogram = _fan_forward_impl(
        cotangent, src_pos, det_center, det_u_vec,
        n_det, detector_spacing, 1.0
    )
    return (grad_sinogram,
            mx.zeros_like(src_pos), mx.zeros_like(det_center), mx.zeros_like(det_u_vec),
            None, None, None, None)


@mx.custom_function
def fan_forward_footprint(image, src_pos, det_center, det_u_vec,
                          num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    """Matched footprint-style 2D fan-beam forward projection."""
    return _fan_forward_footprint_impl(
        image,
        src_pos,
        det_center,
        det_u_vec,
        num_detectors,
        detector_spacing,
        voxel_spacing,
    )


@fan_forward_footprint.vjp
def fan_forward_footprint_vjp(primals, cotangent, _):
    image, src_pos, det_center, det_u_vec = primals[:4]
    detector_spacing = primals[5] if len(primals) > 5 else 1.0
    voxel_spacing = primals[6] if len(primals) > 6 else 1.0

    Ny, Nx = image.shape
    grad_image = _fan_backward_footprint_impl(
        cotangent,
        src_pos,
        det_center,
        det_u_vec,
        detector_spacing,
        Ny,
        Nx,
        voxel_spacing,
    )
    return (grad_image,
            mx.zeros_like(src_pos), mx.zeros_like(det_center), mx.zeros_like(det_u_vec),
            None, None, None)


@mx.custom_function
def fan_backward_footprint(sinogram, src_pos, det_center, det_u_vec,
                           detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    """Adjoint of the matched footprint-style 2D fan-beam forward projector."""
    return _fan_backward_footprint_impl(
        sinogram,
        src_pos,
        det_center,
        det_u_vec,
        detector_spacing,
        H,
        W,
        voxel_spacing,
    )


@fan_backward_footprint.vjp
def fan_backward_footprint_vjp(primals, cotangent, _):
    sinogram, src_pos, det_center, det_u_vec = primals[:4]
    detector_spacing = primals[4] if len(primals) > 4 else 1.0
    n_ang, n_det = sinogram.shape
    voxel_spacing = primals[7] if len(primals) > 7 else 1.0

    grad_sinogram = _fan_forward_footprint_impl(
        cotangent,
        src_pos,
        det_center,
        det_u_vec,
        n_det,
        detector_spacing,
        voxel_spacing,
    )
    return (grad_sinogram,
            mx.zeros_like(src_pos), mx.zeros_like(det_center), mx.zeros_like(det_u_vec),
            None, None, None, None)


# ============================================================================
# 3D Cone Beam
# ============================================================================

#: Perturbation step (in the same length unit as src_pos, typically mm) used
#: for the finite-difference source-position VJP.  A value of ~0.5 mm gives
#: good accuracy for source-to-isocenter distances in the 100–1000 mm range.
_FD_EPS_SRC: float = 0.5


def _src_pos_grad_fd(
    volume, src_pos, det_center, det_u_vec, det_v_vec,
    det_u, det_v, du, dv, voxel_spacing,
    cotangent,
    eps: float = _FD_EPS_SRC,
) -> "mx.array":
    """Finite-difference VJP for *src_pos* in cone-beam forward projection.

    Uses 6 forward passes (±ε per spatial axis) to approximate
    ``cotangent^T · ∂F/∂src_pos``.  Because each view's projection depends
    only on that view's source position, all views can be perturbed
    simultaneously with the same Δ, giving the correct per-view gradient
    without any extra passes per additional view.

    Parameters
    ----------
    cotangent : mx.array, shape ``(n_views, det_u, det_v)``
        Upstream gradient from the objective.
    eps : float
        Finite-difference step size in the same units as *src_pos*.

    Returns
    -------
    grad_src_pos : mx.array, shape ``(n_views, 3)``
    """
    # Convert all inputs to plain MLX arrays to detach them from any autograd
    # tape that may be active in the calling context (second-order
    # differentiation is not desired here).
    volume = _as_mx_float_array(volume)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)
    det_v_vec = _as_mx_float_array(det_v_vec)
    cotangent = _as_mx_float_array(cotangent)

    grads = []
    for j in range(3):
        delta_list = [[0.0, 0.0, 0.0]]
        delta_list[0][j] = eps
        delta = mx.array(delta_list, dtype=src_pos.dtype)  # (1, 3) — broadcast

        f_plus = _cone_forward_impl(
            volume, src_pos + delta, det_center, det_u_vec, det_v_vec,
            det_u, det_v, du, dv, voxel_spacing,
        )
        f_minus = _cone_forward_impl(
            volume, src_pos - delta, det_center, det_u_vec, det_v_vec,
            det_u, det_v, du, dv, voxel_spacing,
        )
        # Directional derivative w.r.t. src_pos[:, j]; shape (n_views, det_u, det_v)
        df_dxj = (f_plus - f_minus) / (2.0 * eps)
        # Chain rule: sum cotangent * df_dxj over the detector pixels per view
        grads.append(mx.sum(cotangent * df_dxj, axis=(1, 2)))  # (n_views,)

    return mx.stack(grads, axis=-1)  # (n_views, 3)

def _cone_forward_impl(volume, src_pos, det_center, det_u_vec, det_v_vec,
                       det_u, det_v, du, dv, voxel_spacing=1.0):
    """Raw cone beam forward projection."""
    volume = _as_mx_float_array(volume)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)
    det_v_vec = _as_mx_float_array(det_v_vec)

    D, H, W = volume.shape
    n_views = src_pos.shape[0]
    n_u, n_v = int(det_u), int(det_v)

    # Permute DHW → WHD for kernel (matches CUDA convention)
    volume_perm = mx.transpose(volume, axes=(2, 1, 0))
    # Make contiguous
    volume_perm = mx.array(volume_perm)

    Nx, Ny, Nz = W, H, D
    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    cz = float(Nz * 0.5)

    params, fparams = _projector_params_3d(
        n_views,
        n_u,
        n_v,
        Nx,
        Ny,
        Nz,
        float(du),
        float(dv),
        cx,
        cy,
        cz,
        float(voxel_spacing),
    )

    grid, tg = _grid_3d(n_views, n_u, n_v)

    outputs = cone_3d_forward_kernel(
        inputs=[volume_perm, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(n_views, n_u, n_v)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


def _cone_backward_impl(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                        D, H, W, du, dv, voxel_spacing=1.0):
    """Raw cone beam backprojection."""
    sinogram = _as_mx_float_array(sinogram)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)
    det_v_vec = _as_mx_float_array(det_v_vec)

    n_views, n_u, n_v = sinogram.shape
    Nx, Ny, Nz = int(W), int(H), int(D)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    cz = float(Nz * 0.5)

    params, fparams = _projector_params_3d(
        n_views,
        n_u,
        n_v,
        Nx,
        Ny,
        Nz,
        float(du),
        float(dv),
        cx,
        cy,
        cz,
        float(voxel_spacing),
    )

    grid, tg = _grid_3d(n_views, n_u, n_v)

    outputs = cone_3d_backward_kernel(
        inputs=[sinogram, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(Nx, Ny, Nz)],  # WHD layout
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )

    # Permute WHD → DHW
    vol = mx.transpose(outputs[0], axes=(2, 1, 0))
    return mx.array(vol)


def _cone_geometry_grad_impl(
    volume, cotangent, src_pos, det_center, det_u_vec, det_v_vec,
    det_u, det_v, du, dv, voxel_spacing=1.0
):
    """Analytical VJP of cone_forward w.r.t. src_pos, det_center, det_u_vec, det_v_vec.

    Returns
    -------
    grad_src_pos : ``(n_views, 3)``
    grad_det_center : ``(n_views, 3)``
    grad_det_u_vec : ``(n_views, 3)``
    grad_det_v_vec : ``(n_views, 3)``
    """
    volume = _as_mx_float_array(volume)
    cotangent = _as_mx_float_array(cotangent)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)
    det_v_vec = _as_mx_float_array(det_v_vec)

    D, H, W = volume.shape
    n_views = src_pos.shape[0]
    n_u, n_v = int(det_u), int(det_v)

    # Same WHD permutation as the forward kernel
    volume_perm = mx.array(mx.transpose(volume, axes=(2, 1, 0)))
    Nx, Ny, Nz = W, H, D

    params, fparams = _projector_params_3d(
        n_views, n_u, n_v, Nx, Ny, Nz,
        float(du), float(dv),
        float(Nx * 0.5), float(Ny * 0.5), float(Nz * 0.5),
        float(voxel_spacing),
    )

    grid, tg = _grid_3d(n_views, n_u, n_v)

    outputs = cone_3d_geometry_grad_kernel(
        inputs=[volume_perm, cotangent, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(n_views, 3), (n_views, 3), (n_views, 3), (n_views, 3)],
        output_dtypes=[_MX_DTYPE, _MX_DTYPE, _MX_DTYPE, _MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0], outputs[1], outputs[2], outputs[3]


def _cone_forward_footprint_impl(volume, src_pos, det_center, det_u_vec, det_v_vec,
                                 det_u, det_v, du, dv, voxel_spacing=1.0):
    """Approximate separable-footprint cone-beam forward projector."""
    volume = _as_mx_float_array(volume)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)
    det_v_vec = _as_mx_float_array(det_v_vec)

    D, H, W = volume.shape
    n_views = src_pos.shape[0]
    n_u, n_v = int(det_u), int(det_v)

    volume_perm = mx.array(mx.transpose(volume, axes=(2, 1, 0)))

    Nx, Ny, Nz = W, H, D
    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    cz = float(Nz * 0.5)

    params, fparams = _projector_params_3d(
        n_views,
        n_u,
        n_v,
        Nx,
        Ny,
        Nz,
        float(du),
        float(dv),
        cx,
        cy,
        cz,
        float(voxel_spacing),
    )

    grid, tg = _grid_3d(n_views, Ny, Nz)

    outputs = cone_3d_footprint_forward_kernel(
        inputs=[volume_perm, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(n_views, n_u, n_v)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


def _cone_backward_footprint_impl(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                                  D, H, W, du, dv, voxel_spacing=1.0):
    """Adjoint of the approximate separable-footprint cone-beam forward projector."""
    sinogram = _as_mx_float_array(sinogram)
    src_pos = _as_mx_float_array(src_pos)
    det_center = _as_mx_float_array(det_center)
    det_u_vec = _as_mx_float_array(det_u_vec)
    det_v_vec = _as_mx_float_array(det_v_vec)

    n_views, n_u, n_v = sinogram.shape
    Nx, Ny, Nz = int(W), int(H), int(D)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    cz = float(Nz * 0.5)

    params, fparams = _projector_params_3d(
        n_views,
        n_u,
        n_v,
        Nx,
        Ny,
        Nz,
        float(du),
        float(dv),
        cx,
        cy,
        cz,
        float(voxel_spacing),
    )

    grid, tg = _grid_3d(Nx, Ny, Nz)

    outputs = cone_3d_footprint_backward_kernel(
        inputs=[sinogram, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(Nx, Ny, Nz)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )

    vol = mx.transpose(outputs[0], axes=(2, 1, 0))
    return mx.array(vol)


@mx.custom_function
def cone_forward(volume, src_pos, det_center, det_u_vec, det_v_vec,
                 det_u=128, det_v=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    """Differentiable 3D cone beam forward projection.

    Parameters
    ----------
    volume : mx.array
        3D input volume, shape ``(D, H, W)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 3)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 3)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 3)``.
    det_v_vec : mx.array
        Detector v-direction unit vectors, shape ``(n_views, 3)``.
    det_u, det_v : int
        Number of detector elements along u and v axes.
    du, dv : float
        Physical spacing between detector elements along u and v.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    sino : mx.array
        Shape ``(n_views, det_u, det_v)``.
    """
    return _cone_forward_impl(volume, src_pos, det_center, det_u_vec, det_v_vec,
                              det_u, det_v, du, dv, voxel_spacing)


@cone_forward.vjp
def cone_forward_vjp(primals, cotangent, _):
    volume, src_pos, det_center, det_u_vec, det_v_vec = primals[:5]
    det_u_int = int(primals[5]) if len(primals) > 5 else 128
    det_v_int = int(primals[6]) if len(primals) > 6 else 128
    du = primals[7] if len(primals) > 7 else 1.0
    dv = primals[8] if len(primals) > 8 else 1.0
    voxel_spacing = primals[9] if len(primals) > 9 else 1.0

    D, H, W = volume.shape
    grad_volume = _cone_backward_impl(
        cotangent, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, du, dv, voxel_spacing
    )

    if _GEOMETRY_VJP_ENABLED:
        grad_src_pos, grad_det_center, grad_det_u_vec, grad_det_v_vec = \
            _cone_geometry_grad_impl(
                volume, cotangent, src_pos, det_center, det_u_vec, det_v_vec,
                det_u_int, det_v_int, du, dv, voxel_spacing,
            )
    else:
        grad_src_pos = _src_pos_grad_fd(
            volume, src_pos, det_center, det_u_vec, det_v_vec,
            det_u_int, det_v_int, du, dv, voxel_spacing,
            cotangent,
        )
        grad_det_center = mx.zeros_like(det_center)
        grad_det_u_vec = mx.zeros_like(det_u_vec)
        grad_det_v_vec = mx.zeros_like(det_v_vec)

    return (grad_volume,
            grad_src_pos, grad_det_center,
            grad_det_u_vec, grad_det_v_vec,
            None, None, None, None, None)


@mx.custom_function
def cone_backward(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                  D=128, H=128, W=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    """Differentiable 3D cone beam backprojection.

    Parameters
    ----------
    sinogram : mx.array
        3D projection data, shape ``(n_views, det_u, det_v)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 3)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 3)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 3)``.
    det_v_vec : mx.array
        Detector v-direction unit vectors, shape ``(n_views, 3)``.
    D, H, W : int
        Output volume dimensions (depth, height, width).
    du, dv : float
        Physical spacing between detector elements.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    vol : mx.array
        Shape ``(D, H, W)``.
    """
    return _cone_backward_impl(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                               D, H, W, du, dv, voxel_spacing)


@cone_backward.vjp
def cone_backward_vjp(primals, cotangent, _):
    sinogram, src_pos, det_center, det_u_vec, det_v_vec = primals[:5]
    du = primals[8] if len(primals) > 8 else 1.0
    dv = primals[9] if len(primals) > 9 else 1.0
    voxel_spacing = primals[10] if len(primals) > 10 else 1.0

    n_views, n_u, n_v = sinogram.shape

    grad_sinogram = _cone_forward_impl(
        cotangent, src_pos, det_center, det_u_vec, det_v_vec,
        n_u, n_v, du, dv, voxel_spacing
    )
    return (grad_sinogram,
            mx.zeros_like(src_pos), mx.zeros_like(det_center),
            mx.zeros_like(det_u_vec), mx.zeros_like(det_v_vec),
            None, None, None, None, None, None)


@mx.custom_function
def cone_forward_footprint(volume, src_pos, det_center, det_u_vec, det_v_vec,
                           det_u=128, det_v=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    """Matched voxel-footprint cone-beam forward projection."""
    return _cone_forward_footprint_impl(
        volume,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u,
        det_v,
        du,
        dv,
        voxel_spacing,
    )


@cone_forward_footprint.vjp
def cone_forward_footprint_vjp(primals, cotangent, _):
    volume, src_pos, det_center, det_u_vec, det_v_vec = primals[:5]
    det_u_int = int(primals[5]) if len(primals) > 5 else 128
    det_v_int = int(primals[6]) if len(primals) > 6 else 128
    du = primals[7] if len(primals) > 7 else 1.0
    dv = primals[8] if len(primals) > 8 else 1.0
    voxel_spacing = primals[9] if len(primals) > 9 else 1.0

    D, H, W = volume.shape
    grad_volume = _cone_backward_footprint_impl(
        cotangent,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        D,
        H,
        W,
        du,
        dv,
        voxel_spacing,
    )
    grad_src_pos = _src_pos_grad_fd(
        volume, src_pos, det_center, det_u_vec, det_v_vec,
        det_u_int, det_v_int, du, dv, voxel_spacing,
        cotangent,
    )
    return (grad_volume,
            grad_src_pos, mx.zeros_like(det_center),
            mx.zeros_like(det_u_vec), mx.zeros_like(det_v_vec),
            None, None, None, None, None)


@mx.custom_function
def cone_backward_footprint(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                            D=128, H=128, W=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    """Adjoint of the matched voxel-footprint cone-beam forward projector."""
    return _cone_backward_footprint_impl(
        sinogram,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        D,
        H,
        W,
        du,
        dv,
        voxel_spacing,
    )


@cone_backward_footprint.vjp
def cone_backward_footprint_vjp(primals, cotangent, _):
    sinogram, src_pos, det_center, det_u_vec, det_v_vec = primals[:5]
    det_u = int(sinogram.shape[1])
    det_v = int(sinogram.shape[2])
    du = primals[8] if len(primals) > 8 else 1.0
    dv = primals[9] if len(primals) > 9 else 1.0
    voxel_spacing = primals[10] if len(primals) > 10 else 1.0

    grad_sinogram = _cone_forward_footprint_impl(
        cotangent,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u,
        det_v,
        du,
        dv,
        voxel_spacing,
    )
    return (grad_sinogram,
            mx.zeros_like(src_pos), mx.zeros_like(det_center),
            mx.zeros_like(det_u_vec), mx.zeros_like(det_v_vec),
            None, None, None, None, None, None)
