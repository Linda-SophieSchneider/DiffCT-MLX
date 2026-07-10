"""Torch / CUDA backend.

Thin adapter over the upstream ``diffct`` (dev-line) package: it re-exports the
numba-CUDA autograd projectors as plain functional operators with the unified
signatures, exposes the ``geometry`` and ``analytical`` submodules, and provides
:data:`xp` — an MLX-``core``-shaped array namespace implemented on top of torch
so the higher-level reconstruction layer is written once for both backends.
"""

from types import SimpleNamespace

import numpy as _np
import torch

from diffct import (
    ParallelProjectorFunction,
    ParallelBackprojectorFunction,
    FanProjectorFunction,
    FanBackprojectorFunction,
    ConeProjectorFunction,
    ConeBackprojectorFunction,
)
from diffct import geometry, analytical  # noqa: F401  (re-exported)
from diffct.footprint import (
    ParallelFootprintProjectorFunction,
    ParallelFootprintBackprojectorFunction,
    FanFootprintProjectorFunction,
    FanFootprintBackprojectorFunction,
    ConeFootprintProjectorFunction,
    ConeFootprintBackprojectorFunction,
    ConeFootprintBackprojectorSparseFunction,
)

NAME = "torch"

float32 = torch.float32

#: Default device for newly-created arrays (mirrors MLX unified memory, which has
#: no explicit device). Torch needs one; volumes/masks default here so the
#: reconstruction layer never has to thread a device through.
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_default_device(device):
    """Set the device used for arrays created via ``xp``/:func:`as_array`."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = torch.device(device)


def get_default_device():
    return _DEFAULT_DEVICE


# --- Conversions ------------------------------------------------------------

def as_array(data, dtype=None, device=None):
    """Create a backend array from array-like ``data``.

    Existing tensors keep their device; array-likes land on ``device`` (or the
    default device). float64 is narrowed to float32 to match MLX defaults.
    """
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.as_tensor(_np.asarray(data))
        if t.dtype == torch.float64:
            t = t.float()
        t = t.to(_DEFAULT_DEVICE if device is None else device)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def to_numpy(x):
    """Return a host numpy copy of a backend array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return _np.asarray(x)


def device_of(x):
    """Return the device of a backend array (or ``None``)."""
    return x.device if isinstance(x, torch.Tensor) else None


def clamp_min(x, value):
    """Elementwise ``max(x, value)``."""
    return torch.clamp(x, min=float(value))


# --- xp: MLX-core-shaped array namespace over torch -------------------------

def _xp_array(value, dtype=None):
    return as_array(value, dtype=dtype)


def _xp_zeros(shape, dtype=None):
    return torch.zeros(tuple(shape), dtype=dtype or torch.float32, device=_DEFAULT_DEVICE)


def _xp_ones(shape, dtype=None):
    return torch.ones(tuple(shape), dtype=dtype or torch.float32, device=_DEFAULT_DEVICE)


def _xp_zeros_like(x):
    return torch.zeros_like(x)


def _xp_maximum(a, b):
    if not isinstance(b, torch.Tensor):
        return torch.clamp(a, min=float(b))
    if not isinstance(a, torch.Tensor):
        return torch.clamp(b, min=float(a))
    return torch.maximum(a, b)


def _xp_minimum(a, b):
    if not isinstance(b, torch.Tensor):
        return torch.clamp(a, max=float(b))
    if not isinstance(a, torch.Tensor):
        return torch.clamp(b, max=float(a))
    return torch.minimum(a, b)


def _operand(v, ref):
    if isinstance(v, torch.Tensor):
        return v
    return torch.as_tensor(v, dtype=torch.float32, device=ref.device)


def _xp_where(cond, a, b):
    return torch.where(cond, _operand(a, cond), _operand(b, cond))


def _xp_eval(*_args):
    # Torch is eager; nothing to force. Present for MLX API parity.
    return None


def _xp_norm(x):
    return float(torch.linalg.norm(x))


def _xp_arange(n, dtype=None):
    return torch.arange(int(n), dtype=dtype or torch.float32, device=_DEFAULT_DEVICE)


def _xp_grad(fn):
    """Return a function computing the gradient of scalar ``fn`` w.r.t. its arg.

    Mirrors ``mlx.core.grad`` using torch autograd, including for outputs that do
    not depend on the input (``mx.grad`` returns zeros there; torch would raise).
    """
    def _g(x):
        x = x.detach().clone().requires_grad_(True)
        y = fn(x)
        if not (isinstance(y, torch.Tensor) and y.requires_grad):
            return torch.zeros_like(x)
        (grad,) = torch.autograd.grad(y, x, allow_unused=True)
        return grad if grad is not None else torch.zeros_like(x)
    return _g


# --- Extra ops used by the operator / functional / solver layers ------------

def _xp_ones_like(x):
    return torch.ones_like(x)


def _xp_sum(x, axis=None):
    return torch.sum(x) if axis is None else torch.sum(x, dim=axis)


def _xp_min(x, axis=None):
    return torch.min(x) if axis is None else torch.min(x, dim=axis).values


def _xp_max(x, axis=None):
    return torch.max(x) if axis is None else torch.max(x, dim=axis).values


def _xp_clip(x, lo=None, hi=None):
    if lo is None and hi is None:
        return x
    return torch.clamp(
        x,
        min=None if lo is None else float(lo),
        max=None if hi is None else float(hi),
    )


def _xp_reshape(x, shape):
    return torch.reshape(x, tuple(int(s) for s in shape))


def _xp_concatenate(arrays, axis=0):
    return torch.cat([a for a in arrays], dim=axis)


def _xp_stack(arrays, axis=0):
    return torch.stack([a for a in arrays], dim=axis)


def _xp_real(x):
    return torch.real(x)


def _xp_sort(x, axis=-1):
    return torch.sort(x, dim=axis).values


def _xp_floor(x):
    return torch.floor(x)


def _xp_take(a, indices):
    """Gather from a flattened ``a`` at (float or int) ``indices`` (returns their shape)."""
    return torch.take(a, indices.to(torch.long))


def _fftfreq_torch(n):
    n = int(n)
    k = torch.arange(n, dtype=torch.float32, device=_DEFAULT_DEVICE)
    n_half = (n + 1) // 2
    return torch.where(k < n_half, k, k - n) / float(n)


#: On-device FFT sub-namespace (mirrors ``numpy.fft`` / ``mlx.core.fft``).
_fft_ns = SimpleNamespace(
    fft=lambda x, axis=-1: torch.fft.fft(x, dim=axis),
    ifft=lambda x, axis=-1: torch.fft.ifft(x, dim=axis),
    fftfreq=_fftfreq_torch,
)


xp = SimpleNamespace(
    array=_xp_array,
    zeros=_xp_zeros,
    ones=_xp_ones,
    zeros_like=_xp_zeros_like,
    ones_like=_xp_ones_like,
    maximum=_xp_maximum,
    minimum=_xp_minimum,
    where=_xp_where,
    eval=_xp_eval,
    norm=_xp_norm,
    grad=_xp_grad,
    sum=_xp_sum,
    mean=torch.mean,
    min=_xp_min,
    sqrt=torch.sqrt,
    exp=torch.exp,
    log=torch.log,
    log1p=torch.log1p,
    sign=torch.sign,
    clip=_xp_clip,
    reshape=_xp_reshape,
    concatenate=_xp_concatenate,
    stack=_xp_stack,
    sort=_xp_sort,
    floor=_xp_floor,
    take=_xp_take,
    real=_xp_real,
    square=torch.square,
    max=_xp_max,
    abs=torch.abs,
    arange=_xp_arange,
    astype=lambda x, dtype: x.to(dtype),
    moveaxis=torch.movedim,
    int64=torch.int64,
    cos=torch.cos,
    sin=torch.sin,
    arctan=torch.arctan,
    fft=_fft_ns,
    float32=torch.float32,
)


# --- Functional line-based projectors (unified signatures) ------------------

def parallel_forward(image, ray_dir, det_origin, det_u_vec,
                     num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    return ParallelProjectorFunction.apply(
        image, ray_dir, det_origin, det_u_vec,
        num_detectors, detector_spacing, voxel_spacing)


def parallel_backward(sinogram, ray_dir, det_origin, det_u_vec,
                      detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    return ParallelBackprojectorFunction.apply(
        sinogram, ray_dir, det_origin, det_u_vec,
        detector_spacing, H, W, voxel_spacing)


def fan_forward(image, src_pos, det_center, det_u_vec,
                num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    return FanProjectorFunction.apply(
        image, src_pos, det_center, det_u_vec,
        num_detectors, detector_spacing, voxel_spacing)


def fan_backward(sinogram, src_pos, det_center, det_u_vec,
                 detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    return FanBackprojectorFunction.apply(
        sinogram, src_pos, det_center, det_u_vec,
        detector_spacing, H, W, voxel_spacing)


def cone_forward(volume, src_pos, det_center, det_u_vec, det_v_vec,
                 det_u=128, det_v=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    return ConeProjectorFunction.apply(
        volume, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, du, dv, voxel_spacing)


def cone_backward(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                  D=128, H=128, W=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    return ConeBackprojectorFunction.apply(
        sinogram, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, du, dv, voxel_spacing)


# --- Separable-footprint projectors ----------------------------------------
# Native CUDA footprint kernels for all three geometries (parallel/fan/cone),
# including the sparse cone backprojection variant (``indices=``).

def parallel_forward_footprint(image, ray_dir, det_origin, det_u_vec,
                               num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    return ParallelFootprintProjectorFunction.apply(
        image, ray_dir, det_origin, det_u_vec,
        num_detectors, detector_spacing, voxel_spacing)


def parallel_backward_footprint(sinogram, ray_dir, det_origin, det_u_vec,
                                detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    return ParallelFootprintBackprojectorFunction.apply(
        sinogram, ray_dir, det_origin, det_u_vec,
        detector_spacing, H, W, voxel_spacing)


def fan_forward_footprint(image, src_pos, det_center, det_u_vec,
                          num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    return FanFootprintProjectorFunction.apply(
        image, src_pos, det_center, det_u_vec,
        num_detectors, detector_spacing, voxel_spacing)


def fan_backward_footprint(sinogram, src_pos, det_center, det_u_vec,
                           detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    return FanFootprintBackprojectorFunction.apply(
        sinogram, src_pos, det_center, det_u_vec,
        detector_spacing, H, W, voxel_spacing)


def cone_forward_footprint(volume, src_pos, det_center, det_u_vec, det_v_vec,
                           det_u=128, det_v=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    return ConeFootprintProjectorFunction.apply(
        volume, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, du, dv, voxel_spacing)


def cone_backward_footprint(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                            D=128, H=128, W=128, du=1.0, dv=1.0, voxel_spacing=1.0,
                            indices=None):
    # With `indices` (flat C-order (D,H,W) voxel positions), evaluate only those
    # voxels and return a 1D vector; otherwise return the dense (D,H,W) volume.
    if indices is not None:
        return ConeFootprintBackprojectorSparseFunction.apply(
            sinogram, indices, src_pos, det_center, det_u_vec, det_v_vec,
            D, H, W, du, dv, voxel_spacing)
    return ConeFootprintBackprojectorFunction.apply(
        sinogram, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, du, dv, voxel_spacing)
