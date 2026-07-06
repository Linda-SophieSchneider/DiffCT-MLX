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


xp = SimpleNamespace(
    array=_xp_array,
    zeros=_xp_zeros,
    ones=_xp_ones,
    zeros_like=_xp_zeros_like,
    maximum=_xp_maximum,
    minimum=_xp_minimum,
    where=_xp_where,
    eval=_xp_eval,
    norm=_xp_norm,
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


# Footprint projectors are not (yet) available in the CUDA backend. They are
# intentionally left undefined here; ``diffct_mlx.projectors`` provides a
# documented line-based fallback so user code that calls the *_footprint API
# keeps running across backends.
