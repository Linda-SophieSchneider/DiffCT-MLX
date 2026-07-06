"""Torch / CUDA backend.

Thin adapter over the upstream ``diffct`` (dev-line) package: it re-exports the
numba-CUDA autograd projectors as plain functional operators with the unified
signatures, and exposes the array namespace (``xp``) plus the ``geometry`` and
``analytical`` submodules used by the higher-level reconstruction layer.
"""

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

#: Array namespace. For now this is the ``torch`` module itself; the high-level
#: layer only relies on the subset of the array API that maps cleanly onto both
#: torch and mlx.core. It is centralised here so the abstraction can grow.
xp = torch


def as_array(data, dtype=None, device=None):
    """Create a backend array from array-like ``data``."""
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.as_tensor(data)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def to_numpy(x):
    """Return a host numpy copy of a backend array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    import numpy as np
    return np.asarray(x)


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
