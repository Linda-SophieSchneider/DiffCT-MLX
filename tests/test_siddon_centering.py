"""Regression tests for the Siddon projector geometry convention.

Pins the fix of the vendored Metal Siddon kernels, which interpolated the
image/volume on nodes at integer indices instead of accumulating the
traversed cell — effectively shifting the volume by half a voxel per axis
against the footprint kernels and the standardized convention (cell k spans
``[k - c, k + 1 - c)``, center at ``k + 0.5 - c``). A correct projector maps
a perfectly centered object to a view-independent sinogram centered at
``n_det / 2``; the broken kernels oscillated by up to ±1.24 px with the view
angle. Also pins the removed stray ``voxel_spacing`` factor that made the
Metal cone forward/backward pair non-adjoint for ``voxel_spacing != 1``.

Backend-neutral: runs on MLX (Apple Silicon) and torch/CUDA alike.
"""

import math

import numpy as np
import pytest

import diffct_mlx as dct
from diffct_mlx.backend import xp, active as _b


def _skip_if_unusable():
    if dct.backend == "torch":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA device required")


# Avoid exactly axis-aligned views: there the disk edge falls onto voxel
# boundaries and the half-open floor() interval biases the centroid by
# ±0.5 px in both engines (shared, convention-independent behavior).
_START = 0.13

_N = 64
_NDET = 96


def _centered_disk_2d():
    g = np.arange(_N, dtype=np.float32)
    c = (_N - 1) / 2.0
    X, Y = np.meshgrid(g, g, indexing="ij")
    return xp.array((((X - c) ** 2 + (Y - c) ** 2) <= 20.0**2).astype(np.float32))


def _centered_sphere_3d():
    g = np.arange(_N, dtype=np.float32)
    c = (_N - 1) / 2.0
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    inside = ((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) <= 20.0**2
    return xp.array(inside.astype(np.float32))


def _detector_centroids(sino_np, axis):
    """Per-view centroid along one detector axis of an (n_views, ...) array."""
    idx = np.arange(sino_np.shape[axis], dtype=np.float64)
    out = []
    for view in sino_np:
        profile = view.sum(axis=tuple(a for a in range(view.ndim) if a != axis - 1))
        out.append(float((profile @ idx) / profile.sum()))
    return out


def test_parallel_siddon_centered_disk_projects_centered():
    _skip_if_unusable()
    img = _centered_disk_2d()
    rd, do, du = dct.circular_trajectory_2d_parallel(8, start_angle=_START)
    sino = _b.to_numpy(dct.parallel_forward(img, rd, do, du, _NDET, 1.0))
    cents = _detector_centroids(sino, axis=1)
    assert max(abs(c - _NDET / 2) for c in cents) < 0.1
    assert max(cents) - min(cents) < 0.1


def test_fan_siddon_centered_disk_projects_centered():
    _skip_if_unusable()
    img = _centered_disk_2d()
    src, det_c, det_u = dct.circular_trajectory_2d_fan(8, 400.0, 700.0, start_angle=_START)
    sino = _b.to_numpy(dct.fan_forward(img, src, det_c, det_u, _NDET, 1.0))
    cents = _detector_centroids(sino, axis=1)
    assert max(abs(c - _NDET / 2) for c in cents) < 0.1
    assert max(cents) - min(cents) < 0.1


def test_cone_siddon_centered_sphere_projects_centered():
    _skip_if_unusable()
    vol = _centered_sphere_3d()
    src, det_c, du_v, dv_v = dct.circular_trajectory_3d(8, 400.0, 700.0, start_angle=_START)
    sino = _b.to_numpy(
        dct.cone_forward(vol, src, det_c, du_v, dv_v, _NDET, _NDET, 1.0, 1.0)
    )
    for axis in (1, 2):
        cents = _detector_centroids(sino, axis=axis)
        assert max(abs(c - _NDET / 2) for c in cents) < 0.1
        assert max(cents) - min(cents) < 0.1


def test_cone_siddon_adjoint_holds_for_anisotropic_voxel_spacing():
    """Forward had a stray ``voxel_spacing`` factor its adjoint lacked."""
    _skip_if_unusable()
    rng = np.random.default_rng(0)
    src, det_c, du_v, dv_v = dct.circular_trajectory_3d(12, 200.0, 400.0)
    x = xp.array(rng.standard_normal((32, 32, 32)).astype(np.float32))
    y = xp.array(rng.standard_normal((12, 48, 48)).astype(np.float32))
    for vs in (1.0, 0.7):
        Ax = dct.cone_forward(x, src, det_c, du_v, dv_v, 48, 48, 1.0, 1.0, vs)
        Aty = dct.cone_backward(y, src, det_c, du_v, dv_v, 32, 32, 32, 1.0, 1.0, vs)
        lhs = float((Ax * y).sum())
        rhs = float((x * Aty).sum())
        assert math.isfinite(lhs) and abs(lhs) > 0
        assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_cone_siddon_matches_footprint_on_smooth_phantom():
    """Both cone projector modes must share one geometry convention."""
    _skip_if_unusable()
    phantom = dct.shepp_logan_phantom(fov_radius=32.0)
    vol = phantom.voxelize((_N, _N, _N), voxel_spacing=1.0)
    src, det_c, du_v, dv_v = dct.circular_trajectory_3d(24, 400.0, 700.0, start_angle=_START)
    sid = _b.to_numpy(dct.cone_forward(vol, src, det_c, du_v, dv_v, _NDET, _NDET, 1.0, 1.0))
    fpt = _b.to_numpy(
        dct.cone_forward_footprint(vol, src, det_c, du_v, dv_v, _NDET, _NDET, 1.0, 1.0)
    )
    corr = float(np.corrcoef(sid.ravel(), fpt.ravel())[0, 1])
    assert corr > 0.99
