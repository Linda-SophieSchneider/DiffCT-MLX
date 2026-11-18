"""Amplitude contract of the separable-footprint projectors.

The footprint kernels used to distribute every voxel with weights that sum to
one over the detector cells it touches (``overlap / footprint_width``). The
result was not a line integral: the cone output scaled with
``(detector cell size at the isocentre / voxel size)**2``, and the parallel and
fan outputs additionally carried a view-angle dependent factor of up to
``sqrt(2)``. The kernels now use a mass-conserving amplitude, so a footprint
sinogram is a voxel-unit line integral averaged over the detector cell, on the
same scale as the Siddon projector, for any detector spacing, voxel spacing and
magnification. These tests pin that contract and the adjointness of the pairs.

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


def _disc(n, voxel_spacing, radius_mm):
    ax = (np.arange(n, dtype=np.float32) - 0.5 * (n - 1)) * voxel_spacing
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return xp.array(((xx * xx + yy * yy) <= radius_mm**2).astype(np.float32))


def _sphere(n, voxel_spacing, radius_mm, z_offset_mm=0.0):
    ax = (np.arange(n, dtype=np.float32) - 0.5 * (n - 1)) * voxel_spacing
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
    inside = (xx * xx + yy * yy + (zz - z_offset_mm) ** 2) <= radius_mm**2
    return xp.array(inside.astype(np.float32))


def _per_view_mass_ratio(footprint, siddon):
    f = _b.to_numpy(footprint).astype(np.float64)
    s = _b.to_numpy(siddon).astype(np.float64)
    axes = tuple(range(1, f.ndim))
    return f.sum(axis=axes) / s.sum(axis=axes)


def _central_ratio(footprint, siddon, half=2):
    f = _b.to_numpy(footprint).astype(np.float64)
    s = _b.to_numpy(siddon).astype(np.float64)
    sl = [slice(None)] + [slice(n // 2 - half, n // 2 + half) for n in f.shape[1:]]
    axes = tuple(range(1, f.ndim))
    return f[tuple(sl)].mean(axis=axes) / s[tuple(sl)].mean(axis=axes)


# (image size, voxel spacing, detector cells, detector spacing)
_CASES_2D = [(128, 1.0, 192, 1.0), (128, 1.0, 384, 0.5), (128, 1.0, 96, 2.0), (256, 0.5, 384, 0.5)]


@pytest.mark.parametrize("n,vs,n_det,ds", _CASES_2D)
def test_parallel_footprint_matches_siddon_amplitude(n, vs, n_det, ds):
    _skip_if_unusable()
    img = _disc(n, vs, 20.0)
    ray, det_o, det_u = dct.circular_trajectory_2d_parallel(8)  # 0, 22.5, 45, ... degrees
    sid = dct.parallel_forward(img, ray, det_o, det_u, n_det, ds, vs)
    fpt = dct.parallel_forward_footprint(img, ray, det_o, det_u, n_det, ds, vs)
    assert np.allclose(_per_view_mass_ratio(fpt, sid), 1.0, atol=0.01)
    assert np.allclose(_central_ratio(fpt, sid), 1.0, atol=0.02)


@pytest.mark.parametrize("n,vs,n_det,ds", [(128, 1.0, 256, 1.0), (128, 1.0, 512, 0.5), (128, 1.0, 128, 2.0), (256, 0.5, 512, 0.5)])
def test_fan_footprint_matches_siddon_amplitude(n, vs, n_det, ds):
    _skip_if_unusable()
    img = _disc(n, vs, 20.0)
    src, det_c, det_u = dct.circular_trajectory_2d_fan(8, 500.0, 900.0)
    sid = dct.fan_forward(img, src, det_c, det_u, n_det, ds, vs)
    fpt = dct.fan_forward_footprint(img, src, det_c, det_u, n_det, ds, vs)
    assert np.allclose(_per_view_mass_ratio(fpt, sid), 1.0, atol=0.01)
    assert np.allclose(_central_ratio(fpt, sid), 1.0, atol=0.02)


# (volume size, voxel spacing, sid, sdd, detector cells, detector spacing)
_CASES_3D = [
    (96, 1.0, 500.0, 900.0, 192, 0.5),    # detector cell at isocentre much smaller than a voxel
    (96, 1.0, 500.0, 900.0, 64, 2.5),     # ... larger than a voxel
    (128, 0.6, 500.0, 900.0, 128, 1.0),   # non-unit voxel spacing
    (96, 0.8, 250.0, 1000.0, 160, 2.0),   # magnification 4
]


@pytest.mark.parametrize("n,vs,sid,sdd,n_det,ds", _CASES_3D)
def test_cone_footprint_matches_siddon_amplitude(n, vs, sid, sdd, n_det, ds):
    _skip_if_unusable()
    vol = _sphere(n, vs, 20.0)
    src, det_c, det_u, det_v = dct.circular_trajectory_3d(6, sid, sdd, start_angle=0.13)
    sidd = dct.cone_forward(vol, src, det_c, det_u, det_v, n_det, n_det, ds, ds, vs)
    fpt = dct.cone_forward_footprint(vol, src, det_c, det_u, det_v, n_det, n_det, ds, ds, vs)
    assert np.allclose(_per_view_mass_ratio(fpt, sidd), 1.0, atol=0.01)
    assert np.allclose(_central_ratio(fpt, sidd), 1.0, atol=0.02)
    # absolute scale: a central chord of 40 mm is 40 / vs in voxel units
    centre = float(_central_ratio(fpt, sidd)[0] * _b.to_numpy(sidd)[0, n_det // 2 - 2 : n_det // 2 + 2, n_det // 2 - 2 : n_det // 2 + 2].mean())
    assert abs(centre * vs / 40.0 - 1.0) < 0.03


def test_cone_footprint_amplitude_holds_off_axis():
    """Oblique incidence on the flat detector (large cone angle, off-centre object)."""
    _skip_if_unusable()
    n, vs = 96, 1.0
    vol = _sphere(n, vs, 12.0, z_offset_mm=28.0)
    src, det_c, det_u, det_v = dct.circular_trajectory_3d(6, 150.0, 300.0, start_angle=0.13)
    sidd = dct.cone_forward(vol, src, det_c, det_u, det_v, 256, 256, 1.0, 1.0, vs)
    fpt = dct.cone_forward_footprint(vol, src, det_c, det_u, det_v, 256, 256, 1.0, 1.0, vs)
    assert np.allclose(_per_view_mass_ratio(fpt, sidd), 1.0, atol=0.015)


def _rel_adjoint_error(Ax, y, x, Aty):
    lhs = float((Ax * y).sum())
    rhs = float((x * Aty).sum())
    assert math.isfinite(lhs) and abs(lhs) > 0
    return abs(lhs - rhs) / abs(lhs)


def test_footprint_pairs_stay_adjoint_for_general_spacings():
    _skip_if_unusable()
    rng = np.random.default_rng(0)
    for vs, ds in ((1.0, 1.0), (0.7, 1.3), (1.4, 0.5)):
        x2 = xp.array(rng.standard_normal((40, 40)).astype(np.float32))
        y2 = xp.array(rng.standard_normal((9, 72)).astype(np.float32))
        ray, det_o, det_u = dct.circular_trajectory_2d_parallel(9)
        err = _rel_adjoint_error(
            dct.parallel_forward_footprint(x2, ray, det_o, det_u, 72, ds, vs), y2,
            x2, dct.parallel_backward_footprint(y2, ray, det_o, det_u, ds, 40, 40, vs))
        assert err < 5e-4
        src, det_c, det_u = dct.circular_trajectory_2d_fan(9, 200.0, 400.0)
        err = _rel_adjoint_error(
            dct.fan_forward_footprint(x2, src, det_c, det_u, 72, ds, vs), y2,
            x2, dct.fan_backward_footprint(y2, src, det_c, det_u, ds, 40, 40, vs))
        assert err < 5e-4
        x3 = xp.array(rng.standard_normal((24, 24, 24)).astype(np.float32))
        y3 = xp.array(rng.standard_normal((7, 40, 40)).astype(np.float32))
        src, det_c, det_u, det_v = dct.circular_trajectory_3d(7, 200.0, 400.0)
        err = _rel_adjoint_error(
            dct.cone_forward_footprint(x3, src, det_c, det_u, det_v, 40, 40, ds, ds, vs), y3,
            x3, dct.cone_backward_footprint(y3, src, det_c, det_u, det_v, 24, 24, 24, ds, ds, vs))
        assert err < 5e-4
