"""Detector-pixel and line-integral conventions of the operator factories.

The kernels place detector pixel i at ``(i - N/2)·du`` and return line
integrals in voxel units (the upstream diffct conventions). DiffCT-MLX 1.x and
scanner geometry files use ``(i - (N-1)/2)·du`` and physical line integrals.
Since 3.0 the kernels use ``detector_convention="centered"`` and
``line_integrals="physical"``; ``"legacy"`` and ``"voxel"`` reproduce 2.x.
These tests pin both against exact analytic line integrals,
adjointness and the simulation chain.

Backend-neutral: runs on MLX (Apple Silicon) and torch/CUDA alike.
"""

import numpy as np
import pytest

import diffct_mlx as dct
from diffct_mlx.backend import xp, active as _b
from diffct_mlx.reconstruction_algorithms.cases import make_cone_3d_operators


def _skip_if_unusable():
    if dct.backend == "torch":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA device required")


_START = 0.13
_N = 64
_R = 20.0  # disk / sphere radius in voxels


def _centered_disk_2d():
    g = np.arange(_N, dtype=np.float32)
    c = (_N - 1) / 2.0
    X, Y = np.meshgrid(g, g, indexing="ij")
    return xp.array((((X - c) ** 2 + (Y - c) ** 2) <= _R**2).astype(np.float32))


def _centered_sphere_3d(n=_N, r=_R):
    g = np.arange(n, dtype=np.float32)
    c = (n - 1) / 2.0
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return xp.array((((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) <= r**2).astype(np.float32))


def _cone_geometry(n_views=4):
    return dct.circular_trajectory_3d(n_views, 300.0, 500.0, start_angle=_START)


# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #

def test_kernel_detector_center_shift_and_validation():
    c = np.zeros((2, 3), np.float32)
    u = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (2, 1))
    v = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (2, 1))
    out = dct.kernel_detector_center(c, u, 0.4, v, 0.6, convention="legacy")
    np.testing.assert_allclose(out, [[-0.2, 0.0, -0.3]] * 2, atol=1e-7)
    alias = dct.kernel_detector_center(c, u, 0.4, v, 0.6, convention="kernel")  # 2.3 name
    np.testing.assert_allclose(alias, out, atol=0)
    assert dct.kernel_detector_center(c, u, 0.4, convention="centered") is c
    with pytest.raises(ValueError):
        dct.kernel_detector_center(c, u, 0.4, convention="centre")
    with pytest.raises(ValueError):
        dct.kernel_detector_center(c, u, 0.4, v)  # dv missing
    with pytest.raises(ValueError):
        dct.line_integral_scale("mm", 2.0)


# --------------------------------------------------------------------------- #
# detector pixel centres
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["siddon", "footprint"])
def test_parallel_centered_matches_analytic_chords(mode):
    """A centred disk: centered convention reproduces 2*sqrt(R^2 - t^2)."""
    _skip_if_unusable()
    n_det, du = 80, 0.7
    rd, do, uv = dct.circular_trajectory_2d_parallel(6, start_angle=_START)
    img = _centered_disk_2d()
    t_sym = (np.arange(n_det) - (n_det - 1) / 2) * du
    exact = 2.0 * np.sqrt(np.clip(_R**2 - t_sym**2, 0.0, None))

    errs = {}
    for conv in ("legacy", "centered"):
        A = dct.make_parallel_2d_operator(
            rd, do, uv, image_shape=(_N, _N), num_detectors=n_det, detector_spacing=du,
            projector_mode=mode, detector_convention=conv,
        )
        sino = _b.to_numpy(A @ img)
        errs[conv] = np.linalg.norm(sino - exact[None]) / (np.linalg.norm(exact) * np.sqrt(len(sino)))
        if conv == "centered":
            # a centred object projects mirror-symmetric about (N-1)/2
            np.testing.assert_allclose(sino, sino[:, ::-1], atol=0.05 * exact.max())
    assert errs["centered"] < 0.6 * errs["legacy"], errs


def test_cone_centered_centroid_and_analytic_profile():
    _skip_if_unusable()
    n_u, n_v, du, dv = 72, 70, 1.1, 1.3
    src, dc, uv, vv = _cone_geometry()
    vol = _centered_sphere_3d()
    for conv, expected in (("legacy", (n_u / 2, n_v / 2)), ("centered", ((n_u - 1) / 2, (n_v - 1) / 2))):
        A = dct.make_cone_3d_operator(
            src, dc, uv, vv, volume_shape=(_N, _N, _N), detector_shape=(n_u, n_v),
            du=du, dv=dv, projector_mode="footprint", detector_convention=conv,
        )
        sino = _b.to_numpy(A @ vol)  # (views, u, v)
        iu, iv = np.arange(n_u), np.arange(n_v)
        for view in sino:
            cu = (view.sum(1) @ iu) / view.sum()
            cv = (view.sum(0) @ iv) / view.sum()
            assert abs(cu - expected[0]) < 0.08, (conv, cu)
            assert abs(cv - expected[1]) < 0.08, (conv, cv)


# --------------------------------------------------------------------------- #
# line-integral units
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["siddon", "footprint"])
def test_physical_units_scale_and_adjointness(mode):
    _skip_if_unusable()
    n, s = 32, 0.25
    src, dc, uv, vv = _cone_geometry(3)
    kw = dict(volume_shape=(n, n, n), detector_shape=(40, 40), du=0.4, dv=0.4,
              voxel_spacing=s, projector_mode=mode, detector_convention="centered")
    A_vox = dct.make_cone_3d_operator(src, dc, uv, vv, line_integrals="voxel", **kw)
    A_phy = dct.make_cone_3d_operator(src, dc, uv, vv, line_integrals="physical", **kw)
    assert A_vox.to_physical == pytest.approx(s) and A_phy.to_physical == 1.0
    assert A_phy.subset([0, 2]).line_integrals == "physical"

    rng = np.random.default_rng(0)
    x = xp.array(rng.random((n, n, n), dtype=np.float32))
    y = xp.array(rng.random((3, 40, 40), dtype=np.float32))
    Ax_v, Ax_p = _b.to_numpy(A_vox @ x), _b.to_numpy(A_phy @ x)
    np.testing.assert_allclose(Ax_p, Ax_v * s, rtol=1e-5, atol=1e-6)

    lhs = float(np.sum(Ax_p * _b.to_numpy(y)))
    rhs = float(np.sum(_b.to_numpy(x) * _b.to_numpy(A_phy.T @ y)))
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_physical_units_sphere_chord_in_mm():
    """Central ray through a sphere of radius R voxels: 2*R*voxel_spacing mm."""
    _skip_if_unusable()
    s, n_u = 0.5, 65
    src, dc, uv, vv = _cone_geometry(1)
    A = dct.make_cone_3d_operator(
        src, dc, uv, vv, volume_shape=(_N, _N, _N), detector_shape=(n_u, n_u),
        du=0.5, dv=0.5, voxel_spacing=s, projector_mode="siddon",
        detector_convention="centered", line_integrals="physical",
    )
    centre = _b.to_numpy(A @ _centered_sphere_3d())[0, n_u // 2, n_u // 2]
    assert centre == pytest.approx(2 * _R * s, rel=0.03)


def test_simulate_scan_identical_for_both_unit_conventions():
    _skip_if_unusable()
    n, s = 32, 0.3
    src, dc, uv, vv = _cone_geometry(3)
    kw = dict(volume_shape=(n, n, n), detector_shape=(36, 36), du=0.5, dv=0.5, voxel_spacing=s)
    phantom = 0.02 * _b.to_numpy(_centered_sphere_3d(n, 10.0))
    sinos = [
        _b.to_numpy(dct.simulate_scan(
            phantom, dct.make_cone_3d_operator(src, dc, uv, vv, line_integrals=u, **kw),
            poisson=False,
        ))
        for u in ("voxel", "physical")
    ]
    np.testing.assert_allclose(sinos[0], sinos[1], rtol=1e-5, atol=1e-6)


def test_functional_builders_follow_the_options():
    _skip_if_unusable()
    n, s = 32, 0.25
    src, dc, uv, vv = _cone_geometry(3)
    kw = dict(volume_shape=(n, n, n), detector_shape=(40, 40), du=0.4, dv=0.4, voxel_spacing=s,
              projector_mode="footprint", detector_convention="centered", line_integrals="physical")
    A = dct.make_cone_3d_operator(src, dc, uv, vv, **kw)
    fwd, back, back_all = make_cone_3d_operators(src, dc, uv, vv, **kw)
    x = xp.array(np.random.default_rng(1).random((n, n, n), dtype=np.float32))
    full = _b.to_numpy(A @ x)
    np.testing.assert_allclose(_b.to_numpy(fwd(x, 1)), full[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_b.to_numpy(fwd.project_slice(x, 0, 3)), full, rtol=1e-5, atol=1e-6)
    y = A @ x
    np.testing.assert_allclose(_b.to_numpy(back_all(y)), _b.to_numpy(A.T @ y), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        _b.to_numpy(back(y[1], 1)), _b.to_numpy(A.subset([1]).T @ y[1:2]), rtol=1e-4, atol=1e-4)
