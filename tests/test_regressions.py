"""Regression tests for the 2026-07 review/debug session.

Each test pins a specific fixed defect; see CHANGELOG for the full list.
All run on the torch backend (CUDA where needed).
"""

import math

import numpy as np
import pytest
import torch

import diffct_mlx as dct
from diffct_mlx.backend import xp, active as _b


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


# --------------------------------------------------------------------------- #
# backend shim
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_tv_gradient_degenerate_dims_returns_zeros():
    """xp.grad on a graph-disconnected objective must return zeros (mx.grad
    parity), not raise — hit by single-slice volumes (1, H, W)."""
    _skip_if_no_cuda()
    g = dct.tv_gradient(torch.rand(1, 8, 8, device="cuda"))
    assert float(g.abs().max()) == 0.0


def test_xp_clip_both_none_is_identity():
    x = torch.rand(5)
    assert xp.clip(x) is x


def test_box_infinite_bounds_prox_identity():
    x = torch.rand(6)
    out = dct.Box(float("-inf"), float("inf")).prox(x, 1.0)
    assert torch.allclose(out, x)


def test_xp_max_axis_returns_values():
    m = xp.max(torch.rand(3, 4), axis=0)
    assert isinstance(m, torch.Tensor) and m.shape == (4,)


def test_xp_has_int64_astype_moveaxis():
    a = xp.arange(10, dtype=xp.int64)
    assert a.dtype == torch.int64
    assert xp.astype(a, xp.float32).dtype == torch.float32
    assert xp.moveaxis(torch.zeros(2, 3, 4), 1, -1).shape == (2, 4, 3)


# --------------------------------------------------------------------------- #
# operators / solvers
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_subset_slice_metadata_consistent():
    _skip_if_no_cuda()
    ray, deto, detu = dct.circular_trajectory_2d_parallel(36)
    A = dct.make_parallel_2d_operator(ray, deto, detu, image_shape=(32, 32),
                                      num_detectors=32)
    S = A.subset(slice(0, 10))
    y = S @ torch.rand(32, 32, device="cuda")
    assert S.n_views == 10
    assert S.range_shape == tuple(y.shape)
    with pytest.raises(TypeError):
        A.subset(7)


@pytest.mark.cuda
def test_rls_honors_weights():
    """rls used to double-pop `weights` and silently run unweighted."""
    _skip_if_no_cuda()
    ray, deto, detu = dct.circular_trajectory_2d_parallel(24)
    A = dct.make_parallel_2d_operator(ray, deto, detu, image_shape=(24, 24),
                                      num_detectors=24)
    x_true = xp.array(dct.shepp_logan_2d((24, 24)))
    b = A @ x_true
    # corrupt half the views; weights that zero them out must change the result
    b_bad = b + 0.0
    b_bad[:12] = b_bad[:12] + 5.0
    w = xp.ones(A.range_shape)
    w[:12] = 0.0
    x_w = dct.rls(A, b_bad, weights=w, beta=1e-4, iterations=30)
    x_u = dct.rls(A, b_bad, beta=1e-4, iterations=30)
    assert float(xp.norm(x_w - x_u)) > 1e-3


@pytest.mark.cuda
def test_rwls_rejects_nonsmooth_regularizer():
    _skip_if_no_cuda()
    ray, deto, detu = dct.circular_trajectory_2d_parallel(12)
    A = dct.make_parallel_2d_operator(ray, deto, detu, image_shape=(16, 16),
                                      num_detectors=16)
    b = xp.zeros(A.range_shape)
    with pytest.raises(ValueError, match="non-smooth"):
        dct.rwls(A, b, regularizer=dct.L1Norm(1.0), iterations=1)


@pytest.mark.cuda
def test_sart_warm_start_uses_measurements_in_first_iteration():
    """Warm-started SART used to no-op iteration 0 (N-1 effective iterations)."""
    _skip_if_no_cuda()
    ray, deto, detu = dct.circular_trajectory_2d_parallel(36)
    A = dct.make_parallel_2d_operator(ray, deto, detu, image_shape=(32, 32),
                                      num_detectors=32)
    ph = xp.array(dct.shepp_logan_2d((32, 32)))
    sino = A @ ph
    measured = [sino[i] for i in range(36)]
    x0 = np.full((32, 32), 0.25, dtype=np.float32)
    params = dct.SARTParameters(volume_shape=(32, 32), iteration_count=1,
                                initial_volume=x0)
    out = dct.run_sart(measured,
                       lambda v, i: (A @ v)[i],
                       lambda p, i: A.subset([i]).T @ xp.reshape(p, (1, -1)),
                       params, show_progress=False)
    # one iteration from a constant warm start must move toward the data
    assert float(xp.norm(out - xp.array(x0))) > 1e-3


# --------------------------------------------------------------------------- #
# rebinning
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_parker_weights_match_vendored_reference():
    """rebinning.parker_weights had a factor-2 feathering error."""
    _skip_if_no_cuda()
    from diffct_mlx.rebinning import parker_weights, detector_fan_angles
    from diffct.analytical import parker_weights as parker_ref

    sdd, spacing, nd, nv = 600.0, 1.0, 128, 300
    gam = detector_fan_angles(nd, spacing, sdd)
    gmax = float(torch.max(torch.abs(gam)))
    rng = math.pi + 2 * gmax
    betas = torch.arange(nv, dtype=torch.float32) * (rng / (nv - 1))
    w_new = parker_weights(nv, gam, rng).cpu().numpy()
    w_ref = parker_ref(betas.cuda(), nd, spacing, sdd).cpu().numpy()
    assert np.abs(w_new - w_ref).max() < 1e-4


@pytest.mark.cuda
def test_interp_detector_int64_indices_identity_on_large_array():
    """Flat gather indices used to be built in float32 (rounds above 2^24
    elements): an identity resample of a large stack must be exact."""
    _skip_if_no_cuda()
    from diffct_mlx.rebinning import _interp_detector
    rows, D = 70_000, 256                      # 17.9M elements > 2^24
    sino = torch.rand(rows, D, device="cuda")
    idx = xp.arange(D)                          # identity mapping
    out = _interp_detector(sino, idx)
    assert float((out - sino).abs().max()) == 0.0


@pytest.mark.cuda
def test_curved_flat_resamples_u_axis_for_3d_stacks():
    """(views, u, v) stacks must be rebinned along axis 1 (u), not the last axis."""
    _skip_if_no_cuda()
    from diffct_mlx.rebinning import curved_to_flat
    sino = torch.zeros(4, 32, 8, device="cuda")
    sino[:, 16, :] = 1.0                        # impulse along u
    out = curved_to_flat(sino, sdd=500.0, det_spacing_angle=1e-3)
    assert out.shape == sino.shape
    # the impulse must stay in the u axis (spread over u, constant over v)
    spread_v = float((out.std(dim=2)).max())
    assert spread_v < 1e-5


@pytest.mark.cuda
def test_fan_to_parallel_guards():
    _skip_if_no_cuda()
    from diffct_mlx.rebinning import fan_to_parallel
    one_view = torch.rand(1, 32, device="cuda")
    with pytest.raises(ValueError):
        fan_to_parallel(one_view, sid=500.0, sdd=800.0, detector_spacing=1.0,
                        source_angles=np.array([0.0]), out_angles=np.array([0.0]))


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_scatter_correction_does_not_mix_views():
    _skip_if_no_cuda()
    from diffct_mlx.physics import scatter_correction
    intensity = torch.ones(4, 16, 16, device="cuda")
    intensity[2] = 5.0
    out = scatter_correction(intensity, fraction=0.1, radius=4)
    # each view is constant -> per-view blur is exact -> correction is exact
    assert torch.allclose(out[0], torch.full_like(out[0], 0.9), atol=1e-5)
    assert torch.allclose(out[2], torch.full_like(out[2], 4.5), atol=1e-5)


@pytest.mark.cuda
def test_detector_deblur_preserves_dc():
    """Wiener deblur used to shrink all values by ~reg (DC gain 1/(1+reg))."""
    _skip_if_no_cuda()
    from diffct_mlx.physics import detector_deblur
    sino = torch.full((2, 32, 32), 3.0, device="cuda")
    out = detector_deblur(sino, sigma=1.0, reg=1e-2)
    assert float((out - sino).abs().max()) < 1e-4


def test_auto_voxel_spacing_bounds_lateral_axes_by_u_fov():
    from diffct_mlx.real_measured_data_helper import auto_voxel_spacing_from_detector
    # portrait detector: fov_v (vertical) much larger than fov_u
    vs = auto_voxel_spacing_from_detector(
        volume_shape=(64, 256, 64), detector_shape_uv=(128, 512),
        detector_pitch_u_mm=1.0, detector_pitch_v_mm=1.0,
        magnification=2.0, fov_margin_mm=0.0)
    # height (y=256, lateral) must fit the u FOV: (128*1/2)/256 = 0.25
    assert abs(vs - 0.25) < 1e-6


def test_estimate_cone_isocenter_single_view():
    from diffct_mlx.real_measured_data_helper import estimate_cone_isocenter
    iso = estimate_cone_isocenter(np.array([[0.0, -100.0, 0.0]]),
                                  np.array([[0.0, 100.0, 0.0]]))
    assert np.all(np.isfinite(iso))


# --------------------------------------------------------------------------- #
# vendored engine conventions
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_cone_footprint_matches_siddon_detector_grid():
    """Cone footprint kernels used (n-1)/2 detector centering (half-pixel shift
    vs every other family); sinogram centers of mass must now coincide."""
    _skip_if_no_cuda()
    D = H = W = 48
    NV, DU, DV = 6, 144, 144
    vol = torch.as_tensor(np.asarray(
        dct.shepp_logan_3d(D, H, W).detach().cpu() if isinstance(dct.shepp_logan_3d(D, H, W), torch.Tensor)
        else dct.shepp_logan_3d(D, H, W)), dtype=torch.float32, device="cuda")
    src, detc, detu, detv = dct.circular_trajectory_3d(NV, 150.0, 300.0)
    args = (vol, src, detc, detu, detv, DU, DV, 1.0, 1.0, 1.0)
    sS = dct.cone_forward(*args).cpu().numpy()
    sF = dct.cone_forward_footprint(*args).cpu().numpy()
    u = np.arange(DU)
    comS = (sS.sum(axis=(0, 2)) * u).sum() / sS.sum()
    comF = (sF.sum(axis=(0, 2)) * u).sum() / sF.sum()
    assert abs(comS - comF) < 0.05


@pytest.mark.cuda
def test_sparse_cone_footprint_guards():
    _skip_if_no_cuda()
    src, detc, detu, detv = dct.circular_trajectory_3d(4, 150.0, 300.0)
    sino = torch.zeros(4, 16, 16, device="cuda")
    empty = torch.zeros(0, dtype=torch.long, device="cuda")
    out = dct.cone_backward_footprint(sino, src, detc, detu, detv,
                                      16, 16, 16, 1.0, 1.0, 1.0, indices=empty)
    assert out.shape == (0,)
    with pytest.raises(ValueError, match="int32"):
        dct.cone_backward_footprint(sino, src, detc, detu, detv,
                                    1300, 1300, 1300, 1.0, 1.0, 1.0,
                                    indices=torch.zeros(1, dtype=torch.long, device="cuda"))


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_conveyor_propagates_worker_exceptions():
    """A failing compute_fn used to deadlock the conveyor forever."""
    _skip_if_no_cuda()
    from diffct_mlx.orchestration.out_of_core import _conveyor_run

    def read_fn(cid):
        return cid

    def compute_fn(cid, inp, dev):
        raise RuntimeError("boom")

    def write_fn(cid, out):
        pass

    with pytest.raises(RuntimeError, match="conveyor failed"):
        _conveyor_run([(0, 1), (1, 2)], read_fn, compute_fn, write_fn, gpus=[0])


# --------------------------------------------------------------------------- #
# MLX backend static parity (no Apple hardware needed)
# --------------------------------------------------------------------------- #

def test_mlx_backend_static_parity():
    """The MLX xp namespace must expose the same attribute set as the torch one,
    and the Metal adapter must export all 12 unified projector names. Checked
    via AST so it runs without mlx installed."""
    import ast
    from pathlib import Path

    pkg = Path(dct.__file__).parent
    tree = ast.parse((pkg / "backend" / "_mlx.py").read_text())

    xp_keys = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "xp" and isinstance(node.value, ast.Call):
                    xp_keys = {kw.arg for kw in node.value.keywords if kw.arg}
        if isinstance(node, ast.ImportFrom) and node.module and "projectors" in node.module:
            imported |= {a.name for a in node.names}

    torch_keys = {k for k in vars(xp) if not k.startswith("_")}
    assert xp_keys == torch_keys, (
        f"xp parity broken: only-torch={sorted(torch_keys - xp_keys)}, "
        f"only-mlx={sorted(xp_keys - torch_keys)}")

    expected = {
        "parallel_forward", "parallel_backward",
        "parallel_forward_footprint", "parallel_backward_footprint",
        "fan_forward", "fan_backward",
        "fan_forward_footprint", "fan_backward_footprint",
        "cone_forward", "cone_backward",
        "cone_forward_footprint", "cone_backward_footprint",
    }
    assert expected <= imported, f"missing projector exports: {expected - imported}"

    # vendored Metal implementation is present
    for rel in ("backend/metal/projectors.py", "backend/metal/geometry.py",
                "backend/metal/kernels/cone_beam.py"):
        assert (pkg / rel).exists(), rel
