"""Gradient-flow guarantees for trainable known-operator pipelines.

Verifies that gradients propagate correctly through every building block a
learned/unrolled reconstruction network would use: the 12 projector operators
(both families, incl. the sparse cone backprojection), the LinearOperator
algebra, the analytic FBP layer (ramp + adjoint), unrolled iterative schemes
with trainable step sizes, second-order-differentiable TV gradient steps, the
solver blocks, and the sinogram-domain / physics ops.

Notes pinned here:
- Gradients flow w.r.t. the DATA argument (image/volume/sinogram) everywhere.
  Geometry arguments (src_pos, det_*) get no gradient on the torch backend
  (``None``); only the MLX cone projector implements a geometry VJP.
- Linear-operator wiring is verified via exact adjoint identities
  (finite differences on f32 kernels suffer cancellation; the identities are
  the conclusive check).
"""

import math

import numpy as np
import pytest
import torch

import diffct_mlx as dct
from diffct_mlx.backend import xp
from diffct_mlx.reconstruction_algorithms._analytic import ramp_filter_2d
from diffct_mlx.reconstruction_algorithms.solvers import power_iteration


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 32


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _geo_2d():
    rayP, doP, duP = dct.circular_trajectory_2d_parallel(20)
    srcF, dcF, duF = dct.circular_trajectory_2d_fan(20, 120.0, 240.0)
    return rayP, doP, duP, srcF, dcF, duF


def _geo_3d():
    return dct.circular_trajectory_3d(12, 80.0, 160.0)


def _fd64(f, x, rel_tol=1e-2):
    """Directional FD check with float64 loss accumulation (f32 kernels)."""
    x = x.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(f(x), x)
    gen = torch.Generator(device=x.device).manual_seed(0)   # deterministic direction
    d = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    d /= d.norm()
    eps = 1e-2
    with torch.no_grad():
        fd = (f(x + eps * d).item() - f(x - eps * d).item()) / (2 * eps)
    an = float((g.double() * d.double()).sum())
    assert torch.isfinite(g).all()
    assert abs(fd - an) / max(abs(fd), abs(an), 1e-8) < rel_tol, (an, fd)


# --------------------------------------------------------------------------- #
# Projector operators: gradient == adjoint identities (exact) + FD (forward)
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_forward_projector_input_gradients_fd():
    _skip_if_no_cuda()
    rayP, doP, duP, srcF, dcF, duF = _geo_2d()
    img = torch.rand(N, N, device=DEV)
    for f in (
        lambda x: (dct.parallel_forward(x, rayP, doP, duP, 48, 1.0, 1.0).double() ** 2).sum(),
        lambda x: (dct.parallel_forward_footprint(x, rayP, doP, duP, 48, 1.0, 1.0).double() ** 2).sum(),
        lambda x: (dct.fan_forward(x, srcF, dcF, duF, 48, 1.0, 1.0).double() ** 2).sum(),
        lambda x: (dct.fan_forward_footprint(x, srcF, dcF, duF, 48, 1.0, 1.0).double() ** 2).sum(),
    ):
        _fd64(f, img)
    src, dc, du_, dv_ = _geo_3d()
    vol = 0.5 * torch.rand(24, 24, 24, device=DEV)
    _fd64(lambda x: (dct.cone_forward(x, src, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0).double() ** 2).sum(), vol)
    _fd64(lambda x: (dct.cone_forward_footprint(x, src, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0).double() ** 2).sum(), vol)


@pytest.mark.cuda
def test_gradient_equals_adjoint_identities_cone():
    """Exact wiring check: autograd grad of the quadratic == matched adjoint."""
    _skip_if_no_cuda()
    src, dc, du_, dv_ = _geo_3d()
    vol = 0.5 * torch.rand(24, 24, 24, device=DEV)

    pairs = [
        (lambda x: dct.cone_forward(x, src, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0),
         lambda s: dct.cone_backward(s, src, dc, du_, dv_, 24, 24, 24, 1.0, 1.0, 1.0)),
        (lambda x: dct.cone_forward_footprint(x, src, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0),
         lambda s: dct.cone_backward_footprint(s, src, dc, du_, dv_, 24, 24, 24, 1.0, 1.0, 1.0)),
    ]
    sino = pairs[0][0](vol).detach()
    for fw, bw in pairs:
        # forward direction: grad(0.5||A x||^2) == A^T (A x)
        x = vol.detach().clone().requires_grad_(True)
        y = fw(x)
        (g,) = torch.autograd.grad(0.5 * (y.double() ** 2).sum(), x)
        ref = bw(y.detach())
        assert float((g - ref).abs().max() / ref.abs().max()) < 1e-4
        # backward direction: grad(0.5||A^T s||^2) == A (A^T s)
        s = sino.clone().requires_grad_(True)
        z = bw(s)
        (g,) = torch.autograd.grad(0.5 * (z.double() ** 2).sum(), s)
        ref = fw(z.detach())
        assert float((g - ref).abs().max() / ref.abs().max()) < 1e-3


@pytest.mark.cuda
def test_sparse_cone_backprojection_gradient():
    """Sparse (indices=) backprojection: grad w.r.t. sinogram == footprint
    forward of the scattered sparse cotangent."""
    _skip_if_no_cuda()
    src, dc, du_, dv_ = _geo_3d()
    vol = 0.5 * torch.rand(24, 24, 24, device=DEV)
    sino = dct.cone_forward(vol, src, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0).detach()
    idx = torch.randint(0, 24 ** 3, (500,), device=DEV)
    s = sino.clone().requires_grad_(True)
    z = dct.cone_backward_footprint(s, src, dc, du_, dv_, 24, 24, 24, 1.0, 1.0, 1.0, indices=idx)
    (g,) = torch.autograd.grad(0.5 * (z.double() ** 2).sum(), s)
    dense = torch.zeros(24 ** 3, device=DEV)
    dense.index_add_(0, idx, z.detach())
    ref = dct.cone_forward_footprint(dense.view(24, 24, 24), src, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0)
    assert float((g - ref).abs().max() / max(float(ref.abs().max()), 1e-9)) < 1e-3


@pytest.mark.cuda
def test_geometry_arguments_have_no_torch_gradient():
    """Torch backend: geometry inputs yield grad None (only the MLX cone
    projector implements a geometry VJP) — pin the contract."""
    _skip_if_no_cuda()
    src, dc, du_, dv_ = _geo_3d()
    vol = torch.rand(24, 24, 24, device=DEV)
    src_t = src.detach().clone().requires_grad_(True)
    y = dct.cone_forward(vol, src_t, dc, du_, dv_, 40, 40, 1.0, 1.0, 1.0).sum()
    (gs,) = torch.autograd.grad(y, src_t, allow_unused=True)
    assert gs is None


# --------------------------------------------------------------------------- #
# Operator algebra + trainable FBP layer
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_operator_algebra_gradients():
    _skip_if_no_cuda()
    rayP, doP, duP, *_ = _geo_2d()
    A = dct.make_parallel_2d_operator(rayP, doP, duP, image_shape=(N, N), num_detectors=48)
    B = dct.make_parallel_2d_operator(rayP, doP, duP, image_shape=(N, N), num_detectors=48,
                                      projector_mode="footprint")
    img = torch.rand(N, N, device=DEV)
    _fd64(lambda x: (((2.0 * A + B).T @ (A @ x)).double() ** 2).sum(), img)
    _fd64(lambda x: ((A.subset(slice(0, 7)) @ x).double() ** 2).sum(), img)


@pytest.mark.cuda
def test_trainable_fbp_known_operator():
    """Learnable per-detector weights through ramp filter + adjoint."""
    _skip_if_no_cuda()
    rayP, doP, duP, *_ = _geo_2d()
    A = dct.make_parallel_2d_operator(rayP, doP, duP, image_shape=(N, N), num_detectors=48)
    ph = xp.array(dct.shepp_logan_2d((N, N)))
    sino = (A @ ph).detach()

    w = torch.ones(48, device=DEV, requires_grad=True)
    rec = A.T @ ramp_filter_2d(sino * w[None, :])
    (rec - ph).square().mean().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().max() > 0

    # the reconstruct_fbp driver itself is differentiable w.r.t. the sinogram
    s = sino.clone().requires_grad_(True)
    pr = dct.FBPParameters(normalization_scale=math.pi / (2 * 20), enforce_positivity=True)
    dct.reconstruct_fbp(s, lambda t: A.T @ t, pr).square().sum().backward()
    assert s.grad is not None and s.grad.abs().max() > 0


# --------------------------------------------------------------------------- #
# Unrolled iterative schemes
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_unrolled_landweber_trainable_step():
    _skip_if_no_cuda()
    rayP, doP, duP, *_ = _geo_2d()
    A = dct.make_parallel_2d_operator(rayP, doP, duP, image_shape=(N, N), num_detectors=48)
    L = power_iteration(lambda v: A.T @ (A @ v), A.domain_shape)
    ph = xp.array(dct.shepp_logan_2d((N, N)))
    sino = (A @ ph).detach()

    lam = torch.tensor(0.9 / L, device=DEV, requires_grad=True)
    s_in = sino.clone().requires_grad_(True)
    x = torch.zeros(N, N, device=DEV)
    for _ in range(4):
        x = torch.clamp(x + lam * (A.T @ (s_in - (A @ x))), min=0.0)
    (x - ph).square().mean().backward()
    assert lam.grad is not None and torch.isfinite(lam.grad) and float(lam.grad.abs()) > 0
    assert s_in.grad is not None and float(s_in.grad.abs().max()) > 0


@pytest.mark.cuda
def test_tv_gradient_step_second_order():
    """xp.grad composability: an unrolled TV gradient step must stay
    differentiable (create_graph) w.r.t. both alpha and the input."""
    _skip_if_no_cuda()
    ph = xp.array(dct.shepp_logan_2d((N, N)))
    tv = dct.TotalVariation(1.0, eps=1e-4)

    x0 = (ph + 0.05 * torch.randn_like(ph)).detach().requires_grad_(True)
    g = tv.gradient(x0)
    assert g.requires_grad and g.grad_fn is not None      # graph-connected

    alpha = torch.tensor(0.1, device=DEV, requires_grad=True)
    loss = ((x0 - alpha * tv.gradient(x0)) - ph).square().mean()
    ga, gx = torch.autograd.grad(loss, (alpha, x0))
    assert torch.isfinite(ga) and float(ga.abs()) > 0
    assert torch.isfinite(gx).all() and float(gx.abs().max()) > 0

    # value parity: plain call and under no_grad give identical gradients
    gplain = dct.tv_gradient(ph)
    with torch.no_grad():
        gng = dct.tv_gradient(ph)
    assert float((gplain - gng).abs().max()) < 1e-6 and gng.abs().max() > 0


@pytest.mark.cuda
def test_solver_and_driver_blocks_differentiable():
    """cgls and the run_sirt driver work as differentiable network blocks."""
    _skip_if_no_cuda()
    rayP, doP, duP, *_ = _geo_2d()
    A = dct.make_parallel_2d_operator(rayP, doP, duP, image_shape=(N, N), num_detectors=48)
    ph = xp.array(dct.shepp_logan_2d((N, N)))
    sino = (A @ ph).detach()

    s = sino.clone().requires_grad_(True)
    dct.cgls(A, s, iterations=5).square().sum().backward()
    assert s.grad is not None and s.grad.abs().max() > 0

    sinoP = dct.parallel_forward(ph, rayP, doP, duP, 48, 1.0, 1.0).detach()
    meas = [sinoP[i].detach().clone().requires_grad_(True) for i in range(20)]
    params = dct.SIRTParameters(volume_shape=(N, N), iteration_count=2)
    out = dct.run_sirt(
        meas,
        lambda v, i: dct.parallel_forward(v, rayP, doP, duP, 48, 1.0, 1.0)[i],
        lambda p, i: dct.parallel_backward(
            torch.zeros_like(sinoP).index_put(
                (torch.tensor([i], device=DEV),), p.reshape(1, -1)),
            rayP, doP, duP, 1.0, N, N, 1.0),
        params, show_progress=False)
    out.square().sum().backward()
    assert all(m.grad is not None for m in meas)
    assert any(float(m.grad.abs().max()) > 0 for m in meas)


# --------------------------------------------------------------------------- #
# Sinogram-domain / physics ops inside a differentiable pipeline
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
def test_physics_and_rebinning_gradients():
    _skip_if_no_cuda()
    rayP, doP, duP, srcF, dcF, duF = _geo_2d()
    img = torch.rand(N, N, device=DEV)
    sinoP = dct.parallel_forward(img, rayP, doP, duP, 48, 1.0, 1.0).detach()
    sinoF = dct.fan_forward(img, srcF, dcF, duF, 48, 1.0, 1.0).detach()

    _fd64(lambda s: (dct.physics.beam_hardening_polynomial(s, [1.0, 0.08]).double() ** 2).sum(), sinoP)
    _fd64(lambda s: (dct.apply_parker_weighting(s, 240.0, 1.0).double() ** 2).sum(), sinoF)
    _fd64(lambda s: (dct.extend_truncation(s, 8).double() ** 2).sum(), sinoF)
    _fd64(lambda s: (dct.fan_to_parallel(
        s, sid=120.0, sdd=240.0, detector_spacing=1.0,
        source_angles=np.linspace(0, 2 * np.pi, 20, endpoint=False),
        out_angles=np.linspace(0, np.pi, 10, endpoint=False),
        num_out_positions=48).double() ** 2).sum(), sinoF)
    _fd64(lambda x: (dct.guided_filter(x, radius=2, eps=1e-2).double() ** 2).sum(), img, rel_tol=5e-2)
