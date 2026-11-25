"""The unified ``dct.Projector`` on the active backend.

Backend-neutral: runs on MLX and on Torch (CPU through the engine's portable
backend, CUDA natively). The footprint cases need MLX or a CUDA device.
"""

import numpy as np
import pytest

import diffct_mlx as dct
from diffct_mlx.backend import active as _b

_RNG = np.random.default_rng(11)


def _cuda():
    if dct.backend != "torch":
        return False
    import torch

    return torch.cuda.is_available()


def _arr(a):
    return _b.as_array(np.asarray(a, dtype=np.float32))


def _np(a):
    return np.asarray(_b.to_numpy(a), dtype=np.float64)


def _rel(a, b):
    a, b = _np(a), _np(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _setup(beam):
    if beam == "parallel":
        traj = dct.circular_trajectory_2d_parallel(9, start_angle=0.21)
        return traj, (40, 40), (60,), 0.7, 0.6
    if beam == "fan":
        traj = dct.circular_trajectory_2d_fan(9, 120.0, 210.0, start_angle=0.21)
        return traj, (40, 40), (80,), 0.7, 0.8
    traj = dct.circular_trajectory_3d(6, 90.0, 160.0, start_angle=0.21)
    return traj, (24, 24, 24), (36, 30), 0.8, (0.9, 1.1)


def _functional(beam, mode, x, traj, det, ds, vs, adjoint=False, vol=None):
    suffix = "_footprint" if mode == "footprint" else ""
    if beam == "cone":
        du, dv = ds
        if adjoint:
            return getattr(dct, f"cone_backward{suffix}")(x, *traj, *vol, du, dv, vs)
        return getattr(dct, f"cone_forward{suffix}")(x, *traj, det[0], det[1], du, dv, vs)
    if adjoint:
        return getattr(dct, f"{beam}_backward{suffix}")(x, *traj, ds, *vol, vs)
    return getattr(dct, f"{beam}_forward{suffix}")(x, *traj, det[0], ds, vs)


def _modes():
    return ["siddon", "footprint"] if dct.backend == "mlx" or _cuda() else ["siddon"]


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("mode", _modes())
def test_projector_matches_functional_api(beam, mode):
    traj, vol, det, vs, ds = _setup(beam)
    x = _arr(_RNG.random(vol))
    P = dct.Projector(traj, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs,
                      projector_mode=mode)
    assert P.projection_shape == (int(traj[0].shape[0]),) + det
    y = P(x)
    assert _rel(y, _functional(beam, mode, x, traj, det, ds, vs)) < 1e-5
    r = _arr(_RNG.random(P.projection_shape))
    assert _rel(P.backproject(r), _functional(beam, mode, r, traj, det, ds, vs, True, vol)) < 1e-5


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_view_chunks_match_one_pass(beam):
    traj, vol, det, vs, ds = _setup(beam)
    x = _arr(_RNG.random(vol))
    P = dct.Projector(traj, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs)
    Pc = dct.Projector(traj, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs,
                       view_chunk_size=4)
    assert _rel(Pc(x), P(x)) < 1e-5
    r = _arr(_RNG.random(P.projection_shape))
    assert _rel(Pc.backproject(r), P.backproject(r)) < 1e-5


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_operator_is_adjoint_and_subsets(beam):
    traj, vol, det, vs, ds = _setup(beam)
    A = dct.Projector(traj, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs).as_operator()
    x = _arr(_RNG.random(vol))
    y = _arr(_RNG.random(A.range_shape))
    lhs = float((_np(A @ x) * _np(y)).sum())
    rhs = float((_np(x) * _np(A.T @ y)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-4
    views = [1, 4, 5]
    assert _rel(A.subset(views) @ x, _np(A @ x)[views]) < 1e-6
    assert A.subset(slice(0, 3)).range_shape == (3,) + det


def test_legacy_conventions_match_operator_factory():
    traj, vol, det, vs, (du, dv) = _setup("cone")
    kw = dict(detector_convention="legacy", line_integrals="voxel")
    A = dct.Projector(traj, vol, det, detector_spacing=(du, dv), voxel_spacing=vs, **kw).as_operator()
    B = dct.make_cone_3d_operator(*traj, volume_shape=vol, detector_shape=det, du=du, dv=dv,
                                  voxel_spacing=vs, projector_mode="siddon", **kw)
    x = _arr(_RNG.random(vol))
    assert _rel(A @ x, B @ x) < 1e-6
    y = _arr(_RNG.random(A.range_shape))
    assert _rel(A.T @ y, B.T @ y) < 1e-6
    assert A.to_physical == pytest.approx(vs)


def test_voxel_units_need_scalar_spacing():
    traj, vol, det, _, ds = _setup("cone")
    with pytest.raises(ValueError, match="scalar voxel_spacing"):
        dct.Projector(traj, vol, det, detector_spacing=ds, voxel_spacing=(0.8, 0.7, 0.6),
                      line_integrals="voxel")


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_geometry_gradient_through_projector(beam):
    """Gradients w.r.t. the trajectory equal those of the functional projector."""
    traj, vol, det, vs, ds = _setup(beam)
    x = _arr(_RNG.random(vol))
    w = _arr(_RNG.random((int(traj[0].shape[0]),) + det))
    P = dct.Projector(traj, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs)
    if dct.backend == "mlx":
        import mlx.core as mx

        g = mx.grad(lambda t: (P.project(x, trajectory=t) * w).sum())(tuple(traj))
        g_ref = mx.grad(lambda t: (_functional(beam, "siddon", x, t, det, ds, vs) * w).sum())(tuple(traj))
    else:
        leaves = [t.detach().clone().requires_grad_(True) for t in traj]
        (dct.Projector(leaves, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs)(x) * w).sum().backward()
        g = [t.grad for t in leaves]
        leaves = [t.detach().clone().requires_grad_(True) for t in traj]
        (_functional(beam, "siddon", x, leaves, det, ds, vs) * w).sum().backward()
        g_ref = [t.grad for t in leaves]
    for a, b in zip(g, g_ref):
        assert np.abs(_np(b)).sum() > 0
        assert _rel(a, b) < 1e-4


def test_parallel3d_trajectory_layout():
    ray, origin, u, v = dct.circular_trajectory_3d_parallel(5, detector_distance=2.0)
    ray, origin, u, v = map(_np, (ray, origin, u, v))
    assert ray.shape == (5, 3)
    np.testing.assert_allclose(ray[:, 2], 0.0)
    np.testing.assert_allclose(v, np.tile([0.0, 0.0, 1.0], (5, 1)))
    np.testing.assert_allclose(np.linalg.norm(ray, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose((ray * u).sum(1), 0.0, atol=1e-6)


@pytest.mark.parametrize("option", [
    dict(devices=["gpu:0", "gpu:1"]),
    dict(distributed=True),
    dict(partition="space"),
])
def test_multi_device_options_need_the_torch_engine(option):
    if dct.backend == "torch" and not _cuda():
        pytest.skip("engine multi-device checks need CUDA")
    if dct.backend == "torch":
        pytest.skip("the engine validates multi-device options itself")
    traj, vol, det, _, ds = _setup("cone")
    with pytest.raises(NotImplementedError, match="Multi-device"):
        dct.Projector(traj, vol, det, detector_spacing=ds, **option)


def test_unknown_keyword_is_rejected():
    traj, vol, det, _, ds = _setup("cone")
    if dct.backend == "torch":
        pytest.skip("the engine rejects unknown keywords itself")
    with pytest.raises(TypeError, match="unexpected"):
        dct.Projector(traj, vol, det, detector_spacing=ds, bogus=1)


@pytest.mark.parametrize("option", [dict(beam="parallel3d"), dict(precision="float64"),
                                    dict(voxel_spacing=(0.8, 0.7, 0.6))])
def test_footprint_model_limits_are_named(option):
    if dct.backend == "torch" and not _cuda():
        pytest.skip("footprint needs CUDA on Torch")
    if dct.backend == "mlx" and option.get("precision") == "float64":
        pytest.skip("MLX maps float64 to float32")
    traj, vol, det, _, ds = _setup("cone")
    if option.get("beam") == "parallel3d":
        traj = dct.circular_trajectory_3d_parallel(4)
    with pytest.raises(NotImplementedError, match="footprint"):
        dct.Projector(traj, vol, det, detector_spacing=ds, projector_mode="footprint", **option)


@pytest.mark.parametrize("mode", _modes())
def test_volume_and_detector_blocks_match_one_pass(mode):
    traj, vol, det, vs, ds = _setup("cone")
    x = _arr(_RNG.random(vol))
    kw = dict(detector_spacing=ds, voxel_spacing=vs, projector_mode=mode)
    P = dct.Projector(traj, vol, det, **kw)
    Pb = dct.Projector(traj, vol, det, view_chunk_size=4, volume_chunk_shape=(7, None, 9),
                       detector_chunk_shape=(11, 13), **kw)
    assert _rel(Pb(x), P(x)) < 1e-5
    r = _arr(_RNG.random(P.projection_shape))
    assert _rel(Pb.backproject(r), P.backproject(r)) < 1e-5


def test_project_into_memmaps(tmp_path):
    if dct.backend == "torch":
        pytest.skip("Torch delegates *_into to the engine")
    traj, vol, det, vs, ds = _setup("cone")
    x = _RNG.random(vol).astype(np.float32)
    vin = np.lib.format.open_memmap(tmp_path / "v.npy", mode="w+", dtype=np.float32, shape=vol)
    vin[:] = x
    P = dct.Projector(traj, vol, det, detector_spacing=ds, voxel_spacing=vs,
                      view_chunk_size=2, volume_chunk_shape=(6, None, None))
    out = np.lib.format.open_memmap(tmp_path / "s.npy", mode="w+", dtype=np.float32,
                                    shape=P.projection_shape)
    assert P.project_into(vin, out) is out
    assert _rel(out, P(_arr(x))) < 1e-5
    vout = np.zeros(vol, np.float32)
    P.backproject_into(out, vout)
    assert _rel(vout, P.backproject(_arr(out))) < 1e-5


def test_leading_batch_dimensions():
    traj, vol, det, vs, ds = _setup("fan")
    x = _RNG.random(vol).astype(np.float32)
    P = dct.Projector(traj, vol, det, beam="fan", detector_spacing=ds, voxel_spacing=vs)
    yb = P(_arr(np.stack([x, 2 * x, 3 * x])[None]))
    assert tuple(yb.shape) == (1, 3) + P.projection_shape
    assert _rel(yb[0, 2], 3 * _np(P(_arr(x)))) < 1e-5


@pytest.mark.skipif(dct.backend != "mlx", reason="MLX paths")
def test_mlx_path_selection():
    traj, vol, det, vs, ds = _setup("cone")
    assert dct.Projector(traj, vol, det).path == "kernel"
    assert dct.Projector(traj, vol, det, voxel_spacing=(0.8, 0.7, 0.6)).path == "rays"
    assert dct.Projector(traj, vol, det, precision="float32").path == "rays"
    assert dct.Projector(dct.circular_trajectory_3d_parallel(4), vol, det,
                         beam="parallel3d").path == "rays"
    with pytest.warns(RuntimeWarning, match="float32"):
        P64 = dct.Projector(traj, vol, det, precision="float64")
    assert P64.precision == "float32" and P64.path == "rays"
    for bad in ("portable", "numba"):
        with pytest.raises(ValueError, match="backend"):
            dct.Projector(traj, vol, det, backend=bad)


@pytest.mark.skipif(dct.backend != "mlx", reason="MLX paths")
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_mlx_ray_kernels_match_frame_kernels(beam):
    """The Metal ray kernels against the frame-based Metal kernels (scalar spacing)."""
    import mlx.core as mx

    traj, vol, det, vs, ds = _setup(beam)
    x = _arr(_RNG.random(vol))
    w = _arr(_RNG.random((int(traj[0].shape[0]),) + det))
    kw = dict(beam=beam, detector_spacing=ds, voxel_spacing=vs)
    Pk = dct.Projector(traj, vol, det, **kw)
    Pr = dct.Projector(traj, vol, det, precision="float32", **kw)
    assert (Pk.path, Pr.path) == ("kernel", "rays")
    assert _rel(Pk(x), Pr(x)) < 1e-5
    assert _rel(Pk.backproject(w), Pr.backproject(w)) < 1e-5
    gk = mx.grad(lambda t: (Pk.project(x, trajectory=t) * w).sum())(tuple(traj))
    gr = mx.grad(lambda t: (Pr.project(x, trajectory=t) * w).sum())(tuple(traj))
    for a, b in zip(gk, gr):
        assert _rel(a, b) < 1e-3


@pytest.mark.skipif(dct.backend != "torch", reason="engine options need the Torch backend")
def test_torch_engine_options_reach_the_engine():
    import torch
    from diffct_mlx import engine

    traj, vol, det, _, ds = _setup("cone")
    t64 = tuple(t.double() for t in traj)
    x = torch.rand(vol, dtype=torch.float64, device=t64[0].device)
    vs = (0.8, 0.7, 0.6)
    P = dct.Projector(t64, vol, det, detector_spacing=ds, voxel_spacing=vs, precision="float64")
    ref = engine.Projector(t64, vol, det, beam="cone", detector_spacing=ds, voxel_spacing=vs,
                           precision="float64")
    y = P(x)
    assert y.dtype == torch.float64
    assert _rel(y, ref(x)) < 1e-12
    assert P.as_operator().voxel_spacing == vs

    tp = dct.circular_trajectory_3d_parallel(5)
    Pp = dct.Projector(tp, (16, 16, 16), (20, 18), beam="parallel3d", detector_spacing=0.9)
    xp3 = torch.rand(16, 16, 16, device=tp[0].device)
    ref3 = engine.Projector(tp, (16, 16, 16), (20, 18), beam="parallel3d", detector_spacing=(0.9, 0.9))
    assert _rel(Pp(xp3), ref3(xp3)) < 1e-6


@pytest.mark.skipif(dct.backend != "torch" or _cuda(), reason="CPU-only Torch check")
def test_torch_footprint_on_cpu_says_cuda():
    traj, vol, det, vs, ds = _setup("cone")
    P = dct.Projector(traj, vol, det, detector_spacing=ds, voxel_spacing=vs, projector_mode="footprint")
    import torch

    with pytest.raises(NotImplementedError, match="CUDA tensors"):
        P(torch.rand(vol))
