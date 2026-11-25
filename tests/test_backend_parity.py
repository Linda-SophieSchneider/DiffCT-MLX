"""Cross-backend parity: MLX/Metal projectors against the Torch engine.

Since 3.0 both backends share the upstream diffct conventions (detector cell
``k`` at ``(k - (N-1)/2) * pitch``, physical line integrals). On a Mac both
run side by side: the MLX backend, and the engine's portable PyTorch backend
on the CPU in float64 as the reference. These tests pin that the two agree in
value, in the adjoint and in the cone geometry gradient.

Runs only where MLX is the active backend and torch is installed.
"""

import numpy as np
import pytest

import diffct_mlx as dct

if dct.backend != "mlx":
    pytest.skip("parity test runs on the MLX backend", allow_module_level=True)
torch = pytest.importorskip("torch")

import mlx.core as mx  # noqa: E402

from diffct_mlx import engine  # noqa: E402

_RNG = np.random.default_rng(7)


def _torch(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def _ref(beam, trajectory, volume_shape, detector_shape, detector_spacing, voxel_spacing):
    return engine.Projector(
        tuple(_torch(a) for a in trajectory), volume_shape, detector_shape, beam=beam,
        detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        backend="torch", precision="float64",
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _phantom(shape, vs):
    axes = [(np.arange(n) - (n - 1) / 2) * vs for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    centre = [0.7, -1.1, 0.4][: len(shape)]
    r2 = sum((g - c) ** 2 for g, c in zip(grids, centre))
    return ((r2 <= (0.3 * min(shape) * vs) ** 2) * 1.0 + 0.2 * _RNG.random(shape)).astype(np.float32)


def test_parallel_matches_engine():
    n, vs, nd, ds = 48, 0.7, 70, 0.6
    traj = [np.asarray(a) for a in dct.circular_trajectory_2d_parallel(9, start_angle=0.21)]
    img = _phantom((n, n), vs)
    got = np.asarray(dct.parallel_forward(mx.array(img), *map(mx.array, traj), nd, ds, vs))
    P = _ref("parallel", traj, (n, n), (nd,), ds, vs)
    assert _rel(got, P(_torch(img)).numpy()) < 2e-4
    y = _RNG.random((9, nd)).astype(np.float32)
    got_b = np.asarray(dct.parallel_backward(mx.array(y), *map(mx.array, traj), ds, n, n, vs))
    assert _rel(got_b, P.backproject(_torch(y)).numpy()) < 2e-4


def test_fan_matches_engine():
    n, vs, nd, ds = 48, 0.7, 90, 0.8
    traj = [np.asarray(a) for a in dct.circular_trajectory_2d_fan(9, 120.0, 210.0, start_angle=0.21)]
    img = _phantom((n, n), vs)
    got = np.asarray(dct.fan_forward(mx.array(img), *map(mx.array, traj), nd, ds, vs))
    P = _ref("fan", traj, (n, n), (nd,), ds, vs)
    assert _rel(got, P(_torch(img)).numpy()) < 2e-4
    y = _RNG.random((9, nd)).astype(np.float32)
    got_b = np.asarray(dct.fan_backward(mx.array(y), *map(mx.array, traj), ds, n, n, vs))
    assert _rel(got_b, P.backproject(_torch(y)).numpy()) < 2e-4


def test_cone_matches_engine():
    n, vs, nu, du = 32, 0.6, 44, 0.75
    traj = [np.asarray(a) for a in dct.circular_trajectory_3d(5, 90.0, 160.0, start_angle=0.21)]
    vol = _phantom((n, n, n), vs)
    got = np.asarray(dct.cone_forward(mx.array(vol), *map(mx.array, traj), nu, nu, du, du, vs))
    P = _ref("cone", traj, (n, n, n), (nu, nu), (du, du), vs)
    assert _rel(got, P(_torch(vol)).numpy()) < 2e-4
    y = _RNG.random((5, nu, nu)).astype(np.float32)
    got_b = np.asarray(dct.cone_backward(mx.array(y), *map(mx.array, traj), n, n, n, du, du, vs))
    assert _rel(got_b, P.backproject(_torch(y)).numpy()) < 2e-4


def test_cone_geometry_gradient_matches_engine():
    """Metal geometry VJP against the engine's exact VJP (float64, CPU).

    Before 3.0 the Metal kernel differentiated a trilinear-interpolated volume
    while the forward kernel is cell-constant Siddon, and was off by ~100 %.
    """
    n, vs, nu, du = 24, 0.8, 36, 0.9
    traj = [np.asarray(a) for a in dct.circular_trajectory_3d(4, 80.0, 140.0, start_angle=0.21)]
    vol = _phantom((n, n, n), vs)
    w = _RNG.random((4, nu, nu)).astype(np.float32)

    def loss(src, dc, uv, vv):
        return (dct.cone_forward(mx.array(vol), src, dc, uv, vv, nu, nu, du, du, vs) * mx.array(w)).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3))(*map(mx.array, traj))

    leaves = [_torch(a).requires_grad_(True) for a in traj]
    P = engine.Projector(tuple(leaves), (n, n, n), (nu, nu), beam="cone", detector_spacing=(du, du),
                         voxel_spacing=vs, backend="torch", precision="float64")
    (P(_torch(vol)) * _torch(w)).sum().backward()
    for name, g, leaf in zip(("src", "det_center", "det_u", "det_v"), grads, leaves):
        ref = leaf.grad.numpy()
        assert _rel(np.asarray(g), ref) < 1e-3, name


def _engine_geometry_grads(beam, traj, img, w, nd, ds, vs):
    leaves = [_torch(a).requires_grad_(True) for a in traj]
    P = engine.Projector(tuple(leaves), img.shape, (nd,), beam=beam, detector_spacing=ds,
                         voxel_spacing=vs, backend="torch", precision="float64")
    (P(_torch(img)) * _torch(w)).sum().backward()
    return [leaf.grad.numpy() for leaf in leaves]


@pytest.mark.parametrize("case", ["standard", "scaled_ray", "far_detector"])
def test_parallel_geometry_gradient_matches_engine(case):
    """Metal parallel Siddon geometry VJP against the engine's exact VJP.

    Before 3.1 the MLX backend returned zero gradients for ray_dir,
    det_origin and det_u_vec.
    """
    n, vs, nd, ds = 40, 0.7, 64, 0.8
    ray, origin, u = [np.asarray(a) for a in dct.circular_trajectory_2d_parallel(7, start_angle=0.21)]
    if case == "far_detector":
        origin = origin + 300.0 * ray   # detector origin far outside the volume
    # The kernel normalises ray_dir, so for R = k r the gradient is grad_r / k.
    # The engine only accepts unit directions; its result is scaled instead.
    k = 2.5 if case == "scaled_ray" else 1.0
    img = _phantom((n, n), vs)
    w = _RNG.random((7, nd)).astype(np.float32)
    refs = _engine_geometry_grads("parallel", [ray, origin, u], img, w, nd, ds, vs)
    refs[0] = refs[0] / k
    traj = [ray * k, origin, u]

    def loss(a, b, c):
        return (dct.parallel_forward(mx.array(img), a, b, c, nd, ds, vs) * mx.array(w)).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2))(*map(mx.array, traj))
    for name, g, ref in zip(("ray_dir", "det_origin", "det_u_vec"), grads, refs):
        assert _rel(np.asarray(g), ref) < 1e-3, name


@pytest.mark.parametrize("case", ["standard", "detector_inside", "distant_source"])
def test_fan_geometry_gradient_matches_engine(case):
    """Metal fan Siddon geometry VJP against the engine's exact VJP.

    Before 3.1 the MLX backend returned zero gradients for src_pos,
    det_center and det_u_vec.
    """
    n, vs, nd, ds = 40, 0.7, 64, 0.8
    # (sid, sdd); the volume half-width is 14 mm.
    sid, sdd = {"standard": (120.0, 210.0), "detector_inside": (60.0, 70.0),
                "distant_source": (5000.0, 5060.0)}[case]
    traj = [np.asarray(a) for a in dct.circular_trajectory_2d_fan(7, sid, sdd, start_angle=0.21)]
    img = _phantom((n, n), vs)
    w = _RNG.random((7, nd)).astype(np.float32)

    def loss(a, b, c):
        return (dct.fan_forward(mx.array(img), a, b, c, nd, ds, vs) * mx.array(w)).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2))(*map(mx.array, traj))
    for name, g, ref in zip(("src_pos", "det_center", "det_u_vec"), grads,
                            _engine_geometry_grads("fan", traj, img, w, nd, ds, vs)):
        assert _rel(np.asarray(g), ref) < 1e-3, name


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_footprint_agrees_with_engine_siddon(beam):
    """Footprint is a different discretisation: same mass and centroid, close values."""
    vs = 0.7
    if beam == "cone":
        n, nd, ds = 32, 40, 0.9
        traj = [np.asarray(a) for a in dct.circular_trajectory_3d(5, 90.0, 160.0, start_angle=0.21)]
        x = _phantom((n, n, n), vs)
        got = np.asarray(dct.cone_forward_footprint(mx.array(x), *map(mx.array, traj), nd, nd, ds, ds, vs))
        ref = _ref("cone", traj, (n, n, n), (nd, nd), (ds, ds), vs)(_torch(x)).numpy()
    else:
        n, nd, ds = 48, 80, 0.7
        gen = dct.circular_trajectory_2d_parallel if beam == "parallel" else (
            lambda v, start_angle: dct.circular_trajectory_2d_fan(v, 120.0, 210.0, start_angle=start_angle))
        traj = [np.asarray(a) for a in gen(9, start_angle=0.21)]
        x = _phantom((n, n), vs)
        fwd = dct.parallel_forward_footprint if beam == "parallel" else dct.fan_forward_footprint
        got = np.asarray(fwd(mx.array(x), *map(mx.array, traj), nd, ds, vs))
        ref = _ref(beam, traj, (n, n), (nd,), ds, vs)(_torch(x)).numpy()
    flat = lambda a: a.reshape(a.shape[0], -1)
    np.testing.assert_allclose(flat(got).sum(1), flat(ref).sum(1), rtol=0.01)
    idx = np.arange(got.shape[1])
    prof = lambda a: a.reshape(a.shape[0], a.shape[1], -1).sum(2)
    cen = lambda a: (prof(a) @ idx) / prof(a).sum(1)
    assert np.abs(cen(got) - cen(ref)).max() < 0.1
    assert _rel(got, ref) < 0.06


def test_parallel3d_trajectory_matches_engine():
    got = dct.circular_trajectory_3d_parallel(7, detector_distance=3.0, start_angle=0.21)
    ref = engine.circular_trajectory_3d_parallel(7, detector_distance=3.0, start_angle=0.21,
                                                 device="cpu")
    for g, r in zip(got, ref):
        np.testing.assert_allclose(np.asarray(g), r.numpy(), atol=1e-6)


# --- dct.Projector on the Metal ray kernels against the engine in float64 ------

def _engine_traj(beam, n_views):
    if beam == "parallel":
        return engine.circular_trajectory_2d_parallel(n_views, start_angle=0.21, device="cpu", dtype=torch.float64)
    if beam == "fan":
        return engine.circular_trajectory_2d_fan(n_views, 60.0, 110.0, start_angle=0.21, device="cpu", dtype=torch.float64)
    if beam == "cone":
        return engine.circular_trajectory_3d(n_views, 60.0, 110.0, start_angle=0.21, device="cpu", dtype=torch.float64)
    return engine.circular_trajectory_3d_parallel(n_views, start_angle=0.21, device="cpu", dtype=torch.float64)


def _offset(traj, beam):
    """Lift the orbit off the z = 0 voxel plane and away from voxel edges.

    A source or pixel exactly on a voxel face plane makes the geometry
    derivative one-sided; float32 and float64 may then pick different sides.
    """
    traj = list(traj)
    if beam in ("cone", "parallel3d"):
        shift = torch.tensor([0.031, -0.017, 0.137], dtype=torch.float64)
        for i in ((0, 1) if beam == "cone" else (1,)):
            traj[i] = traj[i] + shift
    return tuple(traj)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone", "parallel3d"])
@pytest.mark.parametrize("case", ["per_axis", "float32"])
def test_projector_metal_rays_match_engine(beam, case):
    """Metal ray kernels (float32, GPU): values, adjoint and geometry gradients."""
    three = beam in ("cone", "parallel3d")
    vol, det, ds = ((14, 16, 18), (24, 20), (0.9, 1.1)) if three else ((24, 26), (40,), 0.7)
    vs = ((0.6, 0.7, 0.5) if three else (0.7, 0.5)) if case == "per_axis" else 0.7
    traj = _offset(_engine_traj(beam, 5), beam)
    x = _RNG.random(vol)
    w = _RNG.random((5,) + det)
    leaves = [t.clone().requires_grad_(True) for t in traj]
    E = engine.Projector(tuple(leaves), vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs,
                         backend="torch", precision="float64")
    (E(_torch(x)) * _torch(w)).sum().backward()

    t32 = tuple(mx.array(t.detach().numpy().astype(np.float32)) for t in traj)
    kw = dict(precision="float32") if case == "float32" else {}
    P = dct.Projector(t32, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=vs, **kw)
    assert P.path == "rays"
    x32, w32 = mx.array(x.astype(np.float32)), mx.array(w.astype(np.float32))
    assert _rel(P(x32), E(_torch(x)).detach().numpy()) < 1e-5
    assert _rel(P.backproject(w32), E.backproject(_torch(w)).detach().numpy()) < 1e-5
    grads = mx.grad(lambda t: (P.project(x32, trajectory=t) * w32).sum())(t32)
    for g, leaf in zip(grads, leaves):
        ref = leaf.grad.numpy()
        if np.linalg.norm(ref) > 0:
            assert _rel(g, ref) < 1e-4


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_projector_detector_surface_matches_engine(beam):
    """Cylindrical detector on the Metal ray kernels: values, adjoint, radius gradient."""
    three = beam == "cone"
    vol, det, ds = ((14, 16, 18), (24, 20), (1.2, 1.0)) if three else ((20, 22), (40,), 1.1)
    traj = _engine_traj(beam, 4)
    x = _RNG.random(vol)
    w = _RNG.random((4,) + det)
    radius = torch.tensor(110.0, dtype=torch.float64, requires_grad=True)

    def sample_t(u, v, ids, r):
        u, v = u.to(r.dtype), v.to(r.dtype)
        return torch.stack((r * torch.sin(u / r), v, r * (torch.cos(u / r) - 1)), -1)

    E = engine.Projector(traj, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=0.8,
                         detector_surface=engine.ParameterizedSurface(sample_t, (radius,)),
                         backend="torch", precision="float64")
    (E(_torch(x)) * _torch(w)).sum().backward()

    def sample_m(u, v, ids, r):
        a = u / r
        return mx.stack([r * mx.sin(a), v, r * (mx.cos(a) - 1)], -1)

    surface32 = dct.ParameterizedSurface(sample_m, (mx.array(110.0),))
    t32 = tuple(mx.array(t.numpy().astype(np.float32)) for t in traj)
    P32 = dct.Projector(t32, vol, det, beam=beam, detector_spacing=ds, voxel_spacing=0.8,
                        detector_surface=surface32)
    assert P32.path == "rays"
    x32, w32 = mx.array(x.astype(np.float32)), mx.array(w.astype(np.float32))
    assert _rel(P32(x32), E(_torch(x)).detach().numpy()) < 1e-5
    assert _rel(P32.backproject(w32), E.backproject(_torch(w)).detach().numpy()) < 1e-5
    g32 = mx.grad(lambda r: (P32.project(x32, surface_parameters=(r,)) * w32).sum())(mx.array(110.0))
    assert _rel(g32, radius.grad.numpy()) < 1e-3
