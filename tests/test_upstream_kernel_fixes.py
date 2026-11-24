"""Regression tests for the kernel and analytical fixes ported from upstream diffct 2.x.

Backend-neutral: every test goes through the public ``diffct_mlx`` API, so it
runs the Metal kernels on MLX and the numba-CUDA kernels on the torch backend.
Each test fails on the code before the port.

The tests do not depend on the detector-cell convention, ``(k - N/2) * du``
or ``(k - (N - 1)/2) * du``, nor on voxel versus physical units. Single rays
use one detector cell with a negligible pitch, phantoms are uniform,
symmetric or compared against a second projection, and the voxel spacing
is 1. The ramp-filter checks are free of the overall filter scale.

* Fan and cone rays integrate only from the source to the detector pixel.
* The ray parameter is measured from the endpoint nearer the volume centre.
* Ray pieces of any positive length are kept.
* Only direction components below 1e-20 count as axis-parallel.
* The 2D parallel kernel normalizes ``ray_dir``.
* The cone geometry gradient includes the moving detector end of the segment.
* ``custom_trajectory_3d`` rejects a source at the isocenter.
* The MLX-side ramp filter uses the discrete Ram-Lak response.
* The MLX-side Parker weights use the generalized taper and decide a full
  turn from the acquired angles alone.
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


def _arr(a):
    return xp.array(np.asarray(a, dtype=np.float32))


# One detector cell with a negligible pitch sits at the detector centre under
# either detector-cell convention.
_PITCH = 1e-6


def _fan_ray(image, src, det, u=(0.0, 1.0), pitch=_PITCH):
    """Line integral of the single ray from ``src`` to the detector centre."""
    sino = dct.fan_forward(image, _arr([src]), _arr([det]), _arr([u]), 1, pitch)
    return float(_b.to_numpy(sino)[0, 0])


def _cone_ray(volume, src, det, u=(0.0, 1.0, 0.0), v=(0.0, 0.0, 1.0)):
    sino = dct.cone_forward(volume, _arr([src]), _arr([det]), _arr([u]), _arr([v]),
                            1, 1, _PITCH, _PITCH)
    return float(_b.to_numpy(sino)[0, 0, 0])


def _parallel_rays(image, ray_dir, foot, normal):
    sino = dct.parallel_forward(image, _arr(ray_dir), _arr(foot), _arr(normal), 1, _PITCH)
    return _b.to_numpy(sino)[:, 0]


# --------------------------------------------------------------------------- #
# 1. Source-to-detector segment
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src_x, det_x, expected", [
    (-10.0, 100.0, 42.0),   # source inside the volume
    (-100.0, 10.0, 42.0),   # detector inside the volume
    (-10.0, 5.0, 15.0),     # both endpoints inside
])
def test_fan_ray_integrates_only_between_source_and_detector(src_x, det_x, expected):
    _skip_if_unusable()
    ones = _arr(np.ones((64, 64)))
    value = _fan_ray(ones, (src_x, 0.0), (det_x, 0.0))
    # The full-line integral through the 64-voxel box would be 64.
    assert value == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("src_x, det_x, expected", [
    (-10.0, 100.0, 42.0),
    (-100.0, 10.0, 42.0),
    (-10.0, 5.0, 15.0),
])
def test_cone_ray_integrates_only_between_source_and_detector(src_x, det_x, expected):
    _skip_if_unusable()
    ones = _arr(np.ones((64, 64, 64)))
    value = _cone_ray(ones, (src_x, 0.0, 0.0), (det_x, 0.0, 0.0))
    assert value == pytest.approx(expected, rel=1e-5)


def test_fan_backprojection_stays_adjoint_with_endpoints_inside_volume():
    _skip_if_unusable()
    rng = np.random.default_rng(3)
    n_views, n_det = 16, 24
    theta = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)
    direction = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    # Source 8 voxels from the centre, detector 6 voxels on the other side:
    # both lie inside the 32 x 32 image.
    src, det = -8.0 * direction, 6.0 * direction
    x = _arr(rng.standard_normal((32, 32)))
    y = _arr(rng.standard_normal((n_views, n_det)))
    Ax = dct.fan_forward(x, _arr(src), _arr(det), _arr(normal), n_det, 0.5)
    Aty = dct.fan_backward(y, _arr(src), _arr(det), _arr(normal), 0.5, 32, 32)
    lhs = float((Ax * y).sum())
    rhs = float((x * Aty).sum())
    assert abs(lhs - rhs) <= 1e-4 * abs(lhs)


# --------------------------------------------------------------------------- #
# 2. Distant sources
# --------------------------------------------------------------------------- #

def _random_lines_2d(n_lines, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_lines)
    offset = rng.uniform(-12.0, 12.0, n_lines)
    direction = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    return direction, normal, offset[:, None] * normal


@pytest.mark.parametrize("distance", [1e4, 1e6])
def test_fan_distant_source_matches_parallel_line_integral(distance):
    """A far source must give the same line integral as the parallel beam."""
    _skip_if_unusable()
    image = _arr(np.random.default_rng(1).random((32, 32)))
    direction, normal, foot = _random_lines_2d(200, seed=2)
    reference = _parallel_rays(image, direction, foot, normal)
    src = foot - distance * direction
    det = foot + 40.0 * direction
    fan = _b.to_numpy(dct.fan_forward(
        image, _arr(src), _arr(det), _arr(normal), 1, _PITCH))[:, 0]
    # Before the fix the error grew with the distance, 3e-3 at 1e4 and 0.1
    # at 1e6 voxels.
    np.testing.assert_allclose(fan, reference, rtol=1e-4)


def test_cone_distant_source_matches_near_source():
    _skip_if_unusable()
    rng = np.random.default_rng(4)
    volume = _arr(rng.random((32, 32, 32)))
    n = 64
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    phi = rng.uniform(-0.5, 0.5, n)
    direction = np.stack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi),
                          np.sin(phi)], axis=1)
    u = np.stack([-np.sin(theta), np.cos(theta), np.zeros(n)], axis=1)
    v = np.cross(direction, u)
    foot = rng.uniform(-10.0, 10.0, (n, 1)) * u + rng.uniform(-8.0, 8.0, (n, 1)) * v
    det = foot + 40.0 * direction

    def central(src):
        sino = dct.cone_forward(volume, _arr(src), _arr(det), _arr(u), _arr(v),
                                1, 1, _PITCH, _PITCH)
        return _b.to_numpy(sino)[:, 0, 0]

    near = central(foot - 50.0 * direction)
    far = central(foot - 1e6 * direction)
    np.testing.assert_allclose(far, near, rtol=1e-4)


# --------------------------------------------------------------------------- #
# 3. Short ray pieces
# --------------------------------------------------------------------------- #

def test_fan_keeps_ray_pieces_shorter_than_old_epsilon():
    """A ray that clips a voxel corner over 7e-7 voxels must still see it.

    The source and detector lie close to the volume and the ray enters at
    the clipped corner, so the piece is resolved in float32.
    """
    _skip_if_unusable()
    bright = 1e6
    # All four corner voxels are bright, so the test holds for any image
    # axis order. The ray only touches the corner voxel at (-2, -2).
    image = np.zeros((4, 4), dtype=np.float32)
    image[[0, 0, -1, -1], [0, -1, 0, -1]] = bright
    delta = 5e-7
    # The line x + y = -4 + delta cuts the corner (-2, -2) over delta * sqrt(2).
    value = _fan_ray(_arr(image), (-3.0, -1.0 + delta), (-1.0, -3.0 + delta),
                     u=(math.sqrt(0.5), math.sqrt(0.5)), pitch=1e-9)
    expected = delta * math.sqrt(2.0) * bright
    assert 0.5 * expected < value < 1.5 * expected


# --------------------------------------------------------------------------- #
# 4. Axis-parallel threshold
# --------------------------------------------------------------------------- #

def test_parallel_grazing_ray_changes_row_at_the_crossing():
    """A ray with a 5e-7 lateral component crosses the grid line y = 0 at x = 0.

    It must see the same voxels as a steeper ray through the same crossing.
    Under the old 1e-6 threshold it counted as axis-parallel and stayed in
    the lower row for the whole image.
    """
    _skip_if_unusable()
    image = _arr(np.random.default_rng(5).random((4, 4)) + 1.0)
    rays = []
    for slope in (5e-7, 1e-3):
        direction = np.array([1.0, slope]) / math.hypot(1.0, slope)
        normal = np.array([-direction[1], direction[0]])
        # Foot point 100 voxels before the crossing at the origin.
        rays.append((direction, -100.0 * direction, normal))
    direction, foot, normal = (np.stack(c) for c in zip(*rays))
    # A detector-cell offset of 5e-7 would move the crossing of the grazing
    # ray by one voxel, so this test uses a still smaller pitch.
    sino = dct.parallel_forward(image, _arr(direction), _arr(foot), _arr(normal), 1, 1e-12)
    grazing, steep = _b.to_numpy(sino)[:, 0]
    assert grazing == pytest.approx(steep, rel=1e-4)


# --------------------------------------------------------------------------- #
# 5. Unnormalized parallel ray_dir
# --------------------------------------------------------------------------- #

def test_parallel_projection_ignores_ray_dir_length():
    _skip_if_unusable()
    image = _arr(np.random.default_rng(0).random((32, 32)))
    angle = 0.3
    ray_dir = np.array([[math.cos(angle), math.sin(angle)]])
    det_u = np.array([[-math.sin(angle), math.cos(angle)]])
    origin = np.zeros((1, 2))

    def project(rd):
        return _b.to_numpy(dct.parallel_forward(image, _arr(rd), _arr(origin),
                                                _arr(det_u), 48, 1.0))

    unit = project(ray_dir)
    np.testing.assert_allclose(project(3.0 * ray_dir), unit, rtol=1e-5, atol=1e-5)

    y = _arr(np.random.default_rng(1).random((1, 48)))
    back_unit = _b.to_numpy(dct.parallel_backward(y, _arr(ray_dir), _arr(origin),
                                                  _arr(det_u), 1.0, 32, 32))
    back_long = _b.to_numpy(dct.parallel_backward(y, _arr(3.0 * ray_dir), _arr(origin),
                                                  _arr(det_u), 1.0, 32, 32))
    np.testing.assert_allclose(back_long, back_unit, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Cone geometry gradient with the detector inside the volume
# --------------------------------------------------------------------------- #

def test_cone_detector_gradient_along_ray_matches_finite_difference():
    """Moving a detector inside the volume along the ray lengthens every ray.

    The gradient must carry that ``mu(p)`` boundary term. The old full-line
    gradient was about 15 times too small.
    """
    _skip_if_unusable()
    n = 32
    g = np.arange(n) + 0.5 - n / 2
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    volume = _arr(np.exp(-(X**2 + Y**2 + Z**2) / (2.0 * 6.0**2)))
    src = _arr([[-60.0, 0.3, 0.2]])
    det_u = _arr([[0.0, 1.0, 0.0]])
    det_v = _arr([[0.0, 0.0, 1.0]])
    det_center = np.array([[3.0, 0.3, 0.2]], dtype=np.float32)

    def loss(dc):
        return dct.cone_forward(volume, src, dc, det_u, det_v, 8, 8, 1.0, 1.0).sum()

    grad = _b.to_numpy(xp.grad(loss)(_arr(det_center)))[0, 0]
    h = 0.25
    step = np.array([[h, 0.0, 0.0]], dtype=np.float32)
    fd = (float(loss(_arr(det_center + step))) - float(loss(_arr(det_center - step)))) / (2 * h)
    assert grad == pytest.approx(fd, rel=0.05)


# --------------------------------------------------------------------------- #
# 7. custom_trajectory_3d
# --------------------------------------------------------------------------- #

def test_custom_trajectory_3d_rejects_source_at_isocenter():
    def at_isocenter(angles, sid):
        zero = 0.0 * angles
        return xp.stack([zero, zero, zero], axis=1)

    with pytest.raises(ValueError, match="isocenter"):
        dct.custom_trajectory_3d(8, 500.0, 900.0, source_path_fn=at_isocenter)


def test_custom_trajectory_3d_rejects_non_finite_source():
    def with_nan(angles, sid):
        c = xp.cos(angles)
        return xp.stack([sid * c, sid * xp.sin(angles), c / (0.0 * c)], axis=1)

    with pytest.raises(ValueError, match="non-finite"):
        dct.custom_trajectory_3d(8, 500.0, 900.0, source_path_fn=with_nan)


# --------------------------------------------------------------------------- #
# Discrete Ram-Lak ramp filter (MLX-side analytical pipeline)
# --------------------------------------------------------------------------- #

def _ramp(rows, pad_factor):
    from diffct_mlx.reconstruction_algorithms import ramp_filter

    return _b.to_numpy(ramp_filter(_arr(rows), axis=1, pad_factor=pad_factor)).astype(np.float64)


def _nyquist_gain(n):
    """Filter gain at Nyquist, which fixes the overall scale of the filter."""
    alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[None, :]
    return float(_ramp(alternating, 1)[0, 0])


def test_ramp_dc_response_is_that_of_the_finite_ram_lak_kernel():
    """Relative to the Nyquist gain, a constant row keeps a small positive value.

    The sampled ramp forced it to zero. With ``e = 1/2 - 4/pi**2 *
    sum(1/k**2, odd k < n/2)`` the finite kernel has DC gain ``e`` and Nyquist
    gain ``1 - e`` in units of the Nyquist ramp value, so their ratio is the
    same in any scale convention.
    """
    n = 128
    odd = np.arange(1, n // 2, 2, dtype=np.float64)
    e = 0.5 - 4.0 / math.pi**2 * np.sum(1.0 / odd**2)
    expected = e / (1.0 - e)
    dc = _ramp(np.ones((2, n)), 1) / _nyquist_gain(n)
    np.testing.assert_allclose(dc, expected, rtol=1e-3)


@pytest.mark.parametrize("n", [15, 16])
@pytest.mark.parametrize("pad_factor", [2, 3])
def test_ramp_is_linear_convolution_with_ram_lak_kernel(n, pad_factor):
    x = np.linspace(-0.4, 0.9, n)
    # h[0] = 1/2, h[odd k] = -2/(pi*k)**2, h[nonzero even k] = 0, up to scale.
    lag = np.subtract.outer(np.arange(n), np.arange(n))
    kernel = np.where(lag == 0, 0.5, np.where(lag % 2 != 0, -2.0 / (math.pi * np.maximum(np.abs(lag), 1)) ** 2, 0.0))
    expected = kernel @ x
    actual = _ramp(x[None, :], pad_factor)[0]
    scale = float(actual @ expected / (expected @ expected))
    np.testing.assert_allclose(actual, scale * expected, atol=1e-5 * np.abs(expected).max())


def test_ramp_of_gaussian_matches_continuous_ramp_in_shape():
    sigma = 6.0
    u = np.arange(-64, 64, dtype=np.float64)
    projection = math.sqrt(2 * math.pi) * sigma * np.exp(-u**2 / (2 * sigma**2))
    # Continuous ramp |omega| of the Gaussian, without a discrete kernel or FFT.
    omega = np.linspace(0, 8 / sigma, 20001)
    integrand = (omega * np.exp(-sigma**2 * omega**2 / 2))[:, None]
    reference = np.trapezoid(integrand * np.cos(omega[:, None] * u[None, :]), omega, axis=0)
    actual = _ramp(projection[None, :], 2)[0]
    scale = float(actual @ reference / (reference @ reference))
    rms = np.sqrt(np.mean((actual - scale * reference) ** 2)) / np.abs(scale * reference).max()
    assert rms < 1e-4


# --------------------------------------------------------------------------- #
# Parker weights (MLX-side rebinning)
# --------------------------------------------------------------------------- #

def _trapezoid_weights(betas):
    gaps = np.diff(betas)
    w = np.zeros_like(betas)
    w[:-1] += 0.5 * gaps
    w[1:] += 0.5 * gaps
    return w


@pytest.mark.parametrize("coverage_degrees", [240, 330, 350])
def test_parker_overscan_integrates_each_detector_to_pi(coverage_degrees):
    """The fan width must not turn a 330 or 350 degree scan into a full scan,
    and overscan views must keep weights so that every ray counts once."""
    from diffct_mlx.rebinning import detector_fan_angles, parker_weights

    n_views = 2049
    gammas = detector_fan_angles(64, 1.0, 80.0)
    betas = np.linspace(0.0, math.radians(coverage_degrees), n_views)
    weights = _b.to_numpy(parker_weights(n_views, gammas, math.radians(coverage_degrees),
                                         betas=_arr(betas))).astype(np.float64)
    integrated = _trapezoid_weights(betas) @ weights
    np.testing.assert_allclose(integrated, math.pi, atol=5e-5)
    assert weights[0].max() < 1e-5
    assert weights[-1].max() < 1e-5


def test_parker_full_turn_with_narrow_fan_is_one():
    from diffct_mlx.rebinning import detector_fan_angles, parker_weights

    n_views = 360
    betas = np.arange(n_views) * (2 * math.pi / n_views)
    gammas = detector_fan_angles(2, 0.01, 800.0)
    weights = _b.to_numpy(parker_weights(n_views, gammas, 2 * math.pi, betas=_arr(betas)))
    np.testing.assert_allclose(weights, 1.0)
