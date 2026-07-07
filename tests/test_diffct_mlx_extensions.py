"""Regression tests for the diffct_mlx extension layer.

Covers the operator algebra, functionals, solver registry, denoiser / Plug-and-Play
regularizers, physics corrections + forward simulation, the embedded spectrum
library, sinogram-domain geometry (Parker / rebinning / laminography), the GPU
ramp, the analytic phantom engine and geometric self-calibration.

These exercise the torch/CUDA backend and are skipped without a GPU.
"""

import math

import numpy as np
import pytest
import torch

import diffct_mlx as dct
from diffct_mlx.backend import active as _b

xp = _b.xp


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _corr(a, b):
    return float(np.corrcoef(_b.to_numpy(a).ravel(), _b.to_numpy(b).ravel())[0, 1])


def _parallel_case(n=96, ndet=160, nv=180, mode="footprint"):
    ref = xp.array(dct.shepp_logan_2d((n, n)))
    A = dct.make_parallel_2d_operator(*dct.circular_trajectory_2d_parallel(nv),
                                      image_shape=(n, n), num_detectors=ndet, projector_mode=mode)
    return ref, A


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_operator_matched_adjoint_and_autograd():
    _skip_if_no_cuda()
    ref, A = _parallel_case()
    rng = np.random.default_rng(0)
    x = xp.array(rng.standard_normal(A.domain_shape).astype(np.float32))
    y = xp.array(rng.standard_normal(A.range_shape).astype(np.float32))
    lhs, rhs = float(xp.sum((A @ x) * y)), float(xp.sum(x * (A.T @ y)))
    assert abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12) < 5e-3

    b = A @ ref
    xg = x.clone().requires_grad_(True)
    (0.5 * xp.sum((A @ xg - b) ** 2)).backward()
    g_manual = A.T @ (A @ x - b)
    assert float(xp.norm(xg.grad - g_manual)) / (float(xp.norm(g_manual)) + 1e-12) < 1e-3


@pytest.mark.cuda
def test_operator_composition_and_subset():
    _skip_if_no_cuda()
    ref, A = _parallel_case()
    x = ref
    b = A @ x
    scale = float(xp.norm(2.0 * b)) + 1e-12
    # relative tolerance: two separate forward calls differ only by CUDA kernel noise
    assert float(xp.norm((2.0 * A) @ x - 2.0 * b)) / scale < 5e-3
    assert float(xp.norm((A + A) @ x - 2.0 * b)) / scale < 5e-3
    assert tuple((A.subset(slice(0, 20)) @ x).shape) == (20, A.range_shape[1])


# --------------------------------------------------------------------------
# Functionals
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_functionals():
    _skip_if_no_cuda()
    xv = xp.array([1.0, -3.0, 2.0])
    assert float(xp.norm(dct.SquaredL2(2.0).gradient(xv) - 2.0 * xv)) < 1e-5
    assert float(xp.norm(dct.SquaredL2(1.0).prox(xv, 1.0) - xv / 2.0)) < 1e-5
    st = dct.L1Norm(1.0).prox(xp.array([0.3, -0.3, 2.0]), 0.5)
    assert float(xp.norm(st - xp.array([0.0, 0.0, 1.5]))) < 1e-6
    assert float(xp.norm(dct.NonNegativity().prox(xv, 1.0) - xp.array([1.0, 0.0, 2.0]))) < 1e-6
    comp = 0.5 * dct.SquaredL2() + 0.1 * dct.TotalVariation()
    img = xp.array(np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32))
    assert comp.is_smooth and np.isfinite(float(comp.value(img)))


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_solver_registry_and_reconstruction():
    _skip_if_no_cuda()
    ref, A = _parallel_case(nv=140)
    b = A @ ref
    need = {"pcg", "ls", "wls", "rls", "rwls", "dls", "rdls", "mlem", "osem", "mltr", "cgls", "landweber"}
    assert need <= set(dct.list_algorithms())
    assert _corr(dct.reconstruct("cgls", A, b, iterations=30, nonnegative=True), ref) > 0.95
    assert _corr(dct.wls(A, b, iterations=25, nonnegative=True), ref) > 0.95
    assert _corr(dct.mlem(A, b, iterations=60), ref) > 0.95
    assert _corr(dct.osem(A, b, subsets=10, iterations=8), ref) > 0.95
    assert _corr(dct.rwls(A, b, regularizer=dct.TotalVariation(2e-3), iterations=50), ref) > 0.95


@pytest.mark.cuda
def test_mltr_physical_attenuation():
    _skip_if_no_cuda()
    ref, A = _parallel_case(nv=140)
    ref = ref * (4.0 / float(xp.max(A @ ref)))
    b = A @ ref
    I0 = 1.0e4
    counts = xp.array(I0) * xp.exp(-b)
    blank = xp.ones(A.range_shape) * I0
    assert _corr(dct.mltr(A, counts, blank, iterations=150), ref) > 0.95


# --------------------------------------------------------------------------
# Denoisers / Plug-and-Play
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_denoisers_and_pnp():
    _skip_if_no_cuda()
    yy, xx = np.mgrid[0:96, 0:96]
    disk = np.zeros((96, 96), np.float32)
    disk[((yy - 48) ** 2 + (xx - 48) ** 2) < 26 ** 2] = 1.0
    clean = xp.array(disk)
    rng = np.random.default_rng(0)
    noisy = clean + xp.array((0.06 * rng.standard_normal((96, 96))).astype(np.float32))
    base = float(xp.mean((noisy - clean) ** 2))
    for den in (dct.bilateral_filter(noisy, radius=2), dct.median_filter(noisy, 1),
                dct.dictionary_denoise(noisy, patch=8, stride=4)):
        assert float(xp.mean((den - clean) ** 2)) < base

    A = dct.make_parallel_2d_operator(*dct.circular_trajectory_2d_parallel(90),
                                      image_shape=(96, 96), num_detectors=140, projector_mode="footprint")
    b = A @ clean
    seq = dct.RegularizerSequence([dct.Bilateral(radius=1), dct.NonNegativity()])
    assert _corr(dct.rwls(A, b, constraint=seq, iterations=50), clean) > 0.9


# --------------------------------------------------------------------------
# Physics corrections + forward simulation
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_physics_corrections_and_pipeline():
    _skip_if_no_cuda()
    from diffct_mlx import physics as phys
    ref, A = _parallel_case(nv=120)
    p = A @ (ref * (4.0 / float(xp.max(A @ ref))))
    flat = xp.ones(tuple(p.shape)) * 3.0e4
    raw = flat * xp.exp(-p)
    assert float(xp.norm(phys.flat_field(raw, flat) - p)) / float(xp.norm(p)) < 1e-3

    def mse(u, v):
        return float(xp.mean((u - v) ** 2))
    bad = _b.to_numpy(p).copy()
    bad[np.random.default_rng(0).random(bad.shape) < 0.02] += 5.0
    bad = xp.array(bad.astype(np.float32))
    assert mse(phys.bad_pixel_correction(bad, threshold=3.0), p) < mse(bad, p)
    prep = phys.PreprocessingPipeline([phys.FlatField(flat=flat), phys.RingRemoval(radius=6)])
    assert np.isfinite(_b.to_numpy(prep(raw))).all()


@pytest.mark.cuda
def test_forward_simulation_beam_hardening():
    _skip_if_no_cuda()
    from diffct_mlx import physics as phys
    E = np.linspace(20, 120, 60)
    mu = 2.0e5 / E ** 3 + 0.2
    spec_mono = dct.Spectrum.monochromatic(70.0)
    spec_poly = dct.Spectrum(E, np.exp(-((E - 60) ** 2) / 800.0) * (120 - E))
    p = xp.array(np.linspace(0, 4, 50).astype(np.float32))
    assert float(xp.norm(phys.apply_beam_hardening(p, spec_mono, mu) - p)) < 1e-4  # mono = identity
    hardened = _b.to_numpy(phys.apply_beam_hardening(p, spec_poly, mu) - p)
    assert np.all(hardened <= 1e-4)                                                # hardening lowers attenuation

    counts = phys.add_poisson_noise(xp.ones((128, 128)) * 500.0, seed=0)
    assert abs(float(xp.mean(counts)) - 500.0) < 6.0

    n = 96
    ref, A = _parallel_case(n=n, ndet=150, nv=180, mode="siddon")
    sino = dct.simulate_scan(ref, A, spectrum=spec_poly, material_attenuation=mu, I0=3e4, poisson=True, seed=1)
    assert tuple(sino.shape) == A.range_shape and np.isfinite(_b.to_numpy(sino)).all()


# --------------------------------------------------------------------------
# Embedded spectrum library
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_spectrum_library():
    _skip_if_no_cuda()
    from diffct_mlx.physics import spectra
    assert 120.0 in spectra.available_kvps() and "bone" in spectra.available_materials()
    assert abs(spectra.tube_spectrum(120).mean_energy() - 45.3) < 1.5
    assert abs(spectra.tube_spectrum(120, [("Cu", 0.5)]).mean_energy() - 69.1) < 1.5
    mu60 = float(np.interp(60, spectra.energy_grid(), spectra.material_attenuation("water")))
    assert abs(mu60 - 0.206) < 0.02
    assert spectra.preset("industrial_160kVp_Cu1mm").mean_energy() > 90


# --------------------------------------------------------------------------
# Sinogram-domain geometry
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_geometry_rebinning_and_weighting():
    _skip_if_no_cuda()
    from diffct_mlx import rebinning as rb
    sid, sdd, du, n = 500.0, 800.0, 1.0, 128
    ndet = 300
    gmax = math.atan((ndet / 2) * du / sdd)
    ph = xp.array(dct.shepp_logan_2d((n, n)))

    # fan -> parallel rebinning then parallel FBP
    srcF, dcF, duvF = dct.circular_trajectory_2d_fan(360, sid, sdd, 0.0, 2 * math.pi)
    AF = dct.make_fan_2d_operator(srcF, dcF, duvF, image_shape=(n, n), num_detectors=ndet,
                                  detector_spacing=du, projector_mode="siddon")
    sino = AF @ ph
    s_max = sid * math.sin(gmax)
    dp = 2 * s_max / (ndet - 1)
    rd, do, duo = dct.circular_trajectory_2d_parallel(180, 0.0, 0.0, math.pi)
    Ap = dct.make_parallel_2d_operator(rd, do, duo, image_shape=(n, n), num_detectors=ndet,
                                       detector_spacing=dp, projector_mode="siddon")
    par = rb.fan_to_parallel(sino, sid=sid, sdd=sdd, detector_spacing=du,
                             source_angles=rb.view_angles(srcF), out_angles=rb.view_angles(rd),
                             num_out_positions=ndet, flip_s=True)
    pr = dct.FBPParameters(normalization_scale=math.pi / (2 * 180), enforce_positivity=False)
    assert _corr(dct.reconstruct_fbp(par, lambda t: Ap.T @ t, pr), ph) > 0.8

    # curved <-> flat round trip
    dgamma = math.atan(du / sdd)
    back = rb.curved_to_flat(rb.flat_to_curved(sino, sdd=sdd, flat_spacing=du, det_spacing_angle=dgamma),
                             sdd=sdd, det_spacing_angle=dgamma, flat_spacing=du)
    c0, c1 = ndet // 6, 5 * ndet // 6
    assert float(xp.norm(back[:, c0:c1] - sino[:, c0:c1])) / float(xp.norm(sino[:, c0:c1])) < 0.05


@pytest.mark.cuda
def test_laminography_forward_and_iterative():
    _skip_if_no_cuda()
    vol = xp.array(dct.shepp_logan_3d((16, 96, 96)))
    src, dc, duv, dvv = dct.laminography_trajectory_3d(120, sid=600., sdd=900., tilt_deg=30.0)
    A = dct.make_cone_3d_operator(src, dc, duv, dvv, volume_shape=(16, 96, 96),
                                  detector_shape=(140, 140), projector_mode="footprint")
    rec = dct.reconstruct("cgls", A, A @ vol, iterations=12, nonnegative=True)
    assert np.isfinite(_b.to_numpy(rec)).all() and _corr(rec, vol) > 0.5


# --------------------------------------------------------------------------
# GPU ramp
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_gpu_ramp_matches_cpu():
    _skip_if_no_cuda()
    from diffct_mlx.reconstruction_algorithms._analytic import _ramp_numpy, ramp_filter
    s = np.random.default_rng(0).standard_normal((180, 256)).astype(np.float32)
    out = ramp_filter(xp.array(s), axis=1)
    assert out.is_cuda
    rel = np.linalg.norm(_b.to_numpy(out) - _ramp_numpy(s, 1)) / np.linalg.norm(_ramp_numpy(s, 1))
    assert rel < 1e-4


# --------------------------------------------------------------------------
# Analytic phantom engine + self-calibration
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_phantom_engine_analytic_matches_numerical():
    _skip_if_no_cuda()
    shape = (64, 64, 64)
    phantom = dct.shepp_logan_phantom(fov_radius=32.0)
    vol = phantom.voxelize(shape, voxel_spacing=1.0)
    src, dc, duv, dvv = dct.circular_trajectory_3d(72, 400.0, 700.0)
    num = dct.cone_forward(vol, src, dc, duv, dvv, det_u=96, det_v=96, du=1.0, dv=1.0)
    ana = phantom.project(src, dc, duv, dvv, detector_shape=(96, 96), du=1.0, dv=1.0)
    assert _corr(num, ana) > 0.97
    ratio = float(xp.mean(xp.abs(num))) / max(float(xp.mean(xp.abs(ana))), 1e-9)
    assert 0.85 < ratio < 1.15


@pytest.mark.cuda
def test_center_of_rotation_recovery():
    _skip_if_no_cuda()
    ref, A = _parallel_case(n=128, ndet=180, nv=180, mode="siddon")
    sino = A @ ref
    base = dct.calibration.center_of_mass_offset(sino)
    rolled = xp.array(np.roll(_b.to_numpy(sino), 7, axis=1))
    assert abs(dct.estimate_center_of_rotation(rolled, method="com") - base - 7) < 1.5

    pr = dct.FBPParameters(normalization_scale=math.pi / (2 * 180), enforce_positivity=False)

    def recon_at(offset):
        s = xp.array(np.roll(_b.to_numpy(rolled), int(round(offset)), axis=1))
        return dct.reconstruct_fbp(s, lambda t: A.T @ t, pr)

    best, _ = dct.refine_center_by_sharpness(recon_at, range(-10, 11))
    assert abs(best - (-7)) <= 1
