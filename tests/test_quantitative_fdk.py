"""Quantitative FDK: amplitude-true reconstruction on realistic geometry.

Pins the three effects that made the legacy case path non-quantitative
(observed on measured camera data at ~0.10-0.16x of the true attenuation):

1. the unpadded ramp's circular-convolution wrap-around depresses large
   objects with an object-size-DEPENDENT bias (no constant can fix it),
2. the case normalization constant ``pi*sid/(2*sdd*N)`` is a synthetic
   unit-spacing calibration, not physics (real du/voxel/magnification break
   it),
3. the Siddon adjoint is not an FDK backprojector (needs voxel-driven
   ``(sid/U)^2`` weighting).

The quantitative path (``reconstruct_case_fdk`` / ``fdk_back_project``)
combines cosine weights, per-view angular weights, a zero-padded PHYSICAL
ramp (``|f|/du``) and the weighted voxel-driven gather backprojector, and
must recover true attenuation values on arbitrary geometry.
"""

import math

import pytest
import torch

import diffct_mlx as dct
from diffct_mlx.backend import xp
from diffct_mlx.reconstruction_algorithms._analytic import ramp_filter
from diffct_mlx.reconstruction_algorithms.cases import _quantitative_fdk_operators


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.cuda
def test_ramp_padding_removes_object_size_bias():
    """A wide box depressed by circular wrap-around must recover with padding."""
    _skip_if_no_cuda()
    n = 512
    box = torch.zeros(1, n, device=DEV)
    box[0, n // 8: 7 * n // 8] = 1.0          # 75 % of the detector window

    def interior_mean(filtered):
        return float(filtered[0, 3 * n // 8: 5 * n // 8].mean())

    unpadded = interior_mean(ramp_filter(box, axis=1, pad_factor=1))
    padded = interior_mean(ramp_filter(box, axis=1, pad_factor=2))
    reference = interior_mean(ramp_filter(box, axis=1, pad_factor=8))

    # wrap-around depresses the interior to ~50 % of the aperiodic reference;
    # 2x padding must remove at least 70 % of that bias
    assert abs(padded - reference) < 0.3 * abs(unpadded - reference), (
        unpadded, padded, reference)


@pytest.mark.cuda
def test_quantitative_fdk_recovers_true_attenuation_camera_geometry():
    """Cylinder of known mu at the measured-camera geometry
    (du != dv != voxel, magnification 2, object fills 75 % of the FOV):
    the quantitative path must recover mu; the legacy case path must not
    (regression-documenting the motivation)."""
    _skip_if_no_cuda()
    n, num_views = 128, 240
    vs, du_, dv_ = 0.278, 0.556, 0.556
    sid, sdd = 300.0, 600.0                    # magnification 2
    det_u = det_v = 272
    mu = 0.05                                  # 1/mm

    zz, yy, xx = torch.meshgrid(*[torch.arange(n, device=DEV, dtype=torch.float32)] * 3,
                                indexing="ij")
    r = torch.sqrt((xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2)
    cyl = ((r <= 0.375 * n) & ((zz - n / 2 + 0.5).abs() <= 0.35 * n)).float() * mu

    src, dc, duv, dvv = dct.circular_trajectory_3d(num_views, sid, sdd)
    sino = dct.cone_forward(cyl, src, dc, duv, dvv, det_u, det_v, du_, dv_, vs)
    interior = (r <= 0.25 * n) & ((zz - n / 2 + 0.5).abs() <= 0.2 * n)

    # quantitative trio (Siddon sinogram is voxel-unit -> rescale by vs)
    q_weight, q_filter, q_back = _quantitative_fdk_operators(
        src, dc, duv, dvv, (n, n, n), (det_u, det_v), du_, dv_, vs,
        sinogram_scale=vs)
    assert q_back is not None
    rec = q_back(q_filter(q_weight(sino)))
    ratio = float(rec[interior].median()) / mu
    assert 0.92 < ratio < 1.06, ratio

    # legacy path (unpadded DFT ramp + Siddon adjoint + synthetic constant)
    filt = ramp_filter(q_weight(sino), axis=1, pad_factor=1)
    legacy = dct.cone_backward(filt, src, dc, duv, dvv, n, n, n, du_, dv_, vs)
    legacy = legacy * (math.pi * sid) / (2.0 * sdd * num_views)
    legacy_ratio = float(legacy[interior].median()) / mu
    assert legacy_ratio < 0.7, legacy_ratio   # why the quantitative path exists


@pytest.mark.cuda
def test_reconstruct_case_fdk_quantitative_path():
    """build_cone_3d_case carries the quantitative trio; reconstruct_case_fdk
    returns an amplitude-true volume (least-squares scale vs phantom ~ 1)."""
    _skip_if_no_cuda()
    case = dct.build_cone_3d_case(volume_shape=(64, 64, 64), num_views=180,
                                  detector_shape=(160, 160), sid=300.0, sdd=600.0)
    assert case.supports_fdk and case.fdk_back_project is not None

    rec = dct.reconstruct_case_fdk(case)
    ref = case.reference
    scale = float(xp.sum(rec * ref) / xp.sum(ref * ref))
    corr_num = float(xp.sum((rec - xp.mean(rec)) * (ref - xp.mean(ref))))
    corr_den = float(xp.norm(rec - xp.mean(rec)) * xp.norm(ref - xp.mean(ref)))
    assert corr_num / corr_den > 0.9
    assert 0.85 < scale < 1.15, scale


@pytest.mark.cuda
def test_simulate_scan_uses_physical_line_integrals():
    """The physics chain (exp(-p), beam hardening) needs PHYSICAL line
    integrals; simulate_scan must rescale the voxel-unit projector output by
    the operator's voxel_spacing (silently wrong at voxel_spacing != 1
    before)."""
    _skip_if_no_cuda()
    from diffct_mlx.physics import spectra, apply_beam_hardening

    n, vs = 48, 0.278
    zz, yy, xx = torch.meshgrid(*[torch.arange(n, device=DEV, dtype=torch.float32)] * 3,
                                indexing="ij")
    r = torch.sqrt((xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2)
    phantom = ((r <= 0.3 * n) & ((zz - n / 2 + 0.5).abs() <= 0.3 * n)).float() * 0.05

    src, dc, duv, dvv = dct.circular_trajectory_3d(24, 150.0, 300.0)
    A = dct.make_cone_3d_operator(src, dc, duv, dvv, volume_shape=(n, n, n),
                                  detector_shape=(96, 96), du=0.556, dv=0.556,
                                  voxel_spacing=vs)
    assert abs(A.voxel_spacing - vs) < 1e-9          # metadata attached
    assert abs(A.subset(slice(0, 5)).voxel_spacing - vs) < 1e-9

    spec = spectra.preset("industrial_160kVp_Cu1mm")
    mu_e = spectra.material_attenuation("Al", spec.energies)

    sino = dct.simulate_scan(phantom, A, spectrum=spec, material_attenuation=mu_e,
                             poisson=False)
    # reference: the same physics applied to the explicitly physical integrals
    p_phys = (A @ phantom) * vs
    ref = apply_beam_hardening(p_phys, spec, mu_e)
    assert float((sino - ref).abs().max()) < 1e-4


@pytest.mark.cuda
def test_quantitative_fdk_object_size_sweep():
    """The wrap-around bias was object-size-DEPENDENT; the quantitative path
    must stay amplitude-true across object sizes (25/50/75 % of the FOV)."""
    _skip_if_no_cuda()
    n, num_views, vs, du_ = 64, 120, 0.278, 0.556
    det = 200
    mu = 0.05
    src, dc, duv, dvv = dct.circular_trajectory_3d(num_views, 300.0, 600.0)
    zz, yy, xx = torch.meshgrid(*[torch.arange(n, device=DEV, dtype=torch.float32)] * 3,
                                indexing="ij")
    r = torch.sqrt((xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2)
    zmask = (zz - n / 2 + 0.5).abs() <= 0.35 * n

    for frac in (0.125, 0.25, 0.375):          # 25 / 50 / 75 % of the FOV width
        cyl = ((r <= frac * n) & zmask).float() * mu
        sino = dct.cone_forward(cyl, src, dc, duv, dvv, det, det, du_, du_, vs) * vs
        qw, qf, qb = _quantitative_fdk_operators(
            src, dc, duv, dvv, (n, n, n), (det, det), du_, du_, vs, sinogram_scale=1.0)
        rec = qb(qf(qw(sino)))
        interior = (r <= 0.6 * frac * n) & ((zz - n / 2 + 0.5).abs() <= 0.2 * n)
        ratio = float(rec[interior].median()) / mu
        assert 0.90 < ratio < 1.08, (frac, ratio)


@pytest.mark.cuda
def test_quantitative_fdk_detector_pitch_invariance():
    """The reconstructed amplitude must not depend on the detector pitch
    (a du-free DFT-bin ramp scales the result by ~du/2 — evaluated and
    rejected; the physical ramp |f|/du plus the analytic constants is
    pitch-invariant)."""
    _skip_if_no_cuda()
    n, num_views, vs, mu = 64, 120, 0.278, 0.05
    zz, yy, xx = torch.meshgrid(*[torch.arange(n, device=DEV, dtype=torch.float32)] * 3,
                                indexing="ij")
    r = torch.sqrt((xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2)
    cyl = ((r <= 0.375 * n) & ((zz - n / 2 + 0.5).abs() <= 0.35 * n)).float() * mu
    src, dc, duv, dvv = dct.circular_trajectory_3d(num_views, 300.0, 600.0)
    interior = (r <= 0.25 * n) & ((zz - n / 2 + 0.5).abs() <= 0.2 * n)

    ratios = []
    for du_ in (1.0, 0.556, 0.278):
        det = int(n * vs * 2 / du_ * 1.35 / 2) * 2
        sino = dct.cone_forward(cyl, src, dc, duv, dvv, det, det, du_, du_, vs) * vs
        qw, qf, qb = _quantitative_fdk_operators(
            src, dc, duv, dvv, (n, n, n), (det, det), du_, du_, vs, sinogram_scale=1.0)
        rec = qb(qf(qw(sino)))
        ratios.append(float(rec[interior].median()) / mu)

    assert all(0.90 < x < 1.08 for x in ratios), ratios
    assert max(ratios) - min(ratios) < 0.04, ratios     # pitch-invariant
