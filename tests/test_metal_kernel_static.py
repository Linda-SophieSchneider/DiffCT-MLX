"""Static contracts for Metal sources that also run on non-Apple CI."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
METAL = ROOT / "diffct_mlx" / "backend" / "metal"


def test_analytic_geometry_kernel_matches_cuda_conventions():
    """Pin the exact face-crossing VJP of the cell-constant Siddon model.

    Before 3.0 the Metal kernel differentiated a trilinear-interpolated volume
    (``f = mid - 0.5`` cells, ``G1/G2`` quadrature) while the forward kernel is
    cell-constant, and the gradients were off by ~100 %. The kernel now ports
    the engine's ``_cone_3d_geometry_vjp_kernel``; numerical agreement is
    pinned in ``tests/test_backend_parity.py``.
    """
    kernel_text = (METAL / "kernels" / "cone_beam.py").read_text()
    projector_text = (METAL / "projectors.py").read_text()

    start = kernel_text.index('_CONE_3D_GEOMETRY_GRAD_SOURCE = """')
    end = kernel_text.index('"""\n\ncone_3d_geometry_grad_kernel', start)
    source = kernel_text[start:end]

    # No interpolated-volume model.
    for axis in "xyz":
        assert f"float f{axis} = mid_{axis} - 0.5f;" not in source
    assert "G1x" not in source and "dmu_x" not in source

    # Face coefficients c = f_before - f_after, tau derivatives (tau-1)/D, -tau/D.
    assert "float c = first ? -f : (f_prev - f);" in source
    assert "float da = c * (entry_tau_m1 + dtau);" in source
    assert "float db = -c * (entry_tau + dtau);" in source
    # Length term with S = accum / |D|.
    assert "float S = accum * inv_len;" in source
    assert "ga_x -= S * dir_x;" in source and "gb_x += S * dir_x;" in source
    # Same segment clipping as the forward kernel.
    assert "bool from_src =" in source and "ent_axis" in source and "ex_axis" in source

    # Analytic is the cross-backend default; fd and legacy false-like values
    # continue to select the compatibility fallback.
    assert 'os.getenv("DIFFCT_GEOMETRY_VJP", "analytic")' in projector_text
    assert '{"0", "false", "off", "fd"}' in projector_text
