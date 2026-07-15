"""Static contracts for Metal sources that also run on non-Apple CI."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
METAL = ROOT / "diffct_mlx" / "backend" / "metal"


def test_analytic_geometry_kernel_matches_cuda_conventions():
    """Pin the CUDA-validated endpoint-gradient formula in Metal source."""
    kernel_text = (METAL / "kernels" / "cone_beam.py").read_text()
    projector_text = (METAL / "projectors.py").read_text()

    start = kernel_text.index('_CONE_3D_GEOMETRY_GRAD_SOURCE = """')
    end = kernel_text.index('"""\n\ncone_3d_geometry_grad_kernel', start)
    source = kernel_text[start:end]

    # Trilinear interpolation is centered on voxel centers, not index nodes.
    for axis in "xyz":
        assert f"float f{axis} = mid_{axis} - 0.5f;" in source

    # Accumulate the projected moment with the same quadrature as G1/G2;
    # never reintroduce the biased closed-form A/L substitution.
    assert "S += w2 * (dmu_x*dir_x + dmu_y*dir_y + dmu_z*dir_z);" in source
    assert "A_over_L" not in source
    assert "float inv_vs = 1.0f / voxel_spacing;" in source
    assert "G1x + dir_x*S" in source
    assert "G2x - dir_x*S" in source

    # Analytic is the cross-backend default; fd and legacy false-like values
    # continue to select the compatibility fallback.
    assert 'os.getenv("DIFFCT_GEOMETRY_VJP", "analytic")' in projector_text
    assert '{"0", "false", "off", "fd"}' in projector_text
