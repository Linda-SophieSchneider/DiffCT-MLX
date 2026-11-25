# DiffCT-MLX: Differentiable CT on Apple Silicon and NVIDIA GPUs

[![PyPI version](https://img.shields.io/pypi/v/diffct-mlx.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/diffct-mlx.svg?style=flat-square)](https://pypi.org/project/diffct-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://linda-sophie-schneider.de/DiffCT-MLX/)
[![Docs build](https://img.shields.io/github/actions/workflow/status/Linda-SophieSchneider/DiffCT-MLX/docs.yml?branch=main&label=docs&style=flat-square)](https://github.com/Linda-SophieSchneider/DiffCT-MLX/actions)
[![DOI](https://img.shields.io/badge/DOI-10.20944%2Fpreprints202605.1446.v1-blue.svg?style=flat-square)](https://doi.org/10.20944/preprints202605.1446.v1)

**DiffCT-MLX** is a differentiable computed-tomography library with one
package, `diffct_mlx`, and two compute backends. It selects the backend at
import: **Apple MLX / Metal** on Apple Silicon and **Torch / numba-CUDA** on
NVIDIA GPUs. The same script runs on both. Projectors, geometry,
reconstruction and physics have the same names and signatures on either
backend, and the two backends produce the same numbers.

Every projector is differentiable with respect to the data, and the Siddon
projectors also with respect to the geometry. That makes the operators usable
as known operators in learned reconstruction, for trajectory optimization and
for self-calibration. Kernels take per-view source and detector arrays, so
circular, helical, saddle, sinusoidal, laminography and any user-supplied
trajectory run through the same code.

The Torch / CUDA engine is upstream [diffct](https://github.com/sypsyp97/diffct)
2.5.2 by Yipeng Sun (Apache-2.0), vendored and maintained here as
`diffct_mlx.engine`. See [ATTRIBUTION.md](ATTRIBUTION.md) for provenance.
Maintained by [Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider).

📖 **Documentation:** <https://linda-sophie-schneider.de/DiffCT-MLX/>

## 🆕 What is new in 3.0

- **One convention on both backends.** Detector pixel `i` sits at
  `det_center + (i − (N−1)/2)·du`, so `det_center` is the geometric centre of
  the panel. Projections are physical line integrals (µ · length in the unit
  of the geometry). Iterative reconstruction of physical data returns µ in
  1/mm, like the quantitative FDK. These are the conventions of upstream
  diffct ≥ 2.0.
- **Upstream diffct 2.5.2 as the Torch engine.** It brings exact geometry
  gradients, a portable PyTorch backend for CPU, Apple MPS and ROCm with
  `precision="float64"`, the chunked `Projector` with streaming stores,
  per-axis voxel spacing, `parallel3d`, curved detector surfaces, solvers and
  all upstream kernel fixes since 2.0.
- **Exact cone geometry gradient on Metal.** The Metal kernel now uses the
  same face-crossing derivative as the engine. The 2.x kernel differentiated
  a trilinear volume and was off by about 100 %.
- **Footprint projectors on the new engine.** The mass-conserving
  separable-footprint kernels of 2.2 are ported onto the engine in the new
  conventions, including the sparse cone backprojection.
- **Cross-backend parity.** MLX and the float64 engine agree to better than
  2·10⁻⁴ relative L2 in forward and backprojection and to better than 10⁻³ in
  the cone geometry gradient (`tests/test_backend_parity.py`).
- **No more top-level `diffct`.** Installing `diffct-mlx` no longer shadows
  the PyPI package `diffct`. Python ≥ 3.10 is required.

Upgrading from 2.x? See [Migrating from 2.x](#-migrating-from-2x) below.

## 🚀 Installation

```bash
pip install "diffct-mlx[mlx]"                         # Apple Silicon (MLX / Metal)
pip install "diffct-mlx[cuda]" "numba-cuda[cu12]"     # NVIDIA GPUs, CUDA 12
pip install "diffct-mlx[cuda]" "numba-cuda[cu13]"     # NVIDIA GPUs, CUDA 13
pip install "diffct-mlx[torch]"                       # portable PyTorch engine only (CPU, MPS, ROCm)
pip install "diffct-mlx[zarr]"                        # optional compressed out-of-core storage
```

On NVIDIA systems install a CUDA build of PyTorch first, following
[pytorch.org](https://pytorch.org/get-started/locally/).

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/Linda-SophieSchneider/DiffCT-MLX.git
cd DiffCT-MLX
pip install -e ".[mlx]"      # or ".[cuda]"
```

Check the installation:

```python
import diffct_mlx as dct
print(dct.__version__, dct.backend)   # e.g. '3.0.0 mlx' or '3.0.0 torch'
```

Set `DIFFCT_BACKEND=torch` or `DIFFCT_BACKEND=mlx` to force a backend.

## 🧭 Quick start

```python
import diffct_mlx as dct

# Cone-beam projection on either backend
src, det_c, det_u, det_v = dct.circular_trajectory_3d(360, sid=600, sdd=900)
sino = dct.cone_forward(volume, src, det_c, det_u, det_v, 256, 256, 1.0, 1.0, 1.0)

# A ready-made case with data and matched operators, reconstructed by FBP
case = dct.build_parallel_2d_case(image_shape=(256, 256), num_views=180)
reco = dct.reconstruct_fbp(
    case.sinogram, case.back_project_all,
    dct.FBPParameters(normalization_scale=case.fbp_normalization_scale),
    weight_projections=case.fbp_weight,
)

# Operator algebra plus a solver from the registry
A = dct.make_cone_3d_operator(src, det_c, det_u, det_v,
                              volume_shape=(128, 128, 128), detector_shape=(256, 256))
x = dct.reconstruct("cgls", A, sino, iterations=30)
```

## 📐 Projectors

Two projector families exist for parallel, fan and cone beams, with native
kernels and full autograd on both backends.

| Projector | Model | Best for |
|---|---|---|
| **Siddon** (`parallel_forward`, `fan_forward`, `cone_forward`, …) | line integral through a cell-constant volume | analytic reconstruction (FBP, FDK) and geometry gradients |
| **Footprint** (`*_forward_footprint`, `*_backward_footprint`) | separable footprint of the voxel on the detector, with a matched adjoint | iterative reconstruction and measured data |

FBP and FDK invert the X-ray transform, so the thin-ray Siddon model fits them
best. Iterative methods minimise `‖Ax − y‖`. There an area-integrated forward
model with an exact adjoint converges better and matches finite detector
pixels. The `build_*_case` helpers follow this split: `back_project_all`,
used by FBP and FDK, is Siddon, and `forward_single` and `back_single`, used
by the iterative solvers, are footprint. Choose explicitly with
`make_*_operators(..., projector_mode="siddon" | "footprint")`.

On the Torch backend `cone_backward_footprint(..., indices=idx)` evaluates
only the given flattened `(D, H, W)` voxels and returns a 1-D vector, for
masked or region-of-interest problems.

### Quantitative FDK

`dct.reconstruct_case_fdk(case)` returns attenuation in 1/mm. It applies
cosine pre-weights, per-view trapezoidal angular weights from the actual
source angles, a zero-padded physical ramp (`|f|/du`, `pad_factor=2`) and the
voxel-driven `(SID/U)²` FDK backprojector with its `SDD/(2π·SID)` constant.
A cylinder of known µ at a real camera geometry (du 0.556 mm, voxel
0.278 mm, magnification 2, object filling 75 % of the field of view)
reconstructs to 0.99 of the true value.

```python
case = dct.build_measured_cone_3d_case(cfg)
vol_mu = dct.reconstruct_case_fdk(case)        # attenuation in 1/mm
```

All cone cases (synthetic, measured TIFF, npy) carry the quantitative
operators (`case.fdk_weight`, `fdk_filter`, `fdk_back_project`).

## 🧰 Operators, solvers, regularizers and physics

- **Operators** (`diffct_mlx.operators`). A `LinearOperator` algebra:
  `A = dct.make_cone_3d_operator(...)`, then `A @ x`, `A.T @ y`, `A @ B`,
  `A + B`, `2*A`, `A.subset(views)`. Both directions flow autograd.
- **Solver registry.** `dct.reconstruct("wls", A, b, ...)`,
  `dct.list_algorithms()`, and `@dct.register_algorithm("name")` for your
  own. Built in: `cgls`, `landweber`, `ls`, `wls`, `rls`, `rwls`, `pcg`,
  `mlem`, `osem`, `mltr`, `dls`, `rdls`. Also FBP, FDK, SART, SIRT,
  TV-/ASD-/AwTV-POCS and DART.
- **Functionals and regularizers** (`diffct_mlx.functionals`,
  `diffct_mlx.filters`). `SquaredL2`, `L1Norm`, `LpNorm`, `Huber`,
  `TotalVariation`, `AdaptiveWeightedTV`, `NonNegativity`, `Box` and the
  denoisers `Bilateral`, `Guided`, `Median`, `HistogramSparsity`, `Azimuthal`,
  `DictionarySparsity`. Chain them into a `RegularizerSequence` for
  Plug-and-Play.
- **Physics and preprocessing** (`diffct_mlx.physics`). A
  `PreprocessingPipeline` of `FlatField`, `RingRemoval`, `BadPixel`,
  `BeamHardening`, `Deblur`, `Scatter` and `MetalArtifactReduction`.
- **Forward simulation** (`dct.simulate_scan`). Phantom to realistic data with
  a polychromatic spectrum, Poisson noise, detector blur, scatter and rings.
  An embedded, TASMICS-validated spectrum and material-attenuation library
  covers 40–225 kVp (`physics.spectra`).
- **Geometry.** Circular, helical, saddle, sinusoidal, random, laminography
  and custom trajectories; `diffct_mlx.rebinning` for Parker short-scan,
  offset and truncation weighting, fan-to-parallel and curved/flat detector
  rebinning.
- **Analytic phantoms** (`dct.Ellipsoid`, `dct.Phantom`,
  `dct.shepp_logan_phantom`) with exact line integrals for ground truth.
- **Self-calibration** (`dct.calibration`). `estimate_center_of_rotation`,
  `refine_center_by_sharpness`, `apply_center_offset`.

Simulate a polychromatic, noisy cone scan and reconstruct it with
TV-regularized least squares:

```python
import diffct_mlx as dct
from diffct_mlx.physics import spectra

vol = dct.shepp_logan_3d((128, 128, 128))
A = dct.make_cone_3d_operator(*dct.circular_trajectory_3d(360, 600, 900),
        volume_shape=(128, 128, 128), detector_shape=(256, 256), projector_mode="siddon")

spec = spectra.preset("industrial_160kVp_Cu1mm")
mu   = spectra.material_attenuation("bone", spec.energies)
sino = dct.simulate_scan(vol, A, spectrum=spec, material_attenuation=mu,
                         I0=3e4, poisson=True)

recon = dct.rwls(A, sino, regularizer=dct.TotalVariation(1e-3), iterations=60)
```

## 🎓 Gradients and known-operator learning

Every building block is differentiable end to end, so the operators work as
known operators in learned reconstruction: learnable FBP filters, unrolled
Landweber, SIRT or FISTA with trainable step sizes, Plug-and-Play priors.
`tests/test_known_operator_gradflow.py` checks that all projector operators
pass gradients to their data, that the `LinearOperator` algebra, ramp filter
and `reconstruct_fbp` are differentiable, and that unrolled schemes
backpropagate to step sizes and data.

**Geometry gradients.** Forward projectors return gradients with respect to
the per-view geometry arrays (`src_pos`, `det_center` or `det_origin`,
`det_u_vec`, `det_v_vec`, `ray_dir`) when those require grad. The data-only
path pays nothing.

| Projector | Torch / CUDA | MLX / Metal |
|---|---|---|
| Cone Siddon | exact | exact |
| Fan and parallel Siddon | exact | exact |
| Footprint, all beams | finite differences | cone `src_pos` by finite differences, others not yet |

The exact gradients differentiate the discrete cell-constant model. They are
exact wherever a ray does not run along a voxel face or edge. There both
backends return a one-sided derivative. On MLX the Siddon geometry kernels
match the engine's float64 gradients to better than 10⁻³
(`tests/test_backend_parity.py`). `DIFFCT_GEOMETRY_VJP=fd` switches MLX back
to the older behaviour: a finite-difference cone `src_pos` gradient and no
fan or parallel geometry gradients.

**Step sizes for unrolled networks.** A free trainable step can drift into
the divergent regime, where a positivity clamp collapses the iterate to zero
and all gradients vanish. Keep each step inside the stable range:

```python
from diffct_mlx import power_iteration

L = power_iteration(lambda v: A.T @ (A @ v), A.domain_shape)   # ||A^T A||
theta = torch.zeros(K, requires_grad=True)                     # one per iteration
for k in range(K):
    lam_k = (1.8 / L) * torch.sigmoid(theta[k])                # in (0, 1.8/L), starts at 0.9/L
    x = torch.clamp(x + lam_k * (A.T @ (y - A @ x)), min=0.0)
```

## 🗄️ Out-of-core and multi-GPU (CUDA)

`diffct_mlx.orchestration` reconstructs volumes larger than GPU memory and
larger than host RAM. It splits work into memory-budgeted chunks (z-slabs
for backprojection, detector-row bands for forward projection), spreads them
over all GPUs and streams data between disk, RAM and GPU:

```python
from diffct_mlx.orchestration import (
    ConeGeom, chunked_cone_fdk, chunked_sirt, mgpu_sirt,
    open_memmap, set_out_of_core_dir, set_out_of_core_backend)

set_out_of_core_dir("/mnt/bigdisk/scratch")
geom = ConeGeom.from_arrays(src, det_center, det_u_vec, det_v_vec)

sino = open_memmap("/mnt/bigdisk/sino.npy", (n_views, det_u, det_v))
vol  = chunked_cone_fdk(sino, geom, D, H, W, out=open_memmap("/mnt/bigdisk/vol.npy", (D, H, W)))

x = chunked_sirt(sino, geom, D, H, W, det_u, det_v, n_iter=30)   # out-of-core SIRT
x = mgpu_sirt(sino, geom, D, H, W, det_u, det_v, n_iter=30)      # in-VRAM, multi-GPU
```

Arrays that fit a host-RAM budget stay in RAM.
`set_out_of_core_backend("zarr")` swaps the memmap spill for compressed zarr
storage. Chunked and monolithic runs agree to about 10⁻⁶.

## ⚙️ The Torch engine (`diffct_mlx.engine`)

On the Torch backend the full upstream diffct 2.5.2 API is available under
`diffct_mlx.engine`, for features the unified API does not expose yet:

```python
from diffct_mlx import engine

P = engine.Projector((src, det_c, det_u, det_v), (D, H, W), (nu, nv), beam="cone",
                     detector_spacing=(du, dv), voxel_spacing=(sx, sy, sz),
                     backend="torch", precision="float64")   # also "numba" (CUDA) or "auto"
y = P(volume)
x = P.backproject(y)
```

This gives the portable backend on CPU, MPS and ROCm, float64 precision,
chunked and streamed projection, per-axis voxel spacing, `parallel3d`,
`ParameterizedSurface` detectors, `fbp`/`fdk`/`helical_fdk` and the engine's
`cgls`, `sirt` and `tv_reconstruction` solvers. On CUDA tensors the unified
Siddon projectors use the native kernels, on CPU and MPS tensors the portable
backend. The upstream documentation is in [`docs/engine/`](docs/engine/).

## 🔁 Migrating from 2.x

3.0 changes two defaults. Results of 2.x differ by half a detector pixel in
position and by the factor `voxel_spacing` in amplitude.

| | 2.x | 3.0 |
|---|---|---|
| Detector pixel `i` | `(i − N/2)·du` | `(i − (N−1)/2)·du` |
| Line integrals | voxel units | physical (µ · mm) |
| Iterative result of physical data | µ per voxel | µ in 1/mm |
| Torch engine | top-level `diffct` (dev line) | `diffct_mlx.engine` (upstream 2.5.2) |

To reproduce 2.x numbers, build operators with the old conventions:

```python
A = dct.make_cone_3d_operator(..., detector_convention="legacy", line_integrals="voxel")
ops = dct.make_cone_3d_operators(..., detector_convention="legacy", line_integrals="voxel")
```

For direct projector or FBP/FDK calls, shift the detector centre:

```python
det_c_legacy = dct.kernel_detector_center(det_c, det_u, du, det_v, dv, convention="legacy")
```

Projection stacks written by 2.x in voxel units load with
`NpyProjectionsConfig(projection_units="voxel")`. Replace
`import diffct` or `from diffct import ...` with `from diffct_mlx import engine`
or `from diffct_mlx.engine import ...`. If you need the old behaviour
unchanged, pin `diffct-mlx<3`.

## 🧩 Code structure

```text
DiffCT-MLX/
├── diffct_mlx/                 # the package (import this)
│   ├── backend/                # backend selection and the xp array namespace
│   │   ├── _torch.py           #   Torch adapter over diffct_mlx.engine
│   │   ├── _mlx.py             #   MLX adapter
│   │   └── metal/              #   Metal kernels and mx.custom_function projectors
│   ├── engine/                 # Torch / numba-CUDA engine (upstream diffct 2.5.2, vendored)
│   ├── conventions.py          # detector and unit conventions, legacy converters
│   ├── projectors.py           # unified functional projector API
│   ├── geometry.py             # trajectory generators, laminography, JSON loader
│   ├── operators.py            # differentiable LinearOperator algebra
│   ├── functionals.py          # objectives, regularizers, constraints
│   ├── filters.py              # edge-preserving denoisers, Plug-and-Play
│   ├── physics/                # corrections, pipeline, simulation, spectra
│   ├── rebinning.py            # Parker and offset weighting, rebinning
│   ├── calibration.py          # centre-of-rotation self-calibration
│   ├── phantoms/               # voxel phantoms and analytic phantom engine
│   ├── orchestration/          # out-of-core and multi-GPU (CUDA)
│   └── reconstruction_algorithms/  # FBP/FDK, SART/SIRT, POCS, DART, solver registry
├── examples/                   # circular and non-circular trajectory examples
├── tests/                      # unit, parity and regression tests; tests/engine: upstream suite
├── docs/                       # Sphinx sources; docs/engine: upstream engine docs
├── CHANGELOG.md
└── ATTRIBUTION.md
```

## 🧪 Tests

```bash
pytest tests/ -q                             # backend-specific tests skip where they do not apply
pytest tests/engine -q                       # upstream engine suite (Torch)
pytest tests/benchmarks/ --benchmark-only    # opt-in performance suite
```

## 🔀 Releases

Development happens on `main`. Tagged versions are published to
[PyPI](https://pypi.org/project/diffct-mlx/) through trusted publishing, and
`gh-pages` hosts the documentation. Changes are listed in
[CHANGELOG.md](CHANGELOG.md). Please report bugs as
[issues](https://github.com/Linda-SophieSchneider/DiffCT-MLX/issues).

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@article{202605.1446,
  doi       = {10.20944/preprints202605.1446.v1},
  url       = {https://doi.org/10.20944/preprints202605.1446.v1},
  year      = 2026,
  month     = {May},
  publisher = {Preprints},
  author    = {Yipeng Sun and Linda-Sophie Schneider and Chengze Ye and Andreas Maier},
  title     = {diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories},
  journal   = {Preprints}
}
```

## 📄 License

Apache 2.0, see [LICENSE](LICENSE). The vendored engine keeps upstream's
`LICENSE` and `NOTICE` in `diffct_mlx/engine/`.

## 🙏 Acknowledgements

The Torch / CUDA engine is [diffct](https://github.com/sypsyp97/diffct) by
Yipeng Sun. The project was inspired by
[PYRO-NN](https://github.com/csyben/PYRO-NN) and
[geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT).
