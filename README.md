# diffct: Differentiable Computed Tomography Operators

> [!NOTE]
> **This is the `cuda` branch of [DiffCT-MLX](https://github.com/Linda-SophieSchneider/DiffCT-MLX), adapted from the original [diffct](https://github.com/sypsyp97) by Yipeng Sun (Apache-2.0).**
> It is based on diffct's arbitrary-trajectory (`dev`) line, whose module layout and per-view projector convention match the MLX port. This branch hosts the unified **auto-backend** package `diffct_mlx` (MLX/Metal on Apple Silicon, Torch/numba-CUDA elsewhere) so the same code runs on both. Maintained by [Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider).
> See [ATTRIBUTION.md](ATTRIBUTION.md) for full provenance and license details.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

A high-performance, CUDA-accelerated library for CT reconstruction with
end-to-end differentiable operators, supporting both **canonical circular
orbits** and **arbitrary per-view trajectories** (spiral, saddle, random,
custom). Built for optimization and deep-learning integration.

⭐ **Please star this project if you find it useful!**

**Apple/MLX maintenance:** The former `apple` branch is maintained by
[Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider) at
[Linda-SophieSchneider/DiffCT-MLX](https://github.com/Linda-SophieSchneider/DiffCT-MLX).

## 🧭 Using this branch — the unified `diffct_mlx` API

This `cuda` branch adds **`diffct_mlx`**, a single API that auto-selects its
compute backend at import — **Torch / numba-CUDA** on NVIDIA GPUs, **Apple MLX**
on Apple Silicon — so the *same* script runs unchanged on both. It mirrors the
public API of the [DiffCT-MLX `main`](https://github.com/Linda-SophieSchneider/DiffCT-MLX)
(Apple) package name-for-name.

```bash
pip install "diffct-mlx[cuda]"   # NVIDIA GPUs (Torch + numba-CUDA)
pip install "diffct-mlx[mlx]"    # Apple Silicon (MLX)
```

```python
import diffct_mlx as dct
print(dct.backend)               # 'torch' on CUDA, 'mlx' on Apple
# force a backend with the DIFFCT_BACKEND env var ('torch' / 'mlx')

# projectors, geometry and reconstruction — identical calls on either backend
src, det_c, det_u, det_v = dct.circular_trajectory_3d(360, sid=600, sdd=900)
sino = dct.cone_forward(volume, src, det_c, det_u, det_v, 256, 256, 1., 1., 1.)

case = dct.build_parallel_2d_case(image_shape=(256, 256), num_views=180)
reco = dct.reconstruct_fbp(
    case.sinogram, case.back_project_all,
    dct.FBPParameters(normalization_scale=case.fbp_normalization_scale),
    weight_projections=case.fbp_weight,
)
```

**Parity & status.** FBP/FDK, SART/SIRT, TV-/ASD-/AwTV-POCS, DART, phantoms,
trajectory generators and the measured-data helpers are all available and were
verified on NVIDIA GPUs. The **MLX backend** is wired in
(`diffct_mlx/backend/metal/` vendors the Metal kernels and `mx.custom_function`
projectors; the `xp` namespace is attribute-identical to the torch one, pinned
by a static parity test) — runtime validation on Apple-Silicon hardware is the
remaining step.

### Projectors: Siddon vs. separable footprint

Two forward/adjoint projector families are available on every geometry
(parallel, fan, cone), both with native CUDA kernels and full autograd:

| Projector | Model | Best for |
|---|---|---|
| **Siddon** (`parallel_forward`, `fan_forward`, `cone_forward`, …) | line integral (thin ray) | **analytic** reconstruction (FBP/FDK) |
| **Footprint** (`*_forward_footprint`, `*_backward_footprint`) | separable footprint — the finite pixel/voxel area projected onto the detector, with a matched adjoint | **iterative / optimization** (SART, SIRT, POCS, DART) and real measured data |

**Why the split:** analytic FBP/FDK *invert the X-ray transform* (line
integrals), so a thin-ray Siddon model is the consistent, most accurate choice
there. Iterative methods instead minimise `‖A x − y‖`, where a more faithful,
area-integrated forward model with an exact adjoint (footprint) improves
convergence and image quality — and better matches real finite-width detector
pixels. Using footprint for analytic FBP is *slightly worse* (a forward-model
mismatch), and Siddon for iterative on real data is *slightly worse*; hence the
defaults below.

**Defaults.** The `build_*_case` helpers follow this policy automatically: the
synthetic sinogram is a Siddon line-integral "measurement", `back_project_all`
(used by `reconstruct_fbp` / `reconstruct_fdk`) is **Siddon**, and
`forward_single` / `back_single` (used by the iterative algorithms) are
**footprint**. To override, build operators yourself with
`make_*_operators(..., projector_mode="siddon" | "footprint")`.

On the Torch/CUDA backend the footprint cone backprojector also supports **sparse
evaluation**: `cone_backward_footprint(..., indices=idx)` computes only the given
flattened `(D, H, W)` voxels and returns a 1-D vector (useful for masked / DART
subproblems).

### Operators, solvers, regularizers, physics & simulation

The `cuda` branch adds a composable, differentiable toolkit on top of the
projectors (all GPU-resident, verified on NVIDIA GPUs):

- **Operators** (`diffct_mlx.operators`) — a `LinearOperator` algebra: build a
  projector as `A = dct.make_cone_3d_operator(...)`, then `A @ x`, `A.T @ y`,
  `A @ B`, `A + B`, `2*A`, `A.subset(views)`. `A @ x` and `A.T @ y` flow autograd,
  so analytic/iterative reconstruction and learned pipelines share one object.
- **Solver registry** — `dct.reconstruct("wls", A, b, ...)`, `dct.list_algorithms()`;
  register your own with `@dct.register_algorithm("name")`. Built in: `cgls`,
  `landweber`, `ls` / `wls` / `rls` / `rwls` (CG / FISTA least squares), `pcg`
  (preconditioned CG), `mlem`, `osem`, `mltr` (transmission Poisson ML), `dls` / `rdls`.
- **Functionals & regularizers** (`diffct_mlx.functionals`, `diffct_mlx.filters`) —
  `SquaredL2`, `L1Norm`, `LpNorm`, `Huber`, `TotalVariation`, `AdaptiveWeightedTV`,
  `NonNegativity`, `Box`, plus edge-preserving denoisers `Bilateral`, `Guided`,
  `Median`, `HistogramSparsity`, `Azimuthal`, `DictionarySparsity`. Chain them into a
  `RegularizerSequence` and pass `rwls(A, b, constraint=seq)` for Plug-and-Play.
- **Physics & preprocessing** (`diffct_mlx.physics`) — a `PreprocessingPipeline` of
  GPU-native corrections: `FlatField`, `RingRemoval`, `BadPixel`, `BeamHardening`,
  `Deblur`, `Scatter`, `MetalArtifactReduction`.
- **Forward simulation** (`dct.simulate_scan`) — phantom → realistic data with a
  measured or synthetic **spectrum** (beam hardening), Poisson noise, detector blur,
  scatter and ring effects. A physically-based (TASMICS-validated) spectrum +
  material-attenuation library is embedded: `physics.spectra.tube_spectrum(kvp, …)`,
  `physics.spectra.preset(…)`, `physics.spectra.material_attenuation(…)` (40–225 kVp).
- **Geometry** — `laminography_trajectory_3d`, helical (`spiral_trajectory_3d`), and
  `diffct_mlx.rebinning`: Parker short-scan / offset / truncation weighting,
  `fan_to_parallel` and `curved_to_flat` / `flat_to_curved` detector rebinning.
- **Analytic phantom engine** (`dct.Ellipsoid`, `dct.Phantom`, `dct.shepp_logan_phantom`)
  — voxelize *and* project analytically (exact line integrals) for ground-truth tests.
- **Self-calibration** (`dct.calibration`) — `estimate_center_of_rotation`,
  `refine_center_by_sharpness`, `apply_center_offset`.

End-to-end — simulate a realistic cone scan (measured-quality spectrum + noise) and
reconstruct it with a TV-regularized least-squares solver:

```python
import diffct_mlx as dct
from diffct_mlx.physics import spectra

vol = dct.shepp_logan_3d((128, 128, 128))
A = dct.make_cone_3d_operator(*dct.circular_trajectory_3d(360, 600, 900),
        volume_shape=(128, 128, 128), detector_shape=(256, 256), projector_mode="siddon")

spec = spectra.preset("industrial_160kVp_Cu1mm")             # TASMICS-validated spectrum
mu   = spectra.material_attenuation("bone", spec.energies)
sino = dct.simulate_scan(vol, A, spectrum=spec, material_attenuation=mu,
                         I0=3e4, poisson=True)               # beam hardening + Poisson noise

recon = dct.rwls(A, sino, regularizer=dct.TotalVariation(1e-3), iterations=60)
```

### Trainable known-operator pipelines

Every building block is differentiable end-to-end, so the operators can be
embedded directly as known operators in learned reconstruction networks
(learnable FBP filters, unrolled Landweber/SIRT/FISTA with trainable step
sizes, Plug-and-Play priors). Verified by `tests/test_known_operator_gradflow.py`:

- all 12 projector operators propagate gradients w.r.t. their data argument
  (exact `grad(0.5*||Ax||^2) == A^T(Ax)` identities, incl. the sparse cone
  backprojection);
- the `LinearOperator` algebra, ramp filter (`torch.fft` on-device) and the
  `reconstruct_fbp` driver are differentiable — e.g. learnable per-detector
  weights through `A.T @ ramp(w * sino)`;
- unrolled iterative schemes backprop to trainable step sizes **and** to the
  measured data (use a convergent step, e.g. `0.9 / power_iteration(...)`);
  `cgls` and `run_sirt` also work as differentiable blocks;
- `TotalVariation.gradient` / `tv_gradient` are built with `create_graph`
  when the input carries gradients, so unrolled TV gradient steps are
  second-order differentiable (mirrors `mx.grad` composability on MLX).

**Trainable geometry (pose/trajectory optimization).** All Torch forward
projectors provide gradients w.r.t. their per-view geometry arrays
(`src_pos`, `det_center`/`det_origin`, `det_u_vec`, `det_v_vec`, `ray_dir`),
computed only for inputs with `requires_grad` (the data-only path pays
nothing):

- **Cone Siddon** uses an **analytic geometry kernel** (default): one kernel
  pass yields closed-form gradients for all four geometry arrays — the
  per-ray endpoint derivatives of the trilinearly smoothed line integral,
  validated against an exact torch-autograd reference (cos > 0.998) and ~4×
  faster than finite differences. Set `DIFFCT_GEOMETRY_VJP=fd` to force the
  FD path instead.
- **Fan/parallel Siddon and all footprint forwards** use finite-difference
  VJPs (two forward passes per geometry component, magnitude-relative step —
  mm- and m-scale setups both work). The operator layer's default
  `projector_mode="footprint"` is therefore geometry-trainable too.

Semantics to know: the discrete forwards are piecewise-linear in the
geometry, so all of these are *smoothed* gradients — descent-quality in
practice (see the pose-recovery tests), but not second-order differentiable.
Backprojectors remain data-only, and stochastic simulation ops
(`add_poisson_noise`) are not differentiable. On MLX, the cone projector has
an FD `src_pos` VJP by default; its analytic kernel
(`DIFFCT_GEOMETRY_VJP=1`) is currently stale (see CHANGELOG known issues).

**Recommended step-size parametrization for unrolled networks.** A free
trainable step can wander into the divergent regime, where positivity clamps
collapse the iterate to zero and all gradients vanish exactly (a dead
network). Bound each unrolled step inside the stable region instead:

```python
from diffct_mlx import power_iteration

L = power_iteration(lambda v: A.T @ (A @ v), A.domain_shape)   # ||A^T A||
theta = torch.zeros(K, requires_grad=True)                     # one per iteration
for k in range(K):
    lam_k = (1.8 / L) * torch.sigmoid(theta[k])                # lam in (0, 1.8/L), init 0.9/L
    x = torch.clamp(x + lam_k * (A.T @ (y - A @ x)), min=0.0)
```

Equivalently, spectrally normalize once (`A_hat = (1.0 / L**0.5) * A`) and
train steps of order 1.

### Out-of-core & multi-GPU (TB-scale volumes)

`diffct_mlx.orchestration` reconstructs volumes far larger than GPU memory —
and larger than host RAM — by partitioning work into memory-budgeted chunks
(z-slabs for backprojection, detector-row bands for forward projection),
dispatching chunks across all GPUs, and streaming data disk↔RAM↔GPU through an
async conveyor (reader / GPU workers / writer threads with bounded queues):

```python
from diffct_mlx.orchestration import (
    ConeGeom, chunked_cone_fdk, chunked_sirt, chunked_os_sart,
    mgpu_sirt, open_memmap, set_out_of_core_dir, set_out_of_core_backend)

set_out_of_core_dir("/mnt/bigdisk/scratch")     # where >RAM arrays spill
geom = ConeGeom.from_arrays(src, det_center, det_u_vec, det_v_vec)

# FDK for a volume that fits neither VRAM nor RAM: sinogram + volume live on
# disk; the ramp filter, backprojection and positivity all stream in chunks.
sino = open_memmap("/mnt/bigdisk/sino.npy", (n_views, det_u, det_v))
vol  = chunked_cone_fdk(sino, geom, D, H, W, out=open_memmap("/mnt/bigdisk/vol.npy", (D, H, W)))

x = chunked_sirt(sino, geom, D, H, W, det_u, det_v, n_iter=30)   # fully out-of-core SIRT
x = mgpu_sirt(sino, geom, D, H, W, det_u, det_v, n_iter=30)      # in-VRAM, view-parallel multi-GPU
```

The RAM-vs-disk choice is automatic (arrays that fit a host-RAM budget stay in
RAM); `set_out_of_core_backend("zarr")` swaps the raw memmap spill for chunked
+ compressed zarr storage. Verified on 2× RTX PRO 6000: chunked == monolithic
to ~1e-6, 2-GPU == 1-GPU exact, 2048³ cone-FDK end-to-end with a bounded
working set. This path is CUDA-only (it drives the numba-CUDA kernels
directly).

## 🔀 Branches

### `main` — unified auto-backend package
The unified `diffct_mlx` package described above: one API, two engines
(Torch/numba-CUDA and MLX/Metal), auto-selected at import. This is the branch
to use on every platform.

### `cuda` — development line of the unified package
Where the unification work happens before it lands on `main`; after a merge
the two branches are identical.

Kernels take per-view ``(src_pos, det_center, det_u_vec[, det_v_vec])``
arrays instead of closed-form ``sdd / sid / beta`` scalars, so **spiral,
saddle, sinusoidal, laminography, or any user-supplied trajectory** works
without touching the kernels — on both backends.

⚠️ **Note:** not published to PyPI yet. If you find any bugs please
[raise an issue](https://github.com/Linda-SophieSchneider/DiffCT-MLX/issues).

## ✨ Features

- **Fast:** CUDA-accelerated forward and backward projectors (Numba
  CUDA kernels), coalesced memory access for the FDK gather.
- **Differentiable:** End-to-end gradient propagation via
  ``torch.autograd``; every projector / backprojector pair is
  byte-accurate adjoints verified by ``tests/test_adjoint_inner_product.py``
  and ``tests/test_gradcheck.py``.
- **Arbitrary trajectories:** Kernels consume per-view source /
  detector position arrays, so circular, spiral, saddle, sinusoidal
  or any user-supplied orbit works from the same code path. See
  ``diffct.geometry`` for built-in trajectory generators.
- **Analytical reconstruction:** Amplitude-calibrated FBP / FDK
  pipelines via ``ramp_filter_1d``, ``fan_cosine_weights`` /
  ``cone_cosine_weights``, ``parker_weights``,
  ``angular_integration_weights``, and
  ``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
  ``cone_weighted_backproject``. Each wrapper dispatches to a
  dedicated voxel-driven gather kernel with the correct
  ``(sid_n / U_n)^2`` weighting and Fourier-convention constant.
- **Modular:** Library split into ``diffct.projectors``,
  ``diffct.geometry``, ``diffct.analytical``, ``diffct.kernels``,
  ``diffct.utils``, ``diffct.constants``. ``diffct.differentiable``
  is retained as a deprecated backward-compatibility shim.
- **Two projector families:** thin-ray Siddon *and* separable-footprint
  forward/adjoint pairs for every geometry (parallel/fan/cone), all with
  native kernels on both backends, including a sparse cone backprojection
  (``indices=``) for region-of-interest gradients.
- **Out-of-core + multi-GPU:** ``diffct_mlx.orchestration`` streams
  TB-scale volumes through chunked, conveyor-pipelined, multi-GPU
  projection/backprojection/FDK/SIRT/OS-SART with automatic RAM/disk spill
  (memmap or zarr).
- **Tested:** 97 pytest tests covering adjoint identity, gradcheck, smoke,
  accuracy, offset handling, ramp-filter windows, the operator/solver/
  physics stack and regression pins from review sessions. Opt-in
  ``pytest-benchmark`` perf suite under ``tests/benchmarks/``.

## 📐 Supported Geometries

- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

Every geometry supports both canonical circular orbits (via the
``circular_trajectory_*`` helpers) and arbitrary trajectories (any
user-supplied ``(n_views, 2 or 3)`` tensors).

## 🧩 Code Structure

```bash
DiffCT-MLX/
├── diffct_mlx/                # THE unified auto-backend package (import this)
│   ├── backend/               # backend selection + xp array namespace
│   │   ├── _torch.py          #   Torch/CUDA adapter over the vendored diffct
│   │   ├── _mlx.py            #   MLX/Metal adapter
│   │   └── metal/             #   vendored Metal kernels + mx projectors (from main)
│   ├── projectors.py          # unified functional projector API (both families)
│   ├── geometry.py            # trajectory generators + laminography + JSON loader
│   ├── operators.py           # differentiable LinearOperator algebra
│   ├── functionals.py         # objectives / regularizers / constraints
│   ├── filters.py             # edge-preserving denoisers + Plug-and-Play
│   ├── physics/               # corrections, pipeline, simulation, spectra
│   ├── rebinning.py           # Parker/offset weighting, fan→par, curved↔flat
│   ├── calibration.py         # center-of-rotation self-calibration
│   ├── phantoms/              # voxel phantoms + analytic phantom engine
│   ├── orchestration/         # out-of-core + multi-GPU chunking (CUDA)
│   └── reconstruction_algorithms/  # FBP/FDK, SART/SIRT, POCS, DART, solver registry
├── diffct/                    # vendored Torch/numba-CUDA engine (upstream dev line)
│   ├── projectors.py          # autograd Function classes
│   ├── footprint.py           # separable-footprint autograd Functions (+ sparse)
│   ├── analytical.py          # ramp filter, cosine weights, Parker, FBP/FDK wrappers
│   ├── geometry.py            # trajectory generators (circular, spiral, ...)
│   ├── kernels/               # Siddon + footprint + FBP/FDK gather CUDA kernels
│   └── differentiable.py     # deprecated compat shim
├── examples/
│   ├── circular_trajectory/   # canonical circular-orbit examples (fbp/fdk + iterative)
│   ├── non_circular_trajectory/  # spiral / custom trajectory examples
│   └── plot_trajectory.py     # visualise a trajectory generator
├── tests/
│   ├── test_*.py              # adjoint / gradcheck / accuracy / weights / ramp-filter
│   └── benchmarks/            # opt-in pytest-benchmark perf suite
├── docs/                      # Sphinx documentation sources
├── pyproject.toml
├── pytest.ini
├── CHANGELOG.md               # dev-branch change log
├── README.md
└── LICENSE
```

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/), [Numba](https://numba.readthedocs.io/en/stable/user/installing.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)

### Installation

`dev` is not on PyPI — install it from source by cloning the
repository and using an editable install.

**CUDA 12 (recommended):**
```bash
# Clone the repository
git clone https://github.com/Linda-SophieSchneider/DiffCT-MLX.git
cd DiffCT-MLX

# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 12.8.1 as example) and PyTorch, and Numba
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# Install PyTorch, follow: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 12
pip install numba-cuda[cu12]

# Install diffct (editable)
pip install -e .
```

<details>
<summary>CUDA 13 installation</summary>

```bash
git clone https://github.com/Linda-SophieSchneider/DiffCT-MLX.git
cd DiffCT-MLX
conda create -n diffct python=3.12
conda activate diffct
conda install nvidia/label/cuda-13.0.2::cuda-toolkit
# Install PyTorch from https://pytorch.org/get-started/locally/
pip install numba-cuda[cu13]
pip install -e .
```

</details>

<details>
<summary>CUDA 11 installation</summary>

```bash
git clone https://github.com/Linda-SophieSchneider/DiffCT-MLX.git
cd DiffCT-MLX
conda create -n diffct python=3.12
conda activate diffct
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
# Install PyTorch from https://pytorch.org/get-started/locally/
pip install numba-cuda[cu11]
pip install -e .
```

</details>

### Running the tests

```bash
pytest tests/ -q                             # 97 tests, ~5 s
pytest tests/benchmarks/ --benchmark-only    # opt-in perf suite
```

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@article{202605.1446,
  doi       = {10.20944/preprints202605.1446.v1},
  url       = {https://doi.org/10.20944/preprints202605.1446.v1},
  year      = 2026,
  month     = {May},
  publisher = {Preprints},
  author    = {Yipeng Sun and Linda-Sophie Schneider and Chengze ye and Andreas Maier},
  title     = {diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories},
  journal   = {Preprints}
}
```

## 📄 License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project was highly inspired by:

- [PYRO-NN](https://github.com/csyben/PYRO-NN)
- [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)

Issues and contributions are welcome!
