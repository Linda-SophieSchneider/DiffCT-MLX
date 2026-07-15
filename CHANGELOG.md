# Changelog

All notable changes to DiffCT-MLX are documented in this file. Entries up to
1.3.0.dev0 track the vendored ``diffct`` dev line this repository was adapted
from; entries from 2.0.0.dev0 on track the unified ``diffct_mlx`` package.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0] - 2026-07-10

First stable release of the unified package. Includes everything in
[2.0.0.dev0] below; on top of it, the MLX backend is now runtime-validated
on Apple Silicon (the last open item of the unification) and the Torch
backend gains trainable projector geometry.

### Added

- **Analytic geometry gradients for the cone Siddon projector (CUDA)**:
  `_cone_3d_geometry_grad_kernel` computes closed-form per-view gradients
  for `src_pos`, `det_center`, `det_u_vec` and `det_v_vec` in a single
  kernel pass (endpoint derivatives of the trilinearly smoothed line
  integral, with the projected moment accumulated by the same segment
  quadrature as the gradient terms). Default for cone geometry gradients
  (`DIFFCT_GEOMETRY_VJP=fd` forces finite differences); validated against
  an exact torch-autograd `grid_sample` reference (cos > 0.998, norm ratio
  within 1%) and ~4× faster than the FD path.
- **Footprint forward projectors are geometry-trainable** (parallel, fan,
  cone): finite-difference VJPs for their geometry arrays, so the operator
  layer's default `projector_mode="footprint"` — and pipelines built on it,
  e.g. differentiable view-coverage objectives — receive source/detector
  gradients. Backprojectors remain data-only.

### Known issues

- The **Metal** analytic geometry kernel (`cone_3d_geometry_grad_kernel`,
  `DIFFCT_GEOMETRY_VJP=1` on MLX) is stale: it still samples at integer-index
  nodes (pre-d668ede convention) and its closed-form `-A/L` substitution
  carries a quadrature-mismatch bias in the translation gradients. Use the
  default FD `src_pos` VJP on MLX until it is ported to match the CUDA
  kernel (see the FIXME in `diffct_mlx/backend/metal/kernels/cone_beam.py`).

- **Trainable geometry on the Torch backend**: the Siddon projectors
  (parallel/fan/cone) return finite-difference VJPs for their per-view
  geometry arrays (`src_pos`, `det_center`/`det_origin`, `det_u_vec`,
  `det_v_vec`, `ray_dir`) — computed only for inputs that `requires_grad`,
  with magnitude-relative step sizes (mm- and m-scale setups both work).
  Enables pose/trajectory optimization and geometry-aware known-operator
  networks; verified by an exact matched-step contraction test and a
  pose-recovery integration test. Footprint projectors remain data-only;
  the FD gradients are smoothed subgradients (Siddon is piecewise-linear
  in geometry) and not second-order differentiable.
- `power_iteration` re-exported at the top level; README documents the
  bounded step-size parametrization for unrolled networks
  (`(1.8/L)*sigmoid(theta)`).
- Known-operator gradient-flow guarantees: torch `xp.grad` is composable
  (`create_graph` when the input carries gradients — unrolled TV steps are
  second-order differentiable, mirroring `mx.grad`), correct under
  `torch.no_grad()`; pinned by `tests/test_known_operator_gradflow.py`.

### Fixed (MLX runtime validation on Apple Silicon, 2026-07-10)

- **Metal Siddon kernels aligned with the CUDA reference** (parallel, fan
  and cone, forward *and* adjoint): the vendored kernels interpolated the
  image/volume bi-/trilinearly on nodes at **integer** indices — voxel
  centers live at `k + 0.5 - c`, so every Siddon projection was shifted by
  half a voxel per axis against the footprint kernels, the FBP/FDK gathers
  and the analytic phantom projector (view-dependent detector shifts up to
  ±1.24 px at fan/cone magnification 1.75). They now accumulate the
  traversed cell (`image[cell] * seg_len`), exactly like the CUDA kernels.
  **Geometry-affecting** for MLX-Siddon users: results shift by half a
  voxel relative to earlier MLX builds. Siddon-vs-footprint cone
  correlation on a 64³ Shepp-Logan rises from 0.972 to 0.994, case-builder
  FDK from 0.923 to 0.986.
- **Metal cone Siddon forward had a stray `voxel_spacing` factor** its
  adjoint lacked, breaking the forward/adjoint pairing (and CUDA parity)
  for `voxel_spacing != 1`.
- Regression-pinned in the backend-neutral `tests/test_siddon_centering.py`
  (centered-object centroid probes, anisotropic-spacing adjoint,
  Siddon-vs-footprint agreement), which runs on MLX and torch/CUDA alike.


## [2.0.0.dev0] - 2026-07-10

The unified, auto-backend **`diffct_mlx`** package: one API that runs on
Torch/numba-CUDA (NVIDIA) and MLX/Metal (Apple Silicon), selected at import
(`DIFFCT_BACKEND` overrides).

### Added

- **Unified API** with full parity to the Apple-Silicon `main` package:
  functional projectors (Siddon + separable footprint, all geometries,
  sparse cone backprojection), trajectory generators (+ laminography),
  FBP/FDK, SART/SIRT/normalized-SART, TV-/ASD-/AwTV-POCS, DART, case
  builders for simulated/measured/npy data, measured-data helpers.
- **Native CUDA footprint kernels** for parallel/fan/cone plus a sparse cone
  footprint backprojection (`indices=` → 1D vector).
- **Operator/functional/solver framework**: differentiable `LinearOperator`
  algebra with `ProjectionOperator.subset`, functionals
  (L1/Lp/Huber/TV/AwTV/Box/...), solver registry (`cgls`, `pcg`,
  `ls`/`wls`/`rls`/`rwls`, `dls`/`rdls`, `mlem`/`osem`, `mltr`,
  `landweber`) and Plug-and-Play denoisers (bilateral, guided, median,
  histogram-sparsity, azimuthal, dictionary).
- **Physics & simulation**: GPU-native preprocessing (flat field, rings, bad
  pixels, beam hardening, Wiener deblur, scatter, MAR inpainting) with a
  `PreprocessingPipeline`; forward simulation (`simulate_scan`) with an
  embedded, offline-generated X-ray spectrum + material-attenuation library
  (W anode, 40–225 kVp).
- **Sinogram-domain tools**: Parker short-scan and offset-detector
  weighting, truncation extension, fan→parallel and curved↔flat rebinning —
  backend-neutral.
- **Analytic phantom engine** (`Ellipsoid`/`Phantom`, exact cone-beam line
  integrals) and center-of-rotation self-calibration.
- **Out-of-core + multi-GPU orchestration** (CUDA): chunked
  forward/backward/FDK/SIRT/OS-SART with z-slab / v-band shadow geometry,
  async disk↔GPU conveyor, automatic RAM/disk spill (memmap or zarr),
  GPU-FFT streaming ramp filter; validated to 2048³ with a bounded working
  set.
- **MLX backend wiring**: the Metal kernels + `mx.custom_function`
  projectors from `main` are vendored under `diffct_mlx/backend/metal/` and
  adapted in `backend/_mlx.py`; the `xp` namespace is attribute-identical to
  the torch shim (pinned by a static parity test). Runtime validation on
  Apple hardware pending.

### Fixed (review/debug session, 2026-07-10)

- **Detector/voxel grid conventions standardized** to `(k - n/2) * spacing`
  package-wide: the CUDA *and* Metal cone footprint kernels (and the Metal
  cone Siddon kernels) used `(n-1)/2` — a half-pixel shift whenever kernel
  families were mixed; the FBP/FDK gather kernels placed voxel centers at
  `ix - c` instead of `ix + 0.5 - c` — a half-voxel shift of analytic
  volumes against iterative ones. Rebinning helpers aligned to the same
  grid. **Geometry-affecting** for cone-footprint / FDK-gather /
  MLX-cone-Siddon users: results shift by half a pixel/voxel relative to
  earlier builds.
- **Parker weights** in `rebinning` had half the correct feathering
  argument (conjugate-ray sums didn't normalize); now exactly matches the
  validated `diffct.analytical` formula, and real per-view angles can be
  passed via `betas`.
- Rebinning gathers build flat indices in int64 (float32 index math
  silently corrupted arrays above 2^24 elements); `curved_to_flat` /
  `flat_to_curved` resample the u axis (axis 1) of 3D stacks; single-view /
  non-uniform `fan_to_parallel` inputs now raise instead of mis-rebinning.
- Warm-started SART/SIRT no longer skip their first iteration; `rls` no
  longer discards `weights`; `rwls` rejects non-smooth regularizers instead
  of silently ignoring them; AwTV-POCS actually anneals `alpha`; POCS
  drivers reuse the geometry-only sweep cache; `normalized_sart_relaxation`
  is honored (`backprojection_scale` defaults to `None`); DART's masked
  subproblem no longer clips negative residual measurements.
- `build_npy_cone_3d_case` kept detector counts consistent with the data
  axes under `transpose_uv` (non-square detectors crashed; the expected
  stack order is now documented as `(views, v, u)` with the default
  `transpose_uv=True`).
- `scatter_correction`/`add_scatter` blur per view in the detector plane
  only (views used to mix); `box_filter` is separable (+`axes=`) and ~1000×
  cheaper at radius 12 in 3D; `detector_deblur` has unit DC gain (values
  were shrunk by ~`reg`); `auto_voxel_spacing_from_detector` bounds both
  lateral axes by the horizontal FOV; `load_tiff_projections` applies
  `revert` before the log transform; `estimate_cone_isocenter` survives a
  single view.
- Out-of-core conveyor propagates worker exceptions (a failing thread used
  to deadlock the run); auto disk scratch is cleaned up (TB-scale leaks);
  zarr gather writes are serialized; the gather path uploads the filtered
  sinogram once per GPU; `chunked_sirt` takes memmap/zarr inputs lazily;
  memmap outputs are flushed; the caller's CUDA device is restored.
- Backend hardening: auto-detection picks MLX only when Metal is available
  (Linux MLX wheels no longer shadow CUDA); `xp.grad` returns zeros for
  disconnected objectives (single-slice `tv_gradient` crashed); `xp.clip`
  with no bounds, `Box(-inf, inf).prox`, `xp.max(axis=...)` fixed;
  `ProjectionOperator.subset(slice)` reports correct metadata; sparse cone
  backprojection guards int32 overflow and empty index vectors.

### Changed

- Package renamed/versioned as `diffct-mlx 2.0.0.dev0` with extras
  `[cuda]` / `[mlx]` / `[zarr]`; docs (README, Sphinx) rewritten for the
  unified package; dead code removed (unused `_trig_tables`, eager
  deprecated-shim import, never-read parameters).

## [1.3.0.dev0] - 2026-04-14

First sync of the dev (arbitrary-trajectory) branch against the main
branch's 1.2.10 / 1.2.11 analytical reconstruction overhaul. Brings
dev up to functional parity with main 1.2.11 except for the 1.3.0
separable-footprint (SF) projector backends, which rely on closed-form
circular-orbit geometry and are not yet generalised to arbitrary
trajectories.

### Added

#### Analytical reconstruction helpers (``diffct.analytical``)

A new module exposes the following helpers, all trajectory-agnostic so
they work with both the circular trajectories and the arbitrary
``(src_pos, det_center, det_u_vec[, det_v_vec])`` trajectory arrays
that dev-branch kernels already accept:

- ``detector_coordinates_1d`` — detector cell centre coordinates.
- ``angular_integration_weights`` — trapezoidal per-view weights for
  the analytical FBP/FDK sum. Optional ``redundant_full_scan`` flag
  absorbs the ``1/2`` redundancy factor for full scans.
- ``fan_cosine_weights`` / ``cone_cosine_weights`` — per-detector-cell
  ``cos(gamma)`` pre-weight for fan / cone FBP pipelines.
- ``parker_weights`` — Parker short-scan redundancy weights for
  circular fan geometries.
- ``ramp_filter_1d`` — generic 1D ramp filter with ``sample_spacing``,
  ``pad_factor``, ``window`` (``"ram-lak"``, ``"hann"``, ``"hamming"``,
  ``"cosine"``, ``"shepp-logan"``), and ``use_rfft`` options, rescaled
  by ``1 / sample_spacing`` so the output is in physical units.
- ``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
  ``cone_weighted_backproject`` — voxel-driven FBP / FDK backprojection
  wrappers that dispatch to new dedicated gather kernels and apply the
  analytical Fourier-convention constant (``1/(2*pi)`` for parallel,
  ``sdd_mean/(2*pi*sid_mean)`` for fan and cone). These are the
  **recommended path for analytical reconstruction** and replace the
  previous pattern of passing a filtered sinogram through
  ``*BackprojectorFunction.apply``.

#### Voxel-driven FBP / FDK gather kernels

Three new CUDA kernels, each under the dedicated ``fastmath=False``
``_FDK_ACCURACY_DECORATOR`` added to ``diffct.constants``:

- ``_parallel_2d_fbp_backproject_kernel`` — no distance weighting.
- ``_fan_2d_fbp_backproject_kernel`` — ``(|S|/U_n)^2`` weighted,
  where ``U_n`` is the signed distance from the per-view source to
  the voxel along the detector normal.
- ``_cone_3d_fdk_backproject_kernel`` — same pattern in 3D.

Every kernel is voxel-driven: one thread per output pixel/voxel, loops
over views inside, projects the voxel onto the detector using the
per-view ``(src_pos, det_center, det_u_vec[, det_v_vec])`` arrays,
bilinearly samples the filtered sinogram, applies the per-view weight
and accumulates. They are completely separate from the pure Siddon
adjoint kernels that back the autograd path; autograd is untouched.

#### Tests (``tests/``)

The dev branch had no test directory at all prior to this change.
``pytest.ini`` and ``tests/__init__.py`` are new, plus 58 tests across
11 files that mirror the main-branch test layout:

- ``tests/test_adjoint_inner_product.py`` — ``<A x, y> = <x, A^T y>``
  identity for parallel / fan / cone autograd pairs, plus an extra
  ``test_cone_autograd_backward_matches_backprojector_forward`` that
  protects the autograd ``ConeProjectorFunction.backward`` from
  drifting away from the standalone ``ConeBackprojectorFunction``.
- ``tests/test_gradcheck.py`` — ``torch.autograd.gradcheck`` for every
  projector Function with float32-calibrated tolerances.
- ``tests/test_weights.py`` — unit tests for ``detector_coordinates_1d``,
  ``angular_integration_weights``, ``fan_cosine_weights``,
  ``cone_cosine_weights`` and ``parker_weights``.
- ``tests/test_cuda_smoke.py`` — end-to-end smoke tests for every
  Projector / Backprojector Function pair + the analytical
  ``*_weighted_backproject`` wrappers.
- ``tests/test_cone_projector_autograd.py`` — gradient finiteness and
  sparsity guard for the cone projector Function (circular + spiral
  trajectories).
- ``tests/test_fbp_parallel_accuracy.py`` /
  ``tests/test_fbp_fan_accuracy.py`` /
  ``tests/test_fdk_cone_accuracy.py`` — quantitative RMSE and
  amplitude bounds for a full Shepp-Logan FBP / FDK pipeline per
  geometry. These would have tripped on the old dev reconstruction
  path, which silently produced wrong-amplitude volumes.
- ``tests/test_fbp_fan_offsets.py`` /
  ``tests/test_fdk_cone_offsets.py`` — adapted from main's offset
  tests: dev's arbitrary-trajectory kernels have no scalar
  ``detector_offset`` / ``center_offset_*`` parameters, so offsets
  are applied by shifting the trajectory arrays directly. Verifies
  that the FBP / FDK gather kernels handle non-centred trajectories.
- ``tests/test_ramp_filter_windows.py`` — 29 parametrised tests covering
  every ``_ramp_window`` option (DC gain, Nyquist value, non-negativity)
  and the full ``ramp_filter_1d`` end-to-end (shape, DC annihilation,
  rfft vs complex-fft parity, ``sample_spacing`` scaling, high-frequency
  pass-through).

#### Benchmark suite (``tests/benchmarks/``)

Opt-in ``pytest-benchmark`` suite covering every CUDA kernel in the
library (forward projector, pure-adjoint backprojector, and the full
analytical FBP / FDK pipeline) across three sizes for each of the
three geometries. 27 benchmarks total, excluded from the default
``pytest tests/`` run via ``--ignore=tests/benchmarks`` in
``pytest.ini``. Run explicitly with
``pytest tests/benchmarks/ --benchmark-only``.

#### Parker short-scan demos in examples

``examples/circular_trajectory/fbp_fan.py`` and
``examples/circular_trajectory/fdk_cone.py`` now expose an
``apply_parker`` switch. When enabled, the example switches the
trajectory to a minimal ``pi + 2*gamma_max`` short scan and applies
``parker_weights`` to the sinogram before the ramp filter; when
disabled, the pipeline runs a full ``2*pi`` scan with the ``1/2``
redundancy factor absorbed by ``angular_integration_weights``. Both
branches produce correctly amplitude-calibrated reconstructions.

#### Documentation (``docs/source/api.rst``)

New "Analytical Reconstruction Helpers", "Ramp Filter Options", and
"Analytical FBP / FDK architecture" sections describe the new
``diffct.analytical`` module, enumerate every ``ramp_filter_1d``
option, and explain the analytical scale factors used in each
``*_weighted_backproject`` wrapper.

### Fixed

#### FBP / FDK amplitude bugs in the ``circular_trajectory/*.py`` examples

Prior to this release the circular-trajectory FBP / FDK examples
(``examples/circular_trajectory/fbp_parallel.py``,
``examples/circular_trajectory/fbp_fan.py``,
``examples/circular_trajectory/fdk_cone.py``) produced amplitude-wrong
reconstructions because they:

- used ``*BackprojectorFunction.apply`` (the pure Siddon adjoint) as if
  it were an FBP gather, so they missed the ``(sid/U)^2`` / ``(|S|/U_n)^2``
  distance weight that the classical FBP / FDK formula requires;
- hand-rolled a ramp filter missing the ``1/sample_spacing`` scale;
- multiplied by a standalone ``pi/num_views`` normalisation that only
  absorbs the angular step, not the ``1/(2*pi)`` Fourier-convention
  constant.

All three examples now go through the new
``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
``cone_weighted_backproject`` wrappers and end-to-end produce raw MSE
matching the main branch's 1.2.11 release to within rounding:

| Example                                 | Raw MSE   | Range             |
|-----------------------------------------|-----------|-------------------|
| ``circular_trajectory/fbp_parallel.py`` | ~0.00366  | ``[-0.02, 1.00]`` |
| ``circular_trajectory/fbp_fan.py``      | ~0.00220  | ``[-0.10, 1.01]`` |
| ``circular_trajectory/fdk_cone.py``     | ~0.00333  | ``[-0.07, 1.00]`` |

### Notes on the sync from ``main``

The ``main`` branch's 1.2.10 / 1.2.11 / 1.3.0 releases introduced several
changes which this update brings to the dev branch, adapted where
necessary for the arbitrary-trajectory kernel API:

- ``1.2.10`` FDK / FBP voxel-driven gather, ``ramp_filter_1d`` options,
  analytical constants, and the ``(sid/U)^2`` correctness fix: **ported**.
  Because dev's kernels take per-view ``(src_pos, det_center, u_vec,
  v_vec)`` arrays instead of closed-form ``sin(beta)/cos(beta)`` math,
  the new gather kernels use ``U_n = (P - S) . n`` (where
  ``n = u_vec x v_vec``) as the generalisation of the classical ``U``.
  For a canonical circular orbit this reduces to the textbook
  ``sid + x*sin(beta) - y*cos(beta)``.
- ``1.2.10`` cone / fan autograd ``distance_weight=1.0`` bug fix:
  **not applicable**. The dev cone / fan backward kernels were
  already correct (they never had that parameter), so the autograd
  adjoint identity holds on dev and is now protected by
  ``tests/test_adjoint_inner_product.py``.
- ``1.2.11`` gradcheck / ramp window tests / CHANGELOG: **ported**.
- ``1.2.11`` benchmark suite: **not ported yet** (dev's benchmarks
  directory stays empty).
- ``1.3.0`` SF-TR / SF-TT projector backends: **not ported**. SF relies
  on closed-form ``sin(beta)/cos(beta)`` math for the trapezoidal
  transaxial / rectangular axial footprints. Generalising SF to
  arbitrary per-view ``(S, C, u_vec, v_vec)`` trajectories is a
  non-trivial research effort (the "separable" part is not separable
  when the detector is not plane-aligned to the voxel axes) and will
  be tackled in a dedicated follow-up.

[Unreleased]: https://github.com/Linda-SophieSchneider/DiffCT-MLX/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Linda-SophieSchneider/DiffCT-MLX/releases/tag/v2.0.0
