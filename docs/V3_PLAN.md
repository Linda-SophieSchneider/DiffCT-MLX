# DiffCT-MLX 3.0 — migration plan

Branch `v3`. Status markers: [x] done, [~] in progress, [ ] open.

## Decisions (2026-11-25)

- **Own maintenance.** Upstream diffct 2.5.2 (github.com/sypsyp97/diffct,
  commit `982610e`) is re-vendored as `diffct_mlx.engine`. It is not a PyPI
  dependency. The internal name ends the clash with the PyPI package
  `diffct`: up to 2.x our wheel shipped a top-level `diffct` that overwrote
  it. Upstream fixes are merged by hand; `ATTRIBUTION.md` records the commit.
- **Conventions.** 3.0 defaults to the upstream conventions on both backends:
  detector pixel `i` at `(i - (N-1)/2)·du` and physical line integrals
  (µ·mm). Old studies keep `detector_convention="kernel"` and
  `line_integrals="voxel"`, or stay on 2.x.
- **Metal kernel fixes** are ported in the separate session
  `port-upstream-diffct-2-x`, Metal only. The convention switch on Metal
  happens here after that branch is merged.

## What the engine brings (Torch side)

- Exact geometry gradients for all beams (`kernels/geometry_vjp.py`).
- Portable PyTorch backend (`backends.py`) on CPU, CUDA, ROCm and Apple MPS,
  with `precision="fast" | "float32" | "float64"`.
- `Projector` (`operators.py`): chunking, streaming into `TensorStore` /
  `NpyStore`, per-axis voxel spacing, `parallel3d`, detector surfaces,
  batch dimensions, multi-GPU.
- Solvers (`cgls`, `sirt`, `tv_reconstruction`), differentiable `fbp` / `fdk`
  / `helical_fdk`.
- All kernel fixes of upstream 2.0–2.5.

## Steps

1. [x] Vendor the upstream package as `diffct_mlx/engine/`, with its license
   notice. Drop `differentiable.py` (a deprecated shim with absolute imports).
2. [~] Port the upstream tests to `tests/engine/`, importing
   `diffct_mlx.engine`. Run them on CPU and MPS (portable backend) on the Mac
   and on CUDA on Alex.
3. [~] Port our extensions onto the engine, in upstream conventions:
   separable-footprint kernels (parallel, fan, cone; mass-conserving since
   2.2.0), sparse cone footprint backprojection, out-of-core FDK
   (`orchestration/out_of_core.py`). Their geometry gradients use the
   engine's finite-difference helper until an analytic VJP exists.
4. [x] Rewire `diffct_mlx/backend/_torch.py` to the engine. Delete the old
   top-level `diffct/` package and `tests/` that import it, or port them.
5. [ ] Switch the Metal kernels (Siddon, footprint, FDK gathers) to the
   upstream conventions. Flip the defaults in `diffct_mlx.conventions`,
   `operators.py` and `cases.py`. Remove the voxel-unit conversions in
   `simulate_scan`, the case builders and the quantitative FDK path.
6. [ ] Expose the new engine features in the unified API: `precision`,
   per-axis `voxel_spacing`, `parallel3d`, detector surfaces, geometry
   gradients, chunked `Projector`. MLX raises `NotImplementedError` where it
   has no counterpart yet.
7. [ ] Cross-backend parity tests (MLX against portable Torch on the same Mac,
   against CUDA on Alex). Rerun the study equivalence check with 3.0 on both
   backends.
8. [ ] README, migration guide (2.x → 3.0), CHANGELOG, release 3.0.0.

## Status (2026-11-25)

- Engine tests on the Mac (CPU, MPS, cudasim): 2357+ pass. The remaining
  1 failure and 5 errors also occur in a pristine upstream checkout on this
  Mac (no CUDA bindings), so they come from the environment.
- Footprint kernels in cudasim against the engine's float64 Siddon: mass
  within 1.5 % per view, centroid within 0.13 px, adjoint error below 1e-7.
- Not run on CUDA yet: native footprint path, out-of-core FDK gather,
  `tests/engine` CUDA tests. Next: a full run on Alex.
- MLX suite unchanged (36 pass; 2 old `test_xp_*` failures).
- Backend parity is broken until step 5: Torch is centred + physical, MLX
  still uses the kernel convention.

## Open questions

- Dependency floor: the engine needs Python ≥ 3.10 and `numpy < 2.5`.
  3.0 raises `requires-python` to 3.10.
- Which upstream tests to carry. The formal TLA+ models and the docs site
  stay out.
