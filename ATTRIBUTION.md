# Attribution

The Torch / CUDA engine of **DiffCT-MLX** is **adapted from the original
[diffct](https://github.com/sypsyp97/diffct)** by Yipeng Sun et al.

## Source

| | |
|---|---|
| **Upstream project** | [diffct](https://github.com/sypsyp97/diffct) |
| **Original author** | Yipeng Sun (<yipeng.sun@fau.de>) |
| **License** | Apache License 2.0 — see [LICENSE](LICENSE) |
| **Imported from** | upstream `main`, v2.5.2, commit `982610e` (DiffCT-MLX 3.0, as `diffct_mlx.engine`) |
| **Import date** | 2026-11-25 (first import: `dev` v1.3.3.dev0, commit `cb516cf`, 2026-07-06, as top-level `diffct`; retired in 3.0) |
| **Maintained here by** | Linda-Sophie Schneider (<linda-sophie.schneider@fau.de>) |

The `dev` branch was chosen as the basis (rather than `main`) because its
module layout (`projectors`, `geometry`, `analytical`, `kernels`) and its
per-view projector convention (`src_pos, det_center, det_u_vec[, det_v_vec]`)
match the DiffCT-MLX port, enabling a shared public API across backends.

## Relationship between branches

- **`main`** — the Apple Silicon / MLX port of diffct (this repository, DiffCT-MLX).
- **`cuda`** — a clean snapshot of the CUDA / PyTorch implementation from the
  original diffct, being grown into an **auto-backend** package: it detects the
  available backend at import (MLX on Apple Silicon, Torch/CUDA otherwise) and
  dispatches to it, so the same user code runs unchanged on both platforms.

This mirrors the historical relationship in which diffct's former `apple`
branch became the standalone DiffCT-MLX repository.

## Engine in DiffCT-MLX 3.0

From 3.0 the Torch / numba-CUDA engine is upstream diffct 2.5.2, vendored as
`diffct_mlx/engine/` with upstream's `LICENSE` and `NOTICE` next to it. It is
maintained here, independently of upstream releases; upstream fixes are merged
by hand. The internal name ends a clash with the PyPI package `diffct`: up to
2.x the DiffCT-MLX wheel installed its own top-level `diffct`.

Changes against upstream 2.5.2:

- `differentiable.py` (a deprecated import shim) is removed.
- DiffCT-MLX extensions, as new files: separable-footprint projectors
  (`footprint.py`, `kernels/*_footprint.py`), registered in `kernels/__init__.py`.
- Upstream tests run from `tests/engine/` against `diffct_mlx.engine`; upstream
  example helpers they need are in `tests/engine/_upstream/`, and upstream's
  user documentation is in `docs/engine/`.

## Notice of changes (Apache-2.0 §4)

This branch is a modified redistribution of diffct. The initial commit is a
clean snapshot of diffct `dev` (`cb516cf`) taken on 2026-07-06, **without**
upstream commit history. All subsequent modifications — including the
auto-backend restructuring and the ported high-level reconstruction layer —
are tracked in this repository's git history.

- The original `LICENSE` (Apache License 2.0) is retained unchanged.
- Any third-party data attribution shipped with examples is preserved in the
  corresponding `NOTICE` files.

## Embedded spectrum data (XrayPhysics)

The physically-based example X-ray spectra and material attenuation curves in
`diffct_mlx/physics/data/spectra.npz` were **generated offline** with the
LLNL **[XrayPhysics](https://github.com/kylechampley/XrayPhysics)** library
(Kyle Champley, LLNL; **MIT License**), whose tube-spectrum model is validated
against TASMICS. XrayPhysics is used **only** as an offline reference to produce
this static data — it is **not** a runtime dependency and is not imported by the
package. The generator that reproduces the data is committed at
[`tools/generate_spectra.py`](tools/generate_spectra.py). If you rely on these
spectra quantitatively, please also credit XrayPhysics and TASMICS.

## Citation

Please cite the original work. See the citation section of the
[README](README.md) for the BibTeX entry.
