# Attribution

This `cuda` branch of **DiffCT-MLX** is **adapted from the original
[diffct](https://github.com/sypsyp97/diffct)** by Yipeng Sun et al.

## Source

| | |
|---|---|
| **Upstream project** | [diffct](https://github.com/sypsyp97/diffct) |
| **Original author** | Yipeng Sun (<yipeng.sun@fau.de>) |
| **License** | Apache License 2.0 — see [LICENSE](LICENSE) |
| **Imported from** | upstream `dev` branch (arbitrary-trajectory line), v1.3.3.dev0, commit `cb516cf` |
| **Import date** | 2026-07-06 |
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
