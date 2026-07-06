# Attribution

This `cuda` branch of **DiffCT-MLX** is **adapted from the original
[diffct](https://github.com/sypsyp97/diffct)** by Yipeng Sun et al.

## Source

| | |
|---|---|
| **Upstream project** | [diffct](https://github.com/sypsyp97/diffct) |
| **Original author** | Yipeng Sun (<yipeng.sun@fau.de>) |
| **License** | Apache License 2.0 — see [LICENSE](LICENSE) |
| **Imported from** | upstream `main` at v1.3.4, commit `506676b` |
| **Import date** | 2026-07-06 |
| **Maintained here by** | Linda-Sophie Schneider (<linda-sophie.schneider@fau.de>) |

## Relationship between branches

- **`main`** — the Apple Silicon / MLX port of diffct (this repository, DiffCT-MLX).
- **`cuda`** — a clean snapshot of the CUDA / PyTorch implementation from the
  original diffct, maintained here to evolve in a separate development direction.

This mirrors the historical relationship in which diffct's former `apple`
branch became the standalone DiffCT-MLX repository.

## Notice of changes (Apache-2.0 §4)

This branch is a modified redistribution of diffct. The initial commit is a
clean snapshot of diffct v1.3.4 (`506676b`) taken on 2026-07-06, **without**
upstream commit history. All subsequent modifications are tracked in this
repository's git history.

- The original `LICENSE` (Apache License 2.0) is retained unchanged.
- Third-party data attribution for the bundled walnut dataset is preserved in
  [`examples/data/NOTICE`](examples/data/NOTICE).

## Citation

Please cite the original work. See the citation section of the
[README](README.md) for the BibTeX entry.
