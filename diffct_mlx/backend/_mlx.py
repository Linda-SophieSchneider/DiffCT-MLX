"""MLX / Metal backend (Apple Silicon).

Wiring of this backend against the MLX Metal kernels is completed in a later
phase (see ATTRIBUTION.md / the project roadmap). On Apple Silicon the MLX port
lives on the ``main`` branch of DiffCT-MLX; this module will bridge to it so the
unified ``diffct_mlx`` API dispatches to Metal kernels.

Until then, selecting this backend and calling an operator raises a clear error.
It is only ever selected when ``mlx.core`` is importable, so the Torch/CUDA path
is unaffected.
"""

NAME = "mlx"

try:  # pragma: no cover - only meaningful on Apple Silicon
    import mlx.core as _mx
    xp = _mx
except Exception:  # pragma: no cover
    xp = None


def _pending(*_args, **_kwargs):
    raise NotImplementedError(
        "The MLX backend is not wired up on this branch yet. "
        "Use the DiffCT-MLX 'main' branch on Apple Silicon, or set "
        "DIFFCT_BACKEND=torch to force the CUDA backend."
    )


# Placeholder operators (filled in when the MLX kernels are bridged).
parallel_forward = parallel_backward = _pending
fan_forward = fan_backward = _pending
cone_forward = cone_backward = _pending
geometry = None
analytical = None
