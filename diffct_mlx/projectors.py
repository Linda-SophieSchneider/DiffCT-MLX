"""Unified functional projector API.

Same names and signatures across backends, so user code (and the higher-level
reconstruction algorithms) is written once and runs on MLX or Torch/CUDA
unchanged. Operators dispatch to the active backend
(:mod:`diffct_mlx.backend`).

Both backends implement the full operator set natively, including the
separable-footprint projectors and the sparse cone backprojection
(``cone_backward_footprint(..., indices=...)``). The line-based fallback below
only engages for a backend that lacks a native footprint operator (none of the
shipped backends today) and warns once when it does.
"""

import warnings

from .backend import active as _b, NAME as _BACKEND

__all__ = [
    "parallel_forward", "parallel_backward",
    "parallel_forward_footprint", "parallel_backward_footprint",
    "fan_forward", "fan_backward",
    "fan_forward_footprint", "fan_backward_footprint",
    "cone_forward", "cone_backward",
    "cone_forward_footprint", "cone_backward_footprint",
]

# --- Line-based projectors (available on every backend) ---------------------
parallel_forward = _b.parallel_forward
parallel_backward = _b.parallel_backward
fan_forward = _b.fan_forward
fan_backward = _b.fan_backward
cone_forward = _b.cone_forward
cone_backward = _b.cone_backward


# --- Footprint projectors ---------------------------------------------------
_warned = set()


def _fallback(name, base):
    """Wrap a line-based projector as a footprint stand-in (warn once)."""
    def _wrapped(*args, **kwargs):
        # ``cone_backward_footprint`` accepts an extra ``indices`` kwarg (sparse
        # voxel evaluation, returns a 1D vector). The line-based stand-in cannot
        # honor that contract, so fail loudly instead of returning a dense volume.
        if kwargs.pop("indices", None) is not None:
            raise NotImplementedError(
                f"{name}: sparse evaluation (indices=...) requires the native "
                f"footprint operator, which the '{_BACKEND}' backend lacks."
            )
        if name not in _warned:
            _warned.add(name)
            warnings.warn(
                f"{name}: footprint projectors are not implemented on the "
                f"'{_BACKEND}' backend; falling back to the line-based "
                f"projector. Numerical results differ from the MLX footprint "
                f"model.",
                RuntimeWarning,
                stacklevel=2,
            )
        return base(*args, **kwargs)
    _wrapped.__name__ = name
    return _wrapped


def _resolve_footprint(name, base):
    """Use the backend's native footprint operator if it has one."""
    native = getattr(_b, name, None)
    return native if native is not None else _fallback(name, base)


parallel_forward_footprint = _resolve_footprint(
    "parallel_forward_footprint", parallel_forward)
parallel_backward_footprint = _resolve_footprint(
    "parallel_backward_footprint", parallel_backward)
fan_forward_footprint = _resolve_footprint(
    "fan_forward_footprint", fan_forward)
fan_backward_footprint = _resolve_footprint(
    "fan_backward_footprint", fan_backward)
cone_forward_footprint = _resolve_footprint(
    "cone_forward_footprint", cone_forward)
cone_backward_footprint = _resolve_footprint(
    "cone_backward_footprint", cone_backward)
