"""Trajectory geometry generators, dispatched to the active backend.

Same generator names/signatures on every backend (returning that backend's
array type), so downstream code selecting a trajectory never changes.
"""

from .backend import active as _b

_g = getattr(_b, "geometry", None)

_NAMES = [
    "circular_trajectory_3d",
    "random_trajectory_3d",
    "spiral_trajectory_3d",
    "sinusoidal_trajectory_3d",
    "saddle_trajectory_3d",
    "custom_trajectory_3d",
    "circular_trajectory_2d_fan",
    "sinusoidal_trajectory_2d_fan",
    "custom_trajectory_2d_fan",
    "circular_trajectory_2d_parallel",
    "sinusoidal_trajectory_2d_parallel",
    "custom_trajectory_2d_parallel",
]

if _g is not None:
    for _n in _NAMES:
        _fn = getattr(_g, _n, None)
        if _fn is not None:
            globals()[_n] = _fn

__all__ = [n for n in _NAMES if n in globals()]
