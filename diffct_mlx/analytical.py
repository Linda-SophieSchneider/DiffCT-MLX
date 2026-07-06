"""Analytical weighting / filtering helpers, dispatched to the active backend.

Used internally by the analytic reconstruction algorithms (FBP / FDK). Exposed
as a submodule for advanced users; not part of the top-level public API.
"""

from .backend import active as _b

_a = getattr(_b, "analytical", None)

_NAMES = [
    "detector_coordinates_1d",
    "angular_integration_weights",
    "fan_cosine_weights",
    "cone_cosine_weights",
    "parker_weights",
    "ramp_filter_1d",
    "parallel_weighted_backproject",
    "fan_weighted_backproject",
    "cone_weighted_backproject",
]

if _a is not None:
    for _n in _NAMES:
        _fn = getattr(_a, _n, None)
        if _fn is not None:
            globals()[_n] = _fn

__all__ = [n for n in _NAMES if n in globals()]
