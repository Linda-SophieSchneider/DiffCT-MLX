"""Trajectory geometry generators, dispatched to the active backend.

Same generator names/signatures on every backend (returning that backend's
array type), so downstream code selecting a trajectory never changes. The
arbitrary-trajectory JSON loader is defined here (backend-neutral) since it is
not part of the upstream low-level backends.
"""

import json
import warnings

import numpy as np

from .backend import active as _b
from .real_measured_data_helper import (
    _as_3d_float_array,
    _get_first_present,
    _normalize_detector_vectors,
    diagnose_cone_geometry,
    estimate_cone_isocenter,
)

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


def _compute_detector_from_source_and_sdd(src_np, sdd):
    src_norm = np.linalg.norm(src_np, axis=1, keepdims=True)
    src_norm = np.maximum(src_norm, 1e-6)
    src_unit = src_np / src_norm

    sdd_np = np.asarray(sdd, dtype=np.float32)
    if sdd_np.ndim == 0:
        sdd_np = np.full((src_np.shape[0], 1), float(sdd_np), dtype=np.float32)
    elif sdd_np.ndim == 1:
        sdd_np = sdd_np[:, None]

    det_c_np = -src_unit * (sdd_np - src_norm)

    det_u_np = np.stack(
        [-src_np[:, 1], src_np[:, 0], np.zeros(src_np.shape[0], dtype=np.float32)],
        axis=1,
    )
    det_u_np = _normalize_detector_vectors(
        det_u_np,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    det_v_np = np.cross(src_unit, det_u_np)
    det_v_np = _normalize_detector_vectors(
        det_v_np,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return det_c_np, det_u_np, det_v_np


def load_arbitrary_cone_geometry_from_json(json_path, *, flip_det_u=False,
                                           flip_det_v=False,
                                           warn_on_inconsistency=True,
                                           recenter_to_isocenter=False):
    """Load arbitrary cone trajectory from JSON and return backend geometry arrays.

    If detector center and basis vectors are present in the JSON, they are used
    directly. ``flip_det_v=True`` is useful for scanners whose protocol stores
    the detector row direction with opposite sign.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        src_raw = _get_first_present(
            payload,
            ("src_pos",),
            ("source_positions",),
            ("source_path",),
            ("trajectory", "src_pos"),
            ("trajectory", "source_positions_mm"),
        )
        if src_raw is None:
            raise ValueError("Missing source positions in JSON.")
        src_np = _as_3d_float_array(src_raw, "src_pos")

        det_c = _get_first_present(
            payload,
            ("det_center",),
            ("detector_center",),
            ("trajectory", "det_center"),
            ("trajectory", "detector_centers_mm"),
        )
        det_u = _get_first_present(
            payload,
            ("det_u_vec",),
            ("detector_u_vec",),
            ("trajectory", "det_u_vec"),
            ("trajectory", "detector_u_vec"),
        )
        det_v = _get_first_present(
            payload,
            ("det_v_vec",),
            ("detector_v_vec",),
            ("trajectory", "det_v_vec"),
            ("trajectory", "detector_v_vec"),
        )
        sdd = _get_first_present(
            payload,
            ("sdd",),
            ("source", "source_to_detector_distance_mm"),
            ("source", "source_to_detector_distance"),
        )
    else:
        src_np = _as_3d_float_array(payload, "src_pos")
        det_c = det_u = det_v = sdd = None

    if det_c is None or det_u is None or det_v is None:
        if sdd is None:
            raise ValueError(
                "Missing SDD in JSON. Provide detector geometry or 'sdd'."
            )
        det_c_np, det_u_np, det_v_np = _compute_detector_from_source_and_sdd(src_np, sdd)
        if warn_on_inconsistency:
            warnings.warn(
                "Detector center/u/v missing in JSON; falling back to geometry "
                "constructed from source positions and SDD.",
                stacklevel=2,
            )
    else:
        det_c_np = _as_3d_float_array(det_c, "det_center")
        det_u_np = _normalize_detector_vectors(
            _as_3d_float_array(det_u, "det_u_vec"),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        det_v_np = _normalize_detector_vectors(
            _as_3d_float_array(det_v, "det_v_vec"),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    if flip_det_u:
        det_u_np = -det_u_np
    if flip_det_v:
        det_v_np = -det_v_np

    if recenter_to_isocenter:
        isocenter = estimate_cone_isocenter(src_np, det_c_np)
        print(
            "Recentering geometry: "
            f"estimated isocenter = ({isocenter[0]:.3f}, "
            f"{isocenter[1]:.3f}, {isocenter[2]:.3f}) mm -> (0.000, 0.000, 0.000) mm"
        )
        src_np = src_np - isocenter
        det_c_np = det_c_np - isocenter

    if warn_on_inconsistency and sdd is not None:
        diag = diagnose_cone_geometry(src_np, det_c_np, det_u_np, det_v_np)
        sdd_np = np.asarray(sdd, dtype=np.float32)
        sdd_min = float(np.min(sdd_np))
        sdd_max = float(np.max(sdd_np))
        if (
            abs(diag["sdd_min_mm"] - sdd_min) > 1e-3
            or abs(diag["sdd_max_mm"] - sdd_max) > 1e-3
        ):
            warnings.warn(
                "Loaded detector centers imply an effective SDD range "
                f"[{diag['sdd_min_mm']:.3f}, {diag['sdd_max_mm']:.3f}] mm, "
                f"but JSON metadata reports [{sdd_min:.3f}, {sdd_max:.3f}] mm.",
                stacklevel=2,
            )

    return (
        _b.as_array(src_np, dtype=_b.float32),
        _b.as_array(det_c_np, dtype=_b.float32),
        _b.as_array(det_u_np, dtype=_b.float32),
        _b.as_array(det_v_np, dtype=_b.float32),
    )


def laminography_trajectory_3d(n_views, sid, sdd, tilt_deg=30.0, start_angle=0.0, end_angle=None):
    """Circular laminography trajectory (rotation axis tilted from the beam).

    The source traces a circle of radius ``sid*cos(tilt)`` at constant height
    ``sid*sin(tilt)``; ``tilt_deg=0`` reduces to standard circular cone-beam CT,
    while larger tilts approach planar (laminographic) imaging of flat / slab-like
    objects. Returns backend arrays ``(src_pos, det_center, det_u_vec, det_v_vec)``
    in the same convention as the other generators, so forward projection and
    iterative reconstruction (SART / SIRT / CG / WLS) work on it directly.
    """
    if end_angle is None:
        end_angle = 2.0 * np.pi
    phi = np.linspace(float(start_angle), float(end_angle), int(n_views), endpoint=False).astype(np.float32)
    tilt = np.radians(float(tilt_deg))
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    src_np = (float(sid) * np.stack(
        [cos_t * np.cos(phi), cos_t * np.sin(phi), sin_t * np.ones_like(phi)], axis=1
    )).astype(np.float32)
    det_c_np, det_u_np, det_v_np = _compute_detector_from_source_and_sdd(src_np, sdd)
    return (
        _b.as_array(src_np, dtype=_b.float32),
        _b.as_array(det_c_np, dtype=_b.float32),
        _b.as_array(det_u_np, dtype=_b.float32),
        _b.as_array(det_v_np, dtype=_b.float32),
    )


__all__ = [n for n in _NAMES if n in globals()] + [
    "load_arbitrary_cone_geometry_from_json",
    "laminography_trajectory_3d",
]
