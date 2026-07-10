"""Reusable reconstruction-case builders for example and benchmark scripts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..backend import active as _b

xp = _b.xp

from ..geometry import (
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    load_arbitrary_cone_geometry_from_json,
)
from ..phantoms import shepp_logan_2d, shepp_logan_3d
from ..projectors import (
    fan_backward_footprint,
    cone_backward,
    cone_backward_footprint,
    fan_forward_footprint,
    cone_forward,
    cone_forward_footprint,
    fan_backward,
    fan_forward,
    parallel_backward_footprint,
    parallel_backward,
    parallel_forward_footprint,
    parallel_forward,
)
from ..real_measured_data_helper import (
    apply_detector_geometry_convention,
    auto_voxel_spacing_from_detector,
    load_tiff_projections,
    normalize_volume,
    resize_volume_to_shape,
    shift_detector_center,
    transform_detector_offsets,
)


Array = Any


@dataclass
class ReconstructionCase:
    """Container describing one reconstruction problem instance."""

    name: str
    sinogram: Array
    volume_shape: tuple[int, ...]
    forward_single: Callable[[Array, int], Array]
    back_single: Callable[[Array, int], Array]
    back_project_all: Callable[[Array], Array]
    reference: Array | None = None
    reference_title: str | None = None
    supports_fbp: bool = False
    supports_fdk: bool = False
    fbp_normalization_scale: float | None = None
    iterative_iteration_count: int = 5
    sirt_iteration_count: int = 15
    iterative_sart_iteration_count: int = 2
    iterative_projection_subset_count: int = 1
    iterative_normalized_sart_relaxation: float = 0.9
    iterative_backprojection_scale: float | None = 0.215
    pocs_iterative_update_method: str = "sart"
    iterative_positivity_mode: str = "per_iteration"
    iterative_detector_border_u: int = 0
    iterative_detector_border_v: int = 0
    iterative_volume_border_width: int = 0
    iterative_volume_support_mask: Array | None = None
    iterative_volume_support_mask_mode: str = "always"
    iterative_voxel_sensitivity_normalization: bool = False
    iterative_projection_weights: tuple[float, ...] | None = None
    tv_reg_iteration_count: int = 6
    tv_alpha: float = 0.12
    asd_reg_iteration_count: int = 6
    asd_alpha: float = 0.12
    asd_epsilon: float = 0.05
    awtv_reg_iteration_count: int = 6
    awtv_alpha: float = 0.12
    awtv_epsilon: float = 0.08
    awtv_delta: float = 0.6e-2
    fdk_normalization_scale: float | None = None
    fbp_weight: Callable[[Array], Array] | None = None
    fdk_weight: Callable[[Array], Array] | None = None


@dataclass
class MeasuredConeDataConfig:
    """Configuration for loading measured cone-beam data."""

    data_dir: str | Path
    volume_shape: tuple[int, int, int] = (128, 128, 128)
    voxel_spacing_mm: float | None = None
    target_view_count: int = 360
    target_detector_shape: tuple[int, int] = (256, 256)
    trajectory_json_path: str | Path | None = None
    reference_volume_path: str | Path | None = None
    reference_meta_path: str | Path | None = None
    recenter_to_isocenter: bool = True
    flip_det_u: bool = False
    flip_det_v: bool = False
    transpose_uv: bool = True
    flip_u: bool = False
    flip_v: bool = False
    log_transform: bool = True
    revert: bool = False
    viewwise_i0: bool = True
    air_border_px: int = 16
    subtract_air_baseline: bool = True
    air_baseline_percentile: float = 50.0
    measured_fov_margin_mm: float = 8.0
    use_leap_fov_support_mask: bool = True
    support_mask_sensitivity_threshold_ratio: float = 1e-3
    iterative_volume_support_mask_mode: str = "final"
    iterative_projection_subset_count: int = 1
    iterative_normalized_sart_relaxation: float = 0.9
    iterative_backprojection_scale: float | None = None
    projector_mode: str = "footprint"
    normalize_reference: bool = True


def _center_crop_or_pad_volume(volume, target_shape):
    """Match a volume to a target shape without changing its voxel spacing."""
    volume_np = np.asarray(volume, dtype=np.float32)
    target_shape = tuple(int(x) for x in target_shape)
    if volume_np.shape == target_shape:
        return volume_np

    result = volume_np
    for axis, target_size in enumerate(target_shape):
        current_size = result.shape[axis]
        if current_size > target_size:
            start = (current_size - target_size) // 2
            stop = start + target_size
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, stop)
            result = result[tuple(slices)]

    if result.shape == target_shape:
        return result.astype(np.float32, copy=False)

    pad_width = []
    for current_size, target_size in zip(result.shape, target_shape):
        total_pad = max(0, target_size - current_size)
        before = total_pad // 2
        after = total_pad - before
        pad_width.append((before, after))
    return np.pad(result, pad_width, mode="constant").astype(np.float32, copy=False)


def _trajectory_quadrature_weights(src_pos) -> tuple[float, ...]:
    """Approximate LEAP-style view quadrature weights from source-direction spacing."""
    src_np = _b.to_numpy(src_pos).astype(np.float32)
    if src_np.ndim != 2 or src_np.shape[0] <= 1:
        return tuple(1.0 for _ in range(max(1, int(src_np.shape[0]) if src_np.ndim == 2 else 1)))

    norms = np.linalg.norm(src_np, axis=1, keepdims=True)
    directions = src_np / np.maximum(norms, 1e-6)
    pairwise = np.sum(directions[:-1] * directions[1:], axis=1)
    pairwise = np.clip(pairwise, -1.0, 1.0)
    arc = np.arccos(pairwise).astype(np.float32)

    weights = np.empty(src_np.shape[0], dtype=np.float32)
    weights[0] = arc[0]
    weights[-1] = arc[-1]
    if src_np.shape[0] > 2:
        weights[1:-1] = 0.5 * (arc[:-1] + arc[1:])

    positive = weights > 0.0
    if not np.any(positive):
        return tuple(1.0 for _ in range(src_np.shape[0]))
    mean_weight = float(np.mean(weights[positive]))
    if mean_weight <= 0.0:
        return tuple(1.0 for _ in range(src_np.shape[0]))
    weights = weights / mean_weight
    return tuple(float(value) for value in weights)


def _build_sensitivity_support_mask(
    back_project_all: Callable[[Array], Array],
    sinogram_shape: tuple[int, ...],
    *,
    threshold_ratio: float = 1e-3,
) -> np.ndarray:
    """Build a soft reconstruction-support mask from the cone-beam sensitivity map."""
    detector_coverage = xp.ones(sinogram_shape, dtype=xp.float32)
    sensitivity = _b.to_numpy(back_project_all(detector_coverage)).astype(np.float32)
    positive = sensitivity[sensitivity > 0.0]
    if positive.size == 0:
        return np.ones(sensitivity.shape, dtype=bool)

    threshold_ratio = max(float(threshold_ratio), 0.0)
    threshold = max(float(np.max(positive)) * threshold_ratio, np.finfo(np.float32).eps)
    support_mask = sensitivity >= threshold
    if not np.any(support_mask):
        return np.ones(sensitivity.shape, dtype=bool)
    return support_mask.astype(bool, copy=False)


def _build_leap_style_circular_fov_mask(
    volume_shape: tuple[int, int, int],
    voxel_spacing_mm: float,
    detector_shape_uv: tuple[int, int],
    detector_pitch_u_mm: float,
    src_pos,
    det_center,
    *,
    fov_margin_mm: float = 0.0,
) -> np.ndarray:
    """Build LEAP-style circular XY support from the smallest in-plane detector coverage."""
    src_np = _b.to_numpy(src_pos).astype(np.float32)
    det_center_np = _b.to_numpy(det_center).astype(np.float32)
    sid = np.linalg.norm(src_np, axis=1)
    sdd = np.linalg.norm(det_center_np - src_np, axis=1)
    magnification_inv = sid / np.maximum(sdd, 1e-6)

    det_u_count, _ = detector_shape_uv
    half_fov_u_mm = 0.5 * float(det_u_count) * float(detector_pitch_u_mm) * magnification_inv
    radius_xy_mm = float(np.min(half_fov_u_mm)) - float(fov_margin_mm)
    radius_xy_mm = max(radius_xy_mm, float(voxel_spacing_mm))

    nz, ny, nx = (int(value) for value in volume_shape)
    y_coords = (np.arange(ny, dtype=np.float32) - 0.5 * (ny - 1)) * float(voxel_spacing_mm)
    x_coords = (np.arange(nx, dtype=np.float32) - 0.5 * (nx - 1)) * float(voxel_spacing_mm)

    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    radial_keep = (xx * xx + yy * yy) <= (radius_xy_mm * radius_xy_mm)
    return np.broadcast_to(radial_keep[None, :, :], (nz, ny, nx)).astype(bool, copy=False)


def make_parallel_2d_operators(
    ray_dir: Array,
    det_origin: Array,
    det_u_vec: Array,
    *,
    image_shape: tuple[int, int],
    num_detectors: int,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
    projector_mode: str = "footprint",
) -> tuple[Callable[[Array, int], Array], Callable[[Array, int], Array], Callable[[Array], Array]]:
    """Create single-view and all-view parallel-beam projector wrappers."""
    ny, nx = image_shape
    projector_key = str(projector_mode).strip().lower()
    if projector_key == "siddon":
        forward_op = parallel_forward
        backward_op = parallel_backward
    elif projector_key == "footprint":
        forward_op = parallel_forward_footprint
        backward_op = parallel_backward_footprint
    else:
        raise ValueError(f"Unknown parallel projector_mode: {projector_mode!r}")

    def forward_single(volume: Array, projection_index: int) -> Array:
        return forward_op(
            volume,
            ray_dir[projection_index : projection_index + 1],
            det_origin[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )[0]

    def forward_slice(volume: Array, start: int, stop: int) -> Array:
        return forward_op(
            volume,
            ray_dir[start:stop],
            det_origin[start:stop],
            det_u_vec[start:stop],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )

    def back_single(projection: Array, projection_index: int) -> Array:
        return backward_op(
            xp.array(projection, dtype=xp.float32)[None, :],
            ray_dir[projection_index : projection_index + 1],
            det_origin[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_slice(projection: Array, start: int, stop: int) -> Array:
        return backward_op(
            xp.array(projection, dtype=xp.float32),
            ray_dir[start:stop],
            det_origin[start:stop],
            det_u_vec[start:stop],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_project_all(filtered_sinogram: Array) -> Array:
        return backward_op(
            filtered_sinogram,
            ray_dir,
            det_origin,
            det_u_vec,
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
    back_single.project_slice = back_slice  # type: ignore[attr-defined]
    return forward_single, back_single, back_project_all


def make_fan_2d_operators(
    src_pos: Array,
    det_center: Array,
    det_u_vec: Array,
    *,
    image_shape: tuple[int, int],
    num_detectors: int,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
    projector_mode: str = "footprint",
) -> tuple[Callable[[Array, int], Array], Callable[[Array, int], Array], Callable[[Array], Array]]:
    """Create single-view and all-view fan-beam projector wrappers."""
    ny, nx = image_shape
    projector_key = str(projector_mode).strip().lower()
    if projector_key == "siddon":
        forward_op = fan_forward
        backward_op = fan_backward
    elif projector_key == "footprint":
        forward_op = fan_forward_footprint
        backward_op = fan_backward_footprint
    else:
        raise ValueError(f"Unknown fan projector_mode: {projector_mode!r}")

    def forward_single(volume: Array, projection_index: int) -> Array:
        return forward_op(
            volume,
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )[0]

    def forward_slice(volume: Array, start: int, stop: int) -> Array:
        return forward_op(
            volume,
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )

    def back_single(projection: Array, projection_index: int) -> Array:
        return backward_op(
            xp.array(projection, dtype=xp.float32)[None, :],
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_slice(projection: Array, start: int, stop: int) -> Array:
        return backward_op(
            xp.array(projection, dtype=xp.float32),
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_project_all(filtered_sinogram: Array) -> Array:
        return backward_op(
            filtered_sinogram,
            src_pos,
            det_center,
            det_u_vec,
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
    back_single.project_slice = back_slice  # type: ignore[attr-defined]
    return forward_single, back_single, back_project_all


def make_cone_3d_operators(
    src_pos: Array,
    det_center: Array,
    det_u_vec: Array,
    det_v_vec: Array,
    *,
    volume_shape: tuple[int, int, int],
    detector_shape: tuple[int, int],
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    projector_mode: str = "footprint",
) -> tuple[Callable[[Array, int], Array], Callable[[Array, int], Array], Callable[[Array], Array]]:
    """Create single-view and all-view cone-beam projector wrappers."""
    nz, ny, nx = volume_shape
    det_u_count, det_v_count = detector_shape
    projector_key = str(projector_mode).strip().lower()
    if projector_key == "siddon":
        forward_op = cone_forward
        backward_op = cone_backward
    elif projector_key == "footprint":
        forward_op = cone_forward_footprint
        backward_op = cone_backward_footprint
    else:
        raise ValueError(f"Unknown cone projector_mode: {projector_mode!r}")

    def forward_single(volume: Array, projection_index: int) -> Array:
        return forward_op(
            volume,
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            det_v_vec[projection_index : projection_index + 1],
            det_u=det_u_count,
            det_v=det_v_count,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )[0]

    def forward_slice(volume: Array, start: int, stop: int) -> Array:
        return forward_op(
            volume,
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            det_v_vec[start:stop],
            det_u=det_u_count,
            det_v=det_v_count,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    def back_single(projection: Array, projection_index: int) -> Array:
        return backward_op(
            xp.array(projection, dtype=xp.float32)[None, :, :],
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            det_v_vec[projection_index : projection_index + 1],
            D=nz,
            H=ny,
            W=nx,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    def back_slice(projection: Array, start: int, stop: int) -> Array:
        return backward_op(
            xp.array(projection, dtype=xp.float32),
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            det_v_vec[start:stop],
            D=nz,
            H=ny,
            W=nx,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    def back_project_all(filtered_sinogram: Array) -> Array:
        return backward_op(
            filtered_sinogram,
            src_pos,
            det_center,
            det_u_vec,
            det_v_vec,
            D=nz,
            H=ny,
            W=nx,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
    back_single.project_slice = back_slice  # type: ignore[attr-defined]
    return forward_single, back_single, back_project_all


def build_parallel_2d_case(
    *,
    image_shape: tuple[int, int] = (96, 96),
    num_views: int = 180,
    num_detectors: int = 160,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
) -> ReconstructionCase:
    """Build a Shepp-Logan 2D parallel-beam case."""
    ny, nx = image_shape
    reference = xp.array(shepp_logan_2d((ny, nx)))
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(num_views)
    # Measurements are modeled as line integrals (Siddon). Analytic FBP inverts
    # them with a matched Siddon backprojector; iterative reconstruction uses the
    # more accurate separable-footprint projector pair.
    sinogram = parallel_forward(
        reference,
        ray_dir,
        det_origin,
        det_u_vec,
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    forward_single, back_single, _ = make_parallel_2d_operators(
        ray_dir, det_origin, det_u_vec,
        image_shape=(ny, nx), num_detectors=num_detectors,
        detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        projector_mode="footprint",
    )
    _, _, back_project_all = make_parallel_2d_operators(
        ray_dir, det_origin, det_u_vec,
        image_shape=(ny, nx), num_detectors=num_detectors,
        detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        projector_mode="siddon",
    )

    return ReconstructionCase(
        name="Parallel 2D",
        sinogram=sinogram,
        volume_shape=(ny, nx),
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title="Phantom",
        supports_fbp=True,
        fbp_normalization_scale=math.pi / (2.0 * num_views),
        iterative_voxel_sensitivity_normalization=True,
    )


def build_fan_2d_case(
    *,
    image_shape: tuple[int, int] = (256, 256),
    num_views: int = 360,
    num_detectors: int = 600,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
    sid: float = 500.0,
    sdd: float = 800.0,
) -> ReconstructionCase:
    """Build a Shepp-Logan 2D fan-beam case."""
    ny, nx = image_shape
    reference = xp.array(shepp_logan_2d((ny, nx)))
    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(num_views, sid, sdd)
    # Line-integral (Siddon) measurements; analytic FBP uses a matched Siddon
    # backprojector, iterative uses the footprint projector pair.
    sinogram = fan_forward(
        reference,
        src_pos,
        det_center,
        det_u_vec,
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    detector_coords = (xp.arange(num_detectors) - (num_detectors - 1) / 2) * detector_spacing
    cosine_weights = xp.cos(xp.arctan(detector_coords / sdd)).reshape(1, -1)

    forward_single, back_single, _ = make_fan_2d_operators(
        src_pos, det_center, det_u_vec,
        image_shape=(ny, nx), num_detectors=num_detectors,
        detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        projector_mode="footprint",
    )
    _, _, back_project_all = make_fan_2d_operators(
        src_pos, det_center, det_u_vec,
        image_shape=(ny, nx), num_detectors=num_detectors,
        detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        projector_mode="siddon",
    )

    return ReconstructionCase(
        name="Fan 2D",
        sinogram=sinogram,
        volume_shape=(ny, nx),
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title="Phantom",
        supports_fbp=True,
        fbp_normalization_scale=math.pi / (2.0 * num_views),
        fbp_weight=lambda raw: raw * cosine_weights,
        iterative_voxel_sensitivity_normalization=True,
    )


def build_cone_3d_case(
    *,
    volume_shape: tuple[int, int, int] = (128, 128, 128),
    num_views: int = 360,
    detector_shape: tuple[int, int] = (256, 256),
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    sid: float = 600.0,
    sdd: float = 900.0,
) -> ReconstructionCase:
    """Build a Shepp-Logan 3D cone-beam case."""
    nz, ny, nx = volume_shape
    det_u_count, det_v_count = detector_shape
    reference = xp.array(shepp_logan_3d((nz, ny, nx)))
    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(num_views, sid, sdd)
    # Line-integral (Siddon) measurements; analytic FDK uses a matched Siddon
    # backprojector, iterative uses the footprint projector pair.
    sinogram = cone_forward(
        reference,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u=det_u_count,
        det_v=det_v_count,
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )

    u_coords = (xp.arange(det_u_count) - (det_u_count - 1) / 2) * du
    v_coords = (xp.arange(det_v_count) - (det_v_count - 1) / 2) * dv
    fdk_weights = sdd / xp.sqrt(
        sdd**2 + u_coords.reshape(1, det_u_count, 1) ** 2 + v_coords.reshape(1, 1, det_v_count) ** 2
    )
    cone_weight = lambda raw: raw * fdk_weights

    forward_single, back_single, _ = make_cone_3d_operators(
        src_pos, det_center, det_u_vec, det_v_vec,
        volume_shape=(nz, ny, nx), detector_shape=(det_u_count, det_v_count),
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        projector_mode="footprint",
    )
    _, _, back_project_all = make_cone_3d_operators(
        src_pos, det_center, det_u_vec, det_v_vec,
        volume_shape=(nz, ny, nx), detector_shape=(det_u_count, det_v_count),
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        projector_mode="siddon",
    )

    normalization = (math.pi * sid) / (2.0 * sdd * num_views)
    return ReconstructionCase(
        name="Cone 3D",
        sinogram=sinogram,
        volume_shape=(nz, ny, nx),
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title="Phantom",
        supports_fbp=True,
        supports_fdk=True,
        fbp_normalization_scale=normalization,
        fdk_normalization_scale=normalization,
        fbp_weight=cone_weight,
        fdk_weight=cone_weight,
    )


def build_measured_cone_3d_case(config: MeasuredConeDataConfig) -> ReconstructionCase:
    """Build a 3D cone-beam case from measured TIFF projections and geometry."""
    data_dir = Path(config.data_dir)
    trajectory_json_path = Path(config.trajectory_json_path)
    reference_volume_path = None if config.reference_volume_path is None else Path(config.reference_volume_path)
    reference_meta_path = None if config.reference_meta_path is None else Path(config.reference_meta_path)
    explicit_voxel_spacing = None if config.voxel_spacing_mm is None else float(config.voxel_spacing_mm)

    nz, ny, nx = config.volume_shape
    target_det_u, target_det_v = config.target_detector_shape

    with trajectory_json_path.open("r", encoding="utf-8") as handle:
        geometry_payload = json.load(handle)

    src_pos, det_center, det_u_vec, det_v_vec = load_arbitrary_cone_geometry_from_json(
        trajectory_json_path,
        flip_det_u=config.flip_det_u,
        flip_det_v=config.flip_det_v,
        recenter_to_isocenter=config.recenter_to_isocenter,
    )

    view_stride = max(1, int(np.ceil(src_pos.shape[0] / config.target_view_count)))
    det_v_binning = max(1, int(np.ceil(geometry_payload["detector"]["num_pixels"]["v"] / target_det_v)))
    det_u_binning = max(1, int(np.ceil(geometry_payload["detector"]["num_pixels"]["u"] / target_det_u)))

    measured_sino_np = load_tiff_projections(
        data_dir,
        log_transform=config.log_transform,
        revert=config.revert,
        viewwise_i0=config.viewwise_i0,
        air_border_px=config.air_border_px,
        subtract_air_baseline=config.subtract_air_baseline,
        air_baseline_percentile=config.air_baseline_percentile,
        view_stride=view_stride,
        detector_binning_u=det_u_binning,
        detector_binning_v=det_v_binning,
        debug_visualization=False,
    )

    src_pos = src_pos[::view_stride]
    det_center = det_center[::view_stride]
    det_u_vec = det_u_vec[::view_stride]
    det_v_vec = det_v_vec[::view_stride]

    measured_du = float(geometry_payload["detector"]["pixel_size_mm"]["u"]) * det_u_binning
    measured_dv = float(geometry_payload["detector"]["pixel_size_mm"]["v"]) * det_v_binning
    header_offset_u_px = float(geometry_payload["detector"]["offset_px"].get("horizontal", 0.0) or 0.0) / det_u_binning
    header_offset_v_px = float(geometry_payload["detector"]["offset_px"].get("vertical", 0.0) or 0.0) / det_v_binning
    measured_det_v = measured_sino_np.shape[1]
    measured_det_u = measured_sino_np.shape[2]

    det_u_vec, det_v_vec, measured_du, measured_dv, measured_det_u, measured_det_v = apply_detector_geometry_convention(
        det_u_vec,
        det_v_vec,
        du=measured_du,
        dv=measured_dv,
        det_u=measured_det_u,
        det_v=measured_det_v,
        flip_u=config.flip_u,
        flip_v=config.flip_v,
        transpose_uv=config.transpose_uv,
    )
    header_offset_u_px, header_offset_v_px = transform_detector_offsets(
        header_offset_u_px,
        header_offset_v_px,
        {
            "flip_u": config.flip_u,
            "flip_v": config.flip_v,
            "transpose_uv": config.transpose_uv,
        },
    )
    det_center = shift_detector_center(
        det_center,
        det_u_vec,
        det_v_vec,
        measured_du,
        measured_dv,
        offset_u_px=header_offset_u_px,
        offset_v_px=header_offset_v_px,
    )

    reference_np = None
    reference_title = None
    resized_reference_voxel_spacing = None
    reference_meta = None
    reference_voxel_spacing = None
    source_shape = None
    if reference_meta_path is not None and reference_meta_path.exists():
        reference_meta = json.loads(reference_meta_path.read_text())
        source_shape = tuple(int(value) for value in reference_meta["shape_zyx"])
        reference_voxel_spacing = float(reference_meta["voxel_size_mm"])
        if explicit_voxel_spacing is None:
            resize_factors = tuple(src / dst for src, dst in zip(source_shape, config.volume_shape))
            if max(resize_factors) - min(resize_factors) > 1e-6:
                raise ValueError(
                    f"Reference volume resize must stay isotropic, got factors {resize_factors}."
                )
            resized_reference_voxel_spacing = reference_voxel_spacing * resize_factors[0]

    if reference_volume_path is not None and reference_meta is not None and reference_volume_path.exists():
        reference_volume_np = np.load(reference_volume_path)
        if explicit_voxel_spacing is None:
            reference_np = resize_volume_to_shape(reference_volume_np, config.volume_shape)
        else:
            if reference_voxel_spacing is not None and source_shape is not None:
                spacing_scale = reference_voxel_spacing / explicit_voxel_spacing
                if abs(spacing_scale - 1.0) > 1e-6:
                    resampled_shape = tuple(max(1, int(round(dim * spacing_scale))) for dim in source_shape)
                    reference_volume_np = resize_volume_to_shape(reference_volume_np, resampled_shape)
            reference_np = _center_crop_or_pad_volume(reference_volume_np, config.volume_shape)
        if config.normalize_reference:
            reference_np = normalize_volume(reference_np, upper_percentile=None)
        reference_title = "Reference Volume"

    if explicit_voxel_spacing is not None:
        measured_voxel_spacing = explicit_voxel_spacing
    elif resized_reference_voxel_spacing is not None:
        measured_voxel_spacing = float(resized_reference_voxel_spacing)
    else:
        measured_voxel_spacing = auto_voxel_spacing_from_detector(
            config.volume_shape,
            (measured_det_u, measured_det_v),
            measured_du,
            measured_dv,
            magnification=float(geometry_payload["source"]["magnification"]),
            fov_margin_mm=config.measured_fov_margin_mm,
        )

    forward_single, back_single, back_project_all = make_cone_3d_operators(
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        volume_shape=(nz, ny, nx),
        detector_shape=(measured_det_u, measured_det_v),
        du=measured_du,
        dv=measured_dv,
        voxel_spacing=measured_voxel_spacing,
        projector_mode=config.projector_mode,
    )

    iterative_projection_weights = _trajectory_quadrature_weights(src_pos)
    iterative_volume_support_mask = None
    if config.use_leap_fov_support_mask:
        sensitivity_support_mask = _build_sensitivity_support_mask(
            back_project_all,
            tuple(int(value) for value in measured_sino_np.shape),
            threshold_ratio=float(config.support_mask_sensitivity_threshold_ratio),
        )
        circular_fov_mask = _build_leap_style_circular_fov_mask(
            config.volume_shape,
            measured_voxel_spacing,
            (measured_det_u, measured_det_v),
            measured_du,
            src_pos,
            det_center,
            fov_margin_mm=config.measured_fov_margin_mm,
        )
        iterative_volume_support_mask = xp.array(
            sensitivity_support_mask & circular_fov_mask,
        )

    reference = None if reference_np is None else xp.array(reference_np, dtype=xp.float32)
    sinogram = xp.array(measured_sino_np, dtype=xp.float32)
    return ReconstructionCase(
        name="Measured Cone 3D",
        sinogram=sinogram,
        volume_shape=config.volume_shape,
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title=reference_title,
        iterative_projection_subset_count=int(config.iterative_projection_subset_count),
        iterative_normalized_sart_relaxation=float(config.iterative_normalized_sart_relaxation),
        iterative_backprojection_scale=(
            None if config.iterative_backprojection_scale is None else float(config.iterative_backprojection_scale)
        ),
        pocs_iterative_update_method="normalized_sart",
        iterative_volume_support_mask=iterative_volume_support_mask,
        iterative_volume_support_mask_mode=str(config.iterative_volume_support_mask_mode),
        iterative_voxel_sensitivity_normalization=(str(config.projector_mode).strip().lower() == "footprint"),
        iterative_projection_weights=iterative_projection_weights,
    )


@dataclass
class NpyProjectionsConfig:
    """Configuration for building a cone-beam case from pre-computed .npy line integrals.

    The npy file must contain line integrals (float32). No log transform is
    applied — the array is used as the sinogram directly. With the default
    ``transpose_uv=True`` the stack is expected in detector-image order
    ``[n_views, det_v, det_u]`` (rows = v, matching the TIFF pipeline); set
    ``transpose_uv=False`` if it is already stored ``[n_views, det_u, det_v]``.

    If the source volume has a non-zero background (e.g. Firefly volumes with a
    constant air offset), set subtract_air_baseline=True.  This estimates a
    per-view baseline from the detector border pixels and subtracts it,
    matching the behaviour of the TIFF pipeline's subtract_air_baseline option.
    """

    projections_npy_path: str | Path
    geometry_json_path: str | Path
    volume_shape: tuple[int, int, int] = (384, 384, 384)
    voxel_spacing_mm: float = 0.3
    projector_mode: str = "footprint"
    flip_det_u: bool = False
    flip_det_v: bool = False
    recenter_to_isocenter: bool = True
    flip_u: bool = False
    flip_v: bool = False
    transpose_uv: bool = True
    subtract_air_baseline: bool = False
    air_border_px: int = 16
    air_baseline_percentile: float = 50.0


def build_npy_cone_3d_case(config: NpyProjectionsConfig) -> ReconstructionCase:
    """Build a 3D cone-beam ReconstructionCase from pre-computed line integral projections.

    Reads the geometry from *geometry_json_path* and the sinogram directly from
    *projections_npy_path* — no TIFF loading or log transform involved.
    The npy shape is [n_views, det_u, det_v] and the geometry positions are
    matched 1-to-1 with the npy views (no striding or binning applied).
    Detector pixel offset must be zero; use build_measured_cone_3d_case for
    data with non-zero detector offsets.
    """
    geometry_json = Path(config.geometry_json_path)
    with geometry_json.open("r", encoding="utf-8") as fh:
        geo = json.load(fh)

    src_pos, det_center, det_u_vec, det_v_vec = load_arbitrary_cone_geometry_from_json(
        geometry_json,
        flip_det_u=config.flip_det_u,
        flip_det_v=config.flip_det_v,
        recenter_to_isocenter=config.recenter_to_isocenter,
    )
    src_pos = _b.to_numpy(src_pos)
    det_center = _b.to_numpy(det_center)
    det_u_vec = _b.to_numpy(det_u_vec)
    det_v_vec = _b.to_numpy(det_v_vec)

    sino_np = np.load(Path(config.projections_npy_path), allow_pickle=False).astype(np.float32)
    n_views = sino_np.shape[0]

    if config.subtract_air_baseline:
        border_px = max(0, int(config.air_border_px))
        if border_px > 0:
            border_samples = [
                sino_np[:, :border_px, :].reshape(n_views, -1),
                sino_np[:, -border_px:, :].reshape(n_views, -1),
                sino_np[:, :, :border_px].reshape(n_views, -1),
                sino_np[:, :, -border_px:].reshape(n_views, -1),
            ]
            baseline_source = np.concatenate(border_samples, axis=1)
        else:
            baseline_source = sino_np.reshape(n_views, -1)
        air_baseline = np.percentile(
            baseline_source,
            float(config.air_baseline_percentile),
            axis=1,
        ).astype(np.float32).reshape(n_views, 1, 1)
        sino_np = np.maximum(sino_np - air_baseline, 0.0)

    src_pos = src_pos[:n_views]
    det_center = det_center[:n_views]
    det_u_vec = det_u_vec[:n_views]
    det_v_vec = det_v_vec[:n_views]

    du = float(geo["detector"]["pixel_size_mm"]["u"])
    dv = float(geo["detector"]["pixel_size_mm"]["v"])
    # Detector counts are bound to the data axes: axis 1 is the projector's u
    # axis, axis 2 its v axis. ``transpose_uv`` relabels the *geometry* (vectors
    # and pitches) so that a stack stored in detector-image order
    # (views, v, u) is consumed correctly without copying the data — the counts
    # must therefore NOT be swapped along with the vectors.
    det_u = sino_np.shape[1]
    det_v = sino_np.shape[2]

    det_u_vec, det_v_vec, du, dv, _, _ = apply_detector_geometry_convention(
        det_u_vec, det_v_vec,
        du=du, dv=dv, det_u=det_u, det_v=det_v,
        flip_u=config.flip_u, flip_v=config.flip_v, transpose_uv=config.transpose_uv,
    )

    nz, ny, nx = config.volume_shape
    forward_single, back_single, back_project_all = make_cone_3d_operators(
        src_pos, det_center, det_u_vec, det_v_vec,
        volume_shape=(nz, ny, nx),
        detector_shape=(det_u, det_v),
        du=du,
        dv=dv,
        voxel_spacing=float(config.voxel_spacing_mm),
        projector_mode=str(config.projector_mode),
    )

    iterative_projection_weights = _trajectory_quadrature_weights(src_pos)

    return ReconstructionCase(
        name="NPY Cone 3D",
        sinogram=xp.array(sino_np, dtype=xp.float32),
        volume_shape=config.volume_shape,
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        pocs_iterative_update_method="normalized_sart",
        iterative_voxel_sensitivity_normalization=(
            str(config.projector_mode).strip().lower() == "footprint"
        ),
        iterative_projection_weights=iterative_projection_weights,
    )
