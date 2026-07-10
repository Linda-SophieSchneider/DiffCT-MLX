"""Core helpers for geometry-agnostic iterative reconstruction algorithms.

Backend-neutral port of the MLX implementation: all array operations route
through the active backend's ``xp`` namespace (:mod:`diffct_mlx.backend`), so the
same SART / SIRT / normalized-SART machinery runs on MLX and Torch/CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np

from ..backend import active as _b

xp = _b.xp


ArrayLike = Any
ForwardProjector = Callable[[Any, int], Any]
BackProjector = Callable[[Any, int], Any]
SARTDebugCallback = Callable[[dict], None]
_MATERIALIZE_INTERVAL = 16


@dataclass
class ReconstructionParameters:
    """Parameters shared by SART and the ART-TV family."""

    volume_shape: tuple[int, ...]
    iteration_count: int
    sart_iteration_count: int = 1
    iterative_update_method: str = "sart"
    pixel_extreme_values: tuple[float, float] = (0.0, float("inf"))
    voxel_extreme_values: tuple[float, float] = (-float("inf"), float("inf"))
    initial_volume: ArrayLike | None = None
    enforce_positivity: bool = True
    positivity_mode: str = "per_iteration"
    projection_order: Sequence[int] | None = None
    shuffle_projection_order: bool = True
    projection_order_seed: int = 0
    sart_debug_callback: SARTDebugCallback | None = None
    raylength_thresholding: bool = True
    raylength_quantile: float = 1e-3
    raylength_epsilon: float = 5e-4
    preserve_unmasked_computed_projection: bool = False
    detector_border_u: int = 0
    detector_border_v: int = 0
    volume_border_width: int = 0
    volume_support_mask: ArrayLike | None = None
    volume_support_mask_mode: str = "always"
    voxel_sensitivity_normalization: bool = True
    projection_chunk_size: int | None = None
    projection_subset_count: int = 1
    projection_weights: Sequence[float] | None = None
    normalized_sart_relaxation: float = 0.9
    # None resolves per update method (normalized_sart -> normalized_sart_relaxation,
    # otherwise 1.0); an explicit float always wins.
    backprojection_scale: float | None = None
    dtype: Any = field(default_factory=lambda: _b.float32)


@dataclass
class RegularizationParameters:
    """Parameters shared by TV-based regularization steps."""

    reg_iteration_count: int = 20
    alpha: float = 0.2
    tv_eps: float = 1e-6


def as_mx_array(value: ArrayLike, dtype: Any = None):
    """Convert a NumPy/backend input to the active backend's array."""
    return xp.array(value, dtype=dtype if dtype is not None else _b.float32)


def normalize_measured_projections(
    measured_projections: Sequence[ArrayLike],
    dtype: Any,
) -> list:
    """Convert measured projections once up front."""
    return [as_mx_array(projection, dtype=dtype) for projection in measured_projections]


def validate_reconstruction_inputs(
    params: ReconstructionParameters,
    measured_projections: Sequence[ArrayLike],
) -> None:
    """Validate the generic reconstruction inputs."""
    if not measured_projections:
        raise ValueError("At least one measured projection is required.")
    if not params.volume_shape:
        raise ValueError("volume_shape must not be empty.")
    if any(int(dim) <= 0 for dim in params.volume_shape):
        raise ValueError(f"volume_shape must be positive in every axis, got {params.volume_shape!r}.")
    if params.iteration_count <= 0:
        raise ValueError("iteration_count must be positive.")
    if params.sart_iteration_count <= 0:
        raise ValueError("sart_iteration_count must be positive.")
    if params.iterative_update_method not in {"sart", "sirt", "normalized_sart"}:
        raise ValueError("iterative_update_method must be one of: 'sart', 'sirt', 'normalized_sart'.")
    if params.positivity_mode not in {"per_iteration", "final", "none"}:
        raise ValueError("positivity_mode must be one of: 'per_iteration', 'final', 'none'.")
    if params.raylength_quantile < 0.0 or params.raylength_quantile >= 1.0:
        raise ValueError("raylength_quantile must be in [0, 1).")
    if params.raylength_epsilon <= 0.0:
        raise ValueError("raylength_epsilon must be positive.")
    if params.detector_border_u < 0 or params.detector_border_v < 0:
        raise ValueError("detector border widths must be non-negative.")
    if params.volume_border_width < 0:
        raise ValueError("volume border width must be non-negative.")
    if params.volume_support_mask is not None and tuple(np.shape(params.volume_support_mask)) != tuple(params.volume_shape):
        raise ValueError(
            "volume_support_mask shape does not match volume_shape: "
            f"{tuple(np.shape(params.volume_support_mask))!r} vs {tuple(params.volume_shape)!r}."
        )
    if params.volume_support_mask_mode not in {"always", "final", "none"}:
        raise ValueError("volume_support_mask_mode must be one of: 'always', 'final', 'none'.")
    if params.projection_chunk_size is not None and params.projection_chunk_size <= 0:
        raise ValueError("projection_chunk_size must be positive when specified.")
    if params.projection_subset_count <= 0:
        raise ValueError("projection_subset_count must be positive.")
    if params.projection_weights is not None and len(params.projection_weights) != len(measured_projections):
        raise ValueError(
            "projection_weights must contain one entry per measured projection: "
            f"expected {len(measured_projections)}, got {len(params.projection_weights)}."
        )
    if params.normalized_sart_relaxation <= 0.0:
        raise ValueError("normalized_sart_relaxation must be positive.")
    if params.initial_volume is not None and tuple(np.shape(params.initial_volume)) != tuple(params.volume_shape):
        raise ValueError(
            "initial_volume shape does not match volume_shape: "
            f"{tuple(np.shape(params.initial_volume))!r} vs {tuple(params.volume_shape)!r}."
        )


def projection_order(params: ReconstructionParameters, num_projections: int) -> tuple[int, ...]:
    """Resolve and validate the per-iteration projection traversal order."""
    if params.projection_order is None:
        if not params.shuffle_projection_order:
            return tuple(range(num_projections))
        rng = np.random.default_rng(int(params.projection_order_seed))
        order = np.arange(num_projections, dtype=np.int32)
        rng.shuffle(order)
        return tuple(int(index) for index in order)

    order = tuple(int(index) for index in params.projection_order)
    if len(order) != num_projections:
        raise ValueError(
            "projection_order must contain one entry per measured projection: "
            f"expected {num_projections}, got {len(order)}."
        )
    invalid = [index for index in order if index < 0 or index >= num_projections]
    if invalid:
        raise ValueError(f"projection_order contains invalid indices: {invalid!r}.")
    return order


def projection_subsets(
    order: Sequence[int],
    subset_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Split an ordered traversal into interleaved ordered subsets."""
    order_tuple = tuple(int(index) for index in order)
    if not order_tuple:
        return tuple()
    subset_count = max(1, min(int(subset_count), len(order_tuple)))
    if subset_count == 1:
        return (order_tuple,)
    return tuple(
        tuple(order_tuple[offset::subset_count])
        for offset in range(subset_count)
        if order_tuple[offset::subset_count]
    )


def initialize_volume(params: ReconstructionParameters):
    """Create the reconstruction, ones, and zero reference volumes."""
    if params.initial_volume is None:
        volume = xp.zeros(params.volume_shape, dtype=params.dtype)
    else:
        volume = as_mx_array(params.initial_volume, dtype=params.dtype)
    ones_volume = xp.ones(params.volume_shape, dtype=params.dtype)
    zero_volume = xp.zeros(params.volume_shape, dtype=params.dtype)
    return volume, ones_volume, zero_volume


def clamp_volume(
    volume,
    voxel_extreme_values: tuple[float, float],
    *,
    enforce_positivity: bool,
):
    """Apply configured bounds to a reconstruction volume."""
    lower, upper = voxel_extreme_values
    if enforce_positivity:
        lower = max(lower, 0.0)
    if np.isfinite(lower):
        volume = xp.maximum(volume, float(lower))
    if np.isfinite(upper):
        volume = xp.minimum(volume, float(upper))
    return volume


def should_enforce_positivity(
    params: ReconstructionParameters,
    *,
    stage: str,
) -> bool:
    """Resolve whether positivity should be enforced at the given stage."""
    if not params.enforce_positivity:
        return False
    if params.positivity_mode == "none":
        return False
    if params.positivity_mode == "final":
        return stage == "final"
    return stage in {"iteration", "regularization", "final"}


def clamp_reconstruction_volume(
    volume,
    params: ReconstructionParameters,
    *,
    stage: str,
):
    """Clamp a reconstruction volume according to the configured positivity policy."""
    volume = clamp_volume(
        volume,
        params.voxel_extreme_values,
        enforce_positivity=should_enforce_positivity(params, stage=stage),
    )
    volume = apply_volume_border_mask(volume, params.volume_border_width)
    if should_apply_volume_support_mask(params, stage=stage):
        return apply_volume_support_mask(volume, params.volume_support_mask)
    return volume


def apply_volume_border_mask(volume, border_width: int):
    """Suppress voxels at the reconstruction-volume border."""
    border_width = int(border_width)
    if border_width <= 0:
        return volume
    shape = tuple(int(dimension) for dimension in volume.shape)
    if not shape:
        return volume
    mask = volume_border_keep_mask(shape, border_width)
    return xp.where(mask, volume, 0.0)


def apply_volume_support_mask(volume, support_mask: ArrayLike | None):
    """Suppress voxels outside the configured reconstruction support."""
    if support_mask is None:
        return volume
    mask = support_mask if hasattr(support_mask, "shape") else as_mx_array(support_mask, dtype=_b.float32)
    return xp.where(mask > 0.0, volume, 0.0)


def should_apply_volume_support_mask(
    params: ReconstructionParameters,
    *,
    stage: str,
) -> bool:
    """Resolve whether the configured reconstruction support should be enforced."""
    if params.volume_support_mask is None:
        return False
    if params.volume_support_mask_mode == "none":
        return False
    if params.volume_support_mask_mode == "final":
        return stage == "final"
    return True


@lru_cache(maxsize=32)
def volume_border_keep_mask(shape: tuple[int, ...], border_width: int):
    """Cache a keep-mask that removes a fixed-width border from a volume."""
    mask_np = np.ones(shape, dtype=bool)
    for axis, axis_size in enumerate(shape):
        trim = min(int(border_width), axis_size)
        if trim <= 0:
            continue
        leading = [slice(None)] * len(shape)
        trailing = [slice(None)] * len(shape)
        leading[axis] = slice(0, trim)
        trailing[axis] = slice(axis_size - trim, axis_size)
        mask_np[tuple(leading)] = False
        mask_np[tuple(trailing)] = False
    return xp.array(mask_np)


def print_progress(iteration: int, iteration_count: int) -> None:
    """Print a small textual progress bar."""
    iteration_number = iteration + 1
    finished = int(10.0 * iteration_number / iteration_count)
    remaining = max(0, 10 - finished)
    progress = ("█" * finished) + ("-" * remaining)
    print(
        f"\rProgress: |{progress}| Finished Iterations: {iteration_number} out of {iteration_count}",
        end="",
        flush=True,
    )
    if iteration_number == iteration_count:
        print()


def scalar_norm(value) -> float:
    """Return an array norm as a Python float."""
    return xp.norm(value)


def projection_residual_norm(
    volume,
    measured_projection,
    forward_project: ForwardProjector,
    projection_index: int,
) -> float:
    """Compute the residual norm for one measured projection."""
    computed_projection = forward_project(volume, projection_index)
    return scalar_norm(measured_projection - computed_projection)


def threshold_small_raylengths(
    raylength_projection,
    quantile: float,
):
    """Drop the smallest positive raylength values to improve SART stability.

    On-device (nearest-rank) quantile of the positive raylengths — no host
    round-trip — so this stays on the GPU inside the iterative sweeps.
    """
    if quantile <= 0.0:
        return raylength_projection
    r = raylength_projection
    flat = xp.reshape(r, (-1,))
    total = int(flat.shape[0])
    n_pos = int(xp.sum(xp.where(flat > 0.0, xp.ones_like(flat), xp.zeros_like(flat))))
    if n_pos == 0:
        return r
    ordered = xp.sort(flat)                       # ascending; positives are the top n_pos
    rank = (total - n_pos) + int(float(quantile) * (n_pos - 1))
    rank = min(max(rank, 0), total - 1)
    threshold = ordered[rank]                      # scalar backend value
    mask = (r > 0.0) & (r <= threshold)
    return xp.where(mask, 0.0, r)


def prepare_raylength_projection(
    raylength_projection,
    params: ReconstructionParameters,
):
    """Apply the configured raylength preprocessing once per projection."""
    if not params.raylength_thresholding:
        return raylength_projection
    return threshold_small_raylengths(raylength_projection, params.raylength_quantile)


def compute_sart_correction(
    measured_projection,
    computed_projection,
    raylength_projection,
    params: ReconstructionParameters,
):
    """Compute the SART correction image for one measured projection."""
    clipped_projection = prepare_measured_projection(measured_projection, computed_projection, params)
    epsilon = float(params.raylength_epsilon)
    mask = raylength_projection > epsilon
    numerator = clipped_projection - computed_projection
    denominator = xp.maximum(raylength_projection, epsilon)
    normalized_update = numerator / denominator
    normalized_update = xp.where(mask, normalized_update, 0.0)
    return apply_detector_border_mask(normalized_update, params, fill_value=0.0)


def compute_detector_update_mask(
    raylength_projection,
    params: ReconstructionParameters,
):
    """Return detector pixels that participate in an iterative update."""
    valid = xp.where(raylength_projection > float(params.raylength_epsilon), 1.0, 0.0)
    return apply_detector_border_mask(valid, params, fill_value=0.0)


def normalize_backprojection_by_sensitivity(
    backprojection_volume,
    sensitivity_volume,
    params: ReconstructionParameters,
):
    """Divide voxel updates by the local ray-coverage sensitivity."""
    if not params.voxel_sensitivity_normalization:
        return backprojection_volume
    epsilon = float(params.raylength_epsilon)
    return xp.where(
        sensitivity_volume > epsilon,
        backprojection_volume / xp.maximum(sensitivity_volume, epsilon),
        0.0,
    )


def prepare_measured_projection(
    measured_projection,
    computed_projection,
    params: ReconstructionParameters,
):
    """Apply measurement clipping before the SART normalization step."""
    lower, upper = params.pixel_extreme_values
    clipped_projection = measured_projection
    if np.isfinite(upper):
        clipped_projection = xp.where(clipped_projection > float(upper), computed_projection, clipped_projection)
    if np.isfinite(lower):
        clipped_projection = xp.where(clipped_projection < float(lower), 0.0, clipped_projection)
    return clipped_projection


def apply_detector_border_mask(
    value,
    params: ReconstructionParameters,
    *,
    fill_value: float = 0.0,
):
    """Suppress detector-border pixels in 1D or 2D projection arrays."""
    border_u = int(params.detector_border_u)
    border_v = int(params.detector_border_v)
    if border_u <= 0 and border_v <= 0:
        return value

    shape = tuple(int(dimension) for dimension in value.shape)
    if len(shape) not in {1, 2}:
        return value
    mask = detector_border_keep_mask(shape, border_u, border_v)
    return xp.where(mask, value, fill_value)


@lru_cache(maxsize=64)
def detector_border_keep_mask(
    shape: tuple[int, ...],
    border_u: int,
    border_v: int,
):
    """Cache detector-border keep masks."""
    mask_np = np.ones(shape, dtype=bool)
    if len(shape) == 1:
        if border_u > 0:
            trim_u = min(border_u, shape[0])
            mask_np[:trim_u] = False
            mask_np[-trim_u:] = False
        return xp.array(mask_np)

    trim_u = min(border_u, shape[0]) if border_u > 0 else 0
    trim_v = min(border_v, shape[1]) if border_v > 0 else 0
    if trim_u > 0:
        mask_np[:trim_u, :] = False
        mask_np[-trim_u:, :] = False
    if trim_v > 0:
        mask_np[:, :trim_v] = False
        mask_np[:, -trim_v:] = False
    return xp.array(mask_np)


def _array_stats(name: str, value) -> dict:
    """Return compact numeric stats for one array."""
    value_np = _b.to_numpy(value)
    return {
        f"{name}_min": float(np.min(value_np)),
        f"{name}_max": float(np.max(value_np)),
        f"{name}_mean": float(np.mean(value_np)),
        f"{name}_norm": float(np.linalg.norm(value_np)),
    }


def _raylength_stats(raylength_projection) -> dict:
    """Return SART-relevant stats for the raylength projection."""
    raylength_np = _b.to_numpy(raylength_projection)
    positive_values = raylength_np[raylength_np > 0.0]
    if positive_values.size == 0:
        return {
            "raylength_nonzero_count": 0.0,
            "raylength_positive_min": 0.0,
            "raylength_positive_p001": 0.0,
            "raylength_positive_mean": 0.0,
            "raylength_positive_max": 0.0,
        }
    return {
        "raylength_nonzero_count": float(positive_values.size),
        "raylength_positive_min": float(np.min(positive_values)),
        "raylength_positive_p001": float(np.quantile(positive_values, 1e-3)),
        "raylength_positive_mean": float(np.mean(positive_values)),
        "raylength_positive_max": float(np.max(positive_values)),
    }


def projection_slice_ranges(
    projection_count: int,
    params: ReconstructionParameters,
) -> tuple[tuple[int, int], ...]:
    """Resolve contiguous projection slices for optional chunked projector calls."""
    chunk_size = params.projection_chunk_size
    if chunk_size is None or chunk_size <= 1:
        return tuple((index, index + 1) for index in range(projection_count))
    return tuple(
        (start, min(start + int(chunk_size), projection_count))
        for start in range(0, projection_count, int(chunk_size))
    )


def forward_project_views(
    volume,
    forward_project: ForwardProjector,
    projection_count: int,
    params: ReconstructionParameters,
) -> list:
    """Project a fixed volume over all views, optionally using contiguous chunks."""
    project_slice = getattr(forward_project, "project_slice", None)
    if project_slice is None or params.projection_chunk_size is None or params.projection_chunk_size <= 1:
        return [forward_project(volume, projection_index) for projection_index in range(projection_count)]

    projections = [None] * projection_count
    for start, stop in projection_slice_ranges(projection_count, params):
        chunk_projection = project_slice(volume, start, stop)
        for offset, projection_index in enumerate(range(start, stop)):
            projections[projection_index] = chunk_projection[offset]
    return projections


def precompute_raylength_projections(
    ones_volume,
    forward_project: ForwardProjector,
    params: ReconstructionParameters,
    projection_count: int,
) -> dict:
    """Cache per-view raylength projections for one iterative sweep call."""
    projected_ones = forward_project_views(
        ones_volume,
        forward_project,
        projection_count,
        params,
    )
    raylengths = {}
    for projection_index in range(projection_count):
        prepared_projection = prepare_raylength_projection(projected_ones[int(projection_index)], params)
        xp.eval(prepared_projection)
        raylengths[int(projection_index)] = prepared_projection
    return raylengths


def precompute_sensitivity_volumes(
    raylength_projections: dict,
    back_project: BackProjector,
    params: ReconstructionParameters,
    projection_count: int,
):
    """Precompute per-view voxel-coverage sensitivity volumes for all projections.

    WARNING: stores one full-resolution volume per projection view.  Memory cost is
    projection_count × volume_voxels × 4 bytes — use only when projection_count is
    small (e.g. a few dozen views) or the volume is small.  For large scans (hundreds
    of views, 384³+ volumes) call compute_voxel_sensitivity_volume per-view on the
    fly instead, as SART does.
    """
    if not params.voxel_sensitivity_normalization:
        return None
    sensitivity_volumes = {}
    for projection_index in range(projection_count):
        sv = compute_voxel_sensitivity_volume(
            raylength_projections[int(projection_index)],
            back_project,
            projection_index,
            params,
        )
        sensitivity_volumes[int(projection_index)] = sv
    return sensitivity_volumes


def compute_voxel_sensitivity_volume(
    raylength_projection,
    back_project: BackProjector,
    projection_index: int,
    params: ReconstructionParameters,
):
    """Compute the voxel-coverage normalization volume for one projection."""
    if not params.voxel_sensitivity_normalization:
        return None
    detector_mask = compute_detector_update_mask(raylength_projection, params)
    sensitivity_volume = back_project(detector_mask, int(projection_index))
    sensitivity_volume = apply_volume_border_mask(sensitivity_volume, params.volume_border_width)
    xp.eval(sensitivity_volume)
    return sensitivity_volume


def resolve_backprojection_scale(params: ReconstructionParameters) -> float:
    """Resolve the multiplicative update scale for the configured iterative mode."""
    if params.backprojection_scale is not None:
        return float(params.backprojection_scale)
    if params.iterative_update_method == "normalized_sart":
        return float(params.normalized_sart_relaxation)
    return 1.0


def resolve_projection_weights(
    measured_projections: Sequence,
    params: ReconstructionParameters,
) -> tuple[float, ...]:
    """Resolve per-view quadrature weights for iterative updates."""
    if params.projection_weights is None:
        return tuple(1.0 for _ in range(len(measured_projections)))
    return tuple(float(value) for value in params.projection_weights)


#: Default cap for caching per-view sensitivity volumes across iterations. The
#: raylength/sensitivity maps are geometry-only (independent of the iterate), so
#: caching them avoids recomputing n_views projections every outer iteration.
_SENSITIVITY_CACHE_BUDGET_BYTES = 8 * 1024 ** 3


def _volume_byte_size(shape, dtype=None) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    try:
        itemsize = int(getattr(dtype, "itemsize", 4) or 4)
    except Exception:
        itemsize = 4
    return total * itemsize


def get_cached_raylengths(sweep_cache, ones_volume, forward_project, params, projection_count):
    """Raylength projections, computed once and reused across outer iterations."""
    if sweep_cache is not None and "raylengths" in sweep_cache:
        return sweep_cache["raylengths"]
    raylengths = precompute_raylength_projections(ones_volume, forward_project, params, projection_count)
    if sweep_cache is not None:
        sweep_cache["raylengths"] = raylengths
    return raylengths


def get_cached_sensitivities(sweep_cache, raylength_projections, back_project, params, projection_count):
    """Per-view sensitivity volumes, cached across iterations (memory-gated).

    Returns a ``{view: volume}`` dict, or ``None`` when sensitivity normalization
    is off or the full set would exceed the cache budget (then callers compute
    each per-view sensitivity on the fly, as before).
    """
    if not params.voxel_sensitivity_normalization:
        return None
    if sweep_cache is not None and "sensitivities" in sweep_cache:
        return sweep_cache["sensitivities"]
    need = projection_count * _volume_byte_size(params.volume_shape, params.dtype)
    sensitivities = None
    if need <= _SENSITIVITY_CACHE_BUDGET_BYTES:
        sensitivities = precompute_sensitivity_volumes(raylength_projections, back_project, params, projection_count)
    if sweep_cache is not None:
        sweep_cache["sensitivities"] = sensitivities
    return sensitivities


def get_cached_ns_backprojections(sweep_cache, raylength_projections, back_project, params, projection_count):
    """Per-view detector-mask backprojections for normalized SART (geometry-only).

    Cached across iterations, memory-gated. Returns ``{view: volume}`` or ``None``
    (then callers backproject the detector mask on the fly).
    """
    if sweep_cache is not None and "ns_sensitivity" in sweep_cache:
        return sweep_cache["ns_sensitivity"]
    need = projection_count * _volume_byte_size(params.volume_shape, params.dtype)
    result = None
    if need <= _SENSITIVITY_CACHE_BUDGET_BYTES:
        result = {}
        for projection_index in range(projection_count):
            detector_mask = compute_detector_update_mask(raylength_projections[int(projection_index)], params)
            backprojected = back_project(detector_mask, int(projection_index))
            xp.eval(backprojected)
            result[int(projection_index)] = backprojected
    if sweep_cache is not None:
        sweep_cache["ns_sensitivity"] = result
    return result


def run_sart_sweeps(
    volume,
    measured_projections: Sequence,
    ones_volume,
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: ReconstructionParameters,
    *,
    beta: float = 1.0,
    outer_iteration_index: int = 0,
    sweep_cache: dict | None = None,
):
    """Run the configured number of SART sweeps over all provided projections."""
    scale = resolve_backprojection_scale(params)
    order = projection_order(params, len(measured_projections))
    raylength_projections = get_cached_raylengths(
        sweep_cache, ones_volume, forward_project, params, len(measured_projections)
    )
    # Per-view sensitivity is geometry-only: hoist it out of the sweep loop and
    # reuse across outer iterations (memory-gated; None => compute on the fly).
    sensitivities = get_cached_sensitivities(
        sweep_cache, raylength_projections, back_project, params, len(measured_projections)
    )
    for sweep_index in range(params.sart_iteration_count):
        for step_index, projection_index in enumerate(order):
            measured_projection = measured_projections[projection_index]
            computed_projection = forward_project(volume, projection_index)
            raylength_projection = raylength_projections[int(projection_index)]
            correction_image = compute_sart_correction(
                measured_projection=measured_projection,
                computed_projection=computed_projection,
                raylength_projection=raylength_projection,
                params=params,
            )
            backprojection_volume = back_project(correction_image, projection_index)
            if params.voxel_sensitivity_normalization:
                if sensitivities is not None:
                    sensitivity_volume = sensitivities[int(projection_index)]
                else:
                    sensitivity_volume = compute_voxel_sensitivity_volume(
                        raylength_projection,
                        back_project,
                        int(projection_index),
                        params,
                    )
                backprojection_volume = normalize_backprojection_by_sensitivity(
                    backprojection_volume,
                    sensitivity_volume,
                    params,
                )
            volume = volume + (float(beta) * scale * backprojection_volume)
            volume = clamp_reconstruction_volume(volume, params, stage="sart_update")
            if params.sart_debug_callback is not None:
                debug_stats = {
                    "outer_iteration": float(outer_iteration_index),
                    "sart_sweep": float(sweep_index),
                    "projection_index": float(projection_index),
                    "beta": float(beta),
                    "backprojection_scale": scale,
                    **_array_stats("measured", measured_projection),
                    **_array_stats("computed", computed_projection),
                    **_raylength_stats(raylength_projection),
                    **_array_stats("correction", correction_image),
                    **_array_stats("backprojection", backprojection_volume),
                    **_array_stats("volume", volume),
                }
                params.sart_debug_callback(debug_stats)
            if (step_index + 1) % _MATERIALIZE_INTERVAL == 0:
                xp.eval(volume)
        # Materialize the updated volume after each sweep to keep lazy backends
        # (e.g. MLX/Metal) bounded on large 3D measured-data runs.
        xp.eval(volume)
    return volume


def run_sirt_sweeps(
    volume,
    measured_projections: Sequence,
    ones_volume,
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: ReconstructionParameters,
    *,
    beta: float = 1.0,
    outer_iteration_index: int = 0,
    sweep_cache: dict | None = None,
):
    """Run the configured number of SIRT sweeps over all provided projections."""
    scale = resolve_backprojection_scale(params)
    order = projection_order(params, len(measured_projections))
    raylength_projections = get_cached_raylengths(
        sweep_cache, ones_volume, forward_project, params, len(measured_projections)
    )
    projection_count = float(len(order))
    # Aggregate sensitivity volume is geometry-only; compute once and reuse across
    # outer iterations via the sweep cache.
    sirt_sensitivity = None
    if params.voxel_sensitivity_normalization:
        if sweep_cache is not None and "sirt_sensitivity" in sweep_cache:
            sirt_sensitivity = sweep_cache["sirt_sensitivity"]
        else:
            sirt_sensitivity = xp.zeros(params.volume_shape, dtype=params.dtype)
            for projection_index in range(len(measured_projections)):
                sv = compute_voxel_sensitivity_volume(
                    raylength_projections[int(projection_index)],
                    back_project,
                    projection_index,
                    params,
                )
                sirt_sensitivity = sirt_sensitivity + sv
            xp.eval(sirt_sensitivity)
            if sweep_cache is not None:
                sweep_cache["sirt_sensitivity"] = sirt_sensitivity
    for sweep_index in range(params.sart_iteration_count):
        reference_volume = volume
        accumulated_backprojection = xp.zeros_like(volume)
        for step_index, projection_index in enumerate(order):
            measured_projection = measured_projections[projection_index]
            computed_projection = forward_project(reference_volume, projection_index)
            raylength_projection = raylength_projections[int(projection_index)]
            correction_image = compute_sart_correction(
                measured_projection=measured_projection,
                computed_projection=computed_projection,
                raylength_projection=raylength_projection,
                params=params,
            )
            backprojection_volume = back_project(correction_image, projection_index)
            accumulated_backprojection = accumulated_backprojection + backprojection_volume
            if params.sart_debug_callback is not None:
                debug_stats = {
                    "outer_iteration": float(outer_iteration_index),
                    "sart_sweep": float(sweep_index),
                    "projection_index": float(projection_index),
                    "beta": float(beta),
                    "backprojection_scale": scale,
                    "iterative_update_method": 1.0,
                    **_array_stats("measured", measured_projection),
                    **_array_stats("computed", computed_projection),
                    **_raylength_stats(raylength_projection),
                    **_array_stats("correction", correction_image),
                    **_array_stats("backprojection", backprojection_volume),
                    **_array_stats("volume", reference_volume),
                }
                params.sart_debug_callback(debug_stats)
            if (step_index + 1) % _MATERIALIZE_INTERVAL == 0:
                xp.eval(accumulated_backprojection)
        if sirt_sensitivity is not None:
            averaged_backprojection = normalize_backprojection_by_sensitivity(
                accumulated_backprojection,
                sirt_sensitivity,
                params,
            )
        else:
            averaged_backprojection = accumulated_backprojection / projection_count
        volume = volume + (float(beta) * scale * averaged_backprojection)
        volume = clamp_reconstruction_volume(volume, params, stage="sart_update")
        xp.eval(volume)
    return volume


def run_normalized_sart_sweeps(
    volume,
    measured_projections: Sequence,
    ones_volume,
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: ReconstructionParameters,
    *,
    beta: float = 1.0,
    outer_iteration_index: int = 0,
    sweep_cache: dict | None = None,
):
    """Run LEAP-style normalized SART updates over full passes or ordered subsets."""
    scale = resolve_backprojection_scale(params)
    order = projection_order(params, len(measured_projections))
    projection_weights = resolve_projection_weights(measured_projections, params)
    subsets = projection_subsets(order, params.projection_subset_count)
    raylength_projections = get_cached_raylengths(
        sweep_cache, ones_volume, forward_project, params, len(measured_projections)
    )
    # Per-view detector-mask backprojections are geometry-only; cache across iters.
    ns_sensitivity = get_cached_ns_backprojections(
        sweep_cache, raylength_projections, back_project, params, len(measured_projections)
    )
    epsilon = float(params.raylength_epsilon)

    for sweep_index in range(params.sart_iteration_count):
        for subset_index, subset in enumerate(subsets):
            reference_volume = volume
            accumulated_backprojection = xp.zeros_like(volume)
            accumulated_sensitivity = xp.zeros_like(volume)
            computed_projections = None
            if len(subset) == len(order):
                computed_projections = forward_project_views(
                    reference_volume,
                    forward_project,
                    len(measured_projections),
                    params,
                )
            for step_index, projection_index in enumerate(subset):
                projection_weight = float(projection_weights[int(projection_index)])
                measured_projection = measured_projections[projection_index]
                if computed_projections is None:
                    computed_projection = forward_project(reference_volume, projection_index)
                else:
                    computed_projection = computed_projections[int(projection_index)]
                raylength_projection = raylength_projections[int(projection_index)]
                correction_image = compute_sart_correction(
                    measured_projection=measured_projection,
                    computed_projection=computed_projection,
                    raylength_projection=raylength_projection,
                    params=params,
                )
                backprojection_volume = back_project(correction_image, projection_index)
                accumulated_backprojection = accumulated_backprojection + (projection_weight * backprojection_volume)
                if ns_sensitivity is not None:
                    sensitivity_backprojection = ns_sensitivity[int(projection_index)]
                else:
                    detector_mask = compute_detector_update_mask(raylength_projection, params)
                    sensitivity_backprojection = back_project(detector_mask, projection_index)
                accumulated_sensitivity = accumulated_sensitivity + (
                    projection_weight * sensitivity_backprojection
                )
                if params.sart_debug_callback is not None:
                    debug_stats = {
                        "outer_iteration": float(outer_iteration_index),
                        "sart_sweep": float(sweep_index),
                        "projection_index": float(projection_index),
                        "subset_index": float(subset_index),
                        "subset_size": float(len(subset)),
                        "projection_weight": projection_weight,
                        "beta": float(beta),
                        "backprojection_scale": scale,
                        "iterative_update_method": 2.0,
                        **_array_stats("measured", measured_projection),
                        **_array_stats("computed", computed_projection),
                        **_raylength_stats(raylength_projection),
                        **_array_stats("correction", correction_image),
                        **_array_stats("backprojection", backprojection_volume),
                        **_array_stats("volume", reference_volume),
                    }
                    params.sart_debug_callback(debug_stats)
                if (step_index + 1) % _MATERIALIZE_INTERVAL == 0:
                    xp.eval(accumulated_backprojection, accumulated_sensitivity)
            normalized_update = xp.where(
                accumulated_sensitivity > epsilon,
                accumulated_backprojection / xp.maximum(accumulated_sensitivity, epsilon),
                0.0,
            )
            volume = volume + (float(beta) * scale * normalized_update)
            volume = clamp_reconstruction_volume(volume, params, stage="sart_update")
            xp.eval(volume)
    return volume


def run_iterative_sweeps(
    volume,
    measured_projections: Sequence,
    ones_volume,
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: ReconstructionParameters,
    *,
    beta: float = 1.0,
    outer_iteration_index: int = 0,
    sweep_cache: dict | None = None,
):
    """Dispatch to the configured iterative sweep method."""
    if params.iterative_update_method == "normalized_sart":
        return run_normalized_sart_sweeps(
            volume=volume,
            measured_projections=measured_projections,
            ones_volume=ones_volume,
            forward_project=forward_project,
            back_project=back_project,
            params=params,
            beta=beta,
            outer_iteration_index=outer_iteration_index,
            sweep_cache=sweep_cache,
        )
    if params.iterative_update_method == "sirt":
        return run_sirt_sweeps(
            volume=volume,
            measured_projections=measured_projections,
            ones_volume=ones_volume,
            forward_project=forward_project,
            back_project=back_project,
            params=params,
            beta=beta,
            outer_iteration_index=outer_iteration_index,
            sweep_cache=sweep_cache,
        )
    return run_sart_sweeps(
        volume=volume,
        measured_projections=measured_projections,
        ones_volume=ones_volume,
        forward_project=forward_project,
        back_project=back_project,
        params=params,
        beta=beta,
        outer_iteration_index=outer_iteration_index,
        sweep_cache=sweep_cache,
    )
