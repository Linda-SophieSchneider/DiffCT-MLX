"""Geometry-agnostic SART reconstruction (backend-neutral)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..backend import active as _b
from ._core import (
    ArrayLike,
    BackProjector,
    ForwardProjector,
    ReconstructionParameters,
    clamp_reconstruction_volume,
    initialize_volume,
    normalize_measured_projections,
    print_progress,
    run_iterative_sweeps,
    validate_reconstruction_inputs,
)

xp = _b.xp


@dataclass
class SARTParameters(ReconstructionParameters):
    """SART-specific configuration."""


def reconstruct_sart(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: SARTParameters,
    *,
    show_progress: bool = True,
):
    """Reconstruct with SART using user-provided per-view operators."""
    validate_reconstruction_inputs(params, measured_projections)
    measured = normalize_measured_projections(measured_projections, params.dtype)
    volume, ones_volume, _ = initialize_volume(params)

    # Geometry-only maps (raylengths / sensitivities) are computed once and reused
    # across outer iterations via this cache.
    sweep_cache: dict = {}
    for iteration in range(params.iteration_count):
        skip_first_sart = iteration == 0 and params.initial_volume is not None
        if not skip_first_sart:
            volume = run_iterative_sweeps(
                volume=volume,
                measured_projections=measured,
                ones_volume=ones_volume,
                forward_project=forward_project,
                back_project=back_project,
                params=params,
                outer_iteration_index=iteration,
                sweep_cache=sweep_cache,
            )
        volume = clamp_reconstruction_volume(volume, params, stage="iteration")
        # Force evaluation per outer iteration so long runs do not accumulate an
        # unbounded lazy graph on lazy backends (MLX/Metal).
        xp.eval(volume)
        if show_progress:
            print_progress(iteration, params.iteration_count)

    volume = clamp_reconstruction_volume(volume, params, stage="final")
    xp.eval(volume)
    return volume


run_sart = reconstruct_sart
