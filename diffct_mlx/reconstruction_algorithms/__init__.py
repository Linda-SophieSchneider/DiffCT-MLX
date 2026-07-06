"""Reconstruction algorithms built on the unified projector API.

Ported incrementally from the MLX implementation to run on any backend.
Currently: analytic (FBP / FDK) and iterative (SART / SIRT). Regularized POCS
variants and DART follow in later phases.
"""

from ._analytic import (
    AnalyticalBackProjector,
    AnalyticalFilter,
    AnalyticalReconstructionParameters,
    AnalyticalWeighting,
    ramp_filter,
)
from ._core import (
    BackProjector,
    ForwardProjector,
    ReconstructionParameters,
    RegularizationParameters,
)
from .fbp import FBPParameters, reconstruct_fbp, run_fbp
from .fdk import FDKParameters, reconstruct_fdk, run_fdk
from .sart import SARTParameters, reconstruct_sart, run_sart
from .sirt import SIRTParameters, reconstruct_sirt, run_sirt


# Backwards-compatible aliases (mirrors the MLX package).
Reconstruction_Parameter = ReconstructionParameters
Regularisation_Parameter = RegularizationParameters

__all__ = [
    # analytic
    "AnalyticalBackProjector",
    "AnalyticalFilter",
    "AnalyticalWeighting",
    "AnalyticalReconstructionParameters",
    "ramp_filter",
    "FBPParameters",
    "run_fbp",
    "reconstruct_fbp",
    "FDKParameters",
    "run_fdk",
    "reconstruct_fdk",
    # core
    "BackProjector",
    "ForwardProjector",
    "ReconstructionParameters",
    "RegularizationParameters",
    "Reconstruction_Parameter",
    "Regularisation_Parameter",
    # iterative
    "SARTParameters",
    "run_sart",
    "reconstruct_sart",
    "SIRTParameters",
    "run_sirt",
    "reconstruct_sirt",
]
