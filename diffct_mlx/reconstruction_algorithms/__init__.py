"""Reconstruction algorithms built on the unified projector API.

Ported from the MLX implementation to run on any backend: analytic (FBP / FDK),
iterative (SART / SIRT), regularized POCS variants (TV / ASD / AwTV) and DART.
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
from .cases import (
    MeasuredConeDataConfig,
    NpyProjectionsConfig,
    ReconstructionCase,
    build_cone_3d_case,
    build_fan_2d_case,
    build_measured_cone_3d_case,
    build_npy_cone_3d_case,
    reconstruct_case_fdk,
    build_parallel_2d_case,
    make_cone_3d_operators,
    make_fan_2d_operators,
    make_parallel_2d_operators,
)
from .fbp import FBPParameters, reconstruct_fbp, run_fbp
from .fdk import FDKParameters, reconstruct_fdk, run_fdk
from .sart import SARTParameters, reconstruct_sart, run_sart
from .sirt import SIRTParameters, reconstruct_sirt, run_sirt
from .tv_pocs import TVPOCSParameters, TV_POCS_Parameter, reconstruct_tv_pocs, run_tv_pocs
from .asd_pocs import ASDPOCSParameters, ASD_POCS_Parameter, reconstruct_asd_pocs, run_asd_pocs
from .awtv_pocs import AwTVPOCSParameters, AwTV_POCS_Parameter, reconstruct_awtv_pocs, run_awtv_pocs
from .dart import DARTParameters, reconstruct_dart, run_dart
from ._solver import (
    IterativeReconstructor,
    cgls,
    get_algorithm,
    init_iterate,
    inner,
    landweber,
    list_algorithms,
    reconstruct,
    register_algorithm,
)
from .solvers import (
    dls,
    ls,
    make_subsets,
    mlem,
    mltr,
    osem,
    pcg,
    power_iteration,
    rdls,
    rls,
    rwls,
    wls,
)


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
    # cases / operator builders
    "MeasuredConeDataConfig",
    "NpyProjectionsConfig",
    "ReconstructionCase",
    "build_parallel_2d_case",
    "build_fan_2d_case",
    "build_cone_3d_case",
    "build_measured_cone_3d_case",
    "build_npy_cone_3d_case",
    "reconstruct_case_fdk",
    "make_parallel_2d_operators",
    "make_fan_2d_operators",
    "make_cone_3d_operators",
    # iterative
    "SARTParameters",
    "run_sart",
    "reconstruct_sart",
    "SIRTParameters",
    "run_sirt",
    "reconstruct_sirt",
    # regularized POCS
    "TVPOCSParameters",
    "TV_POCS_Parameter",
    "run_tv_pocs",
    "reconstruct_tv_pocs",
    "ASDPOCSParameters",
    "ASD_POCS_Parameter",
    "run_asd_pocs",
    "reconstruct_asd_pocs",
    "AwTVPOCSParameters",
    "AwTV_POCS_Parameter",
    "run_awtv_pocs",
    "reconstruct_awtv_pocs",
    # discrete
    "DARTParameters",
    "run_dart",
    "reconstruct_dart",
    # solver framework + registry
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
    "reconstruct",
    "inner",
    "init_iterate",
    "IterativeReconstructor",
    "landweber",
    "cgls",
    # statistical / least-squares / CG solvers
    "pcg",
    "ls",
    "wls",
    "rls",
    "rwls",
    "dls",
    "rdls",
    "mlem",
    "osem",
    "mltr",
    "power_iteration",
    "make_subsets",
]
