"""diffct_mlx — unified, auto-backend differentiable CT.

A single API for differentiable CT forward/backward projection and
reconstruction that runs unchanged on two compute backends:

* **MLX / Metal** on Apple Silicon,
* **Torch / numba-CUDA** on NVIDIA GPUs.

The backend is selected automatically at import (override with the
``DIFFCT_BACKEND`` environment variable). Import the package and call the same
functions regardless of platform::

    import diffct_mlx as dct
    print(dct.backend)            # 'torch' or 'mlx'
    sino = dct.cone_forward(vol, src_pos, det_center, det_u, det_v, ...)

The Torch/CUDA engine is adapted from the original ``diffct`` (see
ATTRIBUTION.md); the MLX/Metal engine is the original DiffCT-MLX
implementation. The full reconstruction stack — analytic (FBP/FDK), iterative
(SART/SIRT/POCS/DART), the operator/functional/solver framework, physics and
simulation — is written once against the backend-neutral ``xp`` layer.
"""

from .backend import NAME as backend

from .projectors import (
    parallel_forward,
    parallel_backward,
    parallel_forward_footprint,
    parallel_backward_footprint,
    fan_forward,
    fan_backward,
    fan_forward_footprint,
    fan_backward_footprint,
    cone_forward,
    cone_backward,
    cone_forward_footprint,
    cone_backward_footprint,
)
from . import geometry
from .geometry import __all__ as _geometry_all

# Re-export trajectory generators at top level (mirrors the MLX API).
from .geometry import *  # noqa: F401,F403

from .reconstruction_algorithms import (
    ReconstructionParameters,
    RegularizationParameters,
    Reconstruction_Parameter,
    Regularisation_Parameter,
    BackProjector,
    ForwardProjector,
    FBPParameters,
    run_fbp,
    reconstruct_fbp,
    FDKParameters,
    run_fdk,
    reconstruct_fdk,
    SARTParameters,
    run_sart,
    reconstruct_sart,
    SIRTParameters,
    run_sirt,
    reconstruct_sirt,
    TVPOCSParameters,
    TV_POCS_Parameter,
    run_tv_pocs,
    reconstruct_tv_pocs,
    ASDPOCSParameters,
    ASD_POCS_Parameter,
    run_asd_pocs,
    reconstruct_asd_pocs,
    AwTVPOCSParameters,
    AwTV_POCS_Parameter,
    run_awtv_pocs,
    reconstruct_awtv_pocs,
    DARTParameters,
    run_dart,
    reconstruct_dart,
    MeasuredConeDataConfig,
    NpyProjectionsConfig,
    ReconstructionCase,
    build_parallel_2d_case,
    build_fan_2d_case,
    build_cone_3d_case,
    build_measured_cone_3d_case,
    build_npy_cone_3d_case,
    make_parallel_2d_operators,
    make_fan_2d_operators,
    make_cone_3d_operators,
    register_algorithm,
    get_algorithm,
    list_algorithms,
    reconstruct,
    IterativeReconstructor,
    landweber,
    cgls,
    pcg,
    power_iteration,
    ls,
    wls,
    rls,
    rwls,
    dls,
    rdls,
    mlem,
    osem,
    mltr,
)
from . import operators as operators
from . import functionals as functionals
from .operators import (
    LinearOperator,
    IdentityOperator,
    DiagonalOperator,
    FunctionOperator,
    ProjectionOperator,
    make_parallel_2d_operator,
    make_fan_2d_operator,
    make_cone_3d_operator,
)
from .functionals import (
    Functional,
    SquaredL2,
    Tikhonov,
    L1Norm,
    LpNorm,
    Huber,
    TotalVariation,
    AdaptiveWeightedTV,
    NonNegativity,
    Box,
    soft_threshold,
)
from . import filters as filters
from .filters import (
    Bilateral,
    Guided,
    Median,
    HistogramSparsity,
    Azimuthal,
    DictionarySparsity,
    RegularizerSequence,
    box_filter,
    bilateral_filter,
    guided_filter,
    median_filter,
    histogram_sparsity,
    azimuthal_smooth,
    dictionary_denoise,
)
from . import physics as physics
from .physics import Spectrum, simulate_scan
from . import rebinning as rebinning
from .rebinning import (
    apply_parker_weighting,
    apply_offset_weighting,
    extend_truncation,
    fan_to_parallel,
    curved_to_flat,
    flat_to_curved,
)
from .phantoms import shepp_logan_2d, shepp_logan_3d, Ellipsoid, Phantom, shepp_logan_phantom
from . import calibration as calibration
from .calibration import (
    estimate_center_of_rotation,
    refine_center_by_sharpness,
    apply_center_offset,
)
from .real_measured_data_helper import (
    apply_detector_array_convention,
    apply_detector_geometry_convention,
    apply_upper_left_detector_transform,
    auto_voxel_spacing_from_detector,
    build_upper_left_detector_transform,
    diagnose_cone_geometry,
    estimate_cone_isocenter,
    load_tiff_projections,
    mu_to_hu,
    mu_to_uint16,
    normalize_volume,
    resize_volume_to_shape,
    shift_detector_center,
    transform_detector_offsets,
)
from .regularizers import (
    l2_regularizer,
    normalize_reconstruction_volume,
    tv_regularizer,
    tv_regularizer_3d,
    awtv_regularizer,
    tv_pocs,
    asd_pocs,
    awtv_pocs,
)
from .tv_gradients import (
    tv_gradient,
    weight_d_volume,
    awtv_gradient,
)

__version__ = "2.0.0.dev0"

__all__ = [
    "backend",
    "geometry",
    # Projector functions
    "parallel_forward",
    "parallel_backward",
    "parallel_forward_footprint",
    "parallel_backward_footprint",
    "fan_forward",
    "fan_backward",
    "fan_forward_footprint",
    "fan_backward_footprint",
    "cone_forward",
    "cone_backward",
    "cone_forward_footprint",
    "cone_backward_footprint",
    *_geometry_all,
    # Reconstruction core
    "ReconstructionParameters",
    "RegularizationParameters",
    "Reconstruction_Parameter",
    "Regularisation_Parameter",
    "BackProjector",
    "ForwardProjector",
    # Analytic reconstruction
    "FBPParameters",
    "run_fbp",
    "reconstruct_fbp",
    "FDKParameters",
    "run_fdk",
    "reconstruct_fdk",
    # Iterative reconstruction
    "SARTParameters",
    "run_sart",
    "reconstruct_sart",
    "SIRTParameters",
    "run_sirt",
    "reconstruct_sirt",
    # Regularized POCS reconstruction
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
    # Discrete reconstruction
    "DARTParameters",
    "run_dart",
    "reconstruct_dart",
    # Cases / operator builders
    "MeasuredConeDataConfig",
    "NpyProjectionsConfig",
    "ReconstructionCase",
    "build_parallel_2d_case",
    "build_fan_2d_case",
    "build_cone_3d_case",
    "build_measured_cone_3d_case",
    "build_npy_cone_3d_case",
    "make_parallel_2d_operators",
    "make_fan_2d_operators",
    "make_cone_3d_operators",
    # Operator algebra (composable, differentiable)
    "operators",
    "LinearOperator",
    "IdentityOperator",
    "DiagonalOperator",
    "FunctionOperator",
    "ProjectionOperator",
    "make_parallel_2d_operator",
    "make_fan_2d_operator",
    "make_cone_3d_operator",
    # Functionals (objectives / regularizers / constraints)
    "functionals",
    "Functional",
    "SquaredL2",
    "Tikhonov",
    "L1Norm",
    "LpNorm",
    "Huber",
    "TotalVariation",
    "AdaptiveWeightedTV",
    "NonNegativity",
    "Box",
    "soft_threshold",
    # Denoisers / Plug-and-Play regularizer sequence
    "filters",
    "Bilateral",
    "Guided",
    "Median",
    "HistogramSparsity",
    "Azimuthal",
    "DictionarySparsity",
    "RegularizerSequence",
    "box_filter",
    "bilateral_filter",
    "guided_filter",
    "median_filter",
    "histogram_sparsity",
    "azimuthal_smooth",
    "dictionary_denoise",
    # Solver framework + registry
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
    "reconstruct",
    "IterativeReconstructor",
    "landweber",
    "cgls",
    "pcg",
    "power_iteration",
    "ls",
    "wls",
    "rls",
    "rwls",
    "dls",
    "rdls",
    "mlem",
    "osem",
    "mltr",
    # Physics / preprocessing / simulation
    "physics",
    "Spectrum",
    "simulate_scan",
    # Geometry: sinogram-domain weighting + rebinning
    "rebinning",
    "apply_parker_weighting",
    "apply_offset_weighting",
    "extend_truncation",
    "fan_to_parallel",
    "curved_to_flat",
    "flat_to_curved",
    # Phantoms + analytic phantom engine
    "shepp_logan_2d",
    "shepp_logan_3d",
    "Ellipsoid",
    "Phantom",
    "shepp_logan_phantom",
    # Geometric self-calibration
    "calibration",
    "estimate_center_of_rotation",
    "refine_center_by_sharpness",
    "apply_center_offset",
    # Measured-data helpers
    "apply_detector_array_convention",
    "apply_detector_geometry_convention",
    "apply_upper_left_detector_transform",
    "auto_voxel_spacing_from_detector",
    "build_upper_left_detector_transform",
    "diagnose_cone_geometry",
    "estimate_cone_isocenter",
    "load_tiff_projections",
    "mu_to_hu",
    "mu_to_uint16",
    "normalize_volume",
    "resize_volume_to_shape",
    "shift_detector_center",
    "transform_detector_offsets",
    # Regularizers
    "l2_regularizer",
    "normalize_reconstruction_volume",
    "tv_regularizer",
    "tv_regularizer_3d",
    "awtv_regularizer",
    "tv_pocs",
    "asd_pocs",
    "awtv_pocs",
    # TV gradients
    "tv_gradient",
    "weight_d_volume",
    "awtv_gradient",
]
