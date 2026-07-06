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

This branch is adapted from the original ``diffct`` (see ATTRIBUTION.md). The
high-level reconstruction layer is being ported incrementally; see the package
roadmap for the current parity status.
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

__version__ = "0.1.0.dev0"

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
