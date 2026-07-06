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
]
