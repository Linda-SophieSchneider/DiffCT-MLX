"""Out-of-core + multi-GPU orchestration for large-volume cone-beam CT.

Sits above the single-GPU autograd projectors: partitions host-resident volumes
/ sinograms into chunks sized to fit GPU memory, dispatches chunks across GPUs,
and stitches the partial results on the host. Inference-only (no autograd across
chunk boundaries). See :mod:`diffct_mlx.orchestration.out_of_core`.
"""

from .out_of_core import (
    ConeGeom,
    open_memmap,
    set_out_of_core_dir,
    get_out_of_core_dir,
    ramp_filter_memmap,
    chunked_cone_forward,
    chunked_cone_backward,
    chunked_cone_fdk,
    chunked_sirt,
    chunked_os_sart,
    mgpu_cone_forward,
    mgpu_cone_backproject,
    mgpu_sirt,
    row_range_for_zslab,
    slice_range_for_vband,
)

__all__ = [
    "ConeGeom",
    "open_memmap",
    "set_out_of_core_dir",
    "get_out_of_core_dir",
    "ramp_filter_memmap",
    "chunked_cone_forward",
    "chunked_cone_backward",
    "chunked_cone_fdk",
    "chunked_sirt",
    "chunked_os_sart",
    "mgpu_cone_forward",
    "mgpu_cone_backproject",
    "mgpu_sirt",
    "row_range_for_zslab",
    "slice_range_for_vband",
]
