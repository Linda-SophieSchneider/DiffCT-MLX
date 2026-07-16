DiffCT-MLX: Differentiable CT — One API, Two Backends
=====================================================

**DiffCT-MLX** is a differentiable computed-tomography library with a single
user-facing package, ``diffct_mlx``, that selects its compute engine at import
time:

* **Torch / numba-CUDA** on NVIDIA GPUs (wrapping the vendored ``diffct``
  engine, adapted from `sypsyp97/diffct <https://github.com/sypsyp97/diffct>`_),
* **MLX / Metal** on Apple Silicon.

The same script — projectors, geometry, reconstruction, physics — runs
unchanged on both platforms. Override the automatic choice with the
``DIFFCT_BACKEND`` environment variable (``torch`` / ``mlx``).

**Key Features**
----------------
- **Fully differentiable:** end-to-end gradient propagation through forward
  and backprojection on both backends (``torch.autograd`` /
  ``mx.custom_function`` VJPs).
- **Two projector families:** thin-ray Siddon and separable-footprint
  forward/adjoint pairs for parallel-, fan- and cone-beam geometries,
  including a sparse cone backprojection for region-of-interest gradients.
- **Arbitrary trajectories:** per-view source/detector arrays — circular,
  spiral (helical), saddle, sinusoidal, laminography, or fully custom.
- **Complete reconstruction stack:** quantitative FDK in physical attenuation
  units for synthetic and measured cone-beam data, amplitude-calibrated FBP,
  SART/SIRT/normalized-SART, TV-/ASD-/AwTV-POCS, DART, plus a solver registry
  (``cgls``, ``wls``/``rwls``, ``pcg``, ``mlem``/``osem``, ``mltr``, ...)
  over a composable ``LinearOperator`` algebra with functionals,
  regularizers and Plug-and-Play denoisers.
- **Physics & simulation:** explicit voxel/physical line-integral conventions,
  GPU-native preprocessing (flat field, rings, bad
  pixels, beam hardening, deblur, scatter, MAR) and forward simulation with an
  embedded, physically-validated X-ray spectrum / material-attenuation
  library (40–225 kVp).
- **Out-of-core + multi-GPU (CUDA):** ``diffct_mlx.orchestration`` streams
  TB-scale volumes through chunked, conveyor-pipelined FDK/SIRT/OS-SART with
  automatic RAM/disk (memmap or zarr) spill.

**Two importable packages**
---------------------------
- ``diffct_mlx`` — the unified, auto-backend API. **Import this.**
- ``diffct`` — the low-level Torch/numba-CUDA engine (autograd Functions,
  CUDA kernels, analytical FBP/FDK helpers). Useful for advanced integrations
  on NVIDIA hardware; documented in the API reference.

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   api
   examples

Citation
--------

If you use this library in your research, please cite:

.. code-block:: bibtex

   @article{202605.1446,
     doi       = {10.20944/preprints202605.1446.v1},
     url       = {https://doi.org/10.20944/preprints202605.1446.v1},
     year      = 2026,
     month     = {May},
     publisher = {Preprints},
     author    = {Yipeng Sun and Linda-Sophie Schneider and Chengze Ye and Andreas Maier},
     title     = {diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories},
     journal   = {Preprints}
   }

License
-------

Apache 2.0 — see the `LICENSE
<https://github.com/Linda-SophieSchneider/DiffCT-MLX/blob/main/LICENSE>`_ file
and `ATTRIBUTION.md
<https://github.com/Linda-SophieSchneider/DiffCT-MLX/blob/main/ATTRIBUTION.md>`_
for provenance.
