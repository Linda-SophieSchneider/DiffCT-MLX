Getting Started
===============

This guide walks you through installing **DiffCT-MLX** and running your first
reconstruction with the unified ``diffct_mlx`` package.

Prerequisites
-------------

**One of:**

- an NVIDIA CUDA GPU (compute capability 6.0+; CUDA Toolkit 11+), or
- an Apple-Silicon Mac (M-series).

**Software:** Python 3.10 or later.

Installation
------------

Install the release from PyPI with the extra matching your platform:

.. code-block:: bash

   pip install "diffct-mlx[cuda]"   # NVIDIA GPUs: Torch + numba-CUDA
   # or
   pip install "diffct-mlx[mlx]"    # Apple Silicon: MLX

For an editable development install, clone the repository:

.. code-block:: bash

   git clone https://github.com/Linda-SophieSchneider/DiffCT-MLX.git
   cd DiffCT-MLX
   pip install -e ".[cuda]"     # NVIDIA GPUs: Torch + numba-CUDA
   # or
   pip install -e ".[mlx]"      # Apple Silicon: MLX

On CUDA systems install a CUDA-enabled PyTorch build first (see
`pytorch.org <https://pytorch.org/get-started/locally/>`_) and the
``numba-cuda`` package matching your toolkit (``pip install numba-cuda[cu12]``
or ``[cu13]``). The optional ``[zarr]`` extra enables the compressed
out-of-core storage backend.

**Verify the installation:**

.. code-block:: python

   import diffct_mlx as dct

   print(dct.backend)        # 'torch' on CUDA systems, 'mlx' on Apple Silicon
   print(dct.__version__)

The backend is chosen automatically at import; set the environment variable
``DIFFCT_BACKEND=torch`` or ``DIFFCT_BACKEND=mlx`` to force one.

Quick Start Example
-------------------

The same code runs on either backend — no device management required:

.. code-block:: python

   import math
   import diffct_mlx as dct

   # A 2D Shepp-Logan phantom and a 360-view parallel-beam operator
   n = 256
   phantom = dct.shepp_logan_2d((n, n))

   ray_dir, det_origin, det_u = dct.circular_trajectory_2d_parallel(360)
   A = dct.make_parallel_2d_operator(
       ray_dir, det_origin, det_u,
       image_shape=(n, n), num_detectors=n, projector_mode="siddon",
   )

   sinogram = A @ phantom            # differentiable forward projection
   backproj = A.T @ sinogram         # differentiable adjoint

   # Analytic FBP reconstruction
   params = dct.FBPParameters(normalization_scale=math.pi / (2 * 360))
   reco = dct.reconstruct_fbp(sinogram, lambda s: A.T @ s, params)

   # Or hand the operator to any registered solver
   reco_cgls = dct.reconstruct("cgls", A, sinogram, iterations=30)

Where to go next
----------------

- The **cases API** (``build_parallel_2d_case``, ``build_cone_3d_case``,
  ``build_measured_cone_3d_case``, ...) bundles simulated or measured data
  with matched forward/backprojectors. For cone data,
  ``dct.reconstruct_case_fdk(case)`` selects the quantitative physical FDK
  path when available and returns attenuation in ``1/mm``.
- ``dct.simulate_scan`` produces realistic polychromatic, noisy data from a
  phantom in physical line-integral units; ``diffct_mlx.physics`` corrects
  real measurements.
- ``diffct_mlx.orchestration`` reconstructs volumes larger than GPU memory
  (and larger than host RAM) across multiple GPUs.
- Explore the :doc:`examples` and the :doc:`api` reference.
