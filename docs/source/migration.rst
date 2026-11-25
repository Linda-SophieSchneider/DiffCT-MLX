Migrating from 2.x to 3.0
=========================

DiffCT-MLX 3.0 changes two defaults and replaces the Torch engine. Code that
runs on 2.x still runs on 3.0, but its numbers change. They move by half a
detector pixel in position and by the factor ``voxel_spacing`` in amplitude.

What changed
------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - 2.x
     - 3.0
   * - Detector pixel ``i``
     - ``det_center + (i - N/2)·du``
     - ``det_center + (i - (N-1)/2)·du``
   * - Line integrals
     - voxel units
     - physical (µ · length of the geometry)
   * - Iterative result of physical data
     - µ per voxel
     - µ in 1/mm
   * - Torch engine
     - top-level ``diffct`` (dev line 1.3)
     - ``diffct_mlx.engine`` (upstream diffct 2.5.2)
   * - Python
     - ≥ 3.9
     - ≥ 3.10

In 3.0 ``det_center`` is the geometric centre of the panel on both backends,
the convention of upstream diffct ≥ 2.0. The quantitative FDK path already
returned 1/mm in 2.x and does not change.

Reproducing 2.x results
-----------------------

The operator factories and case builders take the old conventions as
keywords:

.. code-block:: python

   import diffct_mlx as dct

   A = dct.make_cone_3d_operator(
       src, det_c, det_u, det_v,
       volume_shape=(D, H, W), detector_shape=(nu, nv),
       du=du, dv=dv, voxel_spacing=vs,
       detector_convention="legacy", line_integrals="voxel",
   )

``make_parallel_2d_operator``, ``make_fan_2d_operator`` and the
``make_*_operators`` builders accept the same two keywords. The 2.3 name
``detector_convention="kernel"`` is an alias of ``"legacy"``.

For direct projector calls and for FBP/FDK, convert the detector centre:

.. code-block:: python

   det_c_legacy = dct.kernel_detector_center(
       det_c, det_u, du, det_v, dv, convention="legacy")
   sino_2x_grid = dct.cone_forward(vol, src, det_c_legacy, det_u, det_v,
                                   nu, nv, du, dv, vs)
   sino_2x = sino_2x_grid / vs          # voxel units, as 2.x returned

For a parallel beam pass ``det_origin`` in place of ``det_center``.

Projection stacks
-----------------

``.npy`` stacks written by 2.x projectors are in voxel units. Load them with
``NpyProjectionsConfig(projection_units="voxel")`` and they are converted to
physical units. Measured ``-log`` data is physical already and needs no
change.

Imports of the engine
---------------------

The wheel no longer installs a top-level ``diffct`` package, so it no longer
shadows the PyPI package ``diffct``. Replace

.. code-block:: python

   from diffct import ConeProjectorFunction, circular_trajectory_3d

with

.. code-block:: python

   from diffct_mlx.engine import ConeProjectorFunction, circular_trajectory_3d

``diffct.differentiable`` is gone. Geometry generators in the engine default
to ``device="cuda"`` as upstream does. The unified ``diffct_mlx`` generators
choose the device for you.

Behaviour fixes you will notice
-------------------------------

- **Footprint amplitude.** The footprint projectors are mass-conserving
  since 2.2.0, and 3.0 keeps this on the new engine. Coming from 2.1 or the
  dev0 line, expect a change. Those weighted voxels with ``1/M(x)²``, which
  biased off-centre voxels by depth, by about 7 % for a typical micro-CT
  geometry.
- **Metal cone geometry gradient.** It now matches the engine's exact
  gradient. The 2.x kernel differentiated a trilinear volume and was off by
  about 100 %. Pose and trajectory optimization on Apple Silicon behaves
  differently, and correctly.
- **Rays clipped to the source-detector segment.** A source or detector
  inside the volume no longer gives the full-line integral.

Staying on 2.x
--------------

Pin ``diffct-mlx<3`` to keep the old behaviour unchanged. 2.3.0 is the last
2.x release.
