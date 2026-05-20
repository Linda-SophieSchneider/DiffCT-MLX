API Reference
=============

This page summarizes the current package surface on ``main``.

Top-Level Package
-----------------

The public entry point is ``diffct_mlx``.

Core areas exposed there include:

- Projectors and backprojectors for parallel, fan, and cone geometries
- Geometry generators and JSON geometry loading
- Phantom helpers
- Regularizers and TV gradient helpers
- Reconstruction case builders
- Reconstruction algorithms such as SART, TV-POCS, ASD-POCS, AwTV-POCS, FBP, and FDK
- Measured-data helpers for cone-beam workflows

Main Package Files
------------------

- ``diffct_mlx/projectors.py``
- ``diffct_mlx/geometry.py``
- ``diffct_mlx/phantoms/``
- ``diffct_mlx/regularizers.py``
- ``diffct_mlx/tv_gradients.py``
- ``diffct_mlx/real_measured_data_helper.py``
- ``diffct_mlx/reconstruction_algorithms/``

For the exact current exports, inspect:

.. literalinclude:: ../../diffct_mlx/__init__.py
   :language: python
   :linenos:
   :caption: Current top-level exports in ``diffct_mlx/__init__.py``
