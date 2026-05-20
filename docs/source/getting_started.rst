Getting Started
===============

``diffct-mlx`` targets Apple Silicon and provides differentiable projection,
backprojection, geometry helpers, and reconstruction algorithms built on MLX.

Requirements
------------

- Python 3.10 or newer
- Apple Silicon hardware for actual MLX execution
- Dependencies from ``requirements.txt``

Installation
------------

Install the package in a virtual environment:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Quick Start
-----------

The repository ships runnable examples under ``examples/`` for:

- 2D parallel-beam FBP
- 2D fan-beam FBP
- 3D cone-beam FDK
- Iterative reconstruction with non-circular trajectories

Example commands:

.. code-block:: bash

   python examples/circular_trajectory/fbp_parallel.py
   python examples/circular_trajectory/fbp_fan.py
   python examples/circular_trajectory/fdk_cone.py

Project Layout
--------------

- ``diffct_mlx/``: package code
- ``examples/``: runnable geometry and reconstruction demos
- ``diagnose_scripts/``: debugging and diagnostics helpers
- ``tests/``: regression tests

Citation
--------

If you use this library in research, use the citation from ``README.md``.
