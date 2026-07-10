Examples
========

This section demonstrates practical applications of the low-level ``diffct`` engine (Torch/CUDA) for various computed tomography (CT) reconstruction tasks; the same operations are available portably through the unified ``diffct_mlx`` API (see :doc:`getting_started`). Each example provides comprehensive mathematical background, implementation details, and complete working code. The supporting Python scripts live under ``examples/circular_trajectory`` and ``examples/non_circular_trajectory`` in the repository.

The examples are organized into two main categories:

**Analytical Reconstruction Methods**
- Filtered backprojection (FBP) algorithms for direct reconstruction
- Standard analytical approaches used in clinical and research settings

**Iterative Reconstruction Methods**  
- Gradient-based optimization approaches using differentiable operators
- Advanced reconstruction techniques with regularization capabilities

.. toctree::
   :maxdepth: 1
   :caption: Analytical Reconstruction

   fbp_parallel_example
   fbp_fan_example
   fdk_cone_example

.. toctree::
   :maxdepth: 1
   :caption: Iterative Reconstruction

   iterative_reco_parallel_example
   iterative_reco_fan_example
   iterative_reco_cone_example
