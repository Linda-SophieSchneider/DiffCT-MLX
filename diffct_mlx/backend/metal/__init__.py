"""Metal (MLX) implementation of the low-level CT operators.

Vendored from the DiffCT-MLX ``main`` branch (the original Apple-Silicon
package): custom Metal kernels + ``mx.custom_function`` projectors with VJP
support, plus the MLX trajectory generators. The unified backend adapter
(:mod:`diffct_mlx.backend._mlx`) re-exports these under the package-wide
functional operator API.

Only importable where ``mlx`` is installed; the Metal kernels themselves
require an Apple-Silicon GPU to run.
"""

from . import projectors, geometry  # noqa: F401
