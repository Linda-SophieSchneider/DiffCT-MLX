"""Composable, differentiable linear operators for CT.

This module gives the reconstruction stack a small, backend-neutral
``LinearOperator`` algebra so that

* **differentiable pipelines** read like maths — ``sino = A @ volume`` and
  ``vol0 = A.T @ sino`` both flow gradients (the projectors underneath are
  autograd :class:`~torch.autograd.Function`\\s), and
* **solvers** (CG / MLEM / OSEM / WLS / RWLS / …) are written once against an
  abstract ``A`` with a matched adjoint ``A.T`` instead of against a specific
  beam geometry.

Operators compose with the natural Python operators::

    A @ B          # composition  (apply B, then A)
    A @ x          # application   (forward-project an array)
    A.T @ y        # adjoint application (back-project)
    2.0 * A        # scaling
    A + B          # sum (same domain and range)

``ProjectionOperator`` is the concrete CT operator; build one from a geometry
with :func:`make_parallel_2d_operator`, :func:`make_fan_2d_operator`, or
:func:`make_cone_3d_operator`. It also supports ``.subset(view_indices)`` for
ordered-subset solvers.

The existing functional builders (``make_*_operators``) are unchanged; this is
an additive object-oriented layer on top of the same projectors.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .backend import active as _b
from .projectors import (
    cone_backward,
    cone_backward_footprint,
    cone_forward,
    cone_forward_footprint,
    fan_backward,
    fan_backward_footprint,
    fan_forward,
    fan_forward_footprint,
    parallel_backward,
    parallel_backward_footprint,
    parallel_forward,
    parallel_forward_footprint,
)

xp = _b.xp

Array = Any

__all__ = [
    "LinearOperator",
    "CompositeOperator",
    "SumOperator",
    "ScaledOperator",
    "AdjointOperator",
    "IdentityOperator",
    "DiagonalOperator",
    "FunctionOperator",
    "ProjectionOperator",
    "make_parallel_2d_operator",
    "make_fan_2d_operator",
    "make_cone_3d_operator",
]


# ---------------------------------------------------------------------------
# Operator algebra
# ---------------------------------------------------------------------------
class LinearOperator:
    """Abstract linear map with a matched adjoint.

    Subclasses implement :meth:`_forward` and :meth:`_adjoint`. ``domain_shape``
    / ``range_shape`` are advisory (may be ``None``); they are used only for
    introspection and by solvers that allocate an initial iterate.
    """

    domain_shape: tuple[int, ...] | None = None
    range_shape: tuple[int, ...] | None = None

    # -- to implement in subclasses ----------------------------------------
    def _forward(self, x: Array) -> Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def _adjoint(self, y: Array) -> Array:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- public API --------------------------------------------------------
    def forward(self, x: Array) -> Array:
        """Apply the operator (forward map). Accepts any array-like."""
        return self._forward(_b.as_array(x, dtype=_b.float32))

    def adjoint(self, y: Array) -> Array:
        """Apply the adjoint (transpose) operator. Accepts any array-like."""
        return self._adjoint(_b.as_array(y, dtype=_b.float32))

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    @property
    def T(self) -> "LinearOperator":
        """The adjoint operator ``A.T`` (``A.T.forward == A.adjoint``)."""
        return AdjointOperator(self)

    # Alias for readers who prefer a spelled-out name.
    @property
    def adjoint_operator(self) -> "LinearOperator":
        return AdjointOperator(self)

    # -- algebra -----------------------------------------------------------
    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, LinearOperator):
            return CompositeOperator(self, other)
        # Applying to data: accept any array-like (numpy float64 / int / …).
        return self._forward(_b.as_array(other, dtype=_b.float32))

    def __mul__(self, scalar: float) -> "LinearOperator":
        return ScaledOperator(self, scalar)

    __rmul__ = __mul__

    def __add__(self, other: "LinearOperator") -> "LinearOperator":
        return SumOperator(self, other)

    def __sub__(self, other: "LinearOperator") -> "LinearOperator":
        return SumOperator(self, ScaledOperator(other, -1.0))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(domain={self.domain_shape}, range={self.range_shape})"


class AdjointOperator(LinearOperator):
    """The adjoint of another operator (lazy transpose)."""

    def __init__(self, op: LinearOperator):
        self._op = op
        self.domain_shape = op.range_shape
        self.range_shape = op.domain_shape

    def _forward(self, x: Array) -> Array:
        return self._op._adjoint(x)

    def _adjoint(self, y: Array) -> Array:
        return self._op._forward(y)

    @property
    def T(self) -> LinearOperator:
        return self._op


class CompositeOperator(LinearOperator):
    """Operator composition ``A @ B``: ``x -> A(B(x))``."""

    def __init__(self, first: LinearOperator, second: LinearOperator):
        # ``first @ second`` applies ``second`` then ``first``.
        self._first = first
        self._second = second
        self.domain_shape = second.domain_shape
        self.range_shape = first.range_shape

    def _forward(self, x: Array) -> Array:
        return self._first._forward(self._second._forward(x))

    def _adjoint(self, y: Array) -> Array:
        return self._second._adjoint(self._first._adjoint(y))


class SumOperator(LinearOperator):
    """Operator sum ``A + B``: ``x -> A(x) + B(x)``."""

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self._a = a
        self._b = b
        self.domain_shape = a.domain_shape or b.domain_shape
        self.range_shape = a.range_shape or b.range_shape

    def _forward(self, x: Array) -> Array:
        return self._a._forward(x) + self._b._forward(x)

    def _adjoint(self, y: Array) -> Array:
        return self._a._adjoint(y) + self._b._adjoint(y)


class ScaledOperator(LinearOperator):
    """Scalar-scaled operator ``c * A`` (real scalar)."""

    def __init__(self, op: LinearOperator, scalar: float):
        self._op = op
        self._scalar = float(scalar)
        self.domain_shape = op.domain_shape
        self.range_shape = op.range_shape

    def _forward(self, x: Array) -> Array:
        return self._scalar * self._op._forward(x)

    def _adjoint(self, y: Array) -> Array:
        return self._scalar * self._op._adjoint(y)


class IdentityOperator(LinearOperator):
    """Identity map."""

    def __init__(self, shape: tuple[int, ...] | None = None):
        self.domain_shape = shape
        self.range_shape = shape

    def _forward(self, x: Array) -> Array:
        return x

    def _adjoint(self, y: Array) -> Array:
        return y


class DiagonalOperator(LinearOperator):
    """Elementwise scaling by a (real) array ``d``: ``x -> d * x`` (self-adjoint)."""

    def __init__(self, diagonal: Array):
        self._d = xp.array(diagonal, dtype=_b.float32)
        self.domain_shape = tuple(int(s) for s in self._d.shape)
        self.range_shape = self.domain_shape

    def _forward(self, x: Array) -> Array:
        return self._d * x

    def _adjoint(self, y: Array) -> Array:
        return self._d * y


class FunctionOperator(LinearOperator):
    """Wrap an explicit ``forward_fn`` / ``adjoint_fn`` pair as an operator.

    The two callables must form a matched adjoint pair for solver correctness.
    Handy for user-defined maps (sub-sampling, masking, custom transforms).
    """

    def __init__(
        self,
        forward_fn: Callable[[Array], Array],
        adjoint_fn: Callable[[Array], Array],
        *,
        domain_shape: tuple[int, ...] | None = None,
        range_shape: tuple[int, ...] | None = None,
    ):
        self._forward_fn = forward_fn
        self._adjoint_fn = adjoint_fn
        self.domain_shape = domain_shape
        self.range_shape = range_shape

    def _forward(self, x: Array) -> Array:
        return self._forward_fn(x)

    def _adjoint(self, y: Array) -> Array:
        return self._adjoint_fn(y)


# ---------------------------------------------------------------------------
# Projection operator (the concrete CT map)
# ---------------------------------------------------------------------------
class ProjectionOperator(LinearOperator):
    """All-view CT projection ``A`` with a matched back-projection ``A.T``.

    ``forward`` maps a volume/image to the full multi-view sinogram; ``adjoint``
    back-projects a sinogram to volume/image space. Both call the differentiable
    projectors, so operators built here compose into autograd pipelines.

    Use :meth:`subset` to obtain the operator restricted to a subset of views
    (for ordered-subset solvers such as OSEM / OS-SART).
    """

    def __init__(
        self,
        forward_fn: Callable[[Array, Any], Array],
        adjoint_fn: Callable[[Array, Any], Array],
        *,
        domain_shape: tuple[int, ...],
        range_shape: tuple[int, ...],
        n_views: int,
        beam: str | None = None,
        views: Any = None,
    ):
        self._forward_fn = forward_fn
        self._adjoint_fn = adjoint_fn
        self.domain_shape = tuple(int(s) for s in domain_shape)
        self.range_shape = tuple(int(s) for s in range_shape)
        self.n_views = int(n_views)
        self.beam = beam
        self._views = views

    def _forward(self, x: Array) -> Array:
        return self._forward_fn(x, self._views)

    def _adjoint(self, y: Array) -> Array:
        return self._adjoint_fn(y, self._views)

    def subset(self, view_indices: Sequence[int] | Any) -> "ProjectionOperator":
        """Return the operator restricted to ``view_indices`` (a slice/list/array)."""
        if isinstance(view_indices, slice):
            n = len(range(*view_indices.indices(self.n_views)))
        else:
            try:
                n = len(view_indices)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError(
                    "view_indices must be a slice or a sized sequence/array of "
                    f"view indices, got {type(view_indices).__name__}"
                ) from None
        range_shape = (int(n),) + tuple(self.range_shape[1:])
        return ProjectionOperator(
            self._forward_fn,
            self._adjoint_fn,
            domain_shape=self.domain_shape,
            range_shape=range_shape,
            n_views=int(n),
            beam=self.beam,
            views=view_indices,
        )


def _select_pair(beam: str, projector_mode: str):
    key = str(projector_mode).strip().lower()
    table = {
        "parallel": {
            "siddon": (parallel_forward, parallel_backward),
            "footprint": (parallel_forward_footprint, parallel_backward_footprint),
        },
        "fan": {
            "siddon": (fan_forward, fan_backward),
            "footprint": (fan_forward_footprint, fan_backward_footprint),
        },
        "cone": {
            "siddon": (cone_forward, cone_backward),
            "footprint": (cone_forward_footprint, cone_backward_footprint),
        },
    }
    if key not in table[beam]:
        raise ValueError(f"Unknown {beam} projector_mode: {projector_mode!r}")
    return table[beam][key]


def _index(array: Array, views: Any) -> Array:
    return array if views is None else array[views]


def make_parallel_2d_operator(
    ray_dir: Array,
    det_origin: Array,
    det_u_vec: Array,
    *,
    image_shape: tuple[int, int],
    num_detectors: int,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
    projector_mode: str = "footprint",
) -> ProjectionOperator:
    """Build an all-view parallel-beam :class:`ProjectionOperator`."""
    ny, nx = image_shape
    forward_op, backward_op = _select_pair("parallel", projector_mode)
    ray_dir = xp.array(ray_dir)
    det_origin = xp.array(det_origin)
    det_u_vec = xp.array(det_u_vec)
    n_views = int(ray_dir.shape[0])

    def fwd(volume: Array, views: Any) -> Array:
        return forward_op(
            volume, _index(ray_dir, views), _index(det_origin, views), _index(det_u_vec, views),
            num_detectors=num_detectors, detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        )

    def adj(sino: Array, views: Any) -> Array:
        return backward_op(
            xp.array(sino, dtype=xp.float32), _index(ray_dir, views), _index(det_origin, views), _index(det_u_vec, views),
            detector_spacing=detector_spacing, H=ny, W=nx, voxel_spacing=voxel_spacing,
        )

    return ProjectionOperator(
        fwd, adj, domain_shape=(ny, nx), range_shape=(n_views, num_detectors),
        n_views=n_views, beam="parallel",
    )


def make_fan_2d_operator(
    src_pos: Array,
    det_center: Array,
    det_u_vec: Array,
    *,
    image_shape: tuple[int, int],
    num_detectors: int,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
    projector_mode: str = "footprint",
) -> ProjectionOperator:
    """Build an all-view fan-beam :class:`ProjectionOperator`."""
    ny, nx = image_shape
    forward_op, backward_op = _select_pair("fan", projector_mode)
    src_pos = xp.array(src_pos)
    det_center = xp.array(det_center)
    det_u_vec = xp.array(det_u_vec)
    n_views = int(src_pos.shape[0])

    def fwd(volume: Array, views: Any) -> Array:
        return forward_op(
            volume, _index(src_pos, views), _index(det_center, views), _index(det_u_vec, views),
            num_detectors=num_detectors, detector_spacing=detector_spacing, voxel_spacing=voxel_spacing,
        )

    def adj(sino: Array, views: Any) -> Array:
        return backward_op(
            xp.array(sino, dtype=xp.float32), _index(src_pos, views), _index(det_center, views), _index(det_u_vec, views),
            detector_spacing=detector_spacing, H=ny, W=nx, voxel_spacing=voxel_spacing,
        )

    return ProjectionOperator(
        fwd, adj, domain_shape=(ny, nx), range_shape=(n_views, num_detectors),
        n_views=n_views, beam="fan",
    )


def make_cone_3d_operator(
    src_pos: Array,
    det_center: Array,
    det_u_vec: Array,
    det_v_vec: Array,
    *,
    volume_shape: tuple[int, int, int],
    detector_shape: tuple[int, int],
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    projector_mode: str = "footprint",
) -> ProjectionOperator:
    """Build an all-view cone-beam :class:`ProjectionOperator`."""
    nz, ny, nx = volume_shape
    det_u_count, det_v_count = detector_shape
    forward_op, backward_op = _select_pair("cone", projector_mode)
    src_pos = xp.array(src_pos)
    det_center = xp.array(det_center)
    det_u_vec = xp.array(det_u_vec)
    det_v_vec = xp.array(det_v_vec)
    n_views = int(src_pos.shape[0])

    def fwd(volume: Array, views: Any) -> Array:
        return forward_op(
            volume, _index(src_pos, views), _index(det_center, views),
            _index(det_u_vec, views), _index(det_v_vec, views),
            det_u=det_u_count, det_v=det_v_count, du=du, dv=dv, voxel_spacing=voxel_spacing,
        )

    def adj(sino: Array, views: Any) -> Array:
        return backward_op(
            xp.array(sino, dtype=xp.float32), _index(src_pos, views), _index(det_center, views),
            _index(det_u_vec, views), _index(det_v_vec, views),
            D=nz, H=ny, W=nx, du=du, dv=dv, voxel_spacing=voxel_spacing,
        )

    return ProjectionOperator(
        fwd, adj, domain_shape=(nz, ny, nx), range_shape=(n_views, det_u_count, det_v_count),
        n_views=n_views, beam="cone",
    )
