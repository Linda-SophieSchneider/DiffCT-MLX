"""Solver framework: a registry + a common operator-based interface.

Every modern iterative reconstructor here follows one uniform signature::

    x = solver(A, b, *, x0=None, iterations=..., callback=None, **kwargs)

where ``A`` is a :class:`~diffct_mlx.operators.LinearOperator` (with a matched
adjoint ``A.T``) and ``b`` is the measured sinogram. Because everything is
written against the abstract ``A``, a new algorithm is a single function plus a
one-line decorator::

    from diffct_mlx.reconstruction_algorithms import register_algorithm

    @register_algorithm("my_solver")
    def my_solver(A, b, *, iterations=10, **kw):
        x = A.T @ b
        ...
        return x

    x = reconstruct("my_solver", A, b, iterations=20)

The registry is shared with the built-in statistical/CG/least-squares solvers,
so ``list_algorithms()`` enumerates everything available. The classic
case-based ``reconstruct_fbp`` / ``reconstruct_sart`` entry points are
unchanged; this is an additive interface.
"""

from __future__ import annotations

from typing import Any, Callable

from ..backend import active as _b

xp = _b.xp

Array = Any
Solver = Callable[..., Array]

__all__ = [
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
    "reconstruct",
    "inner",
    "init_iterate",
    "IterativeReconstructor",
    "landweber",
    "cgls",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_ALGORITHMS: dict[str, Solver] = {}


def register_algorithm(name: str, fn: Solver | None = None):
    """Register a solver under ``name``.

    Usable as a decorator (``@register_algorithm("cg")``) or directly
    (``register_algorithm("cg", fn)``). Names are case-insensitive.
    """
    def _register(func: Solver) -> Solver:
        _ALGORITHMS[str(name).strip().lower()] = func
        return func

    return _register if fn is None else _register(fn)


def get_algorithm(name: str) -> Solver:
    """Return the solver registered under ``name`` (case-insensitive)."""
    key = str(name).strip().lower()
    if key not in _ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm {name!r}. Registered: {', '.join(sorted(_ALGORITHMS)) or '(none)'}."
        )
    return _ALGORITHMS[key]


def list_algorithms() -> list[str]:
    """List all registered solver names."""
    return sorted(_ALGORITHMS)


def reconstruct(name: str, *args, **kwargs) -> Array:
    """Dispatch to a registered solver by name."""
    return get_algorithm(name)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def inner(a: Array, b: Array) -> Array:
    """Backend-scalar inner product ``<a, b> = sum(a * b)``."""
    return xp.sum(a * b)


def init_iterate(operator, x0: Array | None) -> Array:
    """Return the initial iterate — ``x0`` if given, else zeros of the domain."""
    if x0 is not None:
        return xp.array(x0, dtype=_b.float32)
    if operator.domain_shape is None:
        raise ValueError("Operator has no domain_shape; pass an explicit x0.")
    return xp.zeros(operator.domain_shape, dtype=_b.float32)


def _maybe_callback(callback, iteration: int, x: Array) -> None:
    if callback is not None:
        callback(iteration, x)


class IterativeReconstructor:
    """Optional base class for class-style algorithms.

    Subclass and implement :meth:`step`; :meth:`run` provides the loop,
    callbacks and (optional) non-negativity projection. Function-style solvers
    (registered directly) are equally first-class — this is just a convenience.
    """

    def __init__(self, iterations: int = 10, nonnegative: bool = False):
        self.iterations = int(iterations)
        self.nonnegative = bool(nonnegative)

    def step(self, A, b, x, k):  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, A, b, *, x0: Array | None = None, callback=None) -> Array:
        x = init_iterate(A, x0)
        for k in range(self.iterations):
            x = self.step(A, b, x, k)
            if self.nonnegative:
                x = xp.maximum(x, 0.0)
            xp.eval(x)
            _maybe_callback(callback, k, x)
        return x


# ---------------------------------------------------------------------------
# Reference solvers (validate the framework; workhorses in their own right)
# ---------------------------------------------------------------------------
@register_algorithm("landweber")
def landweber(
    A,
    b,
    *,
    x0: Array | None = None,
    iterations: int = 30,
    step_size: float = 1e-3,
    nonnegative: bool = True,
    callback=None,
) -> Array:
    """Landweber / gradient descent on ``0.5||A x - b||^2``.

    ``x <- x - step_size * A^T (A x - b)`` (optionally projected to ``x >= 0``).
    Simple and robust; ``step_size`` must be below ``2 / ||A||^2``.
    """
    b = xp.array(b, dtype=_b.float32)
    x = init_iterate(A, x0)
    for k in range(int(iterations)):
        residual = A._forward(x) - b
        x = x - float(step_size) * A._adjoint(residual)
        if nonnegative:
            x = xp.maximum(x, 0.0)
        xp.eval(x)
        _maybe_callback(callback, k, x)
    return x


@register_algorithm("cgls")
def cgls(
    A,
    b,
    *,
    x0: Array | None = None,
    iterations: int = 20,
    tol: float = 1e-6,
    nonnegative: bool = False,
    callback=None,
) -> Array:
    """Conjugate gradients on the normal equations ``A^T A x = A^T b`` (CGLS).

    Numerically stable least-squares CG (Björck). ``nonnegative`` applies a
    final clamp only — for hard box constraints per iteration use a proximal
    solver instead.
    """
    b = xp.array(b, dtype=_b.float32)
    x = init_iterate(A, x0)

    r = b - A._forward(x)
    s = A._adjoint(r)
    p = s
    gamma = inner(s, s)
    gamma0 = float(gamma)
    for k in range(int(iterations)):
        q = A._forward(p)
        qq = float(inner(q, q))
        if qq <= 0.0:
            break
        alpha = gamma / qq
        x = x + alpha * p
        r = r - alpha * q
        s = A._adjoint(r)
        gamma_new = inner(s, s)
        beta = gamma_new / gamma
        p = s + beta * p
        gamma = gamma_new
        xp.eval(x)
        _maybe_callback(callback, k, x)
        if gamma0 > 0.0 and float(gamma) / gamma0 < float(tol) ** 2:
            break
    if nonnegative:
        x = xp.maximum(x, 0.0)
    return x
