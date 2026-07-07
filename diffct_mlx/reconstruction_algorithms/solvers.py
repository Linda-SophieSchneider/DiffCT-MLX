"""Statistical, least-squares and conjugate-gradient reconstructors.

All operate on the abstract operator interface (``A`` with matched adjoint
``A.T``) and register into the shared solver registry, so they are reachable via
``reconstruct("wls", A, b, ...)`` and enumerated by ``list_algorithms()``.

Families provided
-----------------
* **Least squares** — ``pcg`` (generic preconditioned CG for an SPD normal
  operator), ``ls`` (plain), ``wls`` (statistically weighted), ``rls``
  (Tikhonov-regularized), all via CG on the normal equations.
* **Regularized WLS** — ``rwls``: FISTA for ``0.5||Ax-b||^2_W + R(x)`` with any
  smooth regularizer ``R`` (TV / Huber / Tikhonov) plus a proximable constraint
  (non-negativity / box). This is the workhorse for few-view / regularized
  problems and accepts the pluggable regularizers from
  :mod:`diffct_mlx.functionals`.
* **Statistical** — ``mlem`` / ``osem`` (emission multiplicative updates) and
  ``mltr`` (transmission Poisson ML via a separable-paraboloidal surrogate).
* **DLS / RDLS** — LEAP-style few-view (regularized) least squares, provided
  here as preconditioned LS / regularized WLS wrappers (see their docstrings).

The classic ``reconstruct_fbp`` / ``reconstruct_sart`` entry points are
unaffected; this module is purely additive.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..backend import active as _b
from ..functionals import Functional, NonNegativity, ZeroFunctional
from ..operators import DiagonalOperator, IdentityOperator, LinearOperator
from ._solver import init_iterate, inner, register_algorithm

xp = _b.xp
Array = Any

__all__ = [
    "pcg",
    "ls",
    "wls",
    "rls",
    "rwls",
    "dls",
    "rdls",
    "mlem",
    "osem",
    "mltr",
    "power_iteration",
    "make_subsets",
]

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def power_iteration(normal_apply, shape, iterations: int = 15) -> float:
    """Estimate the spectral norm of an SPD operator ``H`` via power iteration.

    ``normal_apply(v)`` must return ``H v``. Returns the largest eigenvalue of
    ``H`` (used to pick safe gradient step sizes ``t = 1/L``).
    """
    import numpy as _np

    rng = _np.random.default_rng(0)
    v = xp.array(rng.standard_normal(tuple(int(s) for s in shape)).astype(_np.float32))
    v = v / (float(xp.norm(v)) + _EPS)
    lam = 1.0
    for _ in range(int(iterations)):
        w = normal_apply(v)
        lam = float(xp.norm(w))
        if lam <= 0.0:
            break
        v = w / lam
    return lam


def make_subsets(n_views: int, n_subsets: int) -> list[list[int]]:
    """Interleaved ordered subsets of the view indices (bit-reversal-like spread)."""
    n_subsets = max(1, min(int(n_subsets), int(n_views)))
    return [list(range(offset, int(n_views), n_subsets)) for offset in range(n_subsets)]


def _weight_operator(weights, range_shape) -> LinearOperator:
    if weights is None:
        return IdentityOperator(range_shape)
    return DiagonalOperator(weights)


def _normal_operator(A: LinearOperator, W: LinearOperator, beta: float) -> LinearOperator:
    """SPD normal operator ``A^T W A + beta I``."""
    H = A.T @ (W @ A)
    if beta and beta > 0.0:
        H = H + float(beta) * IdentityOperator(A.domain_shape)
    return H


# ---------------------------------------------------------------------------
# Preconditioned conjugate gradients (SPD normal operator)
# ---------------------------------------------------------------------------
@register_algorithm("pcg")
def pcg(
    H: LinearOperator,
    c: Array,
    *,
    x0: Array | None = None,
    iterations: int = 25,
    tol: float = 1e-6,
    precond=None,
    nonnegative: bool = False,
    callback=None,
) -> Array:
    """Preconditioned CG for the SPD system ``H x = c``.

    ``H`` must be self-adjoint positive (semi-)definite (e.g. a normal operator
    ``A^T W A + beta I``). ``precond`` is an optional callable ``r -> M^{-1} r``
    approximating ``H^{-1}`` (identity when ``None``).
    """
    c = xp.array(c, dtype=_b.float32)
    x = init_iterate(H, x0) if x0 is not None or H.domain_shape is not None else xp.zeros_like(c)
    r = c - H._forward(x)
    z = precond(r) if precond is not None else r
    p = z
    rz = inner(r, z)
    rz0 = float(inner(r, r))
    for k in range(int(iterations)):
        Hp = H._forward(p)
        pHp = float(inner(p, Hp))
        if pHp <= 0.0:
            break
        alpha = rz / pHp
        x = x + alpha * p
        r = r - alpha * Hp
        z = precond(r) if precond is not None else r
        rz_new = inner(r, z)
        beta = rz_new / (rz + _EPS)
        p = z + beta * p
        rz = rz_new
        xp.eval(x)
        if callback is not None:
            callback(k, x)
        if rz0 > 0.0 and float(inner(r, r)) / rz0 < float(tol) ** 2:
            break
    if nonnegative:
        x = xp.maximum(x, 0.0)
    return x


# ---------------------------------------------------------------------------
# Least-squares family (CG on the normal equations)
# ---------------------------------------------------------------------------
@register_algorithm("wls")
def wls(
    A: LinearOperator,
    b: Array,
    *,
    weights: Array | None = None,
    beta: float = 0.0,
    x0: Array | None = None,
    iterations: int = 25,
    tol: float = 1e-6,
    precond=None,
    nonnegative: bool = False,
    callback=None,
) -> Array:
    """(Regularized) weighted least squares ``min 0.5||Ax-b||^2_W + 0.5 beta||x||^2``.

    Solved with preconditioned CG on ``(A^T W A + beta I) x = A^T W b``.
    ``weights`` is a per-measurement diagonal (e.g. statistical weights
    ``w_i = 1/var_i`` ~ counts); ``None`` means unweighted.
    """
    b = xp.array(b, dtype=_b.float32)
    W = _weight_operator(weights, A.range_shape)
    H = _normal_operator(A, W, beta)
    c = A.T @ (W @ b)
    return pcg(
        H, c, x0=x0, iterations=iterations, tol=tol,
        precond=precond, nonnegative=nonnegative, callback=callback,
    )


@register_algorithm("ls")
def ls(A, b, **kwargs):
    """Plain least squares ``min 0.5||Ax-b||^2`` (unweighted, unregularized)."""
    kwargs.pop("weights", None)
    kwargs.pop("beta", None)
    return wls(A, b, weights=None, beta=0.0, **kwargs)


@register_algorithm("rls")
def rls(A, b, *, beta: float = 1e-2, **kwargs):
    """Tikhonov-regularized least squares (ridge): ``+ 0.5 beta ||x||^2``."""
    kwargs.pop("weights", None)
    return wls(A, b, weights=kwargs.pop("weights", None), beta=beta, **kwargs)


# ---------------------------------------------------------------------------
# Regularized weighted least squares with a general regularizer (FISTA)
# ---------------------------------------------------------------------------
@register_algorithm("rwls")
def rwls(
    A: LinearOperator,
    b: Array,
    *,
    regularizer: Functional | None = None,
    constraint: Functional | None = None,
    weights: Array | None = None,
    x0: Array | None = None,
    iterations: int = 50,
    step_size: float | None = None,
    nonnegative: bool = True,
    callback=None,
) -> Array:
    r"""Regularized WLS via FISTA: ``min 0.5||Ax-b||^2_W + R(x)  s.t.  x in C``.

    * ``regularizer`` — a **smooth** :class:`~diffct_mlx.functionals.Functional`
      (``TotalVariation``, ``Huber``, ``Tikhonov``, ...); its gradient is added
      to the data gradient.
    * ``constraint`` — a **proximable** functional applied via its prox each
      step (``NonNegativity`` by default when ``nonnegative``; e.g. ``Box`` or
      ``L1Norm``).
    * ``step_size`` — gradient step; if ``None`` it is set to ``1/L`` with ``L``
      the data-term Lipschitz constant estimated by power iteration.
    """
    b = xp.array(b, dtype=_b.float32)
    W = _weight_operator(weights, A.range_shape)
    reg = regularizer if regularizer is not None else ZeroFunctional()
    if constraint is None:
        constraint = NonNegativity() if nonnegative else ZeroFunctional()

    def data_grad(x):
        return A.T @ (W @ (A @ x - b))

    if step_size is None:
        L = power_iteration(lambda v: A.T @ (W @ (A @ v)), A.domain_shape)
        step_size = 0.9 / (L + _EPS)
    t = float(step_size)

    x = init_iterate(A, x0)
    z = x
    momentum = 1.0
    for k in range(int(iterations)):
        grad = data_grad(z)
        if reg.is_smooth:
            grad = grad + reg.gradient(z)
        x_new = constraint.prox(z - t * grad, t)
        momentum_new = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        z = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
        x, momentum = x_new, momentum_new
        xp.eval(x)
        if callback is not None:
            callback(k, x)
    return x


@register_algorithm("dls")
def dls(A, b, **kwargs):
    """Few-view least squares (LEAP calls this DLS).

    Implemented as preconditioned :func:`ls`; pass ``precond=`` to supply an
    FBP-style / diagonal preconditioner for faster few-view convergence.
    """
    return ls(A, b, **kwargs)


@register_algorithm("rdls")
def rdls(A, b, **kwargs):
    """Regularized few-view least squares (LEAP calls this RDLS).

    Implemented as :func:`rwls`; supply ``regularizer=`` (e.g. TotalVariation)
    for the regularized few-view behaviour.
    """
    return rwls(A, b, **kwargs)


# ---------------------------------------------------------------------------
# Statistical solvers
# ---------------------------------------------------------------------------
@register_algorithm("mlem")
def mlem(
    A: LinearOperator,
    b: Array,
    *,
    x0: Array | None = None,
    iterations: int = 20,
    callback=None,
) -> Array:
    """Maximum-likelihood expectation maximization (emission model).

    Multiplicative, intrinsically non-negative update for ``b >= 0``::

        x <- x / (A^T 1) * A^T ( b / (A x) )
    """
    b = xp.maximum(xp.array(b, dtype=_b.float32), 0.0)
    x = xp.ones(A.domain_shape, dtype=_b.float32) if x0 is None else xp.maximum(xp.array(x0, dtype=_b.float32), _EPS)
    ones_range = xp.ones(A.range_shape, dtype=_b.float32)
    sens = A.T @ ones_range
    sens = xp.maximum(sens, _EPS)
    for k in range(int(iterations)):
        fp = xp.maximum(A @ x, _EPS)
        x = x * (A.T @ (b / fp)) / sens
        x = xp.maximum(x, 0.0)
        xp.eval(x)
        if callback is not None:
            callback(k, x)
    return x


@register_algorithm("osem")
def osem(
    A: LinearOperator,
    b: Array,
    *,
    subsets: int = 8,
    x0: Array | None = None,
    iterations: int = 5,
    callback=None,
) -> Array:
    """Ordered-subsets EM — MLEM accelerated by processing view subsets per sweep."""
    b = xp.maximum(xp.array(b, dtype=_b.float32), 0.0)
    x = xp.ones(A.domain_shape, dtype=_b.float32) if x0 is None else xp.maximum(xp.array(x0, dtype=_b.float32), _EPS)
    subset_views = make_subsets(A.n_views, subsets)
    # Per-subset sensitivity (geometry only) — precompute once.
    sens = []
    for views in subset_views:
        As = A.subset(views)
        sens.append(xp.maximum(As.T @ xp.ones(As.range_shape, dtype=_b.float32), _EPS))
    for k in range(int(iterations)):
        for views, s in zip(subset_views, sens):
            As = A.subset(views)
            bs = b[views]
            fp = xp.maximum(As @ x, _EPS)
            x = x * (As.T @ (bs / fp)) / s
            x = xp.maximum(x, 0.0)
            xp.eval(x)
        if callback is not None:
            callback(k, x)
    return x


@register_algorithm("mltr")
def mltr(
    A: LinearOperator,
    counts: Array,
    blank: Array,
    *,
    x0: Array | None = None,
    iterations: int = 60,
    beta: float = 0.0,
    regularizer: Functional | None = None,
    relaxation: float = 1.0,
    curvature: str = "optimal",
    callback=None,
) -> Array:
    r"""Transmission maximum-likelihood (Poisson) reconstruction.

    Models detected counts ``y_i ~ Poisson(b0_i * exp(-[A x]_i))`` and minimizes
    the negative Poisson log-likelihood with a separable-paraboloidal surrogate
    (De Man / Erdogan-Fessler)::

        l = A x ;  yhat = b0 * exp(-l)
        x <- x + relax * A^T (yhat - y) / A^T ( c * (A 1) )   (then clamp >= 0)

    ``counts`` are the raw measured counts ``y`` and ``blank`` the air/I0 scan
    ``b0`` (both in detector space). For a log-transformed sinogram ``p`` with a
    known ``I0`` use ``counts = I0 * exp(-p)``, ``blank = I0``.

    ``curvature`` selects the surrogate denominator ``c``: ``"optimal"``
    (default, per-iterate ``yhat``) converges fastest for physically-scaled
    attenuation (``l`` up to a few); ``"max"`` uses the precomputed maximum
    curvature ``b0`` (De Man) — always monotone but slow. Under photon
    starvation (very large ``l``) the diagonal-Newton step can overshoot; then
    lower ``relaxation`` or switch to ``"max"``. An optional smooth
    ``regularizer`` (weight ``beta``) is added to the gradient.
    """
    y = xp.maximum(xp.array(counts, dtype=_b.float32), 0.0)
    b0 = xp.maximum(xp.array(blank, dtype=_b.float32), _EPS)
    x = xp.zeros(A.domain_shape, dtype=_b.float32) if x0 is None else xp.array(x0, dtype=_b.float32)
    a_one = A @ xp.ones(A.domain_shape, dtype=_b.float32)  # row sums (A 1)
    relax = float(relaxation)
    curv_max = None
    if str(curvature).lower() == "max":
        # Max over l >= 0 of yhat is at l = 0 (yhat = b0); precompute once.
        curv_max = A.T @ (b0 * a_one)
        curv_max = xp.maximum(curv_max, _EPS)
    for k in range(int(iterations)):
        l = A @ x
        yhat = b0 * xp.exp(-l)
        numerator = A.T @ (yhat - y)
        denom = curv_max if curv_max is not None else xp.maximum(A.T @ (yhat * a_one), _EPS)
        if regularizer is not None and beta and beta > 0.0:
            numerator = numerator - float(beta) * regularizer.gradient(x)
            denom = denom + float(beta)
        x = xp.maximum(x + relax * (numerator / denom), 0.0)
        xp.eval(x)
        if callback is not None:
            callback(k, x)
    return x
