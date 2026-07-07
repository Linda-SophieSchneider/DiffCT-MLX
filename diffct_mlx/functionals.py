"""Functionals: data-fidelity terms, regularizers and constraints.

A :class:`Functional` exposes up to three things about a scalar objective
``f(x)``:

* ``value(x)``    — the scalar (kept as a backend array so it stays autograd-able),
* ``gradient(x)`` — ``\\nabla f(x)`` for smooth ``f``,
* ``prox(x, tau)``— the proximal operator ``argmin_z f(z) + 1/(2 tau)||z-x||^2``.

Smooth functionals (Tikhonov, Huber, smoothed TV) implement ``gradient`` and are
used by gradient/CG solvers; nonsmooth ones (L1, indicator constraints)
implement ``prox`` and are used by proximal / POCS solvers. Functionals compose::

    objective = 0.5 * data_term + 0.1 * TotalVariation()      # scaling + sum

This is the substrate for the pluggable regularizer *sequences* (see
``regularizer_sequences`` / Phase 4), which apply a list of prox/denoise steps.
"""

from __future__ import annotations

from typing import Sequence

from .backend import active as _b
from .tv_gradients import (
    _awtv_objective,
    _tv_objective,
    awtv_gradient,
    tv_gradient,
)

xp = _b.xp

__all__ = [
    "Functional",
    "ScaledFunctional",
    "SumFunctional",
    "ZeroFunctional",
    "SquaredL2",
    "Tikhonov",
    "L1Norm",
    "LpNorm",
    "Huber",
    "TotalVariation",
    "AdaptiveWeightedTV",
    "NonNegativity",
    "Box",
    "soft_threshold",
]


def soft_threshold(x, tau):
    """Elementwise soft-thresholding ``sign(x) * max(|x| - tau, 0)``."""
    return xp.sign(x) * xp.maximum(xp.abs(x) - float(tau), 0.0)


class Functional:
    """Base class for objective / regularizer / constraint terms."""

    #: True when :meth:`gradient` is available (smooth term).
    is_smooth: bool = False
    #: True when :meth:`prox` is available (proximable term).
    is_proximable: bool = False

    def value(self, x):  # pragma: no cover - abstract-ish
        raise NotImplementedError(f"{type(self).__name__} does not define value().")

    def gradient(self, x):
        raise NotImplementedError(
            f"{type(self).__name__} is not smooth; use a proximal/POCS solver."
        )

    def prox(self, x, tau: float):
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form prox; use a gradient solver."
        )

    # -- algebra -----------------------------------------------------------
    def __mul__(self, scalar: float) -> "Functional":
        return ScaledFunctional(self, scalar)

    __rmul__ = __mul__

    def __add__(self, other: "Functional") -> "Functional":
        return SumFunctional([self, other])

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ScaledFunctional(Functional):
    """``c * f`` for a non-negative scalar ``c``."""

    def __init__(self, base: Functional, scalar: float):
        self.base = base
        self.scalar = float(scalar)
        self.is_smooth = base.is_smooth
        self.is_proximable = base.is_proximable

    def value(self, x):
        return self.scalar * self.base.value(x)

    def gradient(self, x):
        return self.scalar * self.base.gradient(x)

    def prox(self, x, tau: float):
        # prox of (c f) at step tau == prox of f at step (c tau).
        return self.base.prox(x, self.scalar * float(tau))


class SumFunctional(Functional):
    """Sum of functionals (smooth iff every term is smooth)."""

    def __init__(self, terms: Sequence[Functional]):
        self.terms = list(terms)
        self.is_smooth = all(t.is_smooth for t in self.terms)
        self.is_proximable = False  # prox of a sum is not separable in general

    def value(self, x):
        total = self.terms[0].value(x)
        for term in self.terms[1:]:
            total = total + term.value(x)
        return total

    def gradient(self, x):
        grad = self.terms[0].gradient(x)
        for term in self.terms[1:]:
            grad = grad + term.gradient(x)
        return grad


class ZeroFunctional(Functional):
    """The zero functional (no-op regularizer / prox is identity)."""

    is_smooth = True
    is_proximable = True

    def value(self, x):
        return xp.sum(x) * 0.0

    def gradient(self, x):
        return xp.zeros_like(x)

    def prox(self, x, tau: float):
        return x


class SquaredL2(Functional):
    r"""``0.5 * weight * ||x - offset||_2^2`` (a.k.a. Tikhonov when offset=0)."""

    is_smooth = True
    is_proximable = True

    def __init__(self, weight: float = 1.0, offset=None):
        self.weight = float(weight)
        self.offset = offset

    def _residual(self, x):
        return x if self.offset is None else x - self.offset

    def value(self, x):
        r = self._residual(x)
        return 0.5 * self.weight * xp.sum(r * r)

    def gradient(self, x):
        return self.weight * self._residual(x)

    def prox(self, x, tau: float):
        a = self.weight * float(tau)
        if self.offset is None:
            return x / (1.0 + a)
        return (x + a * self.offset) / (1.0 + a)


#: Tikhonov (ridge) regularizer is exactly a zero-offset squared-L2 term.
Tikhonov = SquaredL2


class L1Norm(Functional):
    r"""``weight * ||x||_1`` — nonsmooth; prox is soft-thresholding."""

    is_proximable = True

    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def value(self, x):
        return self.weight * xp.sum(xp.abs(x))

    def prox(self, x, tau: float):
        return soft_threshold(x, self.weight * float(tau))


class LpNorm(Functional):
    r"""``weight * sum |x|^p`` for ``p >= 1``.

    Smooth (with a small ``eps`` for stability) when ``p > 1`` — its gradient is
    ``weight * p * sign(x) * (|x| + eps)^{p-1}``. For ``p == 1`` it is the
    (nonsmooth) L1 norm and exposes the soft-threshold prox instead.
    """

    def __init__(self, p: float = 1.0, weight: float = 1.0, eps: float = 1e-8):
        self.p = float(p)
        self.weight = float(weight)
        self.eps = float(eps)
        self.is_smooth = self.p > 1.0
        self.is_proximable = self.p == 1.0

    def value(self, x):
        return self.weight * xp.sum(xp.abs(x) ** self.p)

    def gradient(self, x):
        if self.p <= 1.0:
            raise NotImplementedError("LpNorm.gradient requires p > 1; use prox for p == 1.")
        return self.weight * self.p * xp.sign(x) * (xp.abs(x) + self.eps) ** (self.p - 1.0)

    def prox(self, x, tau: float):
        if self.p != 1.0:
            raise NotImplementedError("LpNorm.prox is closed-form only for p == 1.")
        return soft_threshold(x, self.weight * float(tau))


class Huber(Functional):
    r"""Elementwise Huber penalty (smooth L1 surrogate).

    ``f(x) = weight * sum_i h(x_i)`` with ``h(t) = 0.5 t^2`` for ``|t| <= delta``
    and ``delta(|t| - 0.5 delta)`` otherwise. Robust and differentiable.
    """

    is_smooth = True

    def __init__(self, delta: float = 1.0, weight: float = 1.0):
        self.delta = float(delta)
        self.weight = float(weight)

    def value(self, x):
        d = self.delta
        absx = xp.abs(x)
        quad = 0.5 * x * x
        lin = d * (absx - 0.5 * d)
        return self.weight * xp.sum(xp.where(absx <= d, quad, lin))

    def gradient(self, x):
        d = self.delta
        absx = xp.abs(x)
        return self.weight * xp.where(absx <= d, x, d * xp.sign(x))


class TotalVariation(Functional):
    r"""Isotropic (smoothed) total variation ``weight * TV_eps(x)``.

    Smoothed with ``sqrt(|grad|^2 + eps)`` so it is differentiable; the gradient
    routes through the shared :func:`~diffct_mlx.tv_gradients.tv_gradient`.
    """

    is_smooth = True

    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        self.weight = float(weight)
        self.eps = float(eps)

    def value(self, x):
        return self.weight * _tv_objective(x, eta=self.eps)

    def gradient(self, x):
        return self.weight * tv_gradient(x, eta=self.eps)


class AdaptiveWeightedTV(Functional):
    r"""Edge-adaptive weighted TV ``weight * AwTV_{delta,eps}(x)`` (smooth)."""

    is_smooth = True

    def __init__(self, weight: float = 1.0, delta: float = 0.6e-2, eps: float = 1e-6):
        self.weight = float(weight)
        self.delta = float(delta)
        self.eps = float(eps)

    def value(self, x):
        return self.weight * _awtv_objective(x, delta=self.delta, eta=self.eps)

    def gradient(self, x):
        return self.weight * awtv_gradient(x, delta=self.delta, eta=self.eps)


class NonNegativity(Functional):
    """Indicator of the non-negative orthant; prox clamps negatives to zero."""

    is_proximable = True

    def value(self, x):
        return xp.sum(x) * 0.0  # 0 on the feasible set (assumed)

    def prox(self, x, tau: float):
        return xp.maximum(x, 0.0)


class Box(Functional):
    """Indicator of ``[lo, hi]``; prox clips into the box."""

    is_proximable = True

    def __init__(self, lo: float = 0.0, hi: float = float("inf")):
        self.lo = None if lo is None or lo == float("-inf") else float(lo)
        self.hi = None if hi is None or hi == float("inf") else float(hi)

    def value(self, x):
        return xp.sum(x) * 0.0

    def prox(self, x, tau: float):
        return xp.clip(x, self.lo, self.hi)
