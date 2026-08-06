"""The :class:`DPModel` problem wrapper."""

from __future__ import annotations

import numpy as np

from ._arrays import as_matrix, as_vector, to_julia_matrix, to_julia_vector
from ._julia import dualperspective
from .errors import translate_julia_errors

__all__ = ["DPModel"]


class DPModel:
    r"""A KL-regularized least squares problem.

    Minimizes ``(1/2λ)‖Ax - b‖²_{C⁻¹} + ⟨c, x⟩ + KL(x ‖ q)`` over the probability simplex or
    the nonnegative orthant.

    Args:
        A: Matrix of shape ``(m, n)``.
        b: Right-hand side of length ``m``.
        q: Prior of length ``n``. Defaults to the uniform distribution.
        C: Covariance of the residual, shape ``(m, m)``. Defaults to the identity.
        c: Linear cost of length ``n``.
        lam: Regularization parameter. Also accepted as ``λ``.

    Note:
        ``C`` weights the residual, so it is ``(m, m)`` -- not ``(n, n)``. Releases before
        0.2.0 documented this incorrectly.

    All arrays are copied into Julia, so mutating the NumPy arrays afterwards does not
    affect the model.
    """

    __slots__ = ("_model",)

    @translate_julia_errors
    def __init__(self, A, b, q=None, C=None, c=None, lam=None, **kwargs):
        if "λ" in kwargs:
            if lam is not None:
                raise TypeError("pass either `lam` or `λ`, not both")
            lam = kwargs.pop("λ")
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

        A = as_matrix(A, "A")
        m, n = A.shape
        b = as_vector(b, "b", size=m)

        jl = dualperspective()
        jl_kwargs = {}
        if q is not None:
            jl_kwargs["q"] = to_julia_vector(jl, as_vector(q, "q", size=n))
        if c is not None:
            jl_kwargs["c"] = to_julia_vector(jl, as_vector(c, "c", size=n))
        if C is not None:
            # (m, m): C weights the residual b - Ax, which lives in R^m.
            jl_kwargs["C"] = to_julia_matrix(jl, as_matrix(C, "C", shape=(m, m)))
        if lam is not None:
            jl_kwargs["λ"] = float(lam)

        self._model = jl.DPModel(
            to_julia_matrix(jl, A), to_julia_vector(jl, b), **jl_kwargs
        )

    @classmethod
    def _from_julia(cls, julia_model) -> "DPModel":
        instance = cls.__new__(cls)
        instance._model = julia_model
        return instance

    @property
    def julia_model(self):
        """The underlying Julia ``DPModel``, for calls this wrapper does not cover."""
        return self._model

    @property
    def A(self) -> np.ndarray:
        return np.asarray(self._model.A, dtype=np.float64)

    @property
    def b(self) -> np.ndarray:
        return np.asarray(self._model.b, dtype=np.float64)

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self._model.q, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        """``(m, n)`` -- rows and columns of ``A``."""
        A = self._model.A
        return (int(jl_size(A, 1)), int(jl_size(A, 2)))

    @property
    def lam(self) -> float:
        """Regularization parameter. Assigning to it calls ``regularize!``."""
        return float(self._model.λ)

    @lam.setter
    @translate_julia_errors
    def lam(self, value: float) -> None:
        dualperspective().regularize_b(self._model, float(value))

    @property
    def scale(self) -> float:
        """Total mass. Assigning to it calls ``scale!``."""
        return float(self._model.scale)

    @scale.setter
    @translate_julia_errors
    def scale(self, value: float) -> None:
        dualperspective().scale_b(self._model, float(value))

    @translate_julia_errors
    def solve(self, method: str = "sequential", **kwargs):
        """Solve this model. See :func:`DualPerspective.solve`."""
        from .solvers import solve

        return solve(self, method=method, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        m, n = self.shape
        return f"DPModel(m={m}, n={n}, lam={self.lam:.3e}, scale={self.scale:g})"


def jl_size(array, dim: int) -> int:
    from ._julia import julia_main

    return julia_main().size(array, dim)
