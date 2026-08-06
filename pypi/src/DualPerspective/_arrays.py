"""Conversion and validation of NumPy arrays on the way into Julia."""

from __future__ import annotations

import numpy as np

__all__ = ["as_matrix", "as_vector", "to_julia_matrix", "to_julia_vector"]


def _check_finite(a: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")


def as_vector(x, name: str, size: int | None = None) -> np.ndarray:
    """Validate and coerce ``x`` to a 1-D ``float64`` array.

    Integer and single-precision input is accepted and converted; the model is solved in
    double precision throughout.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {a.shape}")
    if size is not None and a.shape[0] != size:
        raise ValueError(f"{name} must have length {size}, got {a.shape[0]}")
    _check_finite(a, name)
    return a


def as_matrix(x, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Validate and coerce ``x`` to a 2-D ``float64`` array."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {a.shape}")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    _check_finite(a, name)
    return a


def to_julia_vector(jl, a: np.ndarray):
    """Copy ``a`` into a Julia ``Vector{Float64}``.

    The copy is deliberate: the model holds onto these arrays, and sharing the buffer would
    let a later mutation of the caller's NumPy array silently change the problem being
    solved.
    """
    return jl.convert(jl.Vector[jl.Float64], np.array(a, dtype=np.float64, copy=True))


def to_julia_matrix(jl, a: np.ndarray):
    """Copy ``a`` into a Julia ``Matrix{Float64}``.

    NumPy is row-major and Julia is column-major, so the Fortran-ordered copy is what avoids
    a transpose on the Julia side. The explicit element type matters: without it, an integer
    NumPy array converts to a ``Matrix{Int64}`` and fails much later, deep inside a solve.
    """
    return jl.convert(jl.Matrix[jl.Float64], np.asfortranarray(a, dtype=np.float64))
