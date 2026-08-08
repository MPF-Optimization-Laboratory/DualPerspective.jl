"""Python interface to DualPerspective.jl.

Solves Kullback-Leibler regularized least squares problems over the probability simplex or
the nonnegative orthant.

The Julia dependency is pinned in ``juliapkg.json`` and resolved by ``pyjuliapkg`` before
Julia starts; nothing is installed at import time. Julia itself starts lazily, on the first
call that needs it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .errors import DualPerspectiveError
from .model import DPModel
from .results import Solution
from .solvers import METHODS, rand_dp_model, solve, version

try:
    __version__ = _package_version("DualPerspective")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "DPModel",
    "DualPerspectiveError",
    "METHODS",
    "Solution",
    "__version__",
    "rand_dp_model",
    "regularize",
    "scale",
    "solve",
    "version",
]


def regularize(model: DPModel, lam: float) -> None:
    """Set the regularization parameter. Equivalent to ``model.lam = lam``."""
    model.lam = lam


def scale(model: DPModel, scale_factor: float) -> None:
    """Set the total mass. Equivalent to ``model.scale = scale_factor``."""
    model.scale = scale_factor
