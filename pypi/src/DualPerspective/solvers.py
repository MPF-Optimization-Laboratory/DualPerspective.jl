"""Solver entry points."""

from __future__ import annotations

import numpy as np

from ._julia import dualperspective
from .errors import translate_julia_errors
from .model import DPModel
from .results import Solution, from_execution_stats

__all__ = ["solve", "rand_dp_model", "version", "METHODS"]

#: Solver methods, mapped to the name of the Julia algorithm singleton. ``None`` selects the
#: default trust-region Newton-CG method, which takes no algorithm argument.
METHODS = {
    "sequential": "SequentialSolve",
    "level-set": "LevelSet",
    "adaptive-level-set": "AdaptiveLevelSet",
    "self-scaled": "SSTrunkLS",
    "trust-region": None,
}


@translate_julia_errors
def solve(
    model: DPModel,
    method: str = "sequential",
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    verbose: bool = False,
    logging: int = 0,
    t: float | None = None,
    full_output: bool = False,
    **kwargs,
):
    """Solve a :class:`DPModel`.

    Args:
        model: The problem to solve.
        method: One of :data:`METHODS`. The default, ``"sequential"``, matches the behaviour
            of earlier releases.
        atol: Absolute tolerance on ``‖∇d(y)‖``.
        rtol: Relative tolerance, scaled by ``‖b‖``.
        verbose: Print root-finding progress (``"sequential"`` only).
        logging: Solver log verbosity; ``0`` is silent.
        t: Target total mass. Defaults to ``sum(q)``.
        full_output: Return a :class:`~DualPerspective.results.Solution` instead of just the
            primal solution.
        **kwargs: Passed through to the Julia solver.

    Returns:
        The primal solution as a NumPy array, or a :class:`Solution` when ``full_output`` is
        set.

    Note:
        The solver stops when ``‖∇d(y)‖ < atol + rtol*‖b‖``.
    """
    if method not in METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose one of {sorted(METHODS)}"
        )
    if not isinstance(model, DPModel):
        raise TypeError(f"expected a DPModel, got {type(model).__name__}")

    jl = dualperspective()
    jl_model = model.julia_model

    call_kwargs = dict(atol=float(atol), rtol=float(rtol), logging=int(logging), **kwargs)

    if method == "sequential":
        call_kwargs["verbose"] = bool(verbose)

    # The trust-region and self-scaled methods determine the mass themselves.
    if method in ("sequential", "level-set", "adaptive-level-set"):
        call_kwargs["t"] = float(np.sum(model.q)) if t is None else float(t)
    elif t is not None:
        raise TypeError(f"method {method!r} does not take a `t` argument")

    algorithm = METHODS[method]
    if algorithm is None:
        # The default method takes no algorithm argument and has no `verbose` keyword.
        stats = jl.solve_b(jl_model, **call_kwargs)
    else:
        stats = jl.solve_b(jl_model, getattr(jl, algorithm)(), **call_kwargs)

    solution = from_execution_stats(stats)
    return solution if full_output else solution.x


@translate_julia_errors
def rand_dp_model(m: int, n: int, lam: float = 1e-3) -> DPModel:
    """Build a random ``m x n`` model, for examples and tests."""
    jl = dualperspective()
    return DPModel._from_julia(jl.randDPModel(int(m), int(n), λ=float(lam)))


@translate_julia_errors
def version() -> str:
    """Version of the underlying DualPerspective.jl package."""
    return str(dualperspective().version())
