"""Solver results, converted eagerly into plain Python and NumPy objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Solution"]


@dataclass
class Solution:
    """Everything the Julia solver reported about a solve.

    All fields are plain Python or NumPy values: no live Julia references are held, so a
    ``Solution`` stays valid and picklable independently of the Julia session.
    """

    x: np.ndarray
    """Primal solution."""

    residual: np.ndarray
    """Covariance-weighted residual."""

    status: str
    """Termination status, e.g. ``"optimal"`` or ``"max_iter"``."""

    iterations: int
    elapsed_time: float
    primal_obj: float
    dual_obj: float
    optimality: float
    """Final ``‖∇d(y)‖``; the quantity the stopping rule tests."""

    neval_jprod: int
    neval_jtprod: int
    trace: list[dict] = field(default_factory=list)
    """Per-iteration history, empty unless the solver populates one."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Solution(status={self.status!r}, n={self.x.size}, "
            f"iterations={self.iterations}, optimality={self.optimality:.3e}, "
            f"elapsed_time={self.elapsed_time:.3f}s)"
        )

    def to_pandas(self):
        """Return :attr:`trace` as a ``pandas.DataFrame``.

        pandas is not a dependency; this raises if it is not installed. :attr:`trace` itself
        is always a list of dicts, so the return type never depends on what happens to be
        installed.
        """
        import pandas as pd

        return pd.DataFrame(self.trace)


# Done on the Julia side: `df.names` would be read as a *column* named "names", so the
# conversion has to go through DataFrames' own API.
_TO_RECORDS = """
    df -> [Dict{String,Any}(string(k) => v for (k, v) in pairs(r)) for r in eachrow(df)]
"""


def _tracer_to_records(tracer) -> list[dict]:
    """Convert a Julia ``DataFrame`` tracer into a list of plain dicts."""
    if tracer is None:
        return []
    from ._julia import julia_main

    rows = julia_main().seval(_TO_RECORDS)(tracer)
    return [{str(k): _scalar(v) for k, v in row.items()} for row in rows]


def _scalar(v):
    """Unwrap a Julia scalar into its Python equivalent."""
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return v.item()
    except AttributeError:
        return str(v)


def from_execution_stats(stats) -> Solution:
    """Build a :class:`Solution` from a Julia ``ExecutionStats``."""
    return Solution(
        x=np.asarray(stats.solution, dtype=np.float64),
        residual=np.asarray(stats.residual, dtype=np.float64),
        status=str(stats.status),
        iterations=int(stats.iter),
        elapsed_time=float(stats.elapsed_time),
        primal_obj=float(stats.primal_obj),
        dual_obj=float(stats.dual_obj),
        optimality=float(stats.optimality),
        neval_jprod=int(stats.neval_jprod),
        neval_jtprod=int(stats.neval_jtprod),
        trace=_tracer_to_records(getattr(stats, "tracer", None)),
    )
