"""Translation of Julia exceptions into Python ones."""

from __future__ import annotations

import functools
import re

__all__ = ["DualPerspectiveError", "translate_julia_errors"]

# Signature of the SolverCore 0.3.9 `reset!` binding split. Users hitting this see a wall of
# Julia hint text that says nothing about what to do, so say it here instead.
_AMBIGUITY = re.compile(r"UndefVarError.*`?(reset!|solve!)`?.*not defined", re.DOTALL)

_AMBIGUITY_ADVICE = (
    "This is a dependency-resolution problem in the Julia environment, not a problem "
    "with your data. See "
    "https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl/blob/main/UPGRADING.md"
    " and run `python -m DualPerspective.doctor` to see the resolved versions."
)


class DualPerspectiveError(RuntimeError):
    """A failure raised by the underlying Julia package.

    The message is the first line of the Julia exception. The full Julia error text,
    including its backtrace, is available on :attr:`julia_traceback`.
    """

    def __init__(self, message: str, julia_traceback: str = ""):
        super().__init__(message)
        self.julia_traceback = julia_traceback


def _summarize(text: str) -> str:
    """Reduce a Julia error dump to its first meaningful line."""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Stacktrace"):
            return line
    return text.strip() or "unknown Julia error"


def translate_julia_errors(func):
    """Re-raise ``juliacall.JuliaError`` as :class:`DualPerspectiveError`.

    Chaining is preserved, so the original Julia exception remains reachable through
    ``__cause__`` for anyone who wants it.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if type(exc).__name__ != "JuliaError":
                raise
            text = str(exc)
            message = _summarize(text)
            if _AMBIGUITY.search(text):
                message = f"{message}\n\n{_AMBIGUITY_ADVICE}"
            raise DualPerspectiveError(message, julia_traceback=text) from exc

    return wrapper
