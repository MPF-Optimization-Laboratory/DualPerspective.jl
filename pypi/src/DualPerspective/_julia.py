"""Lazy bootstrap of the Julia side.

The Julia dependency is declared in ``juliapkg.json``, which ``pyjuliapkg`` reads from the
installed package and resolves *before* Julia starts. Nothing here calls ``Pkg.add`` at
import time: doing so left every installation to re-resolve against a moving ecosystem,
which is how a breaking SolverCore release reached users as an unexplained
``UndefVarError``.

Set ``DUALPERSPECTIVE_JL_PATH`` to a checkout of DualPerspective.jl to develop against local
sources instead of the pinned registry version. This must be set before the first call into
Julia.
"""

from __future__ import annotations

import os
import threading

__all__ = ["dualperspective", "julia_main", "is_started"]

_MODULE = None
_LOCK = threading.RLock()

_DEV_PATH_VAR = "DUALPERSPECTIVE_JL_PATH"
_LEGACY_DEV_VAR = "DUALPERSPECTIVE_USE_LOCAL"


def _dev_path() -> str | None:
    """Local checkout to develop against, or ``None`` to use the pinned version."""
    path = os.environ.get(_DEV_PATH_VAR)
    if path:
        return os.path.abspath(os.path.expanduser(path))
    # Backwards compatibility: the old flag assumed the package lived inside a checkout.
    if os.environ.get(_LEGACY_DEV_VAR, "").lower() in ("true", "1", "yes"):
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        if os.path.isfile(os.path.join(root, "Project.toml")):
            return root
        raise RuntimeError(
            f"{_LEGACY_DEV_VAR} is set but no DualPerspective.jl checkout was found at "
            f"{root!r}. Set {_DEV_PATH_VAR} to the checkout instead."
        )
    return None


def _register_dev_path() -> None:
    path = _dev_path()
    if path is None:
        return
    import juliapkg

    juliapkg.add(
        "DualPerspective",
        "a1edf54d-33cf-4bd3-8f57-70bdaf668f08",
        dev=True,
        path=path,
    )
    juliapkg.resolve()


def is_started() -> bool:
    """Whether Julia has been started yet."""
    return _MODULE is not None


def dualperspective():
    """Return a handle to the Julia ``DualPerspective`` module, starting Julia if needed.

    Everything is reached through this handle rather than through globals assigned into
    Julia's ``Main``, so we never collide with whatever else a user has loaded there.
    """
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    with _LOCK:
        if _MODULE is None:
            _register_dev_path()
            from juliacall import Main as jl

            jl.seval("import DualPerspective")
            _MODULE = jl.DualPerspective
    return _MODULE


def julia_main():
    """Return Julia's ``Main`` module, starting Julia if needed.

    Intended for diagnostics (see :mod:`DualPerspective.doctor`); prefer
    :func:`dualperspective` for anything else.
    """
    dualperspective()
    from juliacall import Main as jl

    return jl
