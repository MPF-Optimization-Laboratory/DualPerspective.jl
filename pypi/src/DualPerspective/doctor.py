"""Environment diagnostics: ``python -m DualPerspective.doctor``.

Prints everything needed to diagnose a dependency-resolution problem in one paste. Most
failures in this package come from the Julia environment rather than from Python, and the
Julia versions in play are otherwise awkward to discover.
"""

from __future__ import annotations

import os
import platform
import sys

_JULIA_ENV_VARS = (
    "JULIA_DEPOT_PATH",
    "JULIA_PROJECT",
    "PYTHON_JULIACALL_PROJECT",
    "PYTHON_JULIACALL_EXE",
    "PYTHON_JULIACALL_HANDLE_SIGNALS",
    "PYTHON_JULIACALL_THREADS",
    "DUALPERSPECTIVE_JL_PATH",
)

# The packages whose versions actually explain the failures users report.
_KEY_PACKAGES = (
    "DualPerspective",
    "JSOSolvers",
    "SolverCore",
    "SolverTools",
    "NLPModels",
    "LinearOperators",
    "Krylov",
)


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    print("DualPerspective environment report")

    _section("Python")
    print(f"  python           {sys.version.split()[0]} ({platform.platform()})")
    print(f"  executable       {sys.executable}")
    for name in ("DualPerspective", "juliacall", "juliapkg", "numpy"):
        try:
            from importlib.metadata import version as _v

            print(f"  {name:<16} {_v(name)}")
        except Exception:  # noqa: BLE001
            print(f"  {name:<16} (not installed)")

    _section("Environment variables")
    for var in _JULIA_ENV_VARS:
        value = os.environ.get(var)
        print(f"  {var:<32} {value if value else '(unset)'}")

    _section("Julia")
    try:
        from ._julia import julia_main

        jl = julia_main()
        print(f"  version          {jl.string(jl.VERSION)}")
        print(f"  depot            {list(jl.DEPOT_PATH)[0]}")
        print(f"  project          {jl.Base.active_project()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to start Julia: {type(exc).__name__}: {exc}")
        print("\nSee https://github.com/MPF-Optimization-Laboratory/"
              "DualPerspective.jl/blob/main/UPGRADING.md")
        return 1

    _section("Resolved Julia packages")
    try:
        jl.seval("import Pkg")
        deps = jl.seval("Pkg.dependencies()")
        found = {}
        for entry in jl.seval("collect(values(Pkg.dependencies()))"):
            found[str(entry.name)] = str(entry.version)
        for name in _KEY_PACKAGES:
            print(f"  {name:<20} {found.get(name, '(not installed)')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not query Pkg: {type(exc).__name__}: {exc}")

    _section("Smoke test")
    try:
        from .solvers import rand_dp_model, solve

        model = rand_dp_model(10, 5)
        x = solve(model)
        print(f"  solved a 10x5 problem; sum(x) = {x.sum():.6f}")
        print("\nAll good.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\nSee https://github.com/MPF-Optimization-Laboratory/"
              "DualPerspective.jl/blob/main/UPGRADING.md")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
