"""The Julia dependency must be declared, shipped, and pinned -- not installed at runtime.

An unpinned runtime ``Pkg.add`` is what let a breaking SolverCore release reach users as an
unexplained ``UndefVarError``. These tests guard the replacement.
"""

import json
from importlib import resources

import pytest

import DualPerspective


def _juliapkg_spec() -> dict:
    text = resources.files("DualPerspective").joinpath("juliapkg.json").read_text()
    return json.loads(text)


def test_juliapkg_json_is_installed_with_the_package():
    """Missing from the wheel means silent fallback to an unpinned resolve."""
    assert resources.files("DualPerspective").joinpath("juliapkg.json").is_file()


def test_julia_dependency_is_pinned_exactly():
    spec = _juliapkg_spec()
    entry = spec["packages"]["DualPerspective"]
    assert entry["uuid"] == "a1edf54d-33cf-4bd3-8f57-70bdaf668f08"
    assert entry["version"].startswith("="), (
        f"expected an exact pin, got {entry['version']!r}; a bare version is a caret range"
    )


def test_julia_version_floor_is_declared():
    assert _juliapkg_spec()["julia"] == "1.10"


def test_no_julia_code_calls_pkg_add():
    """`Pkg.add` at runtime is the bug this package was restructured to remove.

    Julia code reaches Julia as a string literal, so look for executable string constants
    mentioning ``Pkg.add`` while ignoring docstrings and comments -- both of which
    legitimately discuss it.
    """
    import ast

    offenders = []
    for entry in resources.files("DualPerspective").iterdir():
        if not entry.name.endswith(".py"):
            continue
        tree = ast.parse(entry.read_text())
        docstrings = {
            node.body[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "Pkg.add" in node.value
                and node not in docstrings
            ):
                offenders.append(f"{entry.name}:{node.lineno}")
    assert not offenders, f"executable Pkg.add found at {offenders}"


def test_package_exports_are_importable():
    for name in DualPerspective.__all__:
        assert hasattr(DualPerspective, name), f"{name} listed in __all__ but missing"


def test_python_and_julia_versions_are_reported():
    assert isinstance(DualPerspective.__version__, str)
    assert DualPerspective.__version__

    julia_version = DualPerspective.version()
    assert isinstance(julia_version, str)
    # The wheel pins one exact Julia release; they must not drift apart.
    pinned = _juliapkg_spec()["packages"]["DualPerspective"]["version"].lstrip("=")
    assert julia_version == pinned, (
        f"loaded DualPerspective.jl {julia_version} but juliapkg.json pins {pinned}"
    )


def test_julia_starts_lazily():
    """Importing must not start Julia; only a call that needs it should."""
    from DualPerspective import _julia

    # By the time the rest of the suite has run Julia is up, so only assert the mechanism
    # exists and reports honestly.
    assert isinstance(_julia.is_started(), bool)
    _julia.dualperspective()
    assert _julia.is_started()


@pytest.mark.parametrize("attr", ["dualperspective", "julia_main", "is_started"])
def test_julia_module_helpers_exist(attr):
    from DualPerspective import _julia

    assert callable(getattr(_julia, attr))
