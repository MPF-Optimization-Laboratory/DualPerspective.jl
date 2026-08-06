import numpy as np
import pytest

from DualPerspective import (
    METHODS,
    DPModel,
    DualPerspectiveError,
    Solution,
    rand_dp_model,
    regularize,
    scale,
    solve,
    version,
)

M, N = 10, 5


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


@pytest.fixture
def model():
    return rand_dp_model(M, N)


def test_rand_dp_model_shapes(model):
    assert model.A.shape == (M, N)
    assert model.b.shape == (M,)
    assert model.shape == (M, N)


def test_model_repr(model):
    assert "DPModel(" in repr(model)


def test_model_with_optional_args(model):
    A, b = model.A, model.b
    q = np.random.rand(N)
    q /= q.sum()
    c = np.random.rand(N)
    # C weights the residual b - Ax, so it is (m, m). Releases before 0.2.0 documented
    # this as (n, n); a model built that way could not be solved at all.
    root = np.random.rand(M, M)
    C = root.T @ root + M * np.eye(M)

    built = DPModel(A, b, q=q, C=C, c=c, lam=0.1)
    assert built.A.shape == (M, N)
    assert built.b.shape == (M,)
    assert built.lam == pytest.approx(0.1)


def test_solve_with_non_identity_covariance():
    """The (m, m) covariance must survive all the way through a solve."""
    A = np.random.rand(M, N)
    b = A @ (np.ones(N) / N)
    root = np.random.rand(M, M)
    C = root.T @ root + M * np.eye(M)

    x = solve(DPModel(A, b, C=C, lam=1e-3))
    assert x.shape == (N,)
    assert np.all(np.isfinite(x))


def test_lambda_alias_accepted():
    A = np.random.rand(M, N)
    b = np.random.rand(M)
    assert DPModel(A, b, **{"λ": 0.25}).lam == pytest.approx(0.25)

    with pytest.raises(TypeError, match="not both"):
        DPModel(A, b, lam=0.1, **{"λ": 0.25})


def test_unexpected_keyword_rejected():
    A = np.random.rand(M, N)
    b = np.random.rand(M)
    with pytest.raises(TypeError, match="unexpected keyword"):
        DPModel(A, b, lamda=0.1)


# --- input validation ---------------------------------------------------------------


def test_integer_and_float32_inputs_are_coerced():
    A_int = np.ones((M, N), dtype=np.int64)
    b_int = np.ones(M, dtype=np.int64)
    assert solve(DPModel(A_int, b_int, lam=1e-2)).shape == (N,)

    A32 = np.random.rand(M, N).astype(np.float32)
    b32 = np.random.rand(M).astype(np.float32)
    assert solve(DPModel(A32, b32, lam=1e-2)).shape == (N,)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(b=np.ones(M + 1)), r"b must have length"),
        (dict(q=np.ones(N + 1)), r"q must have length"),
        (dict(c=np.ones(N + 1)), r"c must have length"),
        (dict(C=np.eye(N)), r"C must have shape"),
    ],
)
def test_shape_errors_are_raised_before_julia(kwargs, match):
    A = np.random.rand(M, N)
    args = {"b": np.random.rand(M)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DPModel(A, **args)


def test_dimensionality_errors():
    with pytest.raises(ValueError, match="A must be a 2-D array"):
        DPModel(np.random.rand(M), np.random.rand(M))
    with pytest.raises(ValueError, match="b must be a 1-D array"):
        DPModel(np.random.rand(M, N), np.random.rand(M, 1))


def test_non_finite_input_rejected():
    A = np.random.rand(M, N)
    A[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        DPModel(A, np.random.rand(M))


def test_arrays_are_copied_not_aliased():
    A = np.random.rand(M, N)
    b = np.random.rand(M)
    built = DPModel(A, b, lam=1e-3)
    A[:] = 0.0
    b[:] = 0.0
    assert not np.allclose(built.A, 0.0)
    assert not np.allclose(built.b, 0.0)


# --- solving ------------------------------------------------------------------------


def test_solve_returns_ndarray_by_default(model):
    """Backwards compatibility: existing code indexes the result directly."""
    x = solve(model)
    assert isinstance(x, np.ndarray)
    assert x.shape == (N,)
    assert len(x) == N
    assert np.isfinite(x[0])
    assert not np.any(np.isnan(x))


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_solves(method):
    model = rand_dp_model(M, N)
    x = solve(model, method=method)
    assert x.shape == (N,)
    assert np.all(np.isfinite(x))


def test_unknown_method_rejected(model):
    with pytest.raises(ValueError, match="unknown method"):
        solve(model, method="no-such-method")


def test_t_rejected_for_methods_that_choose_their_own_mass(model):
    with pytest.raises(TypeError, match="does not take a `t` argument"):
        solve(model, method="trust-region", t=1.0)


def test_solve_rejects_non_model():
    with pytest.raises(TypeError, match="expected a DPModel"):
        solve(np.zeros((M, N)))


def test_model_solve_method_matches_free_function(model):
    np.testing.assert_allclose(model.solve(), solve(model))


# --- results ------------------------------------------------------------------------


def test_full_output_returns_solution(model):
    result = solve(model, full_output=True)
    assert isinstance(result, Solution)
    np.testing.assert_allclose(result.x, solve(model))

    assert isinstance(result.status, str) and result.status
    assert isinstance(result.iterations, int)
    assert isinstance(result.elapsed_time, float)
    assert isinstance(result.primal_obj, float)
    assert isinstance(result.dual_obj, float)
    assert isinstance(result.optimality, float)
    assert isinstance(result.neval_jprod, int)
    assert isinstance(result.neval_jtprod, int)
    assert result.residual.shape == (M,)
    assert "Solution(" in repr(result)


def test_trace_is_always_a_list_of_dicts(model):
    """Never a type that depends on whether pandas happens to be installed."""
    trace = solve(model, method="trust-region", full_output=True, trace=True).trace
    assert isinstance(trace, list)
    assert trace, "trust-region should populate a trace"
    assert all(isinstance(row, dict) for row in trace)
    assert "cgits" in trace[0]


def test_solution_holds_no_live_julia_references(model):
    result = solve(model, full_output=True)
    assert type(result.x).__module__ == "numpy"
    assert type(result.residual).__module__ == "numpy"


# --- mutation helpers ---------------------------------------------------------------


def test_regularize_sets_lambda(model):
    regularize(model, 0.1)
    assert model.lam == pytest.approx(0.1)
    model.lam = 0.2
    assert model.lam == pytest.approx(0.2)


def test_scale_sets_mass(model):
    scale(model, 2.0)
    assert model.scale == pytest.approx(2.0)
    model.scale = 3.0
    assert model.scale == pytest.approx(3.0)


# --- errors -------------------------------------------------------------------------


def test_julia_errors_are_translated(model):
    """A Julia failure must arrive as DualPerspectiveError, with the cause preserved.

    LevelSet asserts ``1 < α < 2``, so this trips an error inside Julia rather than in the
    Python argument checks.
    """
    with pytest.raises(DualPerspectiveError) as excinfo:
        solve(model, method="level-set", **{"α": 5.0})

    error = excinfo.value
    assert error.__cause__ is not None
    assert type(error.__cause__).__name__ == "JuliaError"
    assert error.julia_traceback
    assert str(error)


def test_version_reports_a_string():
    v = version()
    assert isinstance(v, str) and v
