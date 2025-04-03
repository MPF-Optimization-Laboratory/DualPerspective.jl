import numpy as np
import pytest
from DualPerspective import DPModel, solve, regularize, rand_dp_model, version

# Set random seeds for reproducibility
np.random.seed(42)

m = 10
n = 5

def test_dp_model_creation():
    """Test basic model creation."""
    model = rand_dp_model(m, n)
    assert model.A.shape == (m, n)
    assert model.b.shape == (m,)

def test_dp_model_with_optional_args():
    """Test model creation with optional arguments."""
    model = rand_dp_model(m, n)
    A = model.A
    b = model.b
    q = np.random.rand(n)
    q /= q.sum()
    C_temp = np.random.rand(n, n)
    C = C_temp.T @ C_temp  # Create positive definite matrix
    c = np.random.rand(n)
    λ = 0.1
    
    model = DPModel(A, b, q=q, C=C, c=c, λ=λ)
    assert model.A.shape == (m, n)
    assert model.b.shape == (m,)

# def test_scale():
#     """Test scaling functionality."""
#     A = np.random.rand(10, 5)
#     b = np.random.rand(10)
#     model = DPModel(A, b)
#     scale_factor = 2.0
#     scale(model, scale_factor)
#     # Note: We can't directly test the internal state, but we can verify it doesn't raise an error

def test_regularize():
    """Test regularization functionality."""
    model = rand_dp_model(m, n)
    λ = 0.1
    regularize(model, λ)
    assert model.model.λ == λ

def test_solve():
    """Test solving functionality."""
    model = rand_dp_model(m, n)
    solution = solve(model, verbose=True)
    assert solution.shape == (n,)
    assert not np.any(np.isnan(solution)) 

def test_rand_dp_model():
    """Test random DPModel creation."""
    model = rand_dp_model(m, n)
    assert model.A.shape == (m, n)
    assert model.b.shape == (m,)

def test_execution_stats():
    """Test that execution statistics are saved after solving."""
    # Create and solve a model
    model = rand_dp_model(m, n)
    solution = solve(model)
    
    # Check that execution_stats is now available
    assert model.execution_stats is not None
    
    # Verify some key properties of execution_stats
    stats = model.execution_stats
    assert hasattr(stats, 'status')
    assert hasattr(stats, 'elapsed_time')
    assert hasattr(stats, 'iter')
    assert hasattr(stats, 'solution')
    assert hasattr(stats, 'primal_obj')
    assert hasattr(stats, 'dual_obj')
    
    # Check that the solution in stats matches the returned solution
    assert np.allclose(np.array(stats.solution), solution)

def test_version():
    """Test version functionality."""
    v = version()
    assert isinstance(v, str)
    assert len(v) > 0
