import numpy as np
from juliacall import Main as jl
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Path
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT_DIR = "/Users/mpf/Documents/projects/Software/DualPerspective.jl"

# Useful for development: environment variable to install the package from the local directory.
USE_LOCAL = os.environ.get('DUALPERSPECTIVE_USE_LOCAL', '').lower() in ('true', '1', 'yes')

def _initialize_julia():
    """Initialize Julia and load DualPerspective."""
    try:
        if USE_LOCAL:
            # Use local Julia project
            jl.seval(f"""
                import Pkg
                Pkg.develop(path="{ROOT_DIR}")
                """)
        else:
            # Use registry version
            jl.seval("""
                import Pkg
                if !haskey(Pkg.project().dependencies, "DualPerspective")
                    Pkg.add("DualPerspective")
                end
                """)

        jl.seval("""
            using DualPerspective
            solve = DualPerspective.solve!
            scale = DualPerspective.scale!
            regularize = DualPerspective.regularize!
            """)
                
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Julia or install DualPerspective: {str(e)}")

# Initialize on module import
_initialize_julia()

class DPModel:
    """Python wrapper for DualPerspective.jl's DPModel."""
    
    def __init__(self, A, b, q=None, C=None, c=None, λ=None):
        """
        Initialize a DPModel.
        
        Args:
            A: Matrix of shape (m, n)
            b: m-vector
            q: Optional prior vector of length n (default: ones(n)/n)
            C: Optional covariance matrix of shape (n, n)
            c: Optional n-vector for linear term (default: ones(n))
            λ: Optional regularization parameter
        """
        # Convert numpy arrays to Julia arrays
        A_jl = jl.convert(jl.Matrix, A)
        b_jl = jl.convert(jl.Vector, b)
        
        kwargs = {}
        if q is not None:
            kwargs['q'] = jl.convert(jl.Vector, q)
        if C is not None:
            kwargs['C'] = jl.convert(jl.Matrix, C)
        if c is not None:
            kwargs['c'] = jl.convert(jl.Vector, c)
        if λ is not None:
            kwargs['λ'] = λ
            
        self.model = jl.DPModel(A_jl, b_jl, **kwargs)
        # Initialize stats to None
        self.stats = None

    @property
    def A(self):
        return np.array(self.model.A)

    @property
    def b(self):
        return np.array(self.model.b)

    @property
    def execution_stats(self):
        """
        Get the execution statistics from the last solve operation.
        
        Returns:
            The ExecutionStats object from the last solve, or None if solve hasn't been called.
        """
        return self.stats

    @classmethod
    def from_julia_model(cls, julia_model):
        """
        Create a DPModel directly from a Julia DPModel object.
        
        Args:
            julia_model: A Julia DPModel object
            
        Returns:
            DPModel: A Python wrapper for the Julia model
        """
        instance = cls.__new__(cls)
        instance.model = julia_model
        instance.stats = None
        return instance

def solve(model, atol=1e-6, rtol=1e-6, verbose=False, logging=0):
    """
    Solve the DualPerspective problem using SequentialSolve algorithm.
    
    Args:
        model: DualPerspectiveModel instance
        atol: Absolute tolerance
        rtol: Relative tolerance
        verbose: Whether to print DualPerspective logging information
        logging: Whether to print DualPerspective logging information
    Returns:
        numpy array containing the solution
    """
    t0 = sum(model.model.q)
    s_model = jl.SequentialSolve()
    result = jl.solve(
        model.model,
        s_model,
        t=t0,
        atol=atol,
        rtol=rtol,
        zverbose=verbose,
        logging=logging
    )
    
    # Save the execution stats to the model
    model.stats = result
    
    return np.array(result.solution)

def scale(model, scale_factor):
    """
    Scale the problem.
    
    Args:
        model: KLLSModel instance
        scale_factor: Scaling factor
    """
    jl.scale(model.model, scale_factor)

def regularize(model, λ):
    """
    Set the regularization parameter.
    
    Args:
        model: DualPerspectiveModel instance
        λ: Regularization parameter
    """
    jl.regularize(model.model, λ)

def rand_dp_model(m, n, λ=1e-3):
    """
    Create a random DPModel with dimensions m x n.
    
    Args:
        m: Number of rows
        n: Number of columns
        λ: Regularization parameter (default: 1e-3)
        
    Returns:
        A DPModel instance with random data
    """
    julia_model = jl.randDPModel(m, n, λ=λ)
    return DPModel.from_julia_model(julia_model)

def version():
    """
    Get the version of the DualPerspective package.
    """
    return jl.DualPerspective.version()
