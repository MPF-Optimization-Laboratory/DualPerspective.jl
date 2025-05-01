```@meta
CurrentModule = DualPerspective
```

# User Guide

This guide provides an overview of DualPerspective.jl and instructions for using the package to solve Kullback-Leibler (KL) regularized least squares problems.

## Model Types

DualPerspective.jl provides several model types for different problem scenarios:

1. **DPModel**: The base model for Kullback-Leibler regularized least squares.
2. **SSModel**: A self-scaling model that adapts regularization parameters.
3. **OTModel**: A model specifically designed for optimal transport problems.
4. **LPModel**: A model for linear programming problems.

## Basic Usage

Here's a simple workflow for using DualPerspective.jl:

1. Create a model instance
2. Configure model parameters if needed
3. Solve the model
4. Analyze the results

```julia
using DualPerspective

# Create a model (this is a simple example)
model = DPModel(A, b, c, λ, q)

# Solve the model
result = solve!(model)

# Access the solution
x_optimal = result.x
```

## Working with Results

The `solve!` function returns an `ExecutionStats` object containing:

- `status`: Solver status (`:first_order`, `:max_iter`, etc.)
- `x`: The optimal solution
- `objective`: Final objective value
- `iter`: Number of iterations
- `time`: Solution time
- Additional diagnostic information

You can visualize results using the `histogram` function if UnicodePlots is available:

```julia
using UnicodePlots
histogram(result)
```

## Advanced Options

The `solve!` function accepts several optional arguments:

- `trace`: Boolean to enable iteration output (default: `false`)
- `max_iter`: Maximum number of iterations (default: `100`)
- `atol`: Absolute tolerance for convergence (default: depends on model type)
- `rtol`: Relative tolerance for convergence (default: depends on model type)

```julia
solve!(model, trace=true, max_iter=200, atol=1e-8)
``` 