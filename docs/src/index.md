```@meta
CurrentModule = DualPerspective
```

# DualPerspective.jl

*Efficient solvers for Kullback-Leibler regularized least squares problems and variations.*

## Overview

This package provides algorithms for solving Kullback-Leibler (KL) regularized least squares problems of the form:

$$\min_{p \in \mathcal{X}} \tfrac{1}{2\lambda} \|Ax - b\|^2 + \langle c, x \rangle + \rent(x \mid q)$$

where $A$ is a linear operator from $\R^n$ to $\RR^m$, $b$ is an $m$-vector of observations, $c$ is an $n$-vector, $\lambda$ is a positive regularization parameter, and $q$ is an $n$-vector with strictly positive entries.


where $\mathcal{X}$ is either:
- The probability simplex: $\Delta := \{ x∈ℝ^n_+ \mid ∑_j x_j=1\}$
- The nonnegative orthant $ℝ^n_+$

The algorithm is based on a trust-region Newton CG method applied to the dual problem.

## Installation

```julia
import Pkg; Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl")
```

For Python users, the package is available on PyPI:

```bash
pip install DualPerspective
```

## Quick Example

Here's how to solve a simple optimal transport problem:

```julia
using DualPerspective, LinearAlgebra, Distances

μsupport = νsupport = range(-2, 2; length=100)
C = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)           # Cost matrix
μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)                    # Start distribution
ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)   # Target distribution

ϵ = 0.01*median(C)                 # Entropy regularization constant
ot = DualPerspective.OTModel(μ, ν, C, ϵ)      # Model initialization
solution = solve!(ot, trace=true)   # Solution to the OT problem          
```

## Contents

```@contents
Pages = [
    "guide.md",
    "examples.md",
    "api.md",
]
Depth = 1
```

## Documentation Development

If you're contributing to the documentation, you can preview your changes locally with automatic updating:

### Setting Up Live Previews

1. Make sure you're in the `docs` directory
2. Activate the docs environment and add required packages:
   ```julia
   julia> using Pkg
   julia> Pkg.activate(".")
   julia> Pkg.add(["LiveServer", "Revise"])
   ```

3. Run the preview script:
   ```julia
   julia> include("preview.jl")
   ```

4. The script will:
   - Build the documentation 
   - Start a server and open your browser to http://localhost:8000/
   - Watch for changes in both the documentation files and source code docstrings
   - Automatically rebuild the documentation and refresh the browser when changes are detected

5. You can edit:
   - Documentation markdown files in `docs/src/`
   - Source code docstrings in `src/`
   - Both will trigger automatic rebuilds and browser refresh

This provides a smooth workflow for documentation development, with real-time previews as you make changes. 