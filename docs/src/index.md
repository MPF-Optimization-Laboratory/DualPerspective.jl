```@meta
CurrentModule = DualPerspective
```

# DualPerspective.jl

*Efficient solvers for Kullback-Leibler regularized least-squares problems and variations.*

## Overview

This package provides efficient algorithms for solving optimization problems with Kullback-Leibler (KL) regularization, with a focus on linear least-squares formulations

$$\min_{p \in \Delta} \tfrac{1}{2\lambda} \|Ax - b\|^2_{C^{-1}} + \ip{c, x} + \KL(x \mid q),$$

where

$$\KL(x \mid q) =
\begin{cases}
\sum_{j=1}^n x_j \log\left(x_j/q_j\right) & \text{if } x\in\Delta\\
+\infty & \text{otherwise,}
\end{cases}$$

is the KL divergence between densities $x$ and $q$ in the probability simplex

$$\Delta:=\{x\in\R^n_+ \mid  \sum_{j=1}^n x_j = 1\}.$$

The problem data is defined by

-  the linear operator $A$ from $\R^n$ to $\R^m$
-  the observation vector $b\in\R^m$
-  the vector $c\in\R^n$
-  the positive-definite linear operator $C$ on $\R^n$
-  the regularization parameter $\lambda>0$.

The operators $A$ and $C$ can be an explicit matrices or linear maps that implement forward and adjoint products, i.e., `A*x` and `A'*y` with vectors `x` and `y`.

A dual-perspective approach provides for stable and efficient solution of this problem, including important problem classes such as

- linear programming
- optimal transport
- least squares (including simplex constraints)
- Haussdorff moment problems


## Installation

The package is available on the Julia General Registry, and can be installed via

```julia
import Pkg; Pkg.add("DualPerspective")
```

For Python users, the package is available on [PyPI](https://pypi.org/project/DualPerspective/):

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