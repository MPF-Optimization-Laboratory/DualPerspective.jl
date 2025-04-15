![Latest Release](https://img.shields.io/github/v/release/MPF-Optimization-Laboratory/DualPerspective.jl?include_prereleases)
![PyPI - Version](https://img.shields.io/pypi/v/DualPerspective)
[![Run_Tests](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl/actions/workflows/run-tests.yml)


# DualPerspective

This package provides an algorithm for solving Kullback-Leibler (KL) regularized least squares problems of the form

$$
\min_{p \in \mathcal{X}} \frac{1}{2\lambda} \|Ax - b\|^2 + \langle c, x \rangle + \mathop{KL}(x \mid q),
$$

where $\mathcal{X}$ is either
- the probability simplex: $\Delta := \{ x∈ℝ^n_+ \mid ∑_j x_j=1\}$) or,
- the nonnegative orthant $ℝ^n_+$.

The algorithm is based on a trust-region Newton CG method on the dual problem.

## Release v0.1.3

The latest release (v0.1.3) includes:
- Official Julia registry support
- Python package available on PyPI

## Usage

### Installation

You can install this package directly from Julia's package manager:

```julia
import Pkg; Pkg.add("DualPerspective")
```

For Python users, install from PyPI:

```bash
pip install DualPerspective
```

## Examples

To solve a simple optimal transport problem:

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

## Extensions

DualPerspective.jl provides the following optional extensions:

### UnicodePlots Extension

Provides plotting capabilities using the UnicodePlots package. To use:

```julia
using DualPerspective, UnicodePlots
# Now you can plot DualPerspective.ExecutionStats objects
histogram(solution)
```

## Developing

### Getting Started

To get started with developing the package, first pull the Github repo

```shell
git clone https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl.git
```

Or better, [create fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) to develop!

### Usage

To be able to use the library in the julia REPL, navigate to the library root folder, and run

```bash
julia --project=. -i -e 'using Pkg; Pkg.activate(".")'
```

This will start the julia REPL, and add the repo to the julia `LOAD_PATH`. To start using, run

```julia
using DualPerspective
```

#### Using in a notebook

To use this package in a notebook while developing, add it manually to your julia `LOAD_PATH` by
```julia
push!(LOAD_PATH, "/path/to/the/library")
```

### Testing

Each pull requested will be tested against the current tests, and new tests are highly encouraged for any new piece of code. To run the tests, run

```bash
julia --project=. -i -e 'using Pkg; Pkg.test()'
```
