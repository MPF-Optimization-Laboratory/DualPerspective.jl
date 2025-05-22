```@meta
CurrentModule = DualPerspective.DensityEstimation
CollapsedDocStrings = true
```

# Density Estimation from Moments

The **density-from-moments** problem seeks to reconstruct an unknown probability density from a empirical moment measurements. When the density is supported on the unit interval, the problem is known as the _Haussdorf moment problem_ [gzyl_HausdorffMomentProblem_2010, gzyl_SuperresolutionMaximumEntropy_2017, gzyl_SearchBestApproximant_2007](@cite).

The approach implemented in `DualPerspective.jl` and its submodule `DensityEstimation` is based on the **maximum entropy principle**: among all densities matching the observed moments, select the one with maximal entropy. This yields a robust, principled estimator that converges to the true density as the number of moments increases [borwein_ConvergenceBestEntropy_1991](@cite).

## Formulation

Here we describe the discrete case, where the density is supported on a finite set of locations $x = (x_1, ..., x_n)$.  Given an unknown density $p = (p_1, ..., p_n)$ (with $p_j \geq 0$, $\sum_j p_j = 1$), the first $m$ moments are given by

```math
\mu_i = \sum_{j=1}^n x_j^i p_j, \quad i = 1,\ldots, m.
```

In practice, we don't observe the true moments $\mu_i$, but instead we might estimate these from samples $\{X^{(k)}\}_{k=1}^N$:

```math
\hat{\mu}_i = \frac{1}{N} \sum_{k=1}^N \left(X^{(k)}\right)^i, \quad i = 1,\ldots, m.
```

We then solve the **maximum entropy** problem

```math
\max_{p\in\Delta} \left\{
    \textstyle\sum_j p_j \log p_j/q_j
    \mid
    \ A p \approx b
\right\}
```

where $\Delta$ is the set of discrete densities of length $n$, $A$ is the **moment operator** with entries $A_{ij} = x_j^i$, the $m$-vector $b$ collects the empirical moments $\hat{\mu}_i$, and $q\in\Delta$ is a reference density, i.e., a _prior_ on the unknown density $p$.

The function `reconstruct` solves this problem and returns the estimated density.

```@docs
reconstruct
```

## Example

```@example density
using Distributions
using CairoMakie
using DualPerspective.DensityEstimation

# Generate a mixture of two Gaussians
means, vars, weights = [1.0, 5.0], [0.7, 2.0], [0.4, 0.6]
fMix = MixtureModel([Normal(means[i], vars[i]) for i in 1:2], weights)

# Sample the mixture and compute the empirical moments
samples = rand(fMix, 10000)
b = [mean(samples .^ i) for i in 1:6]

# Setup a grid
bnds = [minimum(samples), maximum(samples)]
Δx = 0.01
x = range(bnds[1], bnds[2], step=Δx)

# Reconstruct the density and rescale to convert from a probability mass to density
p = reconstruct(b, x; λ=1e-8, rtol=1e-10)
p = p / Δx

# Plot the true and estimated densities
fig = Figure()
ax = Axis(fig[1, 1], title="Mixture of two Gaussians")
lines!(ax, bnds[1]..bnds[2], x -> pdf(fMix, x), label="true density")
scatter!(ax, x, p, label="estimated density", markersize=7, color=:red)
axislegend(ax)
fig
```

## Additional Examples

See `experiments/julia/density-moments.jl` for end-to-end examples, including:

- Biased die (discrete)
- Gaussian and Gaussian mixtures (continuous)
- Empirical moment estimation
