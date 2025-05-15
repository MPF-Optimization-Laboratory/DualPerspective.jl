##
using LinearAlgebra
using Statistics: mean, std
using Random: seed!
using Distributions
using DualPerspective
using DualPerspective.DensityEstimation
using CairoMakie
using QuadGK

##
"""
   gaussian_mixture([K=3], [means=range(-3.0, 3.0, length=K)], [vars=ones(K)], [weights=fill(1.0/K, K)])

Create a Gaussian mixture model with `K` components.

# Arguments
- `K`: Number of Gaussian components (default: 3)
- `means: Vector of means for each component.
- `vars: Vector of variances for each component.
- `weights: Vector of weights for each component.

# Returns
- A MixtureModel object from Distributions.jl

# Example
```julia
mix = gaussian_mixture() # default 3 components

# Create custom mixture
mix = gaussian_mixture(2, [1.0, 5.0], [0.5, 2.0], [0.3, 0.7])
```
"""
function gaussian_mixture(;
      K=3, 
      means=range(-3.0, 3.0, length=K), 
      vars=ones(K), 
      weights=fill(1.0/K, K))
   components = [Normal(means[i], vars[i]) for i in 1:K]
   return MixtureModel(components, weights)
end

"""
    estimate_mixture_bounds(mixture::MixtureModel, k_std=3.0)

Estimates practical lower and upper bounds for a Gaussian mixture model based on the
means and standard deviations of its components.

# Arguments
- `mixture`: The `MixtureModel` containing Gaussian components.
- `k_std`: Number of standard deviations to extend beyond the min/max means.

# Returns
- A tuple `(min_bound, max_bound)`.
"""
function estimate_mixture_bounds(mixture::MixtureModel, k_std=3.0)
    isempty(mixture.components) && return (0.0, 0.0)
    min_val = +Inf
    max_val = -Inf
    for comp in mixture.components
        μ = mean(comp)
        σ = std(comp)
        min_val = min(min_val, μ - k_std * σ)
        max_val = max(max_val, μ + k_std * σ)
    end
    if min_val ≥ max_val
        min_val = max_val - 1.0
        max_val = min_val + 1.0
    end
    return min_val, max_val
end

"""
    moments(f, k; bnds=(-Inf, Inf))

Compute the first `k` moments (μ₁, ..., μₖ) of the density `f` over the interval `bnds` using numerical integration (quadrature).

# Returns
- Vector of length k containing the computed moments

# Example
```julia
moments = moments(x -> exp(-x^2/2)/sqrt(2π), 3, bnds=(-1, 1))
```
"""
function moments(f::Function, m::Int, bnds=(-Inf, Inf))
    x_min, x_max = bnds
    b = zeros(m)
    for i in 1:m
        mom(x) = x^i * f(x)
        b[i], _ = quadgk(mom, x_min, x_max)
    end
    return b
end

"""
    moments(f::Distribution, m::Int, N::Int)

Estimate the first `m` moments of the distribution `f` by sampling `N` points from the distribution.
"""
function moments(f::Distribution, m::Int, N::Int)
   samples = rand(f, N)
   return [mean(samples.^i) for i in 1:m]
end

"""
    density_experiment(f::Distribution, N::Int, m::Int, bnds::Tuple, Δx::Float64, title::String="")

Perform a density estimation experiment by reconstructing a distribution from its moments.

# Arguments
- `f::Distribution`: The true distribution to compare against
- `N::Int`: Number of samples to draw from the distribution
- `m::Int`: Number of moments to compute
- `bnds::Tuple`: Bounds for the x-axis (min, max)
- `Δx::Float64`: Step size for the x-grid
- `title::String`: Title for the plot (default: "")

# Returns
- `Figure`: A plot comparing the true and estimated densities
"""
function density_experiment(f::Distribution, N::Int, m::Int, bnds::Tuple, Δx::Float64, title::String=""; λ=1e-6, kwargs...)
    b = moments(f, m, N)
    xgrid = range(bnds[1], bnds[2], step=Δx)
    p_computed = reconstruct(b, xgrid; λ=λ, kwargs...)

    # Scale the density to match the continuous distribution
    p_density_scale = p_computed / Δx

    # Plot the distribution `f` against the estimated density
    fig = Figure()
    ax = Axis(fig[1, 1], title=title * " (N=$(N), m=$(m))")
    lines!(ax, bnds[1]..bnds[2], x -> pdf(f, x), label="true density")
    scatter!(ax, xgrid, p_density_scale, label="estimated density", markersize=7, color=:red)
    axislegend(ax)
    
    # Set limits based on the bounds
    ax.limits = (bnds[1], bnds[2], nothing, nothing)
    
    return fig
end

################################################
## Biased Die
################################################
p0 = [1/12, 1/12, 1/12, 2/12, 3/12, 4/12]
x = collect(1:6)
b = [x'p0] # 1st moment
p_computed = reconstruct(b, x)
println("μ from true distribution: $(x'p0)")
println("μ from estimated distribution: $(x'p_computed)")

################################################
## Gaussian Reconstruction
################################################
seed!(42)
fN = Normal(1.0, 0.2)
fig = density_experiment(fN, 1000, 2, (0, 2), 0.1, "Gaussian")
display(fig)

################################################
## Gaussian Mixture (2 components)
################################################
seed!(42)
fMixture = gaussian_mixture(K=2, means=[1.0, 5.0], vars=[0.5, 2.0], weights=[0.3, 0.7])
fig = density_experiment(
    fMixture,
    100000, # number of samples
    5, # number of moments
    bnds, # bounds
    0.2, # step size
    "Gaussian Mixture";
    λ=1e-9, # regularization parameter
    logging=0, # logging
    rtol=1e-11, # relative tolerance
)
display(fig)

################################################
## Gaussian Mixture (3 components)
################################################
seed!(42)
fMixture = gaussian_mixture()
bnds = estimate_mixture_bounds(fMixture)
fig = density_experiment(
    fMixture,
    100000, # number of samples
    10, # number of moments
    bnds, # bounds
    0.2, # step size
    "Gaussian Mixture";
    λ=1e-9, # regularization parameter
    logging=1, # logging
    rtol=1e-11, # relative tolerance
)
display(fig)
