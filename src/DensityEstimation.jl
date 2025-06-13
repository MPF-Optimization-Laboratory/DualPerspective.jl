module DensityEstimation
using ..DualPerspective
using LinearAlgebra

export moment_operator, reconstruct 

"""
   moment_operator(x, m) -> A

Compute the moment operator for a given points `x` up to order `m`, for moments i=1:m. (The 0th moment is not included and is assumed to be 1.)

# Arguments
- `x`: n-vector of locations.
- `m`: maximum order of moment to compute.

# Returns
- `A`: m x n matrix for moments.
"""
function moment_operator(x, m)
   A = zeros(m, length(x))
#    α = range(0,2,length=m+1)[2:end]
   for k in 1:m
      A[k, :] = x.^k
    #   A[k,:] = x.^α[k]
   end
   return A
end

@doc raw"""
    reconstruct(μvec::Vector, x_grid::Vector; λ=1e-6, kwargs...)

Reconstruct a density from the moments $μ_1, μ_2,\ldots, μ_m$ contained in the m-vector `μvec`.

# Arguments
- `μvec`: m-vector of moments.
- `x_grid`: grid of points at which to evaluate moment operator. 
- `λ`: regularization parameter (default: 1e-6).
- `kwargs`: passed to the solver. See [`DualPerspective.solve!`](@ref) for details.

# Returns
- `density`: function that estimates the density at the points in `xgrid`.
"""
function reconstruct(μvec::Vector, xgrid; λ=1e-6, kwargs...)
    m = length(μvec)
    b = μvec
    A = moment_operator(xgrid, m)
    model = DPModel(A, b, λ=λ)
    status = solve!(model; kwargs...)
    display(status)
    return status.solution
end

function reconstruct(
    μvec,
    bnds::Tuple=(-1.0, +1.0);
    n_points::Int=100,
    kwargs...
)
    xgrid = range(bnds[1], bnds[2], length=n_points)
    return reconstruct(μvec, xgrid; kwargs...)
end

end # module DensityEstimation
