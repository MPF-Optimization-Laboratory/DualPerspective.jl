"""
    LogExp{T<:AbstractFloat, V1<:AbstractVector{T}, V2<:AbstractVector{T}}

A structure that implements the log-sum-exp (LSE) function with a prior distribution.

The log-sum-exp function is defined as:
```math
\\phi^*(p \\mid q) = \\log\\left(\\sum_{i=1}^n q_i e^{p_i}\\right)
```

where `q` is a prior probability vector. This function is the convex conjugate of the 
Kullback-Leibler divergence and plays a key role in dual perspective reformulations
of optimization problems.

# Fields
- `q::V1`: Prior probability vector (must be non-negative)
- `g::V2`: Buffer for storing the gradient of the log-sum-exp function

# Usage
Construct with a prior vector:
```julia
lse = LogExp(q)
```
Or with a dimension for uniform prior:
```julia
lse = LogExp(n)
```

Evaluate the function at a point `p`:
```julia
value = obj!(lse, p)
gradient = grad(lse)
```

See also [`obj!`](@ref), [`grad`](@ref), [`hess`](@ref), [`kl_divergence`](@ref).
"""
struct LogExp{T<:AbstractFloat, V1<:AbstractVector{T}, V2<:AbstractVector{T}}
    q::V1  # prior
    g::V2  # buffer for gradient
end

@doc raw"""
    kl_divergence(x, x̄)

Compute the Kullback-Leibler divergence between vectors `x` and `x̄`.

# Implementation Details
- Skips entries where `x_j = 0` 
- Returns zero if `x` and `x̄` are identical
- No checks are performed on the validity of the inputs, i.e., that `x` is a probability vector or that `x̄` is positive.
"""
function kl_divergence(x, x̄)
    result = zero(eltype(x))
    for (xi, x̄i) in zip(x, x̄)
        if xi > 0
            result += xi * (log(xi) - log(x̄i))
        end
    end
    return result
end

"""
    LogExp(q)

Construct a `LogExp` object that implements the log-sum-exp (LSE) function with a prior `q`. Evaluate the LSE function at a point `p` with `obj!(lse, p)`; the corresponding gradient is retrieved with `grad(lse)`. The gradient is stored in an internal buffer `g` of the LogExp object; do not modify this buffer.

If no prior is known, instead provide the dimension `n`:

    LogExp(n)

which will use the uniform prior `q = fill(1/n, n)`.
"""
function LogExp(q::AbstractVector)
    @assert (all(ζ -> ζ ≥ eps(eltype(q)), q)) "prior is not nonnegative"
    LogExp(q, similar(q))
end

 LogExp(n::Int) = LogExp(fill(1/n, n))

"""
    obj!(lse, p) 

Evaluates the value and gradient of the log-sum-exp function

    f(p) = log(sum(q_i e^{p_i} for i in 1:n))

where `q` is the prior probability vector. The gradient is stored in an internal buffer `g` of the LogExp object.

# Implementation
This implementation is adapted from MonteCarloMeasurements.jl to incorporate a prior `q`.
It uses the max-normalization trick for numerical stability against under/overflow:

```math
\\log\\left(\\sum_{i=1}^{n} q_i e^{p_i}\\right) = m + \\log\\left(\\sum_{i=1}^{n} q_i e^{p_i - m}\\right)
```

where `m = \\max_i p_i`.

Reference: https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/4f9b688d298157dc24a5b0a518d971221fbe15dd/src/resampling.jl#L10

See also [`grad`](@ref) and [`hess`](@ref).
"""
function obj!(lse::LogExp, p)
    @unpack q, g = lse
    maxval, maxind = myfindmax(p)
    @. g = q * exp(p - maxval)
    Σ = sum_all_but(g, maxind)
    f = log(Σ + q[maxind]) + maxval
    @. g = g / (Σ + q[maxind])
    return f
end

"""
Find the maximum element of `p` and its index. This is significantly faster than the built-in `findmax`.
"""
function myfindmax(p)
    maxval, maxind = p[1], 1
    @inbounds for i in 2:length(p)
        if p[i] > maxval
            maxval, maxind = p[i], i
        end
    end
    return maxval, maxind
end

"""
    sum_all_but(w, i)

Sum all elements of `w` except the `i`-th element. Used by [`obj!`](@ref).
"""
function sum_all_but(w, i)
    temp = w[i]
    w[i] = zero(eltype(w))
    s = sum(w)
    w[i] = temp
    return s
end

"""
    grad(lse::LogExp)

Get the gradient of the log-sum-exp function at the point `p` where
the `lse` objective was last evaluated.
"""
grad(lse::LogExp) = lse.g

"""
    hess(lse::LogExp)

Get the Hessian of the log-sum-exp function at the point `p` where
the `lse` objective was last evaluated.
"""
function hess(lse::LogExp)
    g = lse.g
    return Diagonal(g) - g * g'
end
