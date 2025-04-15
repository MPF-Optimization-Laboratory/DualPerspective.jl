struct LogExpFunction{T<:AbstractFloat, V1<:AbstractVector{T}, V2<:AbstractVector{T}}
    q::V1  # prior
    g::V2  # buffer for gradient
end

"""
    kl_divergence(x, q) -> T where T<:AbstractFloat

Compute the Kullback-Leibler divergence between probability distributions x and q:

```math
KL(x || q) = ∑_i x_i \\log(x_i / q_i) \\quad \\text{for} \\quad x_i > 0
```

# Implementation Details
- Computes KL divergence as `sum(x_i * (log(x_i) - log(q_i)))` for numerical stability
- Skips entries where x_i = 0` 
- Returns zero if x and q are identical
"""
function kl_divergence(x, q)
    result = zero(eltype(x))
    for (xi, qi) in zip(x, q)
        if xi > 0
            result += xi * (log(xi) - log(qi))
        end
    end
    return result
end

"""
Constructor for the LogExpFunction object.

If an n-vector of priors `q` is available:

    LogExpFunction(q)
    
If no prior is known, instead provide the dimension `n`:

    LogExpFunction(n)
"""
function LogExpFunction(q::AbstractVector)
    # @assert (all(ζ -> ζ ≥ 0, q) && sum(q) ≈ 1) "prior is not on the simplex"
    @assert (all(ζ -> ζ ≥ 0, q)) "prior is not nonnegative"
    LogExpFunction(q, similar(q))
end

# TODO: this constructor doesn't respect type stability
# LogExpFunction(n::Int) = LogExpFunction(fill(1/n, n))

"""
    obj!(lse, p) 

Evaluates the value and gradient of the log-sum-exp function

    f(p) = log(sum(q_i e^{p_i} for i in 1:n))

where `q` is the prior probability vector. The gradient is stored in an internal buffer `g` of the LogExpFunction object.

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
function obj!(lse::LogExpFunction, p)
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
Special all-but-i sum used by logΣexp.
"""
function sum_all_but(w, i)
    temp = w[i]
    w[i] = zero(eltype(w))
    s = sum(w)
    w[i] = temp
    return s
end

"""
    grad(lse::LogExpFunction)

Get the gradient of the log-sum-exp function at the point `p` where
the `lse` objective was last evaluated.
"""
grad(lse::LogExpFunction) = lse.g

"""
    hess(lse::LogExpFunction)

Get the Hessian of the log-sum-exp function at the point `p` where
the `lse` objective was last evaluated.
"""
function hess(lse::LogExpFunction)
    g = lse.g
    return Diagonal(g) - g * g'
end
