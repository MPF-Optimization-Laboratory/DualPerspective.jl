"""
    DPModel

The Dual perspective data type holds the data for the KL-regularized linear least-squares model. It extends the [`AbstractNLPModel` interface](https://jso.dev/NLPModels.jl/stable/reference/#NLPModels.AbstractNLPModel).
    
Instantiate the model keyword arguments, e.g.,
```julia
model = DPModel(A=A, b=b; λ=1e-3, C=I)
```
or using the convenience constructor that makes the first two arguments `A` and `b` required, and infers the other arguments from their sizes, e.g.,
```julia
model = DPModel(A, b; λ=1e-3, C=I)
```

# Keyword fields
- `A` (`AbstractMatrix{T}`, required): Constraint matrix defining the linear system.
- `b` (`AbstractVector{T}`, required): Target vector in the linear system Ax ≈ b.
- `c` (`AbstractVector{T}`, default: ones(n)): Cost vector for the objective function.
- `q` (`AbstractVector{T}`, default: fill(1/n, n)): Prior distribution vector for KL divergence term.
- `λ` (`T`, default: √eps): Regularization parameter controlling the strength of the KL term.
- `scale` (`T`, default: one(eltype(A))): Scaling factor for the problem.
- `C` (`AbstractMatrix{T}`, default: I): Positive definite scaling matrix for the linear system.
- `name` (`String`, default: "Dual Perspective Model"): Optional identifier for the problem instance.

# Examples

Create a simple dual perspective model:
```julia
A, b = randn(10, 5), randn(10)
model = DPModel(A=A, b=b)
```
"""
@kwdef mutable struct DPModel{T<:AbstractFloat, M, CT, SB<:AbstractVector{T}, S<:AbstractVector{T}} <: AbstractNLPModel{T, S}
    A::M
    b::SB
    c::S = begin
              m, n = size(A)
              c = ones(eltype(A), n)
            end
    q::S = begin
             m, n = size(A)
             q = similar(b, n)
             q .= 1/n
           end
    λ::T = √eps(eltype(A))
    C::CT = I
    mbuf::S = similar(b) # first m-dimensional buffer for internal computations
    mbuf2::S = similar(b) # second m-dimensional buffer for internal computations
    nbuf::S = similar(q) # n-dimensional buffer for internal computations
    bNrm::T = norm(b) # cached norm of vector b for scaling purposes
    scale::T = one(eltype(A)) # problem scaling factor τ
    lse::LogExp = LogExp(q) # log-sum-exp function for computing KL divergence
    name::String = "Dual Perspective Model" # optional identifier for the problem instance
    meta::NLPModelMeta{T, S} = begin # problem metadata required by NLPModels interface
        m = size(A, 1)
        NLPModelMeta(m, name="Dual Perspective Model")
    end
    counters::Counters = Counters() # performance counters for operation tracking
end

DPModel(A, b; kwargs...) = DPModel(A=A, b=b; kwargs...)

function Base.show(io::IO, kl::DPModel)
    println(io, "KL regularized least-squares"*
                (kl.name == "" ? "" : ": "*kl.name))
    println(io, @sprintf("   m = %10d  bNrm = %7.1e", size(kl.A, 1), kl.bNrm))
    println(io, @sprintf("   n = %10d  λ    = %7.1e", size(kl.A, 2), kl.λ))
    println(io, @sprintf("       %10s  τ    = %7.1e"," ", kl.scale))
end

"""
    regularize!(kl::DPModel{T}, λ::T) where T

Set the regularization parameter of the Perspectron model.
"""
function regularize!(kl::DPModel{T}, λ::T) where T
    kl.λ = λ
    return kl
end

"""
    scale(kl::DPModel)

Get the scaling factor of the Perspectron model.
"""
scale(kl::DPModel) = kl.scale

"""
    scale!(kl::DPModel{T}, scale::T) where T

Set the scaling factor of the Perspectron model.
"""
function scale!(kl::DPModel{T}, scale::T) where T
    @assert scale > 0 "Scale must be positive"
    kl.scale = scale
    return kl
end

function update_y0!(kl::DPModel{T}, y0::AbstractVector{T}) where T
    kl.meta = NLPModelMeta(kl.meta, x0=y0)
end

function NLPModels.reset!(kl::DPModel)
    for f in fieldnames(Counters)
      setfield!(kl.counters, f, 0)
    end
    return kl
end 