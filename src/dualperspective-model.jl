"""
    DPModel{T<:AbstractFloat, M, CT, SB<:AbstractVector{T}, S<:AbstractVector{T}} <: AbstractNLPModel{T, S}

Dual perspective model for optimization problems, extending `AbstractNLPModel` with KL-regularized least squares functionality.

# Fields
- `A::M`: Constraint matrix defining the linear system
- `b::SB`: Target vector in the linear system Ax ≈ b
- `c::S`: Cost vector for the objective function (defaults to ones)
- `q::S`: Prior distribution vector for KL divergence term (defaults to uniform 1/n)
- `λ::T`: Regularization parameter controlling the strength of the KL term (default: √eps)
- `C::CT`: Positive definite scaling matrix for the linear system (default: Identity)
- `mbuf::S`: First m-dimensional buffer for internal computations
- `mbuf2::S`: Second m-dimensional buffer for internal computations
- `nbuf::S`: n-dimensional buffer for internal computations
- `bNrm::T`: Cached norm of vector b for scaling purposes
- `scale::T`: Problem scaling factor (default: 1.0)
- `lse::LogExpFunction`: Log-sum-exp function for computing KL divergence
- `name::String`: Optional identifier for the problem instance
- `meta::NLPModelMeta{T,S}`: Problem metadata required by NLPModels interface
- `counters::Counters`: Performance counters for operation tracking

# Parameters
- `T`: Floating point type for numerical computations
- `M`: Matrix type for the constraint matrix, must be a subtype of AbstractMatrix
- `CT`: Type of the scaling matrix
- `SB`: Vector type for the right-hand side, must be a subtype of AbstractVector{T}
- `S`: Vector type for solution and workspace vectors, must be a subtype of AbstractVector{T}

# Examples
```julia
# Create a simple dual perspective model
A = randn(10, 5)
b = randn(10)
model = DPModel(A=A, b=b)

# Create model with custom regularization
model = DPModel(A=A, b=b, λ=1e-3)

# Simplified model creation:
model = DPModel(A, b)
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
    mbuf::S = similar(b)
    mbuf2::S = similar(b)
    nbuf::S = similar(q)
    bNrm::T = norm(b)
    scale::T = one(eltype(A))
    lse::LogExpFunction = LogExpFunction(q)
    name::String = ""
    meta::NLPModelMeta{T, S} = begin
        m = size(A, 1)
        NLPModelMeta(m, name="Dual Perspective Model")
    end
    counters::Counters = Counters()
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