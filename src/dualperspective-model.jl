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
              c = -ones(eltype(A), n)
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
    box_width = 58  # Adjusted width for better formatting
    
    # Box top
    println(io, "┌" * "─"^(box_width-2) * "┐")
    
    # Title with centered text
    title_padding = max(0, div(box_width - 2 - length(kl.name), 2))
    right_padding = max(0, box_width - 2 - length(kl.name) - title_padding)
    title_line = "│" * " "^title_padding * kl.name * " "^right_padding * "│"
    println(io, title_line)
    
    # Box separator
    println(io, "├" * "─"^(box_width-2) * "┤")
    
    # Model dimensions and parameters with fixed formatting
    println(io, @sprintf("│ %-15s = %8s    %-4s = %10.3e        │", "num rows (m)", string(size(kl.A, 1)), "‖b‖₂", kl.bNrm))
    println(io, @sprintf("│ %-15s = %8s    %-4s = %10.3e        │", "num cols (n)", string(size(kl.A, 2)), "λ", kl.λ))
    println(io, @sprintf("│ %-15s = %8s    %-4s = %10.3e        │", "element type", string(typeof(kl.λ)), "τ", kl.scale))
    
    # Check if C is a UniformScaling matrix
    if kl.C isa UniformScaling
        println(io, @sprintf("│ %-15s = %-8s (%10.3e)                │", "covariance (C)", "uniform", kl.C.λ))
    else
        println(io, @sprintf("│ %-15s = %-8s                         │", "covariance (C)", "custom"))
    end
    
    # Check if q is a uniform vector (all elements are approximately equal)
    if length(kl.q) > 0 && all(x -> x ≈ first(kl.q), kl.q)
        q_val = first(kl.q)
        println(io, @sprintf("│ %-15s = %-8s (%10.3e)              │", "prior (q)", "uniform", q_val))
    else
        q_min, q_max = extrema(kl.q)
        println(io, @sprintf("│ %-15s = %-8s [%8.3e, %8.3e]      │", "prior (q)", "custom", q_min, q_max))
    end
    
    # Check if c is a constant vector (all elements are the same)
    if length(kl.c) > 0 && all(x -> x ≈ first(kl.c), kl.c)
        c_val = first(kl.c)
        println(io, @sprintf("│ %-15s = %-8s (%10.3e)                │", "cost (c)", "constant", c_val))
    else
        c_min, c_max = extrema(kl.c)
        println(io, @sprintf("│ %-15s = %-8s [%8.3e, %8.3e] │", "cost (c)", "variable", c_min, c_max))
    end
    
    # Box bottom
    println(io, "└" * "─"^(box_width-2) * "┘")
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