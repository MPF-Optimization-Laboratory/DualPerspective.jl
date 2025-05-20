"""
Solve the LP problem:

    min ⟨c, x⟩
    subject to Ax = b, x ≥ 0

using the self-scaled solver by solving the regularized problem:

    minimize_{x ∈ ℝ₊ⁿ} (1/(2λ)) * ||Ax - b||² + ⟨c, x⟩ + ε * H(x)

which is equivalent to:

    minimize_{x ∈ ℝ₊ⁿ} (1/(2λε)) * ||Ax - b||² + ⟨c/ε, x⟩ + H(x)

where ε is the relaxation constant, λ is the feasibility constant, and H(x) is the entropy function.
"""
struct LPModel{T, KL<:DPModel{T}}
    kl::KL   # Self-scaled model
    ε::T     # Relaxation constant
    λ::T     # Feasibility constant. TODO: `kl` already has a `λ` field. Remove this and update references to it.
end

"""
Constructs an `LPModel` for the LP problem.

# Arguments
- `A`: Constraint matrix
- `b`: Right-hand side vector
- `c`: Cost vector
- `ε`: Relaxation constant
- `λ`: Feasibility constant
- `kwargs`: Additional keyword arguments

# Returns
An instance of `LPModel`.
"""
function LPModel(
    A,
    b,
    c;
    ε=1e-3,
    λ=1e-3,
    kwargs...
)
    kl = DPModel(A=A, b=b, c=c/ε, λ=λ, kwargs...)
    regularize!(kl, ε * λ)
    return LPModel(kl, ε, λ)
end

"""
Custom display for `LPModel` instances.
"""
function Base.show(io::IO, lp::LPModel)
    println(io, "KL regularized LP" *
                (lp.kl.name == "" ? "" : ": " * lp.kl.name))
    println(io, @sprintf("   m = %10d  bNrm = %7.1e", size(lp.kl.A, 1), lp.kl.bNrm))
    println(io, @sprintf("   n = %10d  λ    = %7.1e", size(lp.kl.A, 2), lp.kl.λ))
    println(io, @sprintf("       %10s  ε    = %7.1e", " ", lp.ε))
end

"""
Adjusts the feasibility constant `λ` in the `LPModel`.

# Arguments
- `lp`: The `LPModel` instance.
- `λ`: New feasibility constant.
"""
function regularize_feasability!(lp::LPModel, λ)
    regularize!(lp.kl, λ * lp.ε)
    lp.λ = λ
end

"""
Adjusts the relaxation constant `ε` in the `LPModel`.

# Arguments
- `lp`: The `LPModel` instance.
- `ε`: New relaxation constant.
"""
function regularize_relaxation!(lp::LPModel, ε)
    regularize!(lp.kl, lp.λ * ε)
    BLAS.scal!(1/ε, lp.kl.c)
    lp.ε = ε
end

"""
Resets the `LPModel` to its initial state.
"""
NLPModels.reset!(lp::LPModel) = NLPModels.reset!(lp.kl)

"""
Solves the LP problem using the self-scaled solver.

# Arguments
- `lp`: The `LPModel` instance.
- `logging`: Level of logging detail (default: `0`).
- `monotone`: Enforce monotonicity (default: `true`).
- `max_time`: Maximum solving time in seconds (default: `30.0`).
- `kwargs`: Additional keyword arguments.

# Returns
A statistics object containing the solution and status.

# Notes
- If the solution is found but the residual norm exceeds `1e-1`, the status is set to `:infeasible`.
"""
solve!(lp::LPModel; kwargs...) = solve!(lp, SequentialSolve(); kwargs...)

function solve!(
    lp::LPModel{T},
    Solver; 
    kwargs...
) where T
    klls_stats = solve!(
        lp.kl,
        Solver;
        kwargs...
    )

    if klls_stats.status == :optimal && norm(klls_stats.residual) > 1e-1
        klls_stats.status = :infeasible
    end

    return klls_stats
end
