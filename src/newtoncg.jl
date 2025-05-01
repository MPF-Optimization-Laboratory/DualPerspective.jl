"""
    lseatyc!(kl::DPModel{T}, y) -> T where T<:AbstractFloat

Compute the log-sum-exp term `f(y) = logΣexp(A'y - c)` used in the dual objective function.

# Arguments
- `kl`: A `DPModel` containing model parameters and buffers
- `y`: Vector at which to evaluate the log-sum-exp term

# Details
This function calculates `logΣexp(A'y - c)` by:
1. Copying `c` to internal buffer `nbuf`
2. Computing `nbuf = A'y - c` in-place using matrix-vector multiplication
3. Evaluating the log-sum-exp operation via `obj!(lse, nbuf)`

The gradient of the log-sum-exp term is automatically computed and stored in the 
`lse` internal buffer, accessible via `grad(lse)`.

# Returns
- The scalar value of the log-sum-exp term, with type matching the model's type parameter T

# Note
This is an in-place operation that modifies internal buffers of the model.
"""
function lseatyc!(kl, y)
    @unpack A, c, nbuf, lse = kl
    nbuf .= c
    LinearAlgebra.mul!(nbuf, A', y, 1, -1)
    return obj!(lse, nbuf)
end

"""
    dObj!(kl::DPModel{T}, y) -> T where T<:AbstractFloat

Compute the dual objective function value at the vector `y`:

    d(y) = -(b∙y - 0.5λ y∙Cy - τ log∑exp(A'y - c) - τlogτ) 

The scale parameter `τ` is taken from the `scale` field of `kl`.

# Returns
- The scalar value of the dual objective, with type matching the model's type parameter `T`.

!!! warning "Objective sign"
    This function implements a dual objective based on **minimization**. 

"""
function dObj!(kl::DPModel, y)
    @unpack b, λ, C, scale = kl 
    increment!(kl, :neval_jtprod)
    d = lseatyc!(kl, y)
    return scale*d - scale*log(scale) + 0.5λ*dot(y, C, y) - b⋅y
end

NLPModels.obj(kl::DPModel, y) = dObj!(kl, y)

"""
Dual objective gradient

   ∇f(y) = τ A∇log∑exp(A'y-c) + λCy - b 

evaluated at `y`. Assumes that the objective was last evaluated at the same point `y`.
"""
function dGrad!(kl::DPModel, y, ∇f)
    @unpack A, b, λ, C, lse, scale = kl
    increment!(kl, :neval_jprod)
    p = grad(lse)
    ∇f .= -b
    if λ > 0
        LinearAlgebra.mul!(∇f, C, y, λ, 1)
    end
    LinearAlgebra.mul!(∇f, A, p, scale, 1)
    return ∇f
end

NLPModels.grad!(kl::DPModel, y, ∇f) = dGrad!(kl, y, ∇f)

"""
Dual objective value and gradient
"""
function dObjGrad!(kl::DPModel, ∇f, y)
    @unpack A, b, λ, C, lse, scale = kl
    increment!(kl, :neval_jprod)
    p = grad(lse)
    ∇f .= -b
    if λ > 0
        LinearAlgebra.mul!(∇f, C, y, λ, 1)
    end
    LinearAlgebra.mul!(∇f, A, p, scale, 1)

    increment!(kl, :neval_jtprod)
    d = lseatyc!(kl, y)
    return scale*d - scale*log(scale) + 0.5λ*dot(y, C, y) - b⋅y 
end

"""
Dual objective gradient
"""
function dHess(kl::DPModel)
    @unpack A, λ, C, lse, scale = kl
    H = hess(lse)
    ∇²dObj = scale*(A*H*A')
    if λ > 0
        ∇²dObj += λ*C
    end
    return ∇²dObj
end

"""
    dHess_prod!(kl::DPModel{T}, z, Hz) where T

Product of the dual objective Hessian with a vector `z`

    Hz ← ∇²d(y)z = τ A∇²log∑exp(A'y)Az + λCz,

where `y` is the point at which the objective was last evaluated.
"""
function dHess_prod!(kl::DPModel, z, Hz)
    @unpack A, λ, C, nbuf, lse, scale = kl
    w = nbuf
    increment!(kl, :neval_jprod)
    increment!(kl, :neval_jtprod)
    g = grad(lse)
    LinearAlgebra.mul!(w, A', z)                 # w =                  A'z
    w .= g.*(w .- (g⋅w))           # w =        (G - gg')(A'z)
    LinearAlgebra.mul!(Hz, A, w, scale, 0)       # v = scale*A(G - gg')(A'z)
    if λ > 0
        LinearAlgebra.mul!(Hz, C, z, λ, 1)       # v += λCz
    end
    return Hz
end

function NLPModels.hprod!(kl::DPModel{T}, ::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(T)) where T
    return Hz = dHess_prod!(kl, z, Hz)
end

"""
    pObj!(kl::DPModel, x)

Compute the primal objective function value of the problem defined by `kl` at point `x`.

# Returns
- The scalar value of the primal objective, with type matching the model's type parameter T

!!! note
    Evaluating the least-squares residual term requires solving a system of linear equations involving the covariance matrix `C`, which is currently computed using the `\\` operator, i.e., `C \\ r`.

"""
function pObj!(kl::DPModel, x)
    @unpack A, b, c, C, q, λ, mbuf, mbuf2 = kl
    r, r2 = mbuf, mbuf2

    # Compute quadratic term ⟨Ax - b, C⁻¹(Ax - b)⟩
    r .= b
    mul!(r, A, x, 1, -1)
    r2 .= C \ r 
    quadratic_term = dot(r, r2)

    return (1/(2λ)) * quadratic_term + dot(c, x) + kl_divergence(x, q)
end

function solve!(
    kl::DPModel{T};
    M=I,
    logging=0,
    max_time::Real=30,
    reset_counters=true,
    solver=TrunkSolver(kl),
    kwargs...) where T
   
    # Reset counters
    if reset_counters
        reset!(kl)    
    end

    # Tracer    
    tracer = DataFrame(iter=Int[], dual_obj=T[], r=T[], Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])

    # RSFN
    f(y) = dObj!(kl, y)
    fg!(grads, y) = dObjGrad!(kl, grads, y)
    H = x -> LinearOperator(T, length(kl.y0), length(kl.y0), true, true, (res, z) -> dHess_prod!(kl, z, res))

    rsfn_stats = rsfn!(kl.y0, f, fg!, H,
        mode=:RNSolver,
        itmax=typemax(Int),
        time_limit=Float64(max_time),
        atol=DEFAULT_PRECISION(T),
        rtol=DEFAULT_PRECISION(T),
        linesearch=true)

    primal_solution = kl.scale .* grad(kl.lse)

    stats = ExecutionStats(
        rsfn_stats.converged,
        rsfn_stats.run_time,
        rsfn_stats.iterations,
        neval_jprod(kl),
        neval_jprod(kl),
        pObj!(kl, primal_solution),
        rsfn_stats.f_seq[end],
        primal_solution,
        (kl.λ).*(kl.y0),
        rsfn_stats.g_seq[end],
        tracer
    )

    # Tracer
    # tracer = DataFrame(iter=Int[], dual_obj=T[], r=T[], Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])
    
    # Callback routine
    # cb(kl, solver, stats) =
        # callback(kl, solver, M, stats, tracer, logging, max_time; kwargs...)
    
    # Call the Trunk solver
    # trunk_stats = SolverCore.solve!(
    #     solver, kl; 
    #     M=M, 
    #     callback=cb, 
    #     atol=zero(T), 
    #     rtol=zero(T), 
    #     max_time=Float64(max_time)
    # )

    # primal_solution = kl.scale .* grad(kl.lse)
    
    # stats = ExecutionStats(
    #     trunk_stats.status,
    #     trunk_stats.elapsed_time,       # elapsed time
    #     trunk_stats.iter,               # number of iterations
    #     neval_jprod(kl),                # number of products with A
    #     neval_jtprod(kl),               # number of products with A'
    #     pObj!(kl, primal_solution),     # primal objective
    #     trunk_stats.objective,          # dual objective
    #     primal_solution,                # primal solution `x`
    #     (kl.λ).*(trunk_stats.solution), # residual r = λy
    #     trunk_stats.dual_feas,          # norm of the gradient of the dual objective
    #     tracer
    # )
end
const newtoncg = solve!

function callback(
    kl::DPModel{T},
    solver,
    M,
    trunk_stats,
    tracer,
    logging,
    max_time;
    atol::T = DEFAULT_PRECISION(T),
    rtol::T = DEFAULT_PRECISION(T),
    max_iter::Int = typemax(Int),
    trace::Bool = false,
    ) where T
    
    dObj = trunk_stats.objective 
    iter = trunk_stats.iter
    r = trunk_stats.dual_feas # = ||∇ dual obj(x)||
    # r = norm(solver.gx)
    Δ = solver.tr.radius
    actual_to_predicted = solver.tr.ratio
    cgits = solver.subsolver.stats.niter
    cgexit = get(cg_msg, solver.subsolver.stats.status, "default")
    ε = atol + rtol * kl.bNrm
    
    # Test exit conditions
    tired = iter >= max_iter
    optimal = r < ε 
    done = tired || optimal
    
    log_items = (iter, dObj, r, Δ, actual_to_predicted, cgits, cgexit) 
    trace && push!(tracer, log_items)
    if logging > 0 && iter == 0
        println("\n", kl)
        println("Solver parameters:")
        @printf("   atol = %7.1e  max time (sec) = %7d\n", atol, max_time)
        @printf("   rtol = %7.1e  target ∥r∥<ε   = %7.1e\n\n", rtol, ε)
        @printf("%7s  %9s  %9s  %9s  %9s  %6s  %10s\n",
        "iter","dual Obj","∥∇dObj∥","Δ","Δₐ/Δₚ","cg its","cg msg")
    end
    if logging > 0 && (mod(iter, logging) == 0 || done)
        @printf("%7d  %9.2e  %9.2e  %9.1e %9.1e  %6d   %10s\n", (log_items...))
    end
    
    if optimal
        trunk_stats.status = :optimal
    elseif tired
        trunk_stats.status = :max_iter
    end
    if trunk_stats.status == :unkown
        return
    end
    
    # Update the preconditioner
    update!(M)
end

const cg_msg = Dict(
"on trust-region boundary" => "⊕",
"found approximate minimum least-squares solution" => "min soln",
"nonpositive curvature detected" => "neg curv",
"solution good enough given atol and rtol" => "✓",
"zero curvature detected" => "zer curv",
"maximum number of iterations exceeded" => "⤒",
"found approximate zero-residual solution" => "zero res",
"user-requested exit" => "user exit",
"time limit exceeded" => "time exit",
"unknown" => ""
)