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

####
#Utility functions for testing ϵ homotopy

function softmax(y, c, A, q)
    z = -c + A'*y
    zmax = maximum(z)
    z = z .- zmax
    e = exp.(z)

    return (q.*e)/dot(q,e)
end

function max12(x)
    max1 = -Inf
    max2 = -Inf

    for xi in x
        if xi > max1
            max2 = max1
            max1 = xi
        elseif xi > max2 && xi != max1
            max2 = xi
        end
    end

    return max1, max2
end

function ϵ_search(kl, c)
    ϵ = 1e-8
    h = log(eps(ϵ)^(2/3))

    z = kl.A'*kl.y0 - c/ϵ

    m1, m2 = max12(z)

    diff = m2 - m1

    while diff < h && ϵ < 1e8
        ϵ *= 2.

        z = kl.A'*kl.y0 - c/ϵ

        m1, m2 = max12(z)

        diff = m2 - m1
    end

    return ϵ
end

####

function solve!(
    kl::DPModel{T};
    max_time::Real=60,
    reset_counters=false,
    atol::T=DEFAULT_PRECISION(T),
    rtol::T=DEFAULT_PRECISION(T),
    max_iter::Int=typemax(Int)-1,
    logging=false,
    kwargs...) where T
   
    # Reset counters
    if reset_counters
        reset!(kl)    
    end

    #Tracer
    tracer = DataFrame(iter=Int[], dual_obj=T[], r=T[], Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])

    f(y) = dObj!(kl, y)
    fg!(grads, y) = begin v=dObj!(kl, y); dGrad!(kl, y, grads); return v end
    H = x -> LinearOperator(T, length(kl.y0), length(kl.y0), true, true, (res, z) -> dHess_prod!(kl, z, res))

    # ϵ = 1.0
    # c = copy(kl.c)
    # λ = kl.λ
    # callback = Returns(nothing)

    # if !iszero(kl.c)
    #     #ϵ callback option 1
    #     # ϵ = ϵ_search(kl, kl.c)
    #     ϵ = norm(kl.c)
    #     # ϵ = 1.
    #     # println("Initial ϵ: ", ϵ)

    #     kl.c ./= ϵ
    #     kl.λ *= ϵ

    #     function ϵ_homotopy()
    #         α = 2
    #         if (true || iszero(c)) && ϵ>1e-8
    #             kl.c ./= α
    #             kl.λ /= α
    #             ϵ /= α
    #         end
    #         # println("ϵ: ", ϵ)
    #     end

    #     #ϵ callback option 2
    #     # c = copy(kl.c)
    #     # λ = kl.λ

    #     # ϵ = ϵ_search(kl, c)
    #     # println("Initial ϵ: ", ϵ)

    #     # kl.c = c/ϵ
    #     # kl.λ = λ*ϵ

    #     # function ϵ_homotopy()
    #     #     ϵ = ϵ_search(kl, c)

    #     #     println("ϵ: ", ϵ)

    #     #     kl.c = c/ϵ
    #     #     kl.λ = λ*ϵ
    #     # end

    #     callback = ϵ_homotopy
    # end

    #Solve
    stats = newton!(kl.y0, f, fg!, H;
                    linesearch=true,
                    itmax=max_iter,
                    time_limit=Float64(max_time),
                    atol=atol,
                    rtol=rtol)

    # if !iszero(c) kl.c .= c; kl.λ = λ end

    # stats = optimize!(kl.y0, f, fg!, H, Val(:newton);
    #                     itmax=max_iter,
    #                     time_limit=Float64(max_time),
    #                     atol=atol,
    #                     rtol=rtol, M=0., linesearch=nothing, posdef=true)

    # stats = optimize!(kl.y0, f, fg!, H, Val(:rsfn);
    #                     itmax=max_iter,
    #                     time_limit=Float64(max_time),
    #                     atol=1e-4,
    #                     rtol=1e-3, M=1e-8, linesearch=nothing)

    # println("Final ϵ: ", ϵ)

    if logging
        show(stats)
        println()
    end

    primal_solution = kl.scale .* grad(kl.lse)

    return stats, primal_solution
end

# const newtoncg = solve!

# function callback(
#     kl::DPModel{T},
#     solver,
#     M,
#     trunk_stats,
#     tracer,
#     logging,
#     max_time;
#     atol::T = DEFAULT_PRECISION(T),
#     rtol::T = DEFAULT_PRECISION(T),
#     max_iter::Int = typemax(Int),
#     trace::Bool = false,
#     ) where T
    
#     dObj = trunk_stats.objective 
#     iter = trunk_stats.iter
#     r = trunk_stats.dual_feas # = ||∇ dual obj(x)||
#     # r = norm(solver.gx)
#     Δ = solver.tr.radius
#     actual_to_predicted = solver.tr.ratio
#     cgits = solver.subsolver.stats.niter
#     cgexit = get(cg_msg, solver.subsolver.stats.status, "default")
#     ε = atol + rtol * kl.bNrm
    
#     # Test exit conditions
#     tired = iter >= max_iter
#     optimal = r < ε 
#     done = tired || optimal
    
#     log_items = (iter, dObj, r, Δ, actual_to_predicted, cgits, cgexit) 
#     trace && push!(tracer, log_items)
#     if logging > 0 && iter == 0
#         println("\n", kl)
#         println("Solver parameters:")
#         @printf("   atol = %7.1e  max time (sec) = %7d\n", atol, max_time)
#         @printf("   rtol = %7.1e  target ∥r∥<ε   = %7.1e\n\n", rtol, ε)
#         @printf("%7s  %9s  %9s  %9s  %9s  %6s  %10s\n",
#         "iter","dual Obj","∥∇dObj∥","Δ","Δₐ/Δₚ","cg its","cg msg")
#     end
#     if logging > 0 && (mod(iter, logging) == 0 || done)
#         @printf("%7d  %9.2e  %9.2e  %9.1e %9.1e  %6d   %10s\n", (log_items...))
#     end
    
#     if optimal
#         trunk_stats.status = :optimal
#     elseif tired
#         trunk_stats.status = :max_iter
#     end
#     if trunk_stats.status == :unkown
#         return
#     end
    
#     # Update the preconditioner
#     update!(M)
# end

# const cg_msg = Dict(
# "on trust-region boundary" => "⊕",
# "found approximate minimum least-squares solution" => "min soln",
# "nonpositive curvature detected" => "neg curv",
# "solution good enough given atol and rtol" => "✓",
# "zero curvature detected" => "zer curv",
# "maximum number of iterations exceeded" => "⤒",
# "found approximate zero-residual solution" => "zero res",
# "user-requested exit" => "user exit",
# "time limit exceeded" => "time exit",
# "unknown" => ""
# )