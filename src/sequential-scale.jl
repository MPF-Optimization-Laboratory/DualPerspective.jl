"""
    value!(kl::DPModel, t; kwargs...)

Compute the dual objective of a Perspectron model with respect to the scaling parameter `t`.
"""
function value!(kl::DPModel{T}, t; prods=[0,0], kwargs...) where T
    t = max(t, eps(T))
    @unpack λ, A = kl
    scale!(kl, t)
    solve!(kl; kwargs...)
    
    # Update product counts
    prods[1] += neval_jprod(kl)
    prods[2] += neval_jtprod(kl)
    
    # Compute derivative of value function
    residual = ((kl.λ).*(kl.y0))
    y = residual/λ
    dv = -(lseatyc!(kl, y) - log(t) - 1)
    
    # Set starting point for next iteration
    update_y0!(kl, residual/λ)
    
    return dv
end

function value!(kl::DPModel, f, dv, hv, t; prods=[0,0], kwargs...)
    @unpack λ, A = kl

    scale!(kl, t[1])
    solve!(kl; kwargs...)
    
    # Update product counts
    prods[1] += neval_jprod(kl)
    prods[2] += neval_jtprod(kl)

    #Dual solution
    residual = ((kl.λ).*(kl.y0))
    y = residual/λ

    #Dual objective value
    f = -dObj!(kl, y)
    
    # Compute derivative of value function
    if !isnothing(dv)
        dv .= -(lseatyc!(kl, y) - log(t[1]) - 1)
    end

    #Hessian
    if !isnothing(hv)
        b = A*grad(kl.lse)

        hv!(res, z) = dHess_prod!(kl, z, res)
        m = size(A,1)
        H = LinearOperator(Float64, m, m, true, true, hv!)

        ω,_ = cg(H, b)

        hv[1,1] = 1/t[1] + b'*ω
        # hv[1,1] = 1/t[1] + (1/kl.λ)*norm(b)^2
    end

    # Set starting point for next iteration
    update_y0!(kl, residual/λ)

    return f
end

struct SequentialSolve end
"""
    solve!(kl::DPModel, ::SequentialSolve; kwargs...) -> ExecutionStats

Solve the KL-regularized least squares problem by finding the optimal scaling parameter `t` 
that maximizes the dual objective. The optimal `t` is found by applying root-finding to the derivative of the dual objective with respect to `t`.

# Arguments
- `kl`: The KL-regularized least squares model to solve
- `::SequentialSolve`: The sequential solve algorithm type

# Keyword Arguments
- `t::Real=1.0`: Initial guess for the scaling parameter
- `rtol::Real=1e-6`: Relative tolerance for the root-finding optimization
- `atol::Real=1e-6`: Absolute tolerance for the root-finding optimization  
- `xatol::Real=1e-6`: Absolute tolerance for convergence in `t`
- `xrtol::Real=1e-6`: Relative tolerance for convergence in `t`
- `δ::Real=1e-2`: Tolerance factor applied to `atol` and `rtol` for the inner optimization
- `verbose::Bool=false`: Whether to print verbose output from root-finding

# Returns
An `ExecutionStats` struct containing:
- Solution status
- Runtime statistics
- Optimal primal and dual solutions
- Residuals and optimality measures
"""
function solve!(
    kl::DPModel{T},
    ::SequentialSolve;
    t=one(T),
    rtol=DEFAULT_PRECISION(T),
    atol=DEFAULT_PRECISION(T), 
    # xatol=DEFAULT_PRECISION(T), # Removed 28 Apr 2025: not clear how to set this
    # xrtol=DEFAULT_PRECISION(T),
    δ=1e-2,
    verbose=false,
    kwargs...
) where T

    # Initialize counters and trackers
    start_time = time()
    prods = [0, 0]
    tracer = DataFrame(
        iter=Int[], 
        scale=T[], 
        vpt=T[], 
        norm∇d=T[],
        cgits=Int[],
        cgmsg=String[]
    )

    # Find optimal t
    start_time = time()

    #Using Newton solve
    value_fgh!(f, g, h, t) = value!(kl, f, g, h, t;
        prods=prods,
        atol=δ*atol,
        rtol=δ*rtol,
        kwargs...
    )
    outer_stats = Optim.optimize(Optim.only_fgh!(value_fgh!), [t], 
        Optim.Newton(linesearch=BackTracking()), 
        Optim.Options(g_abstol=1e-2))

    elapsed_time = time() - start_time

    # Final solve at optimal t
    scale!(kl, t)
    inner_stats = solve!(
        kl;
        atol=δ*atol,
        rtol=δ*rtol,
        reset_counters=false,
        kwargs...
    )

    primal_solution = inner_stats.solution

    stats = ExecutionStats(
        Optim.converged(outer_stats),
        time() - start_time,                  # elapsed time
        Optim.iterations(outer_stats),                 # number of iterations
        prods[1],                      # number of products with A
        prods[2],                      # number of products with A'
        pObj!(kl, primal_solution),      # primal objective
        dObj!(kl, kl.y0),      # dual objective
        Optim.minimizer(outer_stats)[1],
        primal_solution,      # primal solution `x`
        (kl.λ).*(kl.y0),      # residual r = λy
        Optim.g_residual(outer_stats),
        inner_stats.g,    # norm of gradient of the dual objective
        tracer                         # tracer to store iteration info
    )

    return stats
end
