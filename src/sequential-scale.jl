"""
    value!(kl::DPModel, τ; kwargs...) -> (v, dv)

Compute the dual objective value `v` and its derivative `dv` with respect to the scaling parameter `τ`.

!!! note "Minimum scaling parameter"
    The scaling parameter `τ` is clamped to at least `eps(T)` to avoid numerical issues.
"""
function value!(kl::DPModel{T}, τ; prods=[0, 0], kwargs...) where T
    τ = max(τ, eps(T))
    @unpack λ, A = kl
    scale!(kl, τ)
    s = solve!(kl; kwargs...)
    v = -s.dual_obj
    
    # Update product counts
    prods[1] += neval_jprod(kl)
    prods[2] += neval_jtprod(kl)
    
    # Compute derivative of value function
    y = s.residual/λ
    dv = -lseatyc!(kl, y) + log(τ) + 1
    
    # Set starting point for next iteration
    update_y0!(kl, y)
    
    return v, dv
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
- `zverbose::Bool=false`: Whether to print verbose output from root-finding
- `logging::Int=0`: Logging level (0=none, 1=basic, 2=detailed)

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
    rtol=1e-6,
    atol=1e-6, 
    xatol=1e-6,
    xrtol=1e-6,
    δ=1e-2,
    zverbose=false,
    logging=0,
    kwargs...
) where T

    # Initialize counters and trackers
    start_time = time()
    prods = [0, 0]
    tracker = Roots.Tracks()
    tracer = DataFrame(
        iter=Int[], 
        scale=T[], 
        vpt=T[], 
        norm∇d=T[],
        cgits=Int[],
        cgmsg=String[]
    )

    # Find optimal t using root finding
    function dv!(t)
        _, dv = value!(kl, t; prods=prods, atol=δ*atol, rtol=δ*rtol, logging=logging)
        return dv
    end

    t = Roots.find_zero(
        dv!,
        t;
        tracks=tracker,
        atol=atol,
        rtol=rtol,
        xatol=xatol,
        xrtol=xrtol,
        verbose=zverbose
    )

    # Final solve at optimal t
    scale!(kl, t)
    final_run_stats = solve!(
        kl,
        atol=δ*atol,
        rtol=δ*rtol,
        logging=logging,
        reset_counters=false
    )

    stats = ExecutionStats(
        tracker.convergence_flag == :x_converged ? :optimal : :unknown,
        time() - start_time,                  # elapsed time
        tracker.steps,                 # number of iterations
        prods[1],                      # number of products with A
        prods[2],                      # number of products with A'
        zero(T),                       # TODO: primal objective
        final_run_stats.dual_obj,      # dual objective
        final_run_stats.solution,      # primal solution `x`
        final_run_stats.residual,      # residual r = λy
        final_run_stats.optimality,    # norm of gradient of the dual objective
        tracer                         # tracer to store iteration info
    )

    return stats
end
