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
Root-finder outcomes that count as success. `Roots` distinguishes several: `:x_converged`
when the bracket on `t` is tight enough, `:f_converged` when the residual is small,
`:exact_zero` on a hit, and the generic `:converged`. Recognizing only `:x_converged`
reported almost every successful solve as `:unknown`.
"""
const ROOTS_CONVERGED = (:x_converged, :f_converged, :exact_zero, :converged)

"""
    _sequential_status(tracker, final_run_stats) -> Symbol

Termination status for a sequential-scale solve.

The scale `t` is located by root finding and the solution is then produced by an inner
trust-region solve. Report failure if the root find did not converge; otherwise defer to
the inner solve, which tests the criterion that actually governs the returned solution,
`‖∇d(y)‖ < atol + rtol‖b‖`.
"""
function _sequential_status(tracker, final_run_stats)
    tracker.convergence_flag in ROOTS_CONVERGED || return :stalled
    return final_run_stats.status
end

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
        _, dv = value!(kl, t; prods=prods, atol=δ*atol, rtol=δ*rtol, kwargs...)
        return dv
    end

    t = Roots.find_zero(
        dv!,
        t;
        tracks=tracker,
        atol=atol,
        rtol=rtol,
        verbose=verbose
    )

    # Final solve at optimal t
    scale!(kl, t)
    final_run_stats = solve!(
        kl;
        atol=δ*atol,
        rtol=δ*rtol,
        reset_counters=false,
        kwargs...
    )

    stats = ExecutionStats(
        _sequential_status(tracker, final_run_stats),
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
