"""
    value!(kl::DPModel, t; kwargs...)

Compute the dual objective of a Perspectron model with respect to the scaling parameter `t`.
"""
function value!(kl::DPModel{T}, t; jprods=Int[0], jtprods=Int[0], kwargs...) where T
    t = max(t, eps(T))
    @unpack λ, A = kl
    scale!(kl, t)
    s = solve!(kl; kwargs...)
    
    # Update product counts
    jprods[1] += neval_jprod(kl)
    jtprods[1] += neval_jtprod(kl)
    
    # Compute derivative of value function
    y = s.residual/λ
    dv = -(lseatyc!(kl, y) - log(t) - 1)
    
    # Set starting point for next iteration
    update_y0!(kl, s.residual/λ)
    
    return dv
end

function value!(kl::DPModel, f, dv, hv, t; jprods=Int[0], jtprods=Int[0], kwargs...)
    @unpack λ, A = kl

    scale!(kl, t[1])
    s = solve!(kl; kwargs...)
    
    # Update product counts
    jprods[1] += neval_jprod(kl)
    jtprods[1] += neval_jtprod(kl)

    #Dual solution
    y = s.residual/λ

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
    update_y0!(kl, s.residual/λ)

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
    ::SequentialSolve,
    mode;
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

    ss = SSModel(kl)

    # Initialize counters and trackers
    jprods = Int[0]
    jtprods = Int[0]
    tracker = Roots.Tracks()
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

    if mode==:Bisection
        dv!(t) = value!(
            ss.kl, 
            t;
            jprods=jprods,
            jtprods=jtprods,
            atol=δ*atol,
            rtol=δ*rtol,
            logging=logging
        )

        #Using root finding
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

    #Using Newton solve
    elseif mode==:Newton
        value_fgh!(f, g, h, t) = value!(ss.kl, f, g, h, t;
            jprods=jprods,
            jtprods=jtprods,
            atol=δ*atol,
            rtol=δ*rtol,
            logging=logging,
            kwargs...
        )
        stats = Optim.optimize(Optim.only_fgh!(value_fgh!), [t], Optim.Newton(linesearch=LineSearches.BackTracking()), Optim.Options(x_abstol=1e-6, x_reltol=1e-6))

        t = Optim.minimizer(stats)[1]
        println(t)
        show(stats)

    end

    elapsed_time = time() - start_time

    # Final solve at optimal t
    scale!(ss.kl, t)
    final_run_stats = solve!(
        ss.kl,
        atol=δ*atol,
        rtol=δ*rtol,
        logging=logging,
        reset_counters=false
    )

    # Determine solution status
    status = tracker.convergence_flag == :x_converged ? :optimal : :unknown

    stats = ExecutionStats(
        status,
        elapsed_time,                   # elapsed time
        tracker.steps,                  # number of iterations
        jprods[1],                     # number of products with A
        jtprods[1],                    # number of products with A'
        zero(T),                       # TODO: primal objective
        final_run_stats.dual_obj,      # dual objective
        final_run_stats.solution,      # primal solution `x`
        final_run_stats.residual,      # residual r = λy
        final_run_stats.optimality,    # norm of gradient of the dual objective
        tracer                         # tracer to store iteration info
    )

    return stats
end
