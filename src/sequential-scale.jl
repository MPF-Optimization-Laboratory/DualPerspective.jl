function value!(kl::DPModel{T}, f, dv, hv, t; prods=[0,0], kwargs...) where T
    println(t)
    println(f)

    if !isnothing(f)
        scale!(kl, t[1])
        solve!(kl; kwargs...)
        
        # Update product counts
        prods[1] += neval_jprod(kl)
        prods[2] += neval_jtprod(kl)

        #Dual solution
        # residual = ((kl.λ).*(kl.y0))
        # y = residual/λ

        y = kl.y0
        # println(extrema(y))
        #Dual objective value
        f = -dObj!(kl, y)
    end
    
    # Compute derivative of value function
    if !isnothing(dv)
        dv .= -lseatyc!(kl, y) + log(t[1]) + 1
    end

    #Hessian
    if !isnothing(hv)
        @unpack λ, A = kl

        b = A*grad(kl.lse)

        hv!(res, z) = dHess_prod!(kl, z, res)
        m = size(A,1)
        H = LinearOperator(T, m, m, true, true, hv!)

        ω,_ = cg_lanczos(H, b)

        hv[1,1] = 1/t[1] + b'*ω
    end

    # Set starting point for next iteration
    # update_y0!(kl, residual/λ)

    return f
end

function value_f(kl::DPModel{T}, t; prods=[0,0], kwargs...) where T
    scale!(kl, t[1])
    solve!(kl; reset_counters=false, kwargs...)

    y = kl.y0

    #Dual objective value
    f = -dObj!(kl, y)

    return f
end

function value_g(kl::DPModel{T}, t; kwargs...) where T
    scale!(kl, t)
    solve!(kl; reset_counters=false, kwargs...)

    y = kl.y0

    #Dual objective value
    f = -dObj!(kl, y)
    
    dt = -lseatyc!(kl, y) + log(t) + 1

    return dt
end

function value_fg!(kl::DPModel{T}, dt, t; kwargs...) where T
    f = value_f(kl, t; kwargs...)
    
    dt .= -lseatyc!(kl, kl.y0) + log(t[1]) + 1

    return f
end

function value_H(kl::DPModel{T}, t; kwargs...) where T
    @unpack λ, A = kl
    
    b = A*grad(kl.lse)

    hv!(res, z) = dHess_prod!(kl, z, res)
    m = size(A,1)
    H = LinearOperator(T, m, m, true, true, hv!)

    ω,_ = cg_lanczos(H, b)

    hv = zeros(1,1)

    hv[1,1] = 1/t[1] + b'*ω
    
    # hv[1,1] = 1/t[1] + norm(b)^2/λ

    println("Hessian norm: ", norm(hv))

    return LinearOperator(hv)
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
    atol=DEFAULT_PRECISION(T),
    rtol=DEFAULT_PRECISION(T),
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
    start_time = time()

    #ϵ homotopy
    ϵ = 1.0
    c_ = copy(kl.c)
    λ_ = kl.λ
    homotopy = Returns(nothing)

    if !iszero(kl.c)
        ϵ = norm(kl.c)

        kl.c ./= ϵ
        kl.λ *= ϵ

        function ϵ_homotopy()
            α = 2
            if false && ϵ>1e-8
                kl.c ./= α
                kl.λ /= α
                ϵ /= α
            end
        end

        homotopy = ϵ_homotopy
    end

    #Find optimal scale

    #Using Roots.jl
    # g(t) = begin println(t); kl.y0 .= zero(T); return value_g(kl, t; atol=1e-4, rtol=1e-4, kwargs...) end

    # t_min = norm(kl.b)/norm(kl.A)
    # t_max = 1e6
    # # println(g(t_min))
    # # println(g(t_max))
    # # t = find_zero(g, (t_min, t_max), Bisection(); xatol=1e-3, xrtol=1e-4, verbose=verbose)
    # t = find_zero(g, t_min; atol=1e-1, rtol=1e-1, verbose=verbose)

    #Using custom Newton solve
    t_vec = [norm(kl.b)/norm(kl.A)]
    kl_reset() = begin kl.y0 .= zero(T); homotopy(); verbose ? println("Current scale value: ", t_vec[1]) : nothing; return false end

    atol_ = 1e-4
    rtol_ = 1e-4

    f(t) = value_f(kl, t; atol=atol_, rtol=rtol_, kwargs...)
    fg!(dt, t) = value_fg!(kl, dt, t; atol=atol_, rtol=rtol_, kwargs...) 
    H(t) = value_H(kl, t; atol=atol_, rtol=rtol_, kwargs...)

    outer_stats = newton!(t_vec, f, fg!, H;
                            linesearch=false, α=1.0,
                            itmax=100, time_limit=Inf,
                            atol=1e-5, rtol=1e-6,
                            callback=kl_reset)

    t = t_vec[1]

    #Display outer solve statistics
    verbose ? begin println("#################################\nOuter solve\n#################################"); display(outer_stats) end : nothing

    #Final solve at optimal t
    verbose ? println("#################################\nFinal solve at t=$t...\n#################################") : nothing
    
    kl.y0 .= zero(T)
    scale!(kl, t)

    inner_stats, primal_solution = solve!(
        kl;
        atol=atol,
        rtol=rtol,
        reset_counters=false,
        logging=verbose
    )

    kl.c .= c_
    kl.λ = λ_

    stats = ExecutionStats(
        outer_stats.converged && inner_stats.converged,
        time() - start_time,             # elapsed time
        outer_stats.iterations,          # number of iterations
        neval_jprod(kl),                 # number of products with A
        neval_jtprod(kl),                # number of products with A'
        pObj!(kl, primal_solution),      # primal objective
        dObj!(kl, kl.y0),                # dual objective
        t,                               # optimal scale
        primal_solution,                 # primal solution `x`
        (kl.λ).*(kl.y0),                 # residual r = λy
        inner_stats.g_seq[end],          # norm of gradient of the dual solve 'y'
        outer_stats.g_seq[end],          # norm of gradient of the scale solve 't'
        tracer                           # tracer to store iteration info
    )

    # stats = ExecutionStats(
    #     true && inner_stats.converged,
    #     time() - start_time,             # elapsed time
    #     10,          # number of iterations
    #     neval_jprod(kl),                 # number of products with A
    #     neval_jtprod(kl),                # number of products with A'
    #     pObj!(kl, primal_solution),      # primal objective
    #     dObj!(kl, kl.y0),                # dual objective
    #     t,                               # optimal scale
    #     primal_solution,                 # primal solution `x`
    #     (kl.λ).*(kl.y0),                 # residual r = λy
    #     inner_stats.g_seq[end],          # norm of gradient of the dual solve 'y'
    #     0.,          # norm of gradient of the scale solve 't'
    #     tracer                           # tracer to store iteration info
    # )

    return stats, inner_stats, outer_stats
end
