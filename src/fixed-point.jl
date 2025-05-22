"""
Implement fixed-point iteration for solving the dual problem.
"""

using Printf

function print_header()
    println("\nIter |   ‖∇ϕ‖       |   ‖Δy‖     ")
    println("-----|--------------|------------")
end

function fixed_point(
    kl::DPModel{T};
    atol = DEFAULT_PRECISION(T),
    rtol = DEFAULT_PRECISION(T),
    verbose = false,
    max_iter = 1000,
    α = 1.0,
    kwargs...
) where T
    @unpack A, b, C, λ, lse = kl
    y = copy(kl.y0)
    y_old = copy(y)
    t = kl.scale
    ε = atol + rtol * kl.bNrm
    start_time = time()
    tracer = DataFrame(iter=Int[], ∇ϕ=T[], rNrm=T[], Δy=T[])
    k = 0

    # Dual objective gradient
    ∇ϕ = similar(y)
    dObj!(kl, y)
    dGrad!(kl, y, ∇ϕ)
    x = t*grad(lse)
    r = b - A*x

    function converged(k)
        tired = k ≥ max_iter
        optimal = norm(∇ϕ) < ε
        stalled = k > 0 && norm(y - y_old) < atol + rtol * norm(y)
        done = tired || optimal || stalled
        return done
    end

    verbose && print_header()
    verbose && @printf("%4d | %12.4e | %12.4e\n", k, norm(∇ϕ), norm(y - y_old))
    verbose && push!(tracer, (iter=k, ∇ϕ=norm(∇ϕ), rNrm=norm(r), Δy=norm(y - y_old)))

    while !converged(k)
        k += 1

        y_old = copy(y)
        r = b - A*x
        y = (C\r)/λ

        Δy = y - y_old
        y = y_old + α*Δy

        dObj!(kl, y)
        dGrad!(kl, y, ∇ϕ)
        x = t*grad(lse)

        verbose && k % 10 == 0 && k > 0 && print_header()
        verbose && @printf("%4d | %12.4e | %12.4e\n", k, norm(∇ϕ), norm(y - y_old))
        push!(tracer, (iter=k, ∇ϕ=norm(∇ϕ), rNrm=norm(r), Δy=norm(y - y_old)))
    end

    stats = ExecutionStats(
        :temp,
        time() - start_time,
        k,
        neval_jprod(kl),
        neval_jtprod(kl),
        pObj!(kl, x),
        dObj!(kl, y),
        x,
        (kl.λ).*y,
        norm(∇ϕ),
        tracer
    )
    return x, stats
end