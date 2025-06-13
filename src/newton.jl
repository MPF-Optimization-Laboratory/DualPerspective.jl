mutable struct Stats{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    f_evals::I #number of function evaluations
    hvp_evals::I #number of hvp evaluations
    run_time::Float64 #iteration runtime
    f::T #final function value
    g::T #final gradient norm
    krylov_iterations::T #number of Krylov iterations
    solution::S
end

function Stats(type::Type{<:AbstractFloat}=Float64)
    return Stats(false, 0, 0, 0, 0.0, zero(type), zero(type), zero(type), type[])
end

function Base.show(io::IO, stats::Stats)
    print(io, "Converged: ", stats.converged, '\n',
                "Iterations: ", stats.iterations, '\n',
                "Function Evals: ", stats.f_evals, '\n',
                "Hvp Evals: ", stats.hvp_evals, '\n',
                "Run Time (s): ", stats.run_time, '\n',
                "Minimum: ", stats.f, '\n',
                "Gradient Norm: ", stats.g, '\n',
                "Total Krylov Iterations: ", stats.krylov_iterations, '\n')
end

elapsed(tic::UInt64) = (time_ns()-tic)/1e9

#=
Newton solvers
=#
function newton!(x::S, f::F1, fg!::F2, H::L; itmax::I, time_limit::T, α::T=1e0, linesearch::Bool=false, atol::T=1e-5, rtol::T=1e-6, krylov_order::Int=0) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
    workspace = krylov_workspace(Val(:cg), size(x,1), size(x,1), S)

    # if krylov_order == 0
    #     krylov_order = size(x,1)
    # end

    tic = time_ns()

    stats = Stats(T)
    converged = false
    iterations = 0
    nprod = 0

    if linesearch
        ls = BackTracking(order=3)
    end

    grads = similar(x)

    #compute function and gradient
    fval = fg!(grads, x)
    g_norm = norm(grads)

    tol = atol + rtol*g_norm

    #Iterate
    while iterations<itmax+1
        #check gradient norm
        if g_norm <= tol
            converged = true
            break
        end

        #check other exit conditions
        time = elapsed(tic)

        if (time>=time_limit) || (iterations==itmax)
            break
        end

        #step
        Hv = H(x)
        krylov_solve!(workspace, Hv, grads, itmax=krylov_order, timemax=time_limit-time)

        if !issolved(workspace)
            # println("WARNING: Solver failure")
        end

        stats.krylov_iterations += iteration_count(workspace)

        if linesearch
            function ϕ(t)
                stats.f_evals += 1
                return f(x-t*workspace.x)
            end

            function dϕ(t)
                stats.f_evals += 1
                fg!(grads, x-t*workspace.x)
                return dot(grads, -workspace.x)
            end

            function ϕdϕ(t)
                stats.f_evals += 1
                phi = fg!(grads, x-t*workspace.x)
                dphi = dot(grads, -workspace.x)
                return (phi, dphi)
            end  

            α, _ = ls(ϕ, dϕ, ϕdϕ, 1.0, fval, dot(-workspace.x, grads))
        end

        x .-= α*workspace.x

        #compute function and gradient
        fval = fg!(grads, x)
        g_norm = norm(grads)

        #update hvp operations
        nprod += Hv.nprod

        iterations += 1
    end

    #update some more stats
    stats.converged = converged
    stats.iterations = iterations
    stats.f_evals += iterations+1
    stats.hvp_evals = nprod
    stats.run_time = elapsed(tic)
    stats.f = fval
    stats.g = g_norm

    return stats
end