module NonlinearSolveExt

using DualPerspective
using NonlinearSolve
using LinearAlgebra
using Printf
using DataFrames
using UnPack

"""
    nlresidual!(F, yt, ss::SSModel)

Wrapper around [`residual!`](@ref) to match NonlinearSolve.jl's API.

Compute the residual in the self-scaling optimality conditions:

    F(y, t) = [ ∇d(y)
                logexp(A'y) - log(t) - 1 ]

where ∇d(y) = A(tx(y)) + λCy - b and x(y) = ∇logexp(A'y).
"""
function nlresidual!(F, yt, ss::SSModel)
    residual!(ss, yt, F)
end

"""
    nljprod!(Jyt, zα, yt, ss::SSModel)

Wrapper around [`jprod_residual!`](@ref) to match NonlinearSolve.jl's API.

Compute the Jacobian-vector product:

    [ ∇²d(A'y)  Ax  ][ z ] := [ Jy ]  where x:=x(y)
    [ (Ax)'     -1/t][ α ] := [ Jt ]
"""
function nljprod!(Jyt, zα, yt, ss::SSModel)
    jprod_residual!(ss, yt, zα, Jyt)
end

function DualPerspective.solve!(
    ss::SSModel{T},
    ::NewtonEQ;
    y0 = begin
        m = ss.kl.meta.nvar
        zeros(T, m)
    end,
    t0 = one(T),
    atol = DualPerspective.DEFAULT_PRECISION(T),
    rtol = DualPerspective.DEFAULT_PRECISION(T),
    logging=0,
    max_time=30.0,
    max_iter=1000,
    trace::Bool = false,
    kwargs...) where T
    
    NLPModels.reset!(ss) # reset counters
    kl = ss.kl
    m = kl.meta.nvar

    tracer = DataFrame(iter=Int[], scale=T[], vpt=T[], norm∇d=T[], cgits=Int[], cgmsg=String[])
   
    # Setup the NonlinearSolve objects
    nlf = NonlinearFunction(nlresidual!, jvp=nljprod!)
    prob = NonlinearProblem(nlf, vcat(y0, t0), ss)
    nlcache = init(
                prob,
                reltol=rtol,
                abstol=atol,
                show_trace = Val(false),
                store_trace = Val(false),
                NewtonRaphson(
                    # linesearch = RobustNonMonotoneLineSearch(),
                    linesearch = BackTracking(),
                    linsolve = KrylovJL_MINRES(verbose=0, itmax=50),
                )
            )
   
    start_time = time()
    elapsed_time = 0.0
    iter = 0
    while true 
        
        # Break if time is up
        elapsed_time = time() - start_time
        iter += 1
        elapsed_time > max_time && break
        iter > max_iter && break

        # Take a Newton step
        step!(nlcache)

        t = nlcache.u[end]  
        ∇d = @view nlcache.fu[1:m]
        vpt = nlcache.fu[end]
       
        norm∇d = norm(∇d, Inf)
        log_items = (iter, t, vpt, norm∇d, nlcache.descent_cache.lincache.lincache.cacheval.stats.niter, nlcache.descent_cache.lincache.lincache.cacheval.stats.status) 
        trace && push!(tracer, log_items)
        
        if nlcache.retcode ≠ ReturnCode.Default
            break
        end
    end

    # Test nlcache.retcode to return one of the following symbols:
    # :optimal, :max_iter, :unknown
    status = if nlcache.retcode == ReturnCode.Success
        :optimal
    elseif nlcache.retcode == ReturnCode.MaxIters
        :max_iter
    else
        :unknown
    end

    λ = kl.λ
    y = @view nlcache.u[1:m]
    x = kl.scale.*grad(kl.lse)
    ∇d = @view nlcache.fu[1:m]
    r = λ*y
    stats = DualPerspective.ExecutionStats(
        status,
        elapsed_time,       # elapsed time
        iter,               # number of iterations
        DualPerspective.neval_jprod(kl),    # number of products with A
        DualPerspective.neval_jtprod(kl),   # number of products with A'
        zero(T),            # TODO: primal objective
        zero(T),            # dual objective
        x,                  # primal solultion `x`
        r,                  # residual r = λy
        norm(∇d),           # norm of gradient of the dual objective
        tracer              # TODO: tracer 
    ) 
    
    return stats
end

end 