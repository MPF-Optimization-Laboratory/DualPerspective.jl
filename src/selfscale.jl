"""
    residual!(ss, yt, Fx)

Compute the residual in the self-scaling optimality conditions augmented problem, which concatenate the dual residual with the optimal scaling condition:

    F(y, t) = [ ∇d(y)
                logexp(A'y) - log(t) - 1 ]

where

    ∇d(y) = A(tx(y)) + λCy - b
    x(y) = ∇logexp(A'y)
""" 
function NLPModels.residual!(ss::SSModel, yt, Fx)
	increment!(ss, :neval_residual)
    kl = ss.kl
	@unpack A, c, lse = kl
	m = kl.meta.nvar
    
    r = @view Fx[1:m]
	y = @view yt[1:m]
	t =       yt[end]
    
    scale!(kl, t)             # Apply the latest scaling factor
    f = lseatyc!(kl, y)       # f = logΣexp(A'y-c). Needed before grad eval
	dGrad!(kl, y, r)          # r ≡ Fx[1:m] = ∇d(y)	
	Fx[end] = f - log(t) - 1  # optimal scaling condition
	return Fx
end

"""
    Jyt = jprod_residual!(ss, yt, zα, Jyt)

Compute the Jacobian-vector product, 

    (1) [ ∇²d(A'y)  Ax  ][ z ] := [ Jy ]  where x:=x(y)
    (2) [ (Ax)'     -1/t][ α ] := [ Jt ]
"""
function NLPModels.jprod_residual!(ss::SSModel, yt, zα, Jyt)

    kl = ss.kl
    @unpack A, lse, mbuf = kl
    Ax = mbuf
    m = kl.meta.nvar
    x = grad(lse)

    increment!(ss, :neval_jprod_residual)

    Jy = @view Jyt[1:m]
    t = yt[end]
    z = @view zα[1:m]
    α = zα[end]
   
    mul!(Ax, A, x)

    # Equation (1)
    dHess_prod!(kl, z, Jy)  # Jy = ∇²d(A'y)z
    Jy .+= α*Ax             # Jy += αAx
    
    # Equation (2)  
    Jyt[end] = z⋅Ax - α/t

    return Jyt
end

function NLPModels.jtprod_residual!(ss::SSModel, yt, wα, Jyt)
    increment!(ss, :neval_jtprod_residual)
    NLPModels.jprod_residual!(ss, yt, wα, Jyt)
end

#######################################################################
# Nonlinear Least Squares via TrunkLS
#######################################################################

struct SSTrunkLS end

"""
    solve!(ss::SSModel, ::SSTrunkLS; kwargs...)

Solve the self-scaled model using Gauss-Newton, via the TrunkLS algorithm.
"""
function solve!(
    kl::DPModel{T},
    ::SSTrunkLS;
    logging=0,
    monotone=true,
    max_time=30.0,
    atol=zero(T),
    rtol=zero(T),
    reset_counters=true,
    kwargs...) where T

    if reset_counters
        reset!(kl) # reset counters
    end
    ss = SSModel(kl)

    tracer = DataFrame(iter=Int[], scale=T[], vpt=T[], dual_grad=T[], r=T[], Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])

    # Callback routine
    cb(ss::SSModel, solver, stats) =
      callback(ss, solver, stats, tracer, logging, max_time; kwargs...)
    
    trunk_stats =
      trunk(ss; callback=cb, atol=T(0), rtol=T(0), max_time=max_time, monotone=monotone) 

    # Optimality. Report the maximum ∇d(y) and v'(t)
    optimality = sqrt(obj(ss, trunk_stats.solution))

    kl = ss.kl
    x = kl.scale.*grad(kl.lse)
    y = @view trunk_stats.solution[1:end-1]
    stats = ExecutionStats(
        trunk_stats.status,          # status
        trunk_stats.elapsed_time,    # elapsed time
        trunk_stats.iter,            # iterations
        neval_jprod(kl),             # count products with A
        neval_jtprod(kl),            # count products with A'
        zero(T),                     # TODO: primal objective
        trunk_stats.objective,       # dual objective
        x,                           # primal solution `x`
        (kl.λ)*y,                    # residual r = λy
        optimality,                  # norm of the gradient of the dual objective
        tracer                       # TODO: tracer 
    )
    return stats
end

function callback(
    ss::SSModel{T},
    solver,
    trunk_stats,
    tracer,
    logging,
    max_time;
    atol = DEFAULT_PRECISION(T),
    rtol = DEFAULT_PRECISION(T),
    max_iter::Int = typemax(Int),
    trace::Bool = false,
    ) where T
  
    
    # Norm of the (unscaled) dual gradient ∇d(y)
    m = length(ss.kl.b)
    ∇d = @view solver.Fx[1:m]
    norm∇d = norm(∇d, Inf)
    vpt = solver.Fx[end]

    iter = trunk_stats.iter # = number of iterations
    r = trunk_stats.dual_feas # = ||∇F⋅F(y,t)||
    scale = solver.x[end]
    Δ = solver.tr.radius 
    actual_to_predicted = solver.tr.ratio
    cgits = solver.krylov_subsolver.stats.niter
    cgexit = get(cg_msg, solver.krylov_subsolver.stats.status, "default")
    ε = atol + rtol * ss.kl.bNrm
    
    # Test exit conditions
    # TODO: should the optimality check be based on `norm∇d` or norm of the full residual?
    tired = iter >= max_iter
    # optimal = r < ε
    optimal = norm∇d < ε
    done = tired || optimal
    
    log_items = (iter, scale, vpt, norm∇d, r, Δ, actual_to_predicted, cgits, cgexit) 
    trace && push!(tracer, log_items)
    if logging > 0 && iter == 0
        println("\n", ss)
        println("Solver parameters:")
        @printf("   atol = %7.1e  max time (sec) = %7d\n", atol, max_time)
        @printf("   rtol = %7.1e  target ∥r∥<ε   = %7.1e\n\n", rtol, ε)
        @printf("%7s  %8s  %9s  %8s  %8s  %7s  %7s  %4s  %10s\n",
        "iter","t","v'(t)","∥∇d(y)∥","∥∇F⋅F∥","Δ","Δₐ/Δₚ","cg","cg msg")
    end
    if logging > 0 && (mod(iter, logging) == 0 || done)
        @printf("%7d  %8.2e  %9.2e  %8.2e  %8.2e  %7.1e  %7.1e  %4d  %10s\n", (log_items...))
    end
    
    if optimal
        trunk_stats.status = :optimal
    elseif tired
        trunk_stats.status = :max_iter
    end
    if trunk_stats.status == :unkown
        return
    end
end