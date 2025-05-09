# struct LevelSet end

# function solve!(
#     kl::DPModel{T},
#     ::LevelSet;
#     α::T=1.5,
#     σ=kl.bNrm^2 / (2 * kl.λ),
#     t=one(T),
#     logging=0,
#     atol=1e-3,
#     rtol=1e-3,
#     max_time::Float64=30.0,
#     kwargs...
# ) where {T}

#     # Reset counters
#     reset!(kl)

#     @assert 1 < α < 2 "α must be in the open interval (1, 2)"
#     @assert t > 0 "t0 must be positive"

#     scale!(kl, t)

#     it = 0
#     tracer = DataFrame(iter=Int[], l=T[], u=T[], u_over_l=T[], s=T[])
#     l, u, s = zero(T), zero(T), zero(T)
#     solver = TrunkSolver(kl)
#     subsolver_logging = Int(max(0, logging-1))
#     start_time = time()

#     while true
#         it += 1
#         l, u, s = oracle!(kl, α, σ, solver, tracer, logging=subsolver_logging, max_time=max_time) # TODO: weird max time

#         tk = t - l / s
        
#         small_step = abs(tk - t) ≤ atol + t*rtol
#         min_value = u ≤ atol + σ*rtol
#         ratio = log10(u/l) < 1e-3
#         done = small_step || ratio || min_value

#         if logging > 0
#             if it == 1
#                 println("\n\e[1;32mLevel Set Method:\e[0m")
#                 @printf("\e[1;32m%7s  %9s  %9s  %9s  %9s  %9s  %9s\e[0m\n",
#                         "iter", "l", "u", "s", "t", "|Δt|", "log(u/l)")
#                 @printf("\e[1;32m%7s  %9s  %9s  %9s  %9s  %9s  %9s\e[0m\n",
#                         "-----", "-------", "-------", "-------", "-------", "-------", "-------")
#             end
#             @printf("\e[1;32m%7d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.1e\e[0m\n", 
#                     it, l, u, s, tk, abs(tk - t), log10(u/l))
#             if done && small_step
#                 @printf("\n\e[1;31mStop on small Δt\e[0m")
#             elseif done
#                 @printf("\n\e[1;31mStop on small upper bound\e[0m")
#             elseif done && ratio
#                 @printf("\n\e[1;31mStop on small ratio\e[0m")
#             end
#         end

#         # If tk
#         if tk < 0
#             tk = t / 2
#         end

#         if done
#             break
#         end
#         t = tk
#         scale!(kl, t)
#     end

#     final_soln = solve!(
#         kl,
#         logging=subsolver_logging,
#         reset_counters=false,
#         # atol=atol,
#         # rtol=rtol,
#         solver=solver,
#         kwargs...
#     )

#     runtime = time() - start_time

#     stats = ExecutionStats(
#         final_soln.status,
#         runtime,                        # elapsed time
#         it,                             # number of iterations
#         neval_jprod(kl),                # number of products with A
#         neval_jtprod(kl),               # number of products with A'
#         final_soln.primal_obj,          # primal objective
#         final_soln.dual_obj,            # dual objective
#         final_soln.solution,            # primal solution `x`
#         final_soln.residual,            # residual r = λy
#         final_soln.optimality,          # norm of the gradient of the dual objective
#         tracer
#     )
#     return stats
# end

# ##############################################################################
# # Adaptive Level Set
# ##############################################################################

# struct AdaptiveLevelSet end

# function solve!(
#     kl::DPModel{T},
#     ::AdaptiveLevelSet;
#     α::T=1.5,               # upper to lower bound ratio of oracle!
#     t=one(T),
#     logging=0,
#     atol=1e-3,
#     rtol=1e-3,
#     max_time::Float64=30.0,
#     slope_tol=1.0,          # End scale adjustment when slope of the lower minorant is small
#     kwargs...
# ) where {T}
#     # Reset counters
#     reset!(kl)

#     @assert 1 < α < 2 "α must be in the open interval (1, 2)"
#     @assert t > 0 "t0 must be positive"

#     scale!(kl, t)

#     # Initial objective guess
#     σ = 0.0

#     it = 0
#     tracer = DataFrame(iter=Int[], l=T[], u=T[], u_over_l=T[], s=T[])
#     solver = TrunkSolver(kl)
#     subsolver_logging = Int(max(0, logging-1))

#     start_time = time()

#     # Set bounds and slope
#     l, u, s = 0.0, 0.0, 0.0

#     if logging > 0
#         println("Starting t windup from t = $t")
#     end
 
#     while true
#         it += 1
#         l, u, s, y = oracle!(kl, α, σ, solver, tracer, logging=subsolver_logging, max_time=max_time)
#         update_y0!(kl, y)
#         if s > 0
#             t /= 2
#             if logging > 0
#                 println("t = $t")
#             end
#             scale!(kl, t)
#         else 
#             break
#         end
#     end
    
#     if logging > 0 
#         println("t windup is complete at t = $t")
#         println("\n\e[1;32mAdaptive Level Set Method:\e[0m")
#         @printf("\e[1;32m%7s  %9s  %9s  %9s  %9s  %9s\e[0m\n",
#                 "iter", "l", "u", "s", "t", "|Δt|")
#         @printf("\e[1;32m%7s  %9s  %9s  %9s  %9s  %9s\e[0m\n",
#                 "-----", "-------", "-------", "-------", "-------", "-------")
#     end

#     tₐ, sₐ, lₐ = t, s, l

#     while true
#         it += 1
#         l, u, s, y = oracle!(kl, α, σ, solver, tracer, logging=subsolver_logging, max_time=max_time)
        
#         # Check if we overshot, adjust guess if we did
#         if s > 0.0
#             # If we overshot, but the scale is still good, stop
#             if s < slope_tol
#                 break
#             end
#             σ_new = σ + lₐ / 2      # Set the new guess to avg between lower bound and old guess
#             lₐ = lₐ - σ_new + σ     # Since l is relative, fix it
#             t = tₐ - lₐ / sₐ        # Set new t to the new intersect
#             scale!(kl, t)

#             if logging > 0
#                 println("overshot the optimal t, increasing objective estimate to σ=$σ_new")            end
#             σ = σ_new
#             continue
#         end

#         tₐ, sₐ, lₐ = t, s, l
#         update_y0!(kl, y)
#         t_proposed = t - l / s
        
#         small_step = abs(t_proposed - t) ≤ atol + t*rtol
#         min_value = u ≤ atol + σ*rtol
#         done = small_step || min_value

#         if logging > 0
#             @printf("\e[1;32m%7d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e\e[0m\n", 
#                     it, l, u, s, t_proposed, abs(t_proposed - t))
#             if done && small_step
#                 @printf("\n\e[1;31mStop on small Δt\e[0m\n")
#             elseif done
#                 @printf("\n\e[1;31mStop on small upper bound\e[0m\n")
#             end
#         end

#         if done
#             # If we find the root of v(t) - σ, but the slope is still too high
#             # it means that our estimate objective was too high.

#             # If the slope is small enough, break
#             if s > -slope_tol
#                 break
#             end

#             # if the slope is large, the guess is too high
#             # lower the guess by taking a "gradient" step down, any lowering should work here
#             σ_new = σ + s * 1e-2
#             lₐ = lₐ - σ_new + σ
#             l = l - σ_new + σ
#             t = t + (σ_new - σ - l) / s
#             scale!(kl, t)
#             if logging > 0
#                 println("converged at a high objective with slope $s, decreasing guess from $σ to $σ_new")
#             end
#             σ=σ_new
#         end

#         t = t_proposed
#         scale!(kl, t)
#     end

#     final_soln = solve!(
#         kl,
#         logging=subsolver_logging,
#         reset_counters=false,
#         atol=atol,
#         rtol=rtol,
#         kwargs...
#     )

#     runtime = time() - start_time

#     stats = ExecutionStats(
#         final_soln.status,
#         runtime,                        # elapsed time
#         it,                             # number of iterations
#         neval_jprod(kl),                # number of products with A
#         neval_jtprod(kl),               # number of products with A'
#         final_soln.primal_obj,          # primal objective
#         final_soln.dual_obj,            # dual objective
#         final_soln.solution,            # primal solution `x`
#         final_soln.residual,            # residual r = λy
#         final_soln.optimality,          # norm of the gradient of the dual objective
#         tracer
#     )
#     return stats
# end

# ##############################################################################
# # Oracle
# #
# # This methods solves the same problem as the one in newtoncg.jl, 
# # the only difference is that it stops when the primal objective is less than 
# # α times dual objective. Visit newtoncg.jl for optimization documentation
# ##############################################################################

# function oracle!(
#     kl::DPModel{T},
#     α::T,
#     σ::T,
#     solver::TrunkSolver,
#     tracer::DataFrame;
#     logging=0,
#     max_time::Float64=30.0,
#     kwargs...
# ) where {T}
#     # return values (l, u, s)
#     ret = zeros(T, 3)

#     # Reset the solver
#     SolverCore.reset!(solver, kl)

#     # Callback routine
#     cb(kl, solver, stats) =
#         oracle_callback(kl, solver, stats, tracer, logging, α, σ, ret; kwargs...)

#     stats = SolverCore.solve!(solver, kl; x=kl.meta.x0, callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time)

#     return ret[1], ret[2], ret[3], stats.solution
# end

# function oracle_callback(
#     kl::DPModel{T},
#     solver,
#     trunk_stats,
#     tracer,
#     logging,
#     α,
#     σ,
#     ret;
#     atol::T=DEFAULT_PRECISION(T),
#     rtol::T=DEFAULT_PRECISION(T),
#     max_iter::Int=typemax(Int),
#     trace::Bool=false,
# ) where {T}
#     y = solver.x
#     x = kl.scale * grad(kl.lse)
#     dObj = -trunk_stats.objective - σ
#     iter = trunk_stats.iter
#     r = trunk_stats.dual_feas # = ||∇ dual obj(x)||
#     Δ = solver.tr.radius
#     actual_to_predicted = solver.tr.ratio
#     cgits = solver.subsolver.stats.niter
#     cgexit = get(cg_msg, solver.subsolver.stats.status, "default")
#     ε = atol + rtol * kl.bNrm
#     pObj = pObj!(kl, x) - σ

#     # Test exit conditions
#     tired = iter >= max_iter
#     optimal = r < ε
#     done = tired || optimal

#     # Logging & Tracing
#     ratio = abs(dObj) < eps(T) ? zero(T) : log10(abs(pObj / dObj))
#     log_items = (iter, dObj, pObj, ratio, r, Δ, actual_to_predicted, cgits, cgexit)
#     trace && push!(tracer, log_items)
#     if logging > 0 && iter == 0
#         println("Inside loop:")
#         @printf("%7s  %9s  %9s  %9s  %9s  %9s  %9s  %6s  %10s\n",
#             "iter", "dObj-σ", "pObj-σ", "ratio", "∥∇dObj∥", "Δ", "Δₐ/Δₚ", "cg its", "cg msg")
#         @printf("%7s  %9s  %9s  %9s  %9s  %9s  %9s  %6s  %10s\n",
#             "-----", "-------", "-------", "-------", "-------", "---", "-------", "------", "--------")
#     end
#     if logging > 0 && (mod(iter, logging) == 0 || done)
#         @printf("%7d  %9.2e  %9.2e  %9.1e  %9.1e  %9.1e  %9.1e  %6d   %10s\n", (log_items...))
#     end

#     if optimal
#         trunk_stats.status = :optimal
#     elseif tired
#         trunk_stats.status = :max_iter
#     elseif pObj < α * dObj && dObj > 0
#         st = -(lseatyc!(kl, y) - log(kl.scale) - 1)
#         ret .= [dObj, pObj, st]
#         trunk_stats.status = :user # Ends the oracle iterations
#     end
# end