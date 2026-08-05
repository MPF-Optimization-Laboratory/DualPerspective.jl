using DualPerspective
using Random
using Test

# NOTE: the JSO packages below are `import`ed, never `using`ed. `runtests.jl` includes every
# test file into the same `Main`, and `using SolverCore` here would put its competing
# `solve!`/`reset!` exports into `Main` and break every later test file -- which is exactly
# the class of breakage this file guards against.
import JSOSolvers
import LinearOperators
import NLPModels
import SolverCore

# Regression guard for the v0.1.5 fix.
#
# `reset!` is owned by LinearOperators and extended/re-exported by NLPModels. SolverCore
# 0.3.9 forked off its own `reset!`, so `using SolverCore` inside DualPerspective made the
# bare name ambiguous and every solve failed with `UndefVarError: reset! not defined`.
# These tests fail if that `import SolverCore` is ever widened back to a `using`, or if a
# dependency forks the binding again.

@testset "reset! resolves to the NLPModels binding" begin
    @test DualPerspective.reset! === LinearOperators.reset!
    @test DualPerspective.reset! === NLPModels.reset!
    @test DualPerspective.reset! !== SolverCore.reset!
end

@testset "reset! clears model counters" begin
    Random.seed!(1234)
    kl = randDPModel(10, 5; λ=1e-3)
    DualPerspective.solve!(kl, SequentialSolve(); t=sum(kl.q))
    @test NLPModels.neval_jprod(kl) > 0
    NLPModels.reset!(kl)
    @test NLPModels.neval_jprod(kl) == 0
    @test NLPModels.neval_jtprod(kl) == 0
end

# The user-facing contract: `using DualPerspective` on its own must give a usable bare
# `reset!`. Exercised in a fresh module so the surrounding Main imports cannot mask it.
module BareResetScope
using DualPerspective
using Random
Random.seed!(1234)
const kl = randDPModel(10, 5; λ=1e-3)
DualPerspective.solve!(kl, SequentialSolve(); t=sum(kl.q))
const before = DualPerspective.neval_jprod(kl)
reset!(kl)
const after = DualPerspective.neval_jprod(kl)
end

@testset "bare reset! works under plain `using DualPerspective`" begin
    @test BareResetScope.before > 0
    @test BareResetScope.after == 0
end

@testset "JSOSolvers exposes krylov_subsolver" begin
    # Renamed from `subsolver` in JSOSolvers 0.14.1. The callbacks in level-set.jl,
    # newtoncg.jl and selfscale.jl read this field on every iteration, so a further
    # upstream rename must fail here rather than deep inside a solve.
    Random.seed!(1234)
    kl = randDPModel(10, 5; λ=1e-3)
    @test hasproperty(JSOSolvers.TrunkSolver(kl), :krylov_subsolver)
end
