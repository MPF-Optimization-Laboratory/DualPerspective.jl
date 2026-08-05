using DualPerspective
using Random
using Test

# The JSOSolvers `subsolver` -> `krylov_subsolver` rename (0.14.1) went unnoticed because
# nothing exercised the trust-region callbacks with logging or tracing enabled. Those
# callbacks read the Krylov workspace on every iteration, so each solve below is a
# regression gate for that field.
#
# These also cover keyword forwarding: `LevelSet`/`AdaptiveLevelSet` used to splat
# `kwargs...` positionally, so passing any keyword they do not name by hand (such as
# `trace`) raised a MethodError.

solve_quietly(f) = redirect_stdout(devnull) do
    f()
end

@testset "callbacks with logging and tracing enabled" begin
    Random.seed!(1234)
    m, n = 10, 5
    mk() = randDPModel(m, n; λ=1e-3)

    @testset "SequentialSolve" begin
        kl = mk()
        stats = solve_quietly() do
            DualPerspective.solve!(kl, SequentialSolve(); t=sum(kl.q), logging=2, trace=true)
        end
        @test stats.status in (:optimal, :max_iter)
    end

    @testset "LevelSet" begin
        kl = mk()
        stats = solve_quietly() do
            DualPerspective.solve!(kl, LevelSet(); t=sum(kl.q), logging=2, trace=true)
        end
        @test stats.status in (:optimal, :max_iter)
    end

    # LevelSet used to drop the caller's atol/rtol on the floor (they never reached the
    # final solve), so it always converged to the package default instead. Assert the
    # documented contract: the final gradient norm must meet atol + rtol*‖b‖.
    @testset "LevelSet honours atol/rtol" begin
        kl = mk()
        tol = 1e-9
        stats = solve_quietly() do
            DualPerspective.solve!(kl, LevelSet(); t=sum(kl.q), atol=tol, rtol=tol)
        end
        @test stats.optimality < tol + tol * kl.bNrm
    end

    @testset "AdaptiveLevelSet" begin
        kl = mk()
        stats = solve_quietly() do
            DualPerspective.solve!(kl, AdaptiveLevelSet(); t=sum(kl.q), logging=2, trace=true)
        end
        @test stats.status in (:optimal, :max_iter)
    end

    # Only these two populate their tracer, so they are the ones that prove the callback
    # actually reached the Krylov-workspace read rather than short-circuiting earlier.
    @testset "SSTrunkLS" begin
        kl = mk()
        stats = solve_quietly() do
            DualPerspective.solve!(kl, SSTrunkLS(); logging=2, trace=true, atol=1e-6, rtol=1e-6)
        end
        @test size(stats.tracer, 1) > 0
        @test :cgits in propertynames(stats.tracer)
        @test :cgmsg in propertynames(stats.tracer)
    end

    @testset "trust-region Newton-CG" begin
        kl = mk()
        stats = solve_quietly() do
            DualPerspective.solve!(kl; logging=2, trace=true)
        end
        @test stats.status in (:optimal, :max_iter)
        @test size(stats.tracer, 1) > 0
        @test :cgits in propertynames(stats.tracer)
    end
end
