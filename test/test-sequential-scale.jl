using Test
using LinearAlgebra
using Random
using DualPerspective

@testset "SSModel SequentialSolve test case" begin
      Random.seed!(1234)
      tol = 2e-5
      λ = 1e-2
      m, n = 8, 10
      kl = DualPerspective.randDPModel(m, n) 
      A, b = kl.A, kl.b
      regularize!(kl, λ)

      stats = solve!(kl)

      x = stats.solution
      r = stats.residual
      y = r/λ

      @test norm(A*x + r - b) < tol

      rtol = tol
      atol = tol
      ssSoln = solve!(kl, SequentialSolve(), logging=0, atol=atol, rtol=rtol, verbose=false)

      @test ssSoln.status == :optimal
end

@testset "SequentialSolve status reflects convergence" begin
      # The status used to be derived from `tracker.convergence_flag == :x_converged`
      # alone. Roots far more often reports `:f_converged`, so successful solves came back
      # as `:unknown`. A single problem can pass that by luck, so sweep a few.
      Random.seed!(1234)
      for (m, n) in ((10, 5), (50, 100), (8, 10)), λ in (1e-2, 1e-3, 1e-5)
            kl = DualPerspective.randDPModel(m, n; λ=λ)
            stats = solve!(kl, SequentialSolve(); t=sum(kl.q))
            @test stats.status == :optimal
            # The status must agree with the criterion the inner solve actually applies.
            @test stats.optimality < 1e-6 + 1e-6 * kl.bNrm
      end
end

@testset "SequentialSolve status mapping" begin
      # Every Roots success flag must be accepted, and the inner solve's own status passed
      # through rather than overwritten.
      inner(status) = (; status=status)
      for flag in DualPerspective.ROOTS_CONVERGED
            @test DualPerspective._sequential_status((; convergence_flag=flag), inner(:optimal)) == :optimal
            @test DualPerspective._sequential_status((; convergence_flag=flag), inner(:max_iter)) == :max_iter
      end
      for flag in (:not_converged, :nan, :inf, :algorithm_not_run)
            @test DualPerspective._sequential_status((; convergence_flag=flag), inner(:optimal)) == :stalled
      end
end