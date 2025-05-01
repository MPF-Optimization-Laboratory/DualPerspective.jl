using Test
using DualPerspective, NLPModels, LinearAlgebra, Random
import DualPerspective: lseatyc!, obj!, randDPModel

@testset "SSModel SequentialSolve test case" begin
      Random.seed!(1234)
      tol = 2e-5
      λ = 1e-2
      m, n = 8, 10
      kl = randDPModel(m, n) 
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