using Test
using DualPerspective
using LinearAlgebra
using Random
using NLPModels: residual!, jtprod_residual

@testset "SSModel test case" begin
      Random.seed!(1234)
      tol = 2e-5
      λ = 1e-2
      m, n = 8, 10
      kl = DualPerspective.randDPModel(m, n; λ=λ) 
      A, b = kl.A, kl.b

      stats = solve!(kl)

      x = stats.solution
      r = stats.residual
      y = r/λ

      @test norm(A*x + r - b) < tol

      # test `residual!` (called by `residual`)
      ss = SSModel(kl)
      Fy = residual!(ss, [y; 1], similar([y; 1]))
      @test norm(Fy[1:m]) < tol
      Jy = jtprod_residual(ss, [y; 1], [Fy[1:m]; 0])
      @test norm(Jy) < tol

      # Form the Jacobian manually and compare
      t = 1.0
      scale!(kl, t)
      Jtrue = [ DualPerspective.dHess(kl) A*x; (A*x)' -1/t ]
      J = let
            e(i) = let e = zeros(m+1); e[i] = 1; e end
            reduce(hcat, [jtprod_residual(ss, [y; t], e(i)) for i in 1:m+1])
      end
      @test all( .≈(J, Jtrue, atol=1e-6) )

      rtol = tol
      atol = tol
      ssSoln = solve!(kl, SSTrunkLS(), trace=true, logging=0, atol=atol, rtol=rtol)

      x = ssSoln.solution
      r = ssSoln.residual
      y = r/λ
      t = sum(x)

      Fy = residual!(ss, [y; t], similar([y; t]))
      @test norm(Fy[1:m]) < atol + rtol*max(1, kl.bNrm)
      @test norm(A*x + r - b) ≤ atol + rtol*max(1, kl.bNrm)
end