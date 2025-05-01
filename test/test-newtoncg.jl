using DualPerspective, Test, LinearAlgebra, Random
using NPZ, UnPack

@testset "Newton CG for DPModel" begin
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    x0 = [1/i for i in 1:n]
    x0 = x0 / sum(x0)
    @test all(x0 .>= 0)
    @test sum(x0) ≈ 1.0

    b = randn(m)#A*x0
    λ = 1e-3
    data = DPModel(A, b, λ=λ)
    atol = rtol = 1e-6
    st = solve!(data, atol=atol, rtol=rtol)
    x = st.solution; r = st.residual
    @test norm(A*x + r - b) < atol + rtol*norm(b)

    # Add preconditioning
    M = DualPerspective.AAPreconditioner(data)
    st = solve!(data, M=M, logging=0, atol=atol, rtol=rtol)
end

@testset "Newton CG for DPModel with synthetic" begin
    kl = try # needed because of vscode quirks while developing
        npzread("../data/synthetic-UEG_testproblem.npz")
    catch
        npzread("./data/synthetic-UEG_testproblem.npz")
    end

    @unpack A, b_avg, b_std, mu = kl
    b = b_avg
    q = convert(Vector{Float64}, mu)
    q .= max.(q, 1e-13)
    q .= q./sum(q)
    C = inv.(b_std) |> diagm
    λ = 1e-4
    n = length(q)

    # Create and solve the KL problem
    kl = DPModel(A, b, C=C, c=zeros(n), q=q, λ=λ)
    sP = solve!(kl, atol=1e-5, rtol = 1e-5, logging=0, trace=true)
    @test sP.optimality < 1e-5*kl.bNrm

    # Value-function iteration: nonnegative 
    reset!(kl)
    ssSoln = solve!(kl, SequentialSolve(), verbose=false, rtol=1e-6, logging=0, δ=1e-1)
    x1 = ssSoln.solution
    t1 = kl.scale
    v, dv = DualPerspective.value!(kl, t1)
    @test dv < 1e-6

    # Solve the KL problem with the scaling `t1` obtained above
    reset!(kl)
    scale!(kl, t1)
    sPt = solve!(kl, atol=1e-5, rtol = 1e-6, logging=0, trace=false)
    x2, r2 = sPt.solution, sPt.residual
    @test norm(x1 - x2) < 1e-5
    @test norm(A*x2 + C*r2 - b) < 1e-5
end

@testset "Primal-Dual Obj" begin
    # At a solution, primal and dual objectives must be the same
    Random.seed!(1234)

    m, n = 10, 30
    A = randn(m, n)
    b = randn(m)
    λ = 1e-1
    kl = DPModel(A, b, λ=λ)

    atol = rtol = 1e-6
    st = solve!(kl, atol=atol, rtol=rtol)

    pObj = DualPerspective.pObj!(kl, st.solution)
    dObj = DualPerspective.dObj!(kl, st.residual / λ)
    @test isapprox(pObj, -dObj)
end

@testset "Primal-Dual Obj, Scaled" begin
    # At a solution, primal and dual objectives must be the same
    Random.seed!(1234)

    m, n = 10, 30
    A = randn(m, n)
    b = randn(m)
    λ = 1e-1
    kl = DPModel(A, b, λ=λ)
    scale!(kl, 8.0)

    atol = rtol = 1e-6
    st = solve!(kl, atol=atol, rtol=rtol)

    pObj = DualPerspective.pObj!(kl, st.solution)
    dObj = DualPerspective.dObj!(kl, st.residual / λ)
    @test isapprox(pObj, -dObj)
end
