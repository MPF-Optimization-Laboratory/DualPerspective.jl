using DualPerspective, Test, LinearAlgebra, Random
using NPZ, UnPack

@testset "Level Set Method for random DPModel" begin
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    x0 = [1/i for i in 1:n]
    x0 = x0 / sum(x0)
    @test all(x0 .>= 0)
    @test sum(x0) ≈ 1.0

    b = randn(m)#A*x0
    λ = 1e-3
    kl = DPModel(A, b, λ=λ)
    
    # Get the optimal objective value.
    # Assumes that the dual problem is **minimization**, thus the negative sign.
    σ = -solve!(kl, SequentialSolve()).dual_obj

    atol = rtol = 1e-6
    st = solve!(kl, LevelSet(), α=1.5, σ=σ, atol=atol, rtol=rtol)
    x = st.solution; r = st.residual
    @test norm(A*x + r - b) < atol + rtol*norm(b)

    # Add preconditioning
    M = DualPerspective.AAPreconditioner(kl)
    st = solve!(kl, M=M, logging=0, atol=atol, rtol=rtol)
end

@testset "Level Set Method for DPModel with synthetic kl" begin
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
    σ = -solve!(kl, SequentialSolve()).dual_obj # Find the optimal objective value
    sP = solve!(kl, LevelSet(), α=1.5, σ=σ, atol=1e-5, rtol = 1e-5)
    @test sP.optimality < 1e-5*kl.bNrm
end

@testset "Adaptive Level Set Method for DPModel with synthetic kl" begin
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
    sP = solve!(kl, AdaptiveLevelSet(), α=1.5, atol=1e-5, rtol = 1e-5)
    @test sP.optimality < 1e-5*kl.bNrm
end