using DualPerspective, Test, LinearAlgebra, Random

@testset "DPModel correctness" begin
    Random.seed!(1234)
    m, n = 200, 400
    q = (v=rand(n); v/sum(v))
    A = randn(m, n)
    b = randn(m) 
    λ = 1e-3
    data = DPModel(A, b, q=q, λ=λ)

    @test size(data.A) == (m, n)
    @test size(data.b) == (m,)
    @test size(data.q) == (n,)
    @test data.λ == λ
end

@testset "Modifiers" begin
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    b = randn(m)
    data = DPModel(A, b)

    # Test scaling
    @test try
        scale!(data, 0.5)
        true
    catch
        false
    end
    @test data.scale == 0.5

    # Test regularization
    @test try
        regularize!(data, 1e-3)
        true
    catch
        false
    end
    @test data.λ == 1e-3

    # Test initial guess update
    y0 = ones(Float64, m)
    @test try
        update_y0!(data, y0)
        true
    catch
        false
    end
    @test data.meta.x0 == y0
    
end

@testset "Constructor internal consistency" begin
    Random.seed!(1234)
    m, n = 5, 10
    A = randn(m, n)
    b = randn(m)
    model = DPModel(A, b)
    
    # Test that y0 equals meta.x0
    @test model.y0 === model.meta.x0
    
    # Evaluate the model at the initial point to ensure LSE is computed
    _ = DualPerspective.dObj!(model, model.y0)
    
    # Test that x0 equals the gradient of LSE
    x0 = model.scale .* grad(model.lse)
    @test norm(x0) > 0  # Ensure gradient is non-zero
    @test isa(x0, Vector)
    @test size(x0) == (n,)
end
