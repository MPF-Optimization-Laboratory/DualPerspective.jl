using Test
using DualPerspective
using DualPerspective.DensityEstimation

mean(x) = sum(x) / length(x)

@testset "Uniform Die Density Estimation" begin
    p_true = ones(6) / 6
    x_values = 1:6
    m_order = 1
    A = moment_operator(x_values, m_order)
    b = [mean(x_values)] # Corresponds to the first raw moment

    model = DPModel(A, b)
    
    # Solve the moment problem
    status = solve!(model)
    p_computed = status.solution
    
    @test p_computed ≈ p_true
end 