using Test
using DualPerspective
using Statistics # For mean()

"""
   moment_operator(x, m)

Compute the moment operator for given points `x` up to order `m`.

- `x`: n-vector of locations.
- `m`: maximum order of moment to compute.
- `A`: (m)x(n) matrix for moments.
"""
function moment_operator(x, m)
   A = zeros(m, length(x))
   for k in 1:m
      A[k, :] = x.^k
   end
   return A
end

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