using Test
using LinearAlgebra
using Random
using JuMP
using GLPK
using DualPerspective

@testset "Feasable LP" begin
    Random.seed!(1234)
    m = 10 
    n = 20

    A = rand(m, n)

    x_feasible = rand(n)
    b = A * x_feasible

    c = rand(n)

    lp = DualPerspective.LPModel(A, b, c, ε=5e-3, λ=5e-3)
    stats = solve!(lp, trace=true)
    optimal_x_lpmodel = stats.solution

    # Compare with JuMP LP solution
    jump_model = Model(GLPK.Optimizer)
    @variable(jump_model, x[1:n] >= 0)
    @constraint(jump_model, A * x .== b)
    @objective(jump_model, Min, sum(c .* x))
    optimize!(jump_model)
    objective_jump = objective_value(jump_model)
    optimal_x_jump = value.(x)

    # Tests
    @test all(optimal_x_lpmodel .>= 0)  # Non-negativity
    @test norm(A * optimal_x_lpmodel - b, Inf) < 1e-2  # Feasibility
    @test dot(c, optimal_x_lpmodel) ≈ objective_jump rtol=1e-1  # Optimality
end

@testset "Infeasible LP" begin
    Random.seed!(1234)
    m = 10
    n = 20

    A = rand(m, n)

    x_feasible = rand(n)
    b = A * x_feasible

    # Modify b to make the problem infeasible
    # set the first row of A to zero and set a non-zero value in b
    A[1, :] .= 0.0
    b[1] = 1.0  # Since A[1, :] * x = 0 for any x, setting b[1] ≠ 0 makes it infeasible

    c = rand(n)

    lp = DualPerspective.LPModel(A, b, c, ε=5e-1, λ=5e-1)
    stats = solve!(lp)
    @test stats.status == :infeasible

    model = Model(GLPK.Optimizer)

    @variable(model, x[1:n] >= 0)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .== b)

    optimize!(model)

    status = termination_status(model)
    @test status == MOI.INFEASIBLE
end
