```@meta
CurrentModule = DualPerspective
```

# Examples

This section provides examples demonstrating how to use DualPerspective.jl for solving various problems.

## Optimal Transport

Solving an optimal transport problem with entropic regularization:

```julia
using DualPerspective, LinearAlgebra, Distances

# Create supports for the distributions
μsupport = νsupport = range(-2, 2; length=100)

# Compute the cost matrix using squared Euclidean distance
C = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)

# Create source and target distributions
μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)                    # Source distribution
ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)   # Target distribution

# Set the regularization parameter (entropy)
ϵ = 0.01 * median(C)

# Create and solve the optimal transport model
ot = DualPerspective.OTModel(μ, ν, C, ϵ)
solution = solve!(ot, trace=true)

# Visualize the solution if UnicodePlots is available
using UnicodePlots
histogram(solution)
```

## Random DPModel

Creating and solving a random dual perspective model:

```julia
using DualPerspective, LinearAlgebra

# Create a random DPModel
m, n = 10, 20  # dimensions
model = randDPModel(m, n)

# Solve the model
result = solve!(model, trace=true)

# Print summary
println("Status: ", result.status)
println("Objective value: ", result.objective)
println("Iterations: ", result.iter)
``` 