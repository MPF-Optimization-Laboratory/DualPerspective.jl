mutable struct ExecutionStats{T<:AbstractFloat, V<:AbstractVector{T}, S<:AbstractArray{T}, DF}
    converged::Bool
    elapsed_time::T
    iter::Int
    neval_jprod::Int
    neval_jtprod::Int
    primal_obj::T
    dual_obj::T
    scale::T
    solution::S
    residual::V
    inner_optimality::T
    outer_optimality::T
    tracer::DF
end

function mvps(s::ExecutionStats) return s.neval_jprod + s.neval_jtprod end
function residual(s::ExecutionStats) return norm(s.residual) end

function Base.show(io::IO, s::ExecutionStats)
    nprods = s.neval_jprod + s.neval_jtprod
    @printf(io, "Converged:                   %s\n", s.converged ? "true" : "false")
    @printf(io, "Products with A and A': %9d\n"  , nprods)
    @printf(io, "Time elapsed (sec):     %9.1f\n", s.elapsed_time)
    @printf(io, "Outer Iterations:       %9d\n"  , s.iter)
    @printf(io, "Inner Optimality:       %9.1e\n", s.inner_optimality)
    @printf(io, "Outer Optimality:       %9.1e\n", s.outer_optimality)
    @printf(io, "Residual ||Ax-b||₂:     %9.1e\n", norm(s.residual))
    @printf(io, "Scale:                  %9.1e\n", s.scale)
end

"""
    randDPModel(m, n; λ=1e-3) -> DPModel

Generate a random PT model. Arguments:
- `m`: number of rows of the matrix `A`
- `n`: number of columns of the matrix `A`
- `λ`: regularization parameter (default: 1e-3)
"""
function randDPModel(m, n; scale=1e0, λ=1e-3)
    A = randn(m, n)
    xs = rand(n)
    xs ./= sum(xs)
    xs .*= scale
    b = A * xs
    return DPModel(A, b, λ=λ)
end

"""
    histogram(s:ExecutionStats; kwargs...)

Plot a histogram of the solution.
"""
function histogram end

"""
    version() -> String

Return the current version of the DualPerspective package.
"""
function version()
    # Get the path to the package root directory
    package_path = dirname(dirname(pathof(DualPerspective)))
    
    # Path to Project.toml
    project_path = joinpath(package_path, "Project.toml")
    
    # Read the file and find the version line
    version_line = ""
    open(project_path, "r") do file
        for line in eachline(file)
            if startswith(line, "version =") || startswith(line, "version=")
                version_line = line
                break
            end
        end
    end
    
    # Extract the version string (between quotes)
    if !isempty(version_line)
        # Find the position of the first and last quote
        first_quote = findfirst('"', version_line)
        last_quote = findlast('"', version_line)
        
        if first_quote !== nothing && last_quote !== nothing
            return version_line[first_quote+1:last_quote-1]
        end
    end
    
    return "unknown"  # Fallback if version not found
end
