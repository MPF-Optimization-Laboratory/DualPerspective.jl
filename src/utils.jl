mutable struct ExecutionStats{T<:AbstractFloat, V<:AbstractVector{T}, S<:AbstractArray{T}, DF}
    status::String
    elapsed_time::T
    iter::Int
    neval_jprod::Int
    neval_jtprod::Int
    primal_obj::T
    dual_obj::T
    solution::S
    residual::V
    optimality::T
    tracer::DF
end

function Base.show(io::IO, s::ExecutionStats)
    @printf(io, "\n")
    # if s.status == :max_iter 
    #     @printf(io, "Maximum number of iterations reached\n")
    # elseif s.status == :optimal
    #     @printf(io, "Optimality conditions satisfied\n")
    # end
    @printf(io, s.status)
    nprods = s.neval_jprod + s.neval_jtprod
    @printf(io, "Products with A and A': %9d\n"  , nprods)
    @printf(io, "Time elapsed (sec)    : %9.1f\n", s.elapsed_time)
    @printf(io, "||Ax-b||₂             : %9.1e\n", norm(s.residual))
    @printf(io, "Optimality            : %9.1e\n", s.optimality)
end

"""
    randDPModel(m, n; λ=1e-3) -> DPModel

Generate a random PT model. Arguments:
- `m`: number of rows of the matrix `A`
- `n`: number of columns of the matrix `A`
- `λ`: regularization parameter (default: 1e-3)
"""
function randDPModel(m, n; λ=1e-3)
    A = randn(m, n)
    xs = rand(n)
    xs ./= sum(xs)
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
