using Test
using DualPerspective

@testset "Fixed-point for DPModel with synthetic" begin
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
    kl = DPModel(A, b, C=C, q=q, λ=λ)

    α = λ*norm(A, Inf)

    x, stats = fixed_point(kl, verbose=false, α=α, max_iter=100)
    # display(lines(stats.tracer.iter, stats.tracer.∇ϕ, axis=(xlabel="Iteration", ylabel="‖∇ϕ‖", yscale=log10)))

    # println(norm(A*x - b))
    # println(stats)


end

