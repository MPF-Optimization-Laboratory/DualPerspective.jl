using Test
using OptimalTransport
using LinearAlgebra
using Statistics
using Distances: SqEuclidean, pairwise
using DualPerspective

@testset "OTModel correctness relative to sinkhorn" begin
    μsupport = νsupport = range(-2, 2; length=100)
    C2 = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)
    μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)
    ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)


    Tsk, Tkl, klstats = let
        ϵ = 0.01*median(C2)
        Tsk = sinkhorn(μ, ν, C2, ϵ)
        Tkl, klstats = let
            ot = DualPerspective.OTModel(μ, ν, C2, ϵ)
            solve!(ot, trace=true)
        end
        Tsk, Tkl, klstats
    end

    @test isapprox(Tsk, Tkl, atol=1e-5)
end
