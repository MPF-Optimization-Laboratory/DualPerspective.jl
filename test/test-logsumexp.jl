using DualPerspective, Test, LinearAlgebra, Random
import DualPerspective: LogExpFunction, kl_divergence, obj!, grad, hess, myfindmax, sum_all_but
using ForwardDiff

@testset "LogExpFunction Constructor" begin
    # Valid construction tests
    q = [0.2, 0.3, 0.5]
    lse = LogExpFunction(q)
    @test lse.q == q
    @test length(lse.g) == length(q)
    
    # Error handling tests
    @test_throws AssertionError LogExpFunction([-0.1, 0.5, 0.6])
    
    # Unnormalized vector test
    q_unnorm = [1.0, 2.0, 3.0]
    lse_unnorm = LogExpFunction(q_unnorm)
    @test lse_unnorm.q == q_unnorm
end

@testset "KL Divergence" begin
    # Identical distributions
    x = [0.2, 0.3, 0.5]
    @test kl_divergence(x, x) ≈ 0.0 atol=1e-12
    
    # Known example
    p = [0.5, 0.5]
    q = [0.9, 0.1]
    expected_kl = 0.5 * log(0.5/0.9) + 0.5 * log(0.5/0.1)
    @test kl_divergence(p, q) ≈ expected_kl atol=1e-12
    
    # Zero elements in x
    x_with_zero = [0.0, 0.3, 0.7]
    q_no_zero = [0.1, 0.3, 0.6]
    expected_kl_with_zero = 0.3 * log(0.3/0.3) + 0.7 * log(0.7/0.6)
    @test kl_divergence(x_with_zero, q_no_zero) ≈ expected_kl_with_zero atol=1e-12
    
    # Non-negativity property
    Random.seed!(123)
    for _ in 1:10
        x_rand = rand(5)
        x_rand = x_rand / sum(x_rand)
        q_rand = rand(5)
        q_rand = q_rand / sum(q_rand)
        @test kl_divergence(x_rand, q_rand) >= -1e-12  # Allow for numerical precision
    end
end

@testset "LogΣexp Evaluation (obj!)" begin
    Random.seed!(123)
    q = [0.2, 0.3, 0.5]
    lse = LogExpFunction(q)
    
    # Simple case
    z = [1.0, 2.0, 0.5]
    f = obj!(lse, z)
    
    # Direct calculation for verification
    expected_f = log(sum(q .* exp.(z)))
    @test f ≈ expected_f atol=1e-12
    
    # Test with different inputs
    z2 = [-1.0, 0.0, 1.0]
    f2 = obj!(lse, z2)
    expected_f2 = log(sum(q .* exp.(z2)))
    @test f2 ≈ expected_f2 atol=1e-12
end

@testset "Helper Functions" begin
    # myfindmax tests
    arr = [3.0, 1.0, 5.0, 2.0]
    max_val, max_idx = myfindmax(arr)
    @test max_val == 5.0
    @test max_idx == 3
    
    # Edge cases for myfindmax
    arr_single = [7.0]
    max_val_single, max_idx_single = myfindmax(arr_single)
    @test max_val_single == 7.0
    @test max_idx_single == 1
    
    # Duplicates in myfindmax (should return first occurrence)
    arr_dup = [3.0, 5.0, 5.0, 2.0]
    max_val_dup, max_idx_dup = myfindmax(arr_dup)
    @test max_val_dup == 5.0
    @test max_idx_dup == 2
    
    # sum_all_but tests
    w = [1.0, 2.0, 3.0, 4.0]
    s = sum_all_but(w, 2)
    @test s == sum(w) - w[2]
    @test w[2] == 2.0  # Check that w is unchanged
    
    # Edge cases for sum_all_but
    w_single = [5.0]
    s_single = sum_all_but(w_single, 1)
    @test s_single == 0.0
    @test w_single[1] == 5.0
end

@testset "Gradient and Hessian" begin
    q = [0.25, 0.25, 0.5]
    lse = LogExpFunction(q)
    p = [0.1, 0.2, -0.1]
    
    # Compute function, gradient, and Hessian
    f = obj!(lse, p)
    g = grad(lse)
    H = hess(lse)
    
    # Test Hessian properties
    @test H ≈ H' atol=1e-12  # Symmetry
    
    # Create random direction vectors to test positive semi-definiteness
    Random.seed!(456)
    for _ in 1:10
        d = randn(3)
        d = d / norm(d)
        # Ensure d'Hd ≥ 0 for all d (positive semi-definiteness)
        @test d' * H * d >= -1e-12
    end
    
    # Check that H = Diagonal(g) - g*g'
    expected_H = Diagonal(g) - g * g'
    @test H ≈ expected_H atol=1e-12
    
    # Test gradient consistency
    g_after = grad(lse)
    @test g_after == g
end

@testset "Numerical Stability" begin
    q = ones(5) ./ 5
    lse = LogExpFunction(q)
    
    # Test with large values
    p_large = [1000.0, 0.0, 0.0, 0.0, 0.0]
    f_large = obj!(lse, p_large)
    # Should be close to the maximum + small correction
    @test f_large ≈ 1000.0 + log(0.2 + 0.8*exp(-1000.0)) atol=1e-8
    
    # Test with small values
    p_small = [-1000.0, 0.0, 0.0, 0.0, 0.0]
    f_small = obj!(lse, p_small)
    # Should be close to the maximum (0.0)
    @test f_small ≈ 0.0 + log(0.8 + 0.2*exp(-1000.0)) atol=1e-8
    
    # Test with mixed large and small values
    p_mixed = [-1000.0, 1000.0, 0.0, -500.0, 500.0]
    f_mixed = obj!(lse, p_mixed)
    # Should handle this case without numerical issues
    max_p_mixed = maximum(p_mixed)
    expected_f_mixed = max_p_mixed + log(sum(q .* exp.(p_mixed .- max_p_mixed)))
    @test f_mixed ≈ expected_f_mixed atol=1e-8
    
    # Test gradient stability with extreme values
    g_large = grad(lse)
    @test all(isfinite.(g_large))
    @test sum(g_large) ≈ 1.0 atol=1e-12
    
    # Test with skewed prior (one element dominates, others very small)
    q_skewed = [1.0 - 4e-14, 1e-14, 1e-14, 1e-14, 1e-14]
    lse_skewed = LogExpFunction(q_skewed)
    
    # Test with uniform input
    p_uniform = zeros(5)
    f_skewed = obj!(lse_skewed, p_uniform)
    # Expected to be approximately log(q_skewed[1]) since other terms contribute negligibly
    @test f_skewed ≈ log(q_skewed[1]) atol=1e-12
    
    # Test gradient with skewed prior
    g_skewed = grad(lse_skewed)
    @test all(isfinite.(g_skewed))
    @test sum(g_skewed) ≈ 1.0 atol=1e-12
    @test g_skewed[1] ≈ 1.0 atol=1e-12  # First element should dominate
    
    # Test with large value in position of small prior
    p_contrast = [-1.0, 1000.0, -1.0, -1.0, -1.0]  # Large value at position of small prior
    f_contrast = obj!(lse_skewed, p_contrast)
    # Despite q[2] being tiny, the large p[2] should make it visible in the result
    @test f_contrast ≈ 1000.0 + log(q_skewed[2] + sum(q_skewed .* exp.(p_contrast .- 1000.0))) atol=1e-12 rtol=1e-3
    
    # Verify gradient behaves properly even with extreme prior/value combinations
    g_contrast = grad(lse_skewed)
    @test all(isfinite.(g_contrast))
    @test sum(g_contrast) ≈ 1.0 atol=1e-12
end 