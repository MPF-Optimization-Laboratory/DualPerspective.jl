"""
Module for implementing linear operators that map functions to vectors and back.
    
This file implements the MonomialFunctionOperator which performs:

    A*f = [ ∫ₐᵇ x^i f(x) dx ]_{i=1:m}
    
It also implements the adjoint operator A' which maps a vector to a function.
"""

using Test
using Polynomials
using LinearAlgebra
using QuadGK
using ForwardDiff
using BenchmarkTools

"""
    MonomialFunctionOperator{T<:Number}

Linear operator that maps a function to a vector of monomial integrals.
For a function f, computes the vector [∫ₐᵇ x^i f(x) dx]_{i=1:m}

# Fields
- `m::Int`: Number of monomials (from degree 1 to m)
- `domain::Tuple{T,T}`: Integration domain [a,b]
- `rtol::T`: Relative tolerance for quadrature
- `atol::T`: Absolute tolerance for quadrature

# Examples
```julia
# Create operator for first 3 monomials over [0,1]
A = monomial_operator(3, (0.0, 1.0); rtol=1e-10)

# Apply to function exp(x)
moments = A * exp
```
"""
struct MonomialFunctionOperator{T<:Number}
    m::Int                 # Number of monomials
    domain::Tuple{T,T}     # Integration domain [a,b]
    rtol::T                # Relative tolerance for quadrature
    atol::T                # Absolute tolerance for quadrature
    
    function MonomialFunctionOperator{T}(
        m::Int,
        domain::Tuple{T,T};
        rtol::T=sqrt(eps(T)),
        atol::T=sqrt(eps(T))
    ) where T<:Number
        @assert m > 0 "Number of monomials must be positive"
        @assert domain[1] < domain[2] "Domain must be ordered: a < b"
        return new{T}(m, domain, rtol, atol)
    end
end

"""
    monomial_operator(m::Int, domain::Tuple{T,T};
                      rtol=sqrt(eps(T)), atol=zero(T)) where T<:Number

Constructor for MonomialFunctionOperator with type inference from domain.

# Arguments
- `m`: Number of monomials (degrees 1 through m)
- `domain`: Integration domain [a,b]
- `rtol`: Relative tolerance for quadrature
- `atol`: Absolute tolerance for quadrature

# Returns
- `MonomialFunctionOperator{T}`: The constructed operator
"""
function monomial_operator(
    m::Int,
    domain::Tuple{T,T};
    rtol::T=sqrt(eps(T)),
    atol::T=zero(T)
) where T<:Number
    return MonomialFunctionOperator{T}(m, domain; rtol=rtol, atol=atol)
end

"""
    MonomialFunctionAdjointOperator{T<:Number}

Adjoint operator of MonomialFunctionOperator.
Maps a vector of coefficients to a polynomial function.

# Fields
- `parent::MonomialFunctionOperator{T}`: The parent operator being adjointed
"""
struct MonomialFunctionAdjointOperator{T<:Number}
    parent::MonomialFunctionOperator{T}
end

# Size methods - report dimensions of the operator
Base.size(A::MonomialFunctionOperator) = (A.m, :∞)
Base.size(A::MonomialFunctionAdjointOperator) = (:∞, A.parent.m)

# Adjoint methods - create and resolve adjoints
Base.adjoint(A::MonomialFunctionOperator) = MonomialFunctionAdjointOperator(A)
Base.adjoint(A::MonomialFunctionAdjointOperator) = A.parent

"""
    *(A::MonomialFunctionOperator{T}, f::Function) where T<:Number

Forward multiplication: map a function to vector of integrals against monomials.
Computes [∫ₐᵇ x^i f(x) dx]_{i=1:m}

# Arguments
- `A::MonomialFunctionOperator{T}`: The operator
- `f::Function`: Function to integrate against monomials

# Returns
- Vector{T}: Vector of m integrals
"""
function Base.:*(A::MonomialFunctionOperator{T}, f::Function) where T<:Number
    result = zeros(T, A.m)
    a, b = A.domain
    
    for i in 1:A.m
        # Create monomial x^i using Polynomials.jl
        p = variable(:x)^i
        
        # Integrate p(x) * f(x) over [a,b]
        integrand = x -> p(x) * f(x)
        result[i] = quadgk(integrand, a, b; rtol=A.rtol, atol=A.atol)[1]
    end
    
    return result
end

"""
    *(A::MonomialFunctionAdjointOperator{T}, y::AbstractVector) where T<:Number

Adjoint multiplication: map a vector to a polynomial function.
Creates a polynomial from coefficients y and returns a function for its evaluation.

# Arguments
- `A::MonomialFunctionAdjointOperator{T}`: The adjoint operator
- `y::AbstractVector`: Vector of m coefficients

# Returns
- Function: A function x -> p(x) where p is the polynomial with coefficients y
"""
function Base.:*(A::MonomialFunctionAdjointOperator{T}, y::AbstractVector) where T<:Number
    @assert length(y) == A.parent.m "Vector length must match operator dimension"
    p = Polynomial(vcat(0.0, y))
    return p
end

"""
    verify_dot_test(A, f::Function, y::AbstractVector; rtol=1e-10, verbose=false)

Verify the adjoint relationship <A*f, y> = <f, A'*y> through numerical integration.
This function works with any operator A that maps functions to vectors and has an adjoint
that maps vectors to functions.

# Arguments
- `A`: Function-to-vector operator with fields `domain`, `rtol`, and `atol`
- `f::Function`: Test function
- `y::AbstractVector`: Test vector
- `rtol::Real=1e-10`: Relative tolerance for comparison
- `verbose::Bool=false`: Whether to print test results

# Requirements for operator A
- `A * f` must return a vector
- `A'` must return an adjoint operator
- `A' * y` must return a function
- `A.domain` must provide integration limits as a tuple
- `A.rtol` and `A.atol` provide quadrature tolerances

# Returns
- NamedTuple: Results of the test including validity, left and right sides, and difference
"""
function verify_dot_test(
    A,
    f::Function,
    y::AbstractVector;
    rtol::Real=1e-10,
    verbose::Bool=false
)
    # Forward operation: A*f
    Af = A * f
    
    # First inner product: <A*f, y>
    lhs = dot(Af, y)
    
    # Adjoint operation: A'*y
    Aty = A' * y
    
    # Second inner product: <f, A'*y>
    # Integrate f(x)*Aty(x) over domain
    a, b = A.domain
    integrand = x -> f(x) * Aty(x)
    rhs = quadgk(integrand, a, b; rtol=A.rtol, atol=A.atol)[1]
    
    # Print results if verbose option is enabled
    if verbose
        result = (is_valid=is_valid, lhs=lhs, rhs=rhs, diff=abs(lhs-rhs))
        println("\nDot-test results:")
        println("Passed: $(result.is_valid)")
        println("LHS <A*f, y>: $(result.lhs)")
        println("RHS <f, A'*y>: $(result.rhs)")
        println("Difference: $(result.diff)")
    end
    
    return isapprox(lhs, rhs, rtol=rtol)
end

"""
    polynomial_maximum(p::Polynomial, domain::Tuple{T,T}) where T<:Real

Compute the maximum value of a polynomial over a given domain.
Uses root-finding on the derivative to locate critical points.

# Arguments
- `p::Polynomial`: Polynomial to maximize
- `domain::Tuple{T,T}`: Domain [a,b] over which to maximize

# Returns
- Tuple containing (max_value, location)
"""
function polynomial_maximum(p::Polynomial, domain::Tuple{T,T}) where T<:Real
    a, b = domain
    
    # Extract real part of coefficients for root finding
    # This ensures that operations within root-finding are on standard Float types,
    # making it compatible with ForwardDiff when p's coefficients are Dual numbers.
    # The locations of critical points depend only on the real part of the coefficients.
    p_coeffs_values = ForwardDiff.value.(coeffs(p))
    p_real_coeffs = Polynomial(p_coeffs_values)
    
    # Find all critical points of the derivative of the real-coefficient polynomial
    dp_real_coeffs = derivative(p_real_coeffs)
    all_critical_points_values = roots(dp_real_coeffs)
    
    # Filter for real roots within the domain
    # Ensure that we are comparing real parts for domain checks if critical points can be complex
    critical_points_on_domain = filter(x -> isreal(x) && a ≤ real(x) ≤ b, all_critical_points_values)
    # Convert to Real type; note that real() on a Real number is a no-op.
    real_critical_points = real.(critical_points_on_domain)
    
    # Evaluate the original polynomial (p, which might have Dual coefficients) 
    # at critical points (which are Real) and domain endpoints (which are Real).
    all_points_to_check = unique(vcat(a, real_critical_points, b)) # Use unique to avoid re-evaluating identical points
    
    # Ensure all_points_to_check are within the domain strictly, if filter was too loose or for endpoints.
    # This step might be redundant if the filter is precise, but good for robustness.
    # Actually, endpoints must be included regardless, and critical points are already filtered.
    # So, unique(vcat(a, real_critical_points, b)) is correct.

    values_at_points = [p(x_val) for x_val in all_points_to_check]
    
    # Find the maximum value and its location
    # findmax will work with Dual numbers if values_at_points contains them.
    max_val, idx = findmax(values_at_points)
    max_loc = all_points_to_check[idx]
    
    return (value=max_val, location=max_loc)
end

"""
    monomial_operator_example()

Example demonstrating the use of MonomialFunctionOperator and its adjoint.
Tests the operator with exp(x) and verifies the adjoint relationship.

# Returns
- Tuple: The operator, result vector, adjoint function, and dot-test results
"""
function monomial_operator_example()

    # Test with finite domain
    bnds = (-5.0, 5.0)

    # Operator
    A = monomial_operator(4, bnds)
    
    # Uniform Prior
    q(x) = 1.0 / (bnds[2] - bnds[1]) * (bnds[1] ≤ x ≤ bnds[2])

    
    # Test function
    f(x) = (1/sqrt(2π)) * exp(-x^2/2)
    
    # Apply operator to get integrals of monomials against f
    result = A * f
    println("Integrals of x^i * N(0,1) from $(A.domain[1]) to $(A.domain[2]):")
    for (i, val) in enumerate(result)
        # Calculate exact value for the ith moment of N(0,1)
        exact = if iseven(i)
            prod(i-1:-2:1)  # Double factorial (i-1)!!
        else
            0.0
        end
        # println("abs(val-exact)=", abs(val-exact))
        @test isapprox(val, exact, atol=1e-4, rtol=i*1e-4)
    end
    
    # Test adjoint with coefficient vector
    # y = [1.0, 2.0, 3.0, 4.0]
    y = rand(4)
    g = A' * y
    println("\nEvaluating adjoint (polynomial p(x) = 1 + 2x + 3x² + 4x³) at x=0.5: $(g(0.5))")
    
    # Verify dot-test
    @test verify_dot_test(A, f, y, verbose=false)
    
    # Compute the maximum of g(x) over a finite domain
    p = Polynomial(y)  # Convert coefficients to a polynomial
    max_result = polynomial_maximum(p, bnds)
    println("\nPolynomial maximum:")
    println("Maximum value: $(max_result.value)")
    println("Location: $(max_result.location)")
    
    # dObj(y) = logexp(A'y | q) = log(∫ q(x)⋅exp((A'y)(x)) dx)
    dObj(y_vec) = begin
        p_poly = A'y_vec # p_poly is a Polynomial object
        # Ensure pmax_val is of a type compatible with later arithmetic (e.g., Float64)
        T_calc = eltype(y_vec) # Or promote_type(typeof(A.rtol), eltype(y_vec))
        pmax_val = T_calc(polynomial_maximum(p_poly, bnds).value)

        # For objective value
        integrand_obj = x -> q(x) * exp(p_poly(x) - pmax_val)
        sumexp_val = quadgk(integrand_obj, bnds...; atol=A.atol, rtol=A.rtol)[1]
        obj_val = log(sumexp_val) + pmax_val

        # For analytical gradient numerator (vectorized)
        # The j-th component of the gradient numerator is ∫ x^j * q(x) * exp(p_poly(x) - pmax_val) dx
        vector_integrand_grad = x_val -> begin
            px_at_xval = p_poly(x_val)
            common_term = q(x_val) * exp(px_at_xval - pmax_val)
            
            grad_vector_terms = Vector{T_calc}(undef, A.m)
            current_x_power = x_val # for x^1
            for j in 1:A.m
                grad_vector_terms[j] = common_term * current_x_power
                if j < A.m # Avoid overflow if x_val is large and A.m is large, though unlikely here
                    current_x_power *= x_val # for next power x^(j+1)
                end
            end
            return grad_vector_terms
        end
        
        analytical_grad_numerator_vec = quadgk(vector_integrand_grad, bnds...; atol=A.atol, rtol=A.rtol)[1]
        analytical_grad_vec = analytical_grad_numerator_vec / sumexp_val
        
        return obj_val, analytical_grad_vec
    end

    obj_val, analytical_grad = dObj(y)
    println("\nObjective value: $(obj_val)")
    println("Analytical gradient: $(analytical_grad)")

    # Define a function that returns only the objective for ForwardDiff
    objective_for_fd(y_val) = begin
        p_fd = A'y_val
        pmax_fd = polynomial_maximum(p_fd, bnds).value
        
        integrand_fd = x -> q(x)*exp(p_fd(x)-pmax_fd)
        sumexp_fd = quadgk(integrand_fd, bnds..., atol=A.atol, rtol=A.rtol)[1]
        obj_fd = log(sumexp_fd)+pmax_fd
        return obj_fd
    end

    # Compute numerical gradient using ForwardDiff
    numerical_grad = ForwardDiff.gradient(objective_for_fd, y)
    println("Numerical gradient (ForwardDiff): $(numerical_grad)")

    # Compare gradients
    grad_diff = analytical_grad - numerical_grad
    println("Difference (Analytical - Numerical): $(grad_diff)")
    println("Component-wise absolute errors:")
    for i in 1:length(y)
        println("  Component $(i): $(abs(grad_diff[i]))")
    end
    @test isapprox(analytical_grad, numerical_grad, atol=1e-6) # Add a test for comparison

    println("\n--- Benchmarking Gradient Computations ---")

    println("Benchmarking Analytical Gradient:")
    # Interpolate dObj as it's a local closure
    analytical_bench = @benchmarkable $dObj($y)[2] 
    # run returns a Trial object, median operates on this Trial object
    trial_analytical = run(analytical_bench, seconds=1.0) 
    println(median(trial_analytical)) # Get median from the trial results


    println("\nBenchmarking ForwardDiff Gradient:")
    # objective_for_fd is also a local closure and is correctly interpolated here
    forward_diff_bench = @benchmarkable ForwardDiff.gradient($objective_for_fd, $y)
    trial_forward_diff = run(forward_diff_bench, seconds=1.0) 
    println(median(trial_forward_diff)) # Get median from the trial results

end
monomial_operator_example()
