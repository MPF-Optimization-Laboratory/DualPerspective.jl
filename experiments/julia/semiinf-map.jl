"""
Module for implementing linear operators that map functions to vectors and back.
    
This file implements the MonomialFunctionOperator which performs:

    A*f = [ ∫ₐᵇ x^i f(x) dx ]_{i=1:m}
    
It also implements the adjoint operator A' which maps a vector to a function.
"""

using Polynomials
using Integrals
using LinearAlgebra
using Zygote

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
        atol::T=zero(T)
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
        problem = IntegralProblem((x, _) -> integrand(x), a, b)
        result[i] = solve(problem, QuadGKJL(; rtol=A.rtol, atol=A.atol)).u
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
    
    # Create polynomial from coefficients
    p = Polynomial(vcat(0.0, y))
    
    # Return function that evaluates this polynomial
    return p
    # return x -> p(x)
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
    problem = IntegralProblem((x, _) -> integrand(x), a, b)
    rhs = solve(problem, QuadGKJL(; rtol=A.rtol, atol=A.atol)).u
    
    # Check if the two inner products are equal
    is_valid = isapprox(lhs, rhs, rtol=rtol)
    
    result = (is_valid=is_valid, lhs=lhs, rhs=rhs, diff=abs(lhs-rhs))
    
    # Print results if verbose option is enabled
    if verbose
        println("\nDot-test results:")
        println("Passed: $(result.is_valid)")
        println("LHS <A*f, y>: $(result.lhs)")
        println("RHS <f, A'*y>: $(result.rhs)")
        println("Difference: $(result.diff)")
    end
    
    return result
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
    
    # Find all critital points of the polynomial derivative
    dp = derivative(p)
    all_critical_points = roots(dp)
    
    # Filter for real roots within the domain
    critical_points = filter(x -> isreal(x) && a ≤ real(x) ≤ b, all_critical_points)
    critical_points = real.(critical_points) # Convert from complex to real
    
    # Evaluate at critical points and domain endpoints
    all_points = vcat(a, critical_points, b)
    values = [p(x) for x in all_points]
    
    # Find the maximum value and its location
    max_val, idx = findmax(values)
    max_loc = all_points[idx]
    
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
    # Create operator for first 3 monomials over [0,1]
    A = monomial_operator(4, (-Inf, Inf); rtol=1e-10)
    
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
        println("i=$i: computed=$val, exact=$exact")
    end
    
    # Test adjoint with coefficient vector
    y = [1.0, 2.0, 3.0, 4.0]
    g = A' * y
    println("\nEvaluating adjoint (polynomial p(x) = 1 + 2x + 3x² + 4x³) at x=0.5: $(g(0.5))")
    
    # Verify dot-test
    verify_dot_test(A, f, y, verbose=true)
    
    # Compute the maximum of g(x) over a finite domain
    finite_domain = (-5.0, 5.0)
    p = Polynomial(y)  # Convert coefficients to a polynomial
    max_result = polynomial_maximum(p, finite_domain)
    println("\nPolynomial maximum:")
    println("Maximum value: $(max_result.value)")
    println("Location: $(max_result.location)")
   
    
    dObj(y) = begin z1 = A'*y
         p = A'y
         problem = IntegralProblem((x, _) -> p(x)^2, -Inf, Inf)
         log(solve(problem, QuadGKJL(; rtol=1e-10)).u)
    end
    println(dObj(y))
    Zygote.gradient(dObj, y)
end
# Run the example when this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    monomial_operator_example()
end
