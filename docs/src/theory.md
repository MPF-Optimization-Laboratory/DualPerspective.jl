```@meta
CurrentModule = DualPerspective
CollapsedDocStrings = true
```

# Theoretical Development

The DualPerspective package reformulates various problem classes, including nonnegative least-squares, linear programming, optimal transport, and Hausdorff moment recovery, as instances of the regularized relative-entropy problem

```math
\begin{equation}
\min_{x\in\R^n_+}\  \textstyle \ip{c, x} + \tfrac{1}{2\lambda} \|Ax - b\|^2_{C^{-1}} 
+ \sum_{j=1}^n x_j \log\left(x_j/\xbar_j\right),
\label{eq:primal}
\end{equation}
```

where the convention $0\log0=0$ is used. A key idea in this approach is to further reformulate the nonnegative cone constraint $x \in \mathbb{R}^n_+$ as the conic extension of the probability simplex. That is, with the obvious identity

```math 
\R^n_+ = \bigcup_{\tau\ge0} \tau \Delta^n, \quad \Delta^n := \left\{ x \in \mathbb{R}^n_+ \mid \textstyle\sum_{j=1}^n x_j = 1 \right\},
```

we can exploit this structure to develop a powerful solution technique.

This reformulation allows us to approach the original problem \eqref{eq:primal} through a sequence of simpler, compactified problems defined over the probability simplex. Each of these problems corresponds to a specific scale factor $\tau$, and their solutions converge to the solution of the original problem as $\tau$ approaches the appropriate value.

Moreover, these compactified problems admit dual reformulations with highly favorable properties: their objectives are globally Lipschitz-smooth and strongly-convex with uniformly bounded Hessian matrices. These properties enable the development of efficient algorithms with strong convergence guarantees, which we will explore in detail in subsequent sections.

We make the blanket assumption that the reference point $\xbar\in\relint\Delta^n$. This assumption implies that $\xbar$ has full support. Otherwise, any variable $x_j$ with $\xbar_j=0$ can be fixed at zero without affecting the optimal solution $x^*$ of the original problem \eqref{eq:primal}.

## Dual Perspective Model

The [`DualPerspective.DPModel`](@ref) type extends the `AbstractNLPModel` interface to encapsulate the data for the regularized relative-entropy problem \eqref{eq:primal}.

```@docs; canonical=false
DPModel
```

Create a model with a specific regularization parameter `λ`, using the convenience constructor:

```@example
using DualPerspective, Random
A, b = randn(10, 5), randn(10)
model = DPModel(A, b; λ=1e-3)
```

## Value function and compactification

Define the [`Kullback-Leibler (KL)divergence`](@ref DualPerspective.kl_divergence) as the relative-entropy of two discrete densities $x$ and $\xbar$ in the probability simplex $\Delta^n$:

```math
\kappa(x \mid \xbar) :=
\begin{cases}
\sum_{j=1}^n x_j \log\left(x_j/\xbar_j\right) & \text{if } (x,\xbar)\in\Delta^n\times\R^n_{++},\\
+\infty & \text{otherwise.}
\end{cases}
```

```@docs; canonical=false
kl_divergence
```

We introduce a deceptively simple scaling of the KL divergence,

```math
\kappa_\tau(x \mid \xbar):=\tau\kappa(x/\tau\mid\xbar/\tau).
```

The following identify is immediate:

```math
\kappa_\tau(x \mid \xbar) = \kappa(x\mid\xbar) \quad \forall \tau>0.
```

Note that $\kappa_\tau(\cdot\mid\xbar)$ is the perspective transform of $\kappa(\cdot\mid\xbar)$. With this notation, we can now define the primal function $\phi_p:\R^n_+\times\R_+\to\eR$ by

```math
\phi_p(x,\tau):=\ip{c,x} + \tfrac{1}{2\lambda} \|Ax - b\|^2_{C^{-1}} + \kappa_\tau(x \mid \xbar).
```

Observe that $\dom\phi_p=\set{(x,\tau)\mid x\in\tau\Delta^n,\ \tau>0}$ coincides with the set of feasible points of the original problem \eqref{eq:primal}.

We can now rephrase the original problem \eqref{eq:primal} as the minimization of the value function $v:\R_+\to\eR$ over the set of scales $\tau\ge0$, where the value function is defined as the solution of a compactified problem parameterized by $\tau$:

```math
\begin{equation}
  \min_{\tau\ge0} v(\tau)
  \quad \text{with} \quad
  v(\tau):=\min_{x\in\R^n}\ \phi_p(x,\tau).
  \label{eq:compactified-primal}
\end{equation}
```

The function $\phi_p$ is implemented as [`pObj!`](@ref DualPerspective.pObj!).

```@docs; canonical=false
pObj!
```


## Dual representation of the value function

Here we derive a dual reformulation of the value function $v$ in \eqref{eq:compactified-primal}.
Observe that, as a function of the first argument alone, the function $\kappa_\tau(\cdot\mid\xbar)$ that appears in the compactified primal problem \eqref{eq:compactified-primal} is the perspective transform of the function $\kappa(\cdot\mid\xbar)$. The conjugate of this function is given by
```math
(\kappa_\tau)^*(\cdot\mid\xbar) = \tau\kappa^*(\cdot\mid\xbar),
```
where the [`log-sum-exp`](@ref LogExp(::AbstractVector)) function
```math
\kappa^*(z \mid \xbar) := \log\textstyle\sum_{j=1}^n \xbar_j \exp(z_j)
```
is the convex conjugate of the KL divergence $\kappa(\cdot \mid \xbar)$ [beck_FirstOrderMethodsOptimization_2017; Section 4.4.10](@cite). Using Fenchel duality, we may express the value function $v$ in dual form:
```math
\begin{equation}
  v(\tau) = \min_{x\in\R^n} \phi_p(x,\tau) = \max_{y\in\R^m}\ \phi_d(y,\tau),
  \label{eq:compactified-dual}
\end{equation}
```

where the concave dual function is given by

```math
\phi_d(y,\tau) = \ip{b, y} - \tfrac{\lambda}{2} \ip{y, Cy} - \tau\kappa^*(A^T y - c \mid \xbar/\tau).
```

Note the appearance of the scaled reference point $\xbar/\tau$ in the argument of the conjugate function, which follows from the scaling in the definition of $\kappa_\tau(\cdot\mid\xbar)$. Thus we can rewrite the dual function as

```math
\phi_d(y,\tau) = \ip{b, y} - \tfrac{\lambda}{2} \ip{y, Cy} - \tau\kappa^*(A^T y - c \mid \xbar) - \tau\log\tau.
```

Equality holds in \eqref{eq:compactified-dual} because the primal and dual problems are both strictly feasible.

The function $\phi_d$ is implemented as [`dObj!`](@ref DualPerspective.dObj!).

```@docs; canonical=false
dObj!
```

## Analytical properties of the dual objective

The dual objective function possesses a number of favorable analytical properties that enable efficient numerical optimization and provide strong theoretical guarantees on algorithmic performance.

### Differentiability of the value function

The dual representation \eqref{eq:compactified-dual} allows us to analyze the differentiability properties of the value function $v(\tau)$. For each $\tau > 0$, the optimal dual solution $y(\tau)$ satisfies the first-order optimality condition:

```math
\nabla_y \phi_d(y(\tau),\tau) = 0.
```

Because the covariance matrix $C$ is positive definite, the Hessian $\nabla^2_y \phi_d(y,\tau)$ is negative definite with eigenvalues bounded away from zero. This strong concavity property, combined with the fact that $\phi_d$ is twice continuously differentiable with respect to both $y$ and $\tau$, ensures that the implicit function theorem can be applied.

By the implicit function theorem, the mapping $\tau \mapsto y(\tau)$ is continuously differentiable, with

```math
y'(\tau) = -[\nabla^2_y \phi_d(y(\tau),\tau)]^{-1} \nabla_{\tau,y}^2 \phi_d(y(\tau),\tau).
```

Therefore, the value function $v(\tau) = \phi_d(y(\tau),\tau)$ is continuously differentiable, with derivative given by the partial derivative of $\phi_d$ with respect to $\tau$:

```math
\begin{aligned}
v'(\tau) 
&= \partial_\tau \phi_d(y(\tau),\tau) \\
&= -\kappa^*(A^T y(\tau) - c \mid \xbar) + \log\tau + 1.
\end{aligned}
```

Moreover, because $y(\tau)$ is continuously differentiable and $\phi_d$ is twice continuously differentiable, $v(\tau)$ is twice continuously differentiable. This smoothness property is crucial for applying efficient root-finding methods to solve $v'(\tau) = 0$, which is the optimality condition for the original problem \eqref{eq:primal}.

The value function $v(\tau)$ is implemented as the function [`value!`](@ref DualPerspective.value!). Note that this function implements the **negative** of the dual objective because the algorithm used to solve the compactified problem is based on **minimization**.

```@docs; canonical=false
value!
```

### Primal-from-dual solution map

The strong-duality property that furnished the dual representation \eqref{eq:compactified-dual} also furnishes the primal-from-dual solution map. Indeed, for a fixed scale parameter $\tau$, the dual optimal solution $y_\tau$ must satisfy the optimality condition

```math
\begin{equation}
    \nabla_y\phi(y,\tau)=0
    \quad\Longleftrightarrow\quad
    A x(y) + \lambda C y = b \\
\end{equation}
```

where the primal-from dual solution map $y\mapsto x(y)$ from $\R^m$ to $\Delta^n$ is given by

```math
\begin{equation}
    x(y) = \tau\nabla\kappa^*(A^T y - c \mid \xbar) = \tau\frac{ \xbar \odot \exp(A^T y - c)}{\ip{\xbar, \exp(A^T y - c)}}.
    \label{eq:primal-from-dual}
\end{equation}
```

Here, $\exp(\cdot)$ is interpreted as a vector-valued function, and $\odot$ denotes the elementwise product of two vectors.


### Lipschitz-smoothness and strong concavity

The notion of Lipschitz-smoothness and strong concavity (or convexity) plays a fundamental role in establishing convergence rates for optimization algorithms. For second-order Newton-type methods, these properties are especially relevant as they directly influence both the theoretical convergence rate and practical performance. When a function is Lipschitz-smooth (bounded second derivatives), first-order methods exhibit well-defined convergence rates, while strong concavity ensures uniqueness of solutions and quadratic growth conditions. However, to achieve the quadratic convergence that makes Newton's method so powerful, we additionally need Lipschitz continuity of the Hessian (bounded third derivatives). The following proposition establishes these essential properties for our dual objective function with a fixed scale parameter $\tau$, providing the theoretical foundation for implementing efficient second-order methods with guaranteed rapid convergence in a neighborhood of the solution.

Let $\mu_{\min}$ and $\mu_{\max}$ be the smallest and largest eigenvalues of $C$, respectively.

> **Proposition.**
> For any fixed $\tau > 0$, the dual objective function $\phi_d(\cdot,\tau)$ is globally Lipschitz-smooth with modulus $(\tau\|A\|^2 + \lambda\mu_{\max})$, strongly concave with modulus $\lambda\mu_{\min}$, and the Hessian $\nabla^2_y \phi_d(\cdot,\tau)$ is Lipschitz continuous.

The proof of this proposition relies on the analyzing the spectrum of the Hessian of the dual objective:

```math
\nabla^2_y \phi_d(y,\tau) = -\lambda C + \tau AS(x_y)A^T,
```

where

```math
S(x) := \Diag(x) - xx^T
```

maps vectors in the probability simplex $\Delta^n$ to positive semidefinite matrices, and we use the shorthand notation $x_y := x(y)$ to denote the primal-from-dual solution map \eqref{eq:primal-from-dual}.

Observe that for any $x\in\Delta^n$, $S(x)$ has rank $n-1$ and trace $1 - \|x\|_2^2$, and at least one component of the vector must be no larger than $1/n$.

For the tightest bound on the spectral norm, we can examine the extremal case. When $x = (1/n)e$, where $e$ is the vector of all ones, the matrix $S(x)=(1/n)I - (1/n^2)ee^T$ has eigenvalue $1/n$ with multiplicity $n-1$ and eigenvalue 0 with multiplicity 1.

For the general case, let $\mu_1 \geq \ldots \geq \mu_n$ be the eigenvalues of $S(x)$. Because the matrix is positive semidefinite with trace $1 - \|x\|_2^2 \leq 1$, the largest eigenvalue of $S(x)$ is at most $1$:

```math
\|S(x)\|_2 \leq 1 \quad \forall x\in\Delta^n.
```

This implies that the spectral norm of the Hessian is bounded by

```math
\|\nabla^2_y \phi_d(y,\tau)\|_2 \leq \lambda\mu_{\max} + \tau\|A\|_2^2
```

The Lipschitz continuity of the Hessian follows from analyzing how the term $S(x_y)$ changes with $y$. The mapping $y \mapsto x_y$ is the gradient of the log-sum-exp function composed with the affine map $y\mapsto A^Ty - c$. Because the composition of Lipschitz continuous functions preserves Lipschitz continuity, the Hessian $\nabla^2_y \phi_d(y,\tau)$ is Lipschitz continuous with respect to $y$. Strong concavity follows from the positive definiteness of $C$.

These properties have important algorithmic implications:

1. The Lipschitz smoothness ensures that gradient-based methods have well-defined convergence rates.
2. The strong concavity guarantees a unique maximizer and provides a quadratic growth condition away from the solution.
3. The Lipschitz continuity of the Hessian ensures that Newton's method achieves locally quadratic convergence.

When combined, these properties enable the implementation of robust second-order methods with superlinear convergence. For the compactified dual problem, Newton's method converges quadratically in a neighborhood of the optimal solution. Moreover, the uniform boundedness of the Hessian (with respect to $\tau$) provides numerical stability when implementing globalized Newton methods with line search, as the condition number of the Hessian remains well-controlled throughout the iterative process.

## Convergence analysis

- Discuss convergence rate of the method
- Provide theoretical guarantees on solution quality
- Analyze influence of problem parameters on convergence

## Relationship to other methods

- Compare to interior point methods
- Discuss similarities and differences to entropic regularization approaches
- Highlight unique aspects of DualPerspective

## Implementation details

- Overview of key algorithmic components
- Discuss numerical stability and computational efficiency
- Provide links to API documentation for implementation

## Numerical experiments

- Showcase performance on example problems
- Compare to other state-of-the-art solvers
- Discuss scalability and sensitivity to problem parameters
