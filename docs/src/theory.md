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

### Dual Perspective Model

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

## The Perspective Transform

A key idea in the DualPerspective approach is to reformulate the nonnegativity constraint $x \in \mathbb{R}^n_+$ as the conic extension of the probability simplex:
```math 
x \in \bigcup_{\tau\ge0} \tau \Delta^n, \quad \Delta^n := \left\{ x \in \mathbb{R}^n_+ \mid \textstyle\sum_{j=1}^n x_j = 1 \right\}.
```
This reformulation allows us to approximate the solution to \eqref{eq:primal} by solving a sequence of compactified optimization problems over the probability simplex, parameterized by a scale factor $\tau$. The solution is recovered as $\tau \to \gamma_\Delta(x^*)$, where $\gamma_\Delta(x^*)$ is the gauge of $\Delta$ evaluated at the solution $x^*$.

The compactified problems have favorable dual formulations with Lipschitz-smooth and strongly-convex objectives and bounded Hessians. These properties enable the derivation of efficient algorithms with strong convergence guarantees.

## Value function and compactification

Define the [`Kullback-Leibler (KL)divergence`](@ref DualPerspective.kl_divergence) as the relative-entropy of two discrete densities $x$ and $\xbar$ in the probability simplex $\Delta^n$:
```math
\kappa(x \mid \xbar) :=
\begin{cases}
\sum_{j=1}^n x_j \log\left(x_j/\xbar_j\right) & \text{if } (x,\xbar)\in\Delta^n\times\R^n++,\\
+\infty & \text{otherwise.}
\end{cases}
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
Observe that $\dom\phi_p=\set{(x,\tau)\mid x\in\tau\Delta^n,\ \tau>0}$ coincides with the set of feasible points of the original problem \eqref{eq:primal}. We can now rephrase the original problem \eqref{eq:primal} as the minimization of the value function $v:\R_+\to\eR$ over the set of scales $\tau\ge0$, where the value function is defined as the solution of a compactified problem:
```math
\begin{equation}
  \min_{\tau\ge0} v(\tau)
  \quad \text{with} \quad
  v(\tau):=\min_{x\in\R^n}\ \phi_p(x,\tau).
  \label{eq:compactified_primal}
\end{equation}
```

## Dual reformulation

Here we derive a dual reformulation of the value function $v(\tau)$ in \eqref{eq:compactified_primal}.

Observe that, as a function of the first argument alone, the function $\kappa_\tau(\cdot\mid\xbar)$ is the perspective transform of the function $\kappa(\cdot\mid\xbar)$, which has the conjugate
```math
(\kappa_\tau)^*(\cdot\mid\xbar) = \tau\kappa^*(\cdot\mid\xbar),
```
where the [`log-sum-exp`](@ref LogExp(::AbstractVector)) function
```math
\kappa^*(z \mid \xbar) := \log\textstyle\sum_{j=1}^n \xbar_j \exp(z_j)
```
is the convex conjugate of the KL divergence $\kappa(\cdot \mid \xbar)$ [beck_FirstOrderMethodsOptimization_2017; Section 4.4.10](@cite). Using Fenchel duality, we may express the value function $v(\tau)$ in \eqref{eq:compactified_primal} in dual form:
```math
\begin{equation}
  v(\tau) = \min_{x\in\R^n} \phi_p(x,\tau) = \max_{y\in\R^m}\ \phi_d(y,\tau),
  \label{eq:compactified_dual}
\end{equation}
```
where the dual function is given by
```math
\phi_d(y,\tau) = \ip{b, y} - \tfrac{\lambda}{2} \ip{y, Cy} - \tau\kappa^*(A^T y - c \mid \xbar).
```
Equality holds in \eqref{eq:compactified_dual} because the primal and dual problems are both strictly feasible [bonnansPerturbationAnalysisOptimization2000](@cite). Moreover, the optimal values are equal and the primal-dual solutions $(x^\star, y^\star)$ are the unique solution of the equations
```math
\begin{aligned}
    A x + \lambda C y &= b \\
    x &= \tau\nabla\kappa^*(A^T y - c \mid \xbar)
\end{aligned}
```
where
```math
\nabla\kappa^*(z\mid\xbar)=\frac{ \xbar \odot \exp(z)}{\ip{\xbar, \exp(z)}}.
```
Here, $\exp(\cdot)$ is interpreted as a vector-valued function, and $\odot$ denotes the elementwise product of two vectors.

The expression for the gradient $\nabla\kappa^*(z\mid\xbar)$ can be derived using the chain rule:
```math
\begin{aligned}
\nabla\kappa^*(z\mid\xbar) 
&= \nabla\left(\log\textstyle\sum_{j=1}^n \xbar_j \exp(z_j)\right) \\
&= \frac{1}{\sum_{j=1}^n \xbar_j \exp(z_j)} \sum_{j=1}^n \xbar_j \exp(z_j) \nabla z_j \\
&= \frac{ \xbar \odot \exp(z)}{\ip{\xbar, \exp(z)}}.
\end{aligned}
```

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