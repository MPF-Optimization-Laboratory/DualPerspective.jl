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

The function $\phi_p$ is implemented as [`pObj!`](@ref DualPerspective.pObj!).

```@docs; canonical=false
pObj!
```

We can now rephrase the original problem \eqref{eq:primal} as the minimization of the value function $v:\R_+\to\eR$ over the set of scales $\tau\ge0$, where the value function is defined as the solution of a compactified problem parameterized by $\tau$:
```math
\begin{equation}
  \min_{\tau\ge0} v(\tau)
  \quad \text{with} \quad
  v(\tau):=\min_{x\in\R^n}\ \phi_p(x,\tau).
  \label{eq:compactified-primal}
\end{equation}
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
where the dual function is given by
```math
\phi_d(y,\tau) = \ip{b, y} - \tfrac{\lambda}{2} \ip{y, Cy} - \tau\kappa^*(A^T y - c \mid \xbar/\tau).
```
Note the appearance of the scaled reference point $\xbar/\tau$ in the argument of the conjugate function, which follows from the scaling in the definition of $\kappa_\tau(\cdot\mid\xbar)$. Thus we can rewrite the dual function as
```math
\phi_d(y,\tau) = \ip{b, y} - \tfrac{\lambda}{2} \ip{y, Cy} - \tau\kappa^*(A^T y - c \mid \xbar) - \tau\log\tau.
```
Equality holds in \eqref{eq:compactified-dual} because the primal and dual problems are both strictly feasible.

```@docs; canonical=false
dObj!
```

The dual representation \eqref{eq:compactified-dual} allows us to justify differentiability of the value function $v(\tau)$ with respect to the parameter $\tau$. For each $\tau > 0$, the dual objective $\phi_d(\cdot,\tau)$ is strictly concave, since $\kappa^*$ is strictly convex. Moreover, $\phi_d(y,\cdot)$ is differentiable with respect to $\tau$ for each $y \in \R^m$. These properties, together with the compactness of the dual feasible set, imply that the dual optimal solution $y(\tau)$ is unique and continuous in $\tau$ [rockafellar_VariationalAnalysis_2009; Theorem 7.41](@cite). Therefore, by a Danskin-type theorem [rockafellar_VariationalAnalysis_2009; Theorem 10.13](@cite), the value function $v$ is differentiable with respect to $\tau$, with
```math
\begin{aligned}
v'(\tau) 
&= \partial_\tau \phi_d(y(\tau),\tau) \\
&= -\kappa^*(A^T y(\tau) - c \mid \xbar) + \log\tau + 1. 
\end{aligned}
```

The value function $v(\tau)$ is implemented as the function [`value!`](@ref DualPerspective.value!). Note that this function implements the **negative** of the dual objective because the algorithm used to solve the compactified problem is based on **minimization**.

```@docs; canonical=false
value!
```

 Moreover, the optimal values are equal and the primal-dual solutions $(x^\star, y^\star)$ are the unique solution of the equations
```math
\begin{aligned}
    A &x + \lambda C y = b \\
    &x = \tau\nabla\kappa^*(A^T y - c \mid \xbar)
\end{aligned}
```
where
```math
\nabla\kappa^*(z\mid\xbar)=\frac{ \xbar \odot \exp(z)}{\ip{\xbar, \exp(z)}}
```
is the gradient of the conjugate function $\kappa^*$. Here, $\exp(\cdot)$ is interpreted as a vector-valued function, and $\odot$ denotes the elementwise product of two vectors.

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
