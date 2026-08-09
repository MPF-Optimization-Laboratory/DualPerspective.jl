# DualPerspective

Maximum-entropy solutions to underdetermined linear systems, from Python.

Given a wide matrix `A` and measurements `b`, `Ax = b` has infinitely many solutions.
DualPerspective picks the nonnegative one closest — in Kullback-Leibler divergence — to a
prior `q`, while fitting the data to within a tolerance set by a regularization parameter.
It is a Python interface to
[DualPerspective.jl](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl),
which solves the dual problem with Newton-type methods.

## Installation

```bash
pip install DualPerspective
```

Julia itself is installed automatically on first use, via
[juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). Nothing happens at
import time; Julia starts lazily, on the first call that needs it.

## Basic usage

```python
import numpy as np
from DualPerspective import DPModel, solve

np.random.seed(42)
n, m = 200, 100                     # solution dimension, number of measurements
tmp = np.random.rand(n)
x0 = np.pi * tmp / tmp.sum()        # a nonnegative signal of total mass π
A = np.random.rand(m, n)
b = A @ x0                          # measurements

model = DPModel(A, b, lam=1e-4)
x = solve(model)

print(f"sum of solution: {x.sum():.6f} (should be about {np.pi:.6f})")
```

The default solver estimates the total mass from the data rather than taking it as given,
so the recovered solution sums to roughly π even though the prior sums to 1 — that is what
the printed check tests.

`solve` returns a NumPy array. Pass `full_output=True` for the full result:

```python
result = solve(model, full_output=True)
result.x              # primal solution
result.status         # 'optimal', 'max_iter', ...
result.iterations
result.elapsed_time
result.optimality     # final ‖∇d(y)‖, the quantity the stopping rule tests
result.residual       # covariance-weighted residual
result.primal_obj, result.dual_obj
result.trace          # per-iteration history (a list of dicts), when the solver records one
result.to_pandas()    # the trace as a DataFrame, if pandas is installed
```

`model.solve(...)` is equivalent to `solve(model, ...)`. For a quick experiment,
`rand_dp_model(m, n)` builds a random model.

## The model

```python
DPModel(A, b, q=None, C=None, c=None, lam=None)
```

Minimizes `(1/2λ)‖Ax - b‖²_{C⁻¹} + ⟨c, x⟩ + KL(x ‖ q)` over nonnegative `x` of a given
total mass — the probability simplex when that mass is 1.

| Argument | Shape | Meaning | Default |
| --- | --- | --- | --- |
| `A` | `(m, n)` | Forward operator | — |
| `b` | `(m,)` | Measurements | — |
| `q` | `(n,)` | Prior | uniform, summing to 1 |
| `C` | `(m, m)` | Covariance weighting the residual `b - Ax` | identity |
| `c` | `(n,)` | Linear cost | all `-1` |
| `lam` | scalar | Regularization parameter (also accepted as `λ`) | `√eps` |

Integer and single-precision arrays are accepted and converted to double precision. All
arrays are copied into Julia, so mutating them afterwards does not change the model.

`lam` and `scale` are settable properties. `scale` is the total mass of the solution:

```python
model.lam = 1e-4      # regularization parameter
model.scale = 2.0     # total mass
```

## Choosing a solver

```python
solve(model, method="sequential", atol=1e-6, rtol=1e-6,
      t=None, verbose=False, logging=0, full_output=False)
```

| `method` | Algorithm |
| --- | --- |
| `"sequential"` | Sequential scaling (default) |
| `"trust-region"` | Trust-region Newton-CG |
| `"level-set"` | Level-set method |
| `"adaptive-level-set"` | Adaptive level-set method |
| `"self-scaled"` | Self-scaled Gauss-Newton, for unknown total mass |

The solver stops when `‖∇d(y)‖ < atol + rtol*‖b‖`.

- `t` sets the total mass and defaults to `sum(q)`. The `"trust-region"` and
  `"self-scaled"` methods determine the mass themselves and reject it.
- `verbose` prints root-finding progress (`"sequential"` only); `logging` sets solver log
  verbosity, `0` being silent.
- Unrecognized keywords are passed straight through to the Julia solver.

## Versioning

The PyPI package and the Julia package version independently: each wheel pins one exact
Julia release and is tested against it, so the two version numbers differ by design.

```python
import DualPerspective
DualPerspective.__version__   # the Python package
DualPerspective.version()     # the Julia package it pins
```

## Diagnosing problems

```bash
python -m DualPerspective.doctor
```

Prints the Python, Julia and resolved Julia package versions, the relevant environment
variables, and runs a small solve. Include its output in any bug report.

Julia errors surface as `DualPerspectiveError`, whose message is the first line of the Julia
exception; the full Julia backtrace is on `.julia_traceback`, and the original exception is
chained as `__cause__`.

## Running on a cluster

- **Put the Julia depot on fast local storage.** Precompilation is very sensitive to
  filesystem latency, and `~/.julia` on NFS is usually slow and quota-limited:
  ```bash
  export JULIA_DEPOT_PATH=/local/scratch/$USER/julia_depot
  ```
- **Resolve before going offline.** Import the package once on a login node with the same
  `JULIA_DEPOT_PATH` before submitting jobs to nodes without network access.
- **Shared environments.** `PYTHON_JULIACALL_PROJECT` points juliacall at an existing Julia
  project, which must already have a matching PythonCall.jl. Do not mutate an
  admin-managed project.
- **Threads and signals.** `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` avoids segfaults from
  Julia's garbage collector in multi-threaded programs, but interferes with Python's own
  signal handling (including Ctrl-C). It is not set by default; enable it only if you need
  it.

## License

MIT.
