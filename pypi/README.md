# DualPerspective Python Package

Python interface for [DualPerspective.jl](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl),
a Julia package for solving Kullback-Leibler regularized least squares problems.

> **Seeing `juliacall.JuliaError: UndefVarError: reset! not defined`?** This affects every
> Julia release up to v0.1.4 and is fixed in v0.1.5. See
> [UPGRADING.md](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl/blob/main/UPGRADING.md)
> for a workaround that needs no upgrade, and for upgrade instructions.

## Installation

```bash
pip install DualPerspective
```

Julia itself is installed automatically on first use, via
[juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).

### Versioning

The PyPI package and the Julia package version independently. Each wheel pins one exact
Julia release in `juliapkg.json` and is tested against it, so PyPI **0.2.0** shipping
DualPerspective.jl **0.1.5** is expected, not a mismatch.

```python
import DualPerspective
DualPerspective.__version__   # the Python package
DualPerspective.version()     # the Julia package it pins
```

## Basic usage

```python
import numpy as np
from DualPerspective import DPModel, solve

np.random.seed(42)
n, m = 200, 100                     # solution dimension, number of measurements
x0 = np.pi * (tmp := np.random.rand(n)) / np.sum(tmp)
A = np.random.rand(m, n)
b = A @ x0                          # measurements

model = DPModel(A, b, lam=1e-4)
x = solve(model)

print(f"sum of solution: {x.sum():.6f} (should be about {np.pi:.6f})")
```

`solve` returns a NumPy array. Pass `full_output=True` for the full result:

```python
result = solve(model, full_output=True)
result.x            # primal solution
result.status       # 'optimal', 'max_iter', ...
result.iterations
result.optimality   # final ‖∇d(y)‖, the quantity the stopping rule tests
result.residual
result.trace        # per-iteration history (a list of dicts), when the solver records one
result.to_pandas()  # the trace as a DataFrame, if pandas is installed
```

## The model

```python
DPModel(A, b, q=None, C=None, c=None, lam=None)
```

Minimizes `(1/2λ)‖Ax - b‖²_{C⁻¹} + ⟨c, x⟩ + KL(x ‖ q)`.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `A` | `(m, n)` | Forward operator |
| `b` | `(m,)` | Measurements |
| `q` | `(n,)` | Prior; defaults to uniform |
| `C` | **`(m, m)`** | Covariance weighting the residual; defaults to the identity |
| `c` | `(n,)` | Linear cost |
| `lam` | scalar | Regularization parameter (also accepted as `λ`) |

> `C` weights the residual `b - Ax`, so it is `(m, m)`. Releases before 0.2.0 documented it
> incorrectly as `(n, n)`.

Integer and single-precision arrays are accepted and converted to double precision. All
arrays are copied into Julia, so mutating them afterwards does not change the model.

`lam` and `scale` are settable properties:

```python
model.lam = 1e-4      # same as regularize(model, 1e-4)
model.scale = 2.0     # same as scale(model, 2.0)
```

## Choosing a solver

```python
solve(model, method="sequential", atol=1e-6, rtol=1e-6, logging=0, full_output=False)
```

| `method` | Algorithm |
| --- | --- |
| `"sequential"` | Sequential scaling (default) |
| `"trust-region"` | Trust-region Newton-CG |
| `"level-set"` | Level-set method |
| `"adaptive-level-set"` | Adaptive level-set method |
| `"self-scaled"` | Self-scaled Gauss-Newton, for unknown total mass |

The solver stops when `‖∇d(y)‖ < atol + rtol*‖b‖`. Unrecognized keywords are passed straight
through to the Julia solver.

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

## Local development

To run against a checkout of DualPerspective.jl rather than the pinned release:

```bash
export DUALPERSPECTIVE_JL_PATH=/path/to/DualPerspective.jl
```

## Building and publishing

Remove stale artefacts first — `twine upload dist/*` will otherwise try to re-upload every
old build sitting in `dist/`:

```bash
cd pypi
rm -rf build dist *.egg-info
python -m build
unzip -l dist/*.whl | grep juliapkg.json    # the pin must be in the wheel
twine check dist/*
```

Publish only **after** the pinned Julia version has been registered in the General registry;
otherwise the wheel pins a version that cannot be resolved.

## License

MIT.
