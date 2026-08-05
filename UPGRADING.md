# Upgrading DualPerspective

## `UndefVarError: reset! not defined` (affects all releases up to v0.1.4)

### The symptom

Every call to `solve` fails, from Python:

```
juliacall.JuliaError: UndefVarError: `reset!` not defined in `JSOSolvers`
Hint: It looks like two or more modules export different bindings with this name,
resulting in ambiguity. Try explicitly importing it from a particular module, or
qualifying the name with the module it should come from.
Hint: a global variable of this name may be made accessible by importing DataStructures ...
Hint: a global variable of this name may be made accessible by importing LinearOperators ...
Hint: a global variable of this name may be made accessible by importing SolverCore ...
Hint: a global variable of this name may be made accessible by importing SolverTools ...
```

or from Julia:

```
ERROR: UndefVarError: `reset!` not defined in `DualPerspective`
```

The module named in the first line may be `JSOSolvers` or `DualPerspective` depending on
which code path you reach first. Both have the same cause.

### What happened

`reset!` is owned by `LinearOperators`, and `NLPModels` and `SolverTools` extend and
re-export that same function. `SolverCore` did too — until version **0.3.9**, which dropped
its `NLPModels` dependency and began exporting a *separate* `reset!` of its own.

DualPerspective ≤ 0.1.4, and the JSOSolvers 0.12.x it depends on, both bring `NLPModels` and
`SolverCore` into scope with `using` and then call `reset!` unqualified. Once those two names
refer to different functions, Julia can no longer decide which one you meant, and it reports
the name as undefined.

Nothing you did caused this, and it is not specific to your cluster. Any fresh install
resolves the newest compatible dependencies, so the first install after SolverCore 0.3.9 was
released is the one that breaks. An existing environment with a `Manifest.toml` pinning
SolverCore 0.3.8 keeps working, which is why the failure appears when you move machines.

### Fix it now, without upgrading

Constrain `SolverCore` to the last version that shared the binding. This works with the
currently published v0.1.4 and needs no local checkout.

**Python** — run this once, in the same virtual environment, *before* importing
DualPerspective:

```python
import juliapkg

juliapkg.add(
    "SolverCore",
    uuid="ff4d7338-4cf1-434d-91df-b86cb86fb843",
    version="=0.3.8",
)

from DualPerspective import DPModel, solve   # resolution happens here
```

Prefer this over pinning by hand: `juliapkg` may regenerate its `Project.toml` and re-resolve
when its dependency files change, which silently discards a manual `Pkg.pin`.

If you cannot use `juliapkg`, the equivalent is:

```python
from juliacall import Main as jl
jl.seval('''
    import Pkg
    Pkg.add(Pkg.PackageSpec(name="SolverCore", version="0.3.8"))
    Pkg.pin("SolverCore")
''')
```

**Julia:**

```julia
using Pkg
Pkg.add(PackageSpec(name="SolverCore", version="0.3.8"))
Pkg.pin("SolverCore")
```

> **On a shared cluster environment:** if `PYTHON_JULIACALL_PROJECT` points at an
> admin-managed project, do not apply either fix unilaterally — ask whoever owns that
> environment to add the constraint, or point the variable at a project of your own.

### Upgrade properly (v0.1.5 and later)

v0.1.5 removes the ambiguity at its source and no longer needs the pin.

**Python:**

```bash
pip install -U DualPerspective
```

```python
import juliapkg
juliapkg.rm("SolverCore")     # release the temporary pin from above
```

**Julia:**

```julia
using Pkg
Pkg.free("SolverCore")        # release the temporary pin from above
Pkg.update("DualPerspective")
```

**Cluster notes.** Julia writes its package depot to `~/.julia` by default. On a cluster with
a slow or quota-limited NFS home directory, point it at local scratch instead — precompilation
is very sensitive to filesystem latency:

```bash
export JULIA_DEPOT_PATH=/local/scratch/$USER/julia_depot
```

If your compute nodes have no network access, resolve and precompile once on a login node
with the same `JULIA_DEPOT_PATH` before submitting the job.

### Confirm it worked

```julia
using Pkg; Pkg.status()
```

You want to see `DualPerspective v0.1.5` or newer, alongside `JSOSolvers v0.14.1` or newer
and `SolverCore v0.3.10` or newer. Then:

```julia
using DualPerspective
kl = randDPModel(10, 5; λ=1e-3)
DualPerspective.solve!(kl, SequentialSolve(); t=sum(kl.q))
```

> **A note on `using`:** DualPerspective exports `reset!` and `solve!`, and so do several JSO
> packages. If your own code writes `using DualPerspective, SolverCore` (or `JSOSolvers`), a
> bare `reset!` or `solve!` will be ambiguous in *your* namespace for the same reason
> described above. Qualify the call — `DualPerspective.solve!(...)` — or import only what you
> need. `using DualPerspective` on its own is unaffected.

### If it still fails

Open an issue at
<https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl/issues> with the output of:

```julia
using Pkg; Pkg.status(); versioninfo()
```

plus the full traceback and, for Python users, your `python --version` and the value of
`JULIA_DEPOT_PATH` and `PYTHON_JULIACALL_PROJECT` if set.
