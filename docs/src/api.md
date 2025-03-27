```@meta
CurrentModule = DualPerspective
```

# API Reference

This section provides an overview of the main types and functions in DualPerspective.jl.

## Models

DualPerspective.jl provides the following model types:

- `DPModel`: Base model for KL-regularized least squares problems
- `SSModel`: Self-scaling model variant
- `OTModel`: Model for optimal transport problems 
- `LPModel`: Model for linear programming problems
- `randDPModel`: Function to generate random test instances

## Solvers

The package offers several solvers:

- `SSTrunkLS`: Trust-region solver with line search
- `SequentialSolve`: Sequential solver for scaled problems
- `LevelSet`: Level set method solver
- `AdaptiveLevelSet`: Adaptive level set solver

## Core Functions

Key functions for working with models:

- `solve!`: Solve a model
- `scale!`: Apply scaling to a model
- `scale`: Get the scale of a model
- `regularize!`: Apply regularization
- `histogram`: Visualize solution
- `reset!`: Reset a model to initial state
- `update_y0!`: Update starting point

## Auto-generated Documentation

The following section contains auto-generated documentation for exported symbols:

```@autodocs
Modules = [DualPerspective]
Private = false
``` 