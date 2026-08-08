# Changelog

## [0.1.5] - 2026-08-05

### Fixed
- `UndefVarError: reset! not defined` on every solve in a freshly resolved environment.
  SolverCore 0.3.9 dropped its NLPModels dependency and began exporting its own `reset!`,
  distinct from the LinearOperators-owned binding that NLPModels extends and re-exports.
  `using SolverCore` therefore made the bare name ambiguous inside DualPerspective, and
  JSOSolvers 0.12.x has the same defect internally. See [UPGRADING.md](UPGRADING.md) for the
  symptom, the workaround for older releases, and upgrade instructions.
- `FieldError: TrunkSolver has no field subsolver` with JSOSolvers 0.14.1 and later, which
  renamed the field to `krylov_subsolver`. The trust-region callbacks read it on every
  iteration, so every solve was affected.
- Keyword forwarding in `LevelSet` and `AdaptiveLevelSet`: `kwargs...` was splatted without a
  preceding semicolon, so it expanded into positional arguments. Passing any keyword the
  method did not name by hand (such as `trace`) raised a `MethodError`.
- `LevelSet` silently ignored the caller's `atol` and `rtol`, always converging to the
  package default `eps^(1/3)*(1 + ‖b‖)` instead. `AdaptiveLevelSet` was unaffected. **Results
  from `LevelSet` will change**: it now stops when `‖∇d‖ < atol + rtol*‖b‖`, as documented,
  which is looser than the old default for large tolerances and tighter for small ones.

### Changed
- Requires JSOSolvers 0.14.1 or later and Krylov 0.10.1 or later. JSOSolvers 0.12.x can no
  longer be supported: it hits the same `reset!` ambiguity from inside its own module.
- Declares `julia = "1.10"`. The previous `julia = "1"` was never accurate — an unused `Pkg`
  dependency with `Pkg = "1.11.0"` compat silently imposed a Julia 1.11 floor.

### Removed
- The unused `Pkg` dependency.
- Two dead status guards in the solver callbacks. Both compared against `:unkown`, a
  misspelling of `:unknown`, so neither ever fired. Behaviour is unchanged — but note that
  *correcting* the spelling would not have been: in `newtoncg.jl` the guard sat in front of
  the preconditioner update, so fixing it would have restricted `update!` to the final
  callback and quietly degraded `DiagASAPreconditioner`.

## [0.1.4] - 2025-04-03

### Changed
- Python interface updates and version bump.

## [0.1.3] - 2024-03-27

### Changed
- Updated to use official Julia registry installation
- Simplified Python package installation process

## [0.1.2] - 2024-03-24

### Changed
- Renamed package from "Perspectron" to "DualPerspective"
- Renamed model class from "PTModel" to standard usage
- Updated documentation to reflect new naming

### Fixed
- Fixed bug in model initialization and regularization

## [0.1.1] - 2024-03-xx

### Changed
- Interim version with minor improvements and cleanups

## [0.1.0] - 2024-03-16

### Added
- Initial release with core functionality
- Support for KL regularized least squares problems
- Optimal Transport model
- Self-scaling algorithm
- Level-set methods
- UnicodePlots extension
- NonlinearSolve extension
