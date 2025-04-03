# DualPerspective Python Package

Python interface for [DualPerspective.jl](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl), a Julia package for solving large-scale KL divergence problems.

## Installation

```bash
pip install DualPerspective
```

The package automatically:
1. Installs Julia if not already installed (via juliacall)
2. Installs the DualPerspective.jl Julia package from the official registry

TODO:
- Precompiles dependencies for fast performance from the first run

## Basic Usage

```python
import numpy as np
from DualPerspective import DPModel, solve, regularize

# Generate sample data
np.random.seed(42)
n = 200  # dimension of solution
m = 100  # number of measurements
x0 = np.pi * (tmp := np.random.rand(n)) / np.sum(tmp)
A = np.random.rand(m, n)
b = A @ x0  # measurements

# Create and solve the problem
model = DPModel(A, b)
regularize(model, 1e-4)  # Optional: set regularization parameter
solution = solve(model)

print(f"Sum of solution: {np.sum(solution):.6f} (should be ≈ {np.pi:.6f})")
print(f"Optimal solution shape: {solution.shape}")
```

## Features

- Python interface for DualPerspective.jl
- Automatic installation of Julia dependencies
- Integration with NumPy arrays

## Performance Considerations

This package uses Julia's precompilation to ensure good performance. The first import may take slightly longer as it sets up the Julia environment, but subsequent operations should be fast.

## Advanced Usage

### Updating DualPerspective.jl

To reinstall or update the Julia package:

```bash
uv pip install --force-reinstall DualPerspective
```

### Local Development

By default, the Python interface uses the version of `DualPerspective.jl` from the Julia registry. For local development, use the environment variable `DUALPERSPECTIVE_USE_LOCAL`:

```bash
export DUALPERSPECTIVE_USE_LOCAL=true
```

Or, to set it temporarily for a single command:

```bash
DUALPERSPECTIVE_USE_LOCAL=true python [command]
```

## Building and Publishing

### Prerequisites

Install the necessary tools:

```bash
uv pip install --upgrade build twine
```

### Building the Package

Build both wheel and source distributions:

```bash
python -m build
```

This creates distribution files in the `dist/` directory.

### Testing Locally

Before publishing, test the package locally:

```bash
# Install in development mode
source venv/bin/activate
uv pip install -e .

# Or install the built wheel
uv pip install dist/DualPerspective-0.1.1-py3-none-any.whl
```

### Publishing to TestPyPI (Optional)

To test the publishing process:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Install from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ DualPerspective
```

### Publishing to PyPI

Once tested and ready:

```bash
twine upload dist/*
```

### Updating for New Releases

1. Update the version number in `pyproject.toml`
2. Make code changes
3. Rebuild: `python -m build`
4. Upload: `twine upload dist/*`

## Requirements

- Python 3.7+
- NumPy
- juliacall

## License

This project is licensed under the MIT License.