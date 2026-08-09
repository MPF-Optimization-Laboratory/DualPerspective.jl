# Developing the DualPerspective Python package

Maintainer notes for the PyPI wrapper. Users do not need any of this.

## Running against a local Julia checkout

To use a checkout of DualPerspective.jl rather than the pinned release:

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
