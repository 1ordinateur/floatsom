# FloatSOM

Standalone development repository for the core FloatSOM library.

## Scope

This repository contains:

- the `floatsom` core package
- library runtime configuration data
- core library tests

This repository intentionally excludes publication, benchmark, and manuscript infrastructure.

## Layout

- `adapters/`
- `base/`
- `data/`
- `evaluation/`
- `processing/`
- `sampling/`
- `topology/`
- `tests/floatsom/`

## Notes

The package is mapped to the repository root in `pyproject.toml` so the repository can be used as a `floatsom` submodule inside PIBLO without changing existing import paths.
