# FloatSOM tests

From the repository root, install `.[cuda12,dev]` (or the matching CUDA extra)
and run `python -m pytest tests`. The full suite imports CuPy and needs CUDA;
Ray and multi-GPU tests additionally need the corresponding resources.

For a CPU-only installation (`.[dev]`), run:

```bash
python -m pytest tests/floatsom/test_resource_limits.py tests/floatsom/test_local_storage_wipe.py tests/floatsom/test_zarr_utils.py tests/floatsom/test_inference_intent_contracts.py
```

Build validation is separate: `python -m build`, followed by
`python tools/check_distribution.py dist/*.whl`. It checks that all runtime
modules and configuration files are present in the wheel.
