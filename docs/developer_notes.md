# Developer notes

These notes are for people who change FloatSOM. Users do not need them.

## Tests

Most tests need a GPU. Run them in a GPU environment:

```bash
pip install -e ".[ray-cuda12,dev]"
python -m pytest
```

The distributed tests need Ray and GPU resources:

```bash
python -m pytest tests/floatsom/test_gpu_scaling.py tests/floatsom/test_ray_inference_metric.py
```

Single- versus multi-GPU comparisons need at least two usable GPUs. The test
fixtures release a Ray instance that they start themselves, and they keep an
instance that the caller started.

Full-batch parity, random selection and colour processing have separate test
coverage. Results for one processing mode do not show parity for every
configuration. The execution-path concordance analysis of the publication
reports the evaluated full-sampling settings.

## Profile history

The profiles are in `src/floatsom/defaults/profiles/`. Each JSON file has
`sampling method → topology → parameters`, with all six combinations of
`full`/`random` and `hexagonal`/`mst`/`rng`. The original values are kept,
also the differences in initialization.

The old filename `floatsom_min1000_tuned_defaults.json` referred to different
initialization settings in the two repositories. These are now explicitly
separated as `library` and `publication`; neither was overwritten during migration.
The two experimental variants are preserved independently and are not duplicates
of the main publication profile.

The untuned XPySOM comparison baseline belongs to the publication benchmarks,
at `benchmarks/optuna/config/baselines/xpysom-untuned.json` in that repository.
It is not a tuned FloatSOM profile.
