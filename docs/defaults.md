# Tuned defaults

Profiles are installed with the library in `src/floatsom/defaults/profiles/`.
Each JSON file contains `sampling method → topology → parameters`, with all
six combinations of `full`/`random` and `hexagonal`/`mst`/`rng`.
The original values are preserved, including differences in initialization.

| Profile | Purpose and initialization |
| --- | --- |
| `library` | Existing standalone library behavior: PCA for full RNG, random-sampling MST and random-sampling RNG; random initialization elsewhere. |
| `publication` | Main tuned profile from the publication repository; random initialization in all six contexts. |
| `publication-rng-random` | Original RNG-random experiment profile; random-sampling MST uses PCA, all other contexts use random initialization. |
| `publication-rng-mst-config` | Original control assigning MST radius and momentum to RNG; random-sampling MST and RNG use PCA. |

The old filename `floatsom_min1000_tuned_defaults.json` referred to different
initialization settings in the two repositories. These are now explicitly
separated as `library` and `publication`; neither was overwritten during migration.
The two experimental variants are preserved independently and are not duplicates
of the main publication profile.

Select a profile when constructing parameters:

```python
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, TopologyConfig

params = FloatSOMParams(
    input_dim=50,
    defaults_profile="publication",
    sampling_config=SamplingConfig(method="random"),
    topology_config=TopologyConfig(topology_type="rng"),
)
```

Omitting `defaults_profile` retains the `library` profile. Explicit radius,
initialization, decay and momentum settings override the profile. Contexts
outside the six tuned combinations retain the existing untuned fallbacks.

Profiles can be inspected without CUDA or importing the training modules:

```python
from floatsom.defaults import PROFILE_NAMES, load_profile, profile_path

print(PROFILE_NAMES)
print(load_profile("publication")["random"]["rng"])
print(profile_path("publication"))
```

`load_profile` returns the original benchmark keys (`use_momentum`,
`momentum_init`); `resolve_profile_defaults` returns the corresponding
parameter keys (`enable_momentum`, `initial_momentum`). Each call returns fresh
data, so editing a returned dictionary does not modify other runs.

The untuned XPySOM comparison baseline belongs to the publication benchmarks,
at `benchmarks/optuna/config/baselines/xpysom-untuned.json` in that repository.
It is not a tuned FloatSOM profile.
