# Tuned profiles

A profile is a set of tuned default values for the radius, the
initialization, the decay and the momentum. Keep the default profile,
`library`, unless you want the tuned values of the publication.

| Profile | Use when |
| --- | --- |
| `library` (default) | Most work. This is the standalone library behaviour: PCA initialization for full-sampling RNG, random-sampling MST and random-sampling RNG; random initialization for the other combinations. |
| `publication` | You want the main tuned values of the publication. All six combinations use random initialization. |
| `publication-rng-random` | You repeat the RNG-random experiment of the publication. |
| `publication-rng-mst-config` | You repeat the publication control that gives the MST radius and momentum to RNG. |

Each profile has tuned values for six combinations: `full` or `random`
sampling, with `hexagonal`, `mst` or `rng` topology. Other combinations use
the untuned defaults. A setting that you give yourself, for example a radius,
always overrides the profile.

## Select a profile

In `train_som`:

```python
import floatsom

som = floatsom.train_som(data, topology="rng", profile="publication")
```

In the full configuration:

```python
from floatsom import FloatSOMParams, SamplingConfig, TopologyConfig

params = FloatSOMParams(
    input_dim=50,
    defaults_profile="publication",
    sampling_config=SamplingConfig(method="random"),
    topology_config=TopologyConfig(topology_type="rng"),
)
```

## See the values

You can read the profiles with no GPU:

```python
from floatsom import PROFILE_NAMES, load_profile

print(PROFILE_NAMES)
print(load_profile("publication")["random"]["rng"])
```

`load_profile` gives the original benchmark keys (`use_momentum`,
`momentum_init`). `floatsom.defaults.resolve_profile_defaults` gives the
parameter keys (`enable_momentum`, `initial_momentum`). Each call gives new
data, so a change to a returned dict does not change other runs.

The profile files are in `src/floatsom/defaults/profiles/`. For their
history, see the [developer notes](developer_notes.md#profile-history).
