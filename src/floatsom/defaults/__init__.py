"""Bundled, GPU-independent tuned-default profiles. See docs/defaults.md."""
import json
from pathlib import Path
from typing import Any, Dict

PROFILE_NAMES = ("library", "publication", "publication-rng-random", "publication-rng-mst-config")


def profile_path(name: str = "library") -> Path:
    """Return an installed profile's JSON path; reject unknown names."""
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown FloatSOM profile {name!r}. Choose from {PROFILE_NAMES}.")
    return Path(__file__).with_name("profiles") / f"{name}.json"


def load_profile(name: str = "library") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return a fresh copy in the original benchmark JSON schema."""
    payload = json.loads(profile_path(name).read_text(encoding="utf-8"))
    validate_profile(payload)
    return payload


def resolve_profile_defaults(sampling_method: str, topology_type: str,
                             profile: str = "library") -> Dict[str, Any]:
    """Resolve an exact context, using FloatSOM parameter names."""
    normalized = validate_profile(load_profile(profile))
    return dict(normalized.get(str(sampling_method).strip().lower(), {}).get(
        str(topology_type).strip().lower(), {}))

_CONTEXTUAL_FLOATSOM_DEFAULT_SAMPLING_KEYS = {"full", "random"}
_CONTEXTUAL_FLOATSOM_DEFAULT_TOPOLOGY_KEYS = {"hexagonal", "mst", "rng"}
_VALID_DECAY_TYPES = {"exponential", "linear", "sigmoid", "gaussian", "asymptotic", "fixed"}
_VALID_INITIALIZATION_METHODS = {"random", "pca", "pca_sampling", "pca_sampling_snake", "pca_density"}


def validate_profile(payload):
    """Validate and normalize a sampling/topology profile."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Contextual FloatSOM defaults JSON must be an object of "
            "sampling_method->topology->params mappings."
        )

    normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_sampling_key, topology_map in payload.items():
        sampling_key = str(raw_sampling_key).strip().lower()
        if sampling_key not in _CONTEXTUAL_FLOATSOM_DEFAULT_SAMPLING_KEYS:
            raise ValueError(
                f"Unsupported sampling key in contextual FloatSOM defaults: '{raw_sampling_key}'. "
                f"Supported: {sorted(_CONTEXTUAL_FLOATSOM_DEFAULT_SAMPLING_KEYS)}"
            )
        if not isinstance(topology_map, dict):
            raise ValueError(
                f"Sampling entry '{raw_sampling_key}' must map to an object of topology->params mappings."
            )

        normalized_topology_map: Dict[str, Dict[str, Any]] = {}
        for raw_topology_key, raw_param_map in topology_map.items():
            topology_key = str(raw_topology_key).strip().lower()
            if topology_key not in _CONTEXTUAL_FLOATSOM_DEFAULT_TOPOLOGY_KEYS:
                raise ValueError(
                    f"Unsupported topology key in contextual FloatSOM defaults: '{raw_topology_key}'. "
                    f"Supported: {sorted(_CONTEXTUAL_FLOATSOM_DEFAULT_TOPOLOGY_KEYS)}"
                )
            if not isinstance(raw_param_map, dict):
                raise ValueError(
                    f"Topology entry '{raw_sampling_key}:{raw_topology_key}' must map to an object of param->value entries."
                )

            normalized_params: Dict[str, Any] = {}
            for raw_param_name, raw_value in raw_param_map.items():
                param_name = str(raw_param_name).strip()
                if param_name == "initial_radius":
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:initial_radius' "
                            f"must be numeric, got {type(raw_value).__name__}."
                        )
                    normalized_params["initial_radius"] = float(raw_value)
                elif param_name == "radius_decay_type":
                    if not isinstance(raw_value, str) or raw_value not in _VALID_DECAY_TYPES:
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:radius_decay_type' "
                            f"must be one of {sorted(_VALID_DECAY_TYPES)}."
                        )
                    normalized_params["radius_decay_type"] = raw_value
                elif param_name == "initialization_method":
                    if not isinstance(raw_value, str) or raw_value not in _VALID_INITIALIZATION_METHODS:
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:initialization_method' "
                            f"must be one of {sorted(_VALID_INITIALIZATION_METHODS)}."
                        )
                    normalized_params["initialization_method"] = raw_value
                elif param_name == "use_momentum":
                    if not isinstance(raw_value, bool):
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:use_momentum' must be boolean."
                        )
                    normalized_params["enable_momentum"] = raw_value
                elif param_name == "momentum_init":
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:momentum_init' "
                            f"must be numeric, got {type(raw_value).__name__}."
                        )
                    normalized_params["initial_momentum"] = float(raw_value)
                else:
                    raise ValueError(
                        f"Unsupported contextual FloatSOM default key '{param_name}' "
                        f"under '{raw_sampling_key}:{raw_topology_key}'."
                    )

            normalized_topology_map[topology_key] = normalized_params
        normalized[sampling_key] = normalized_topology_map

    return normalized
