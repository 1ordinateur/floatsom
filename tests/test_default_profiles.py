"""CPU-only profile contracts, including the two formerly ambiguous defaults."""
import pytest
from floatsom.defaults import PROFILE_NAMES, load_profile, profile_path, resolve_profile_defaults, validate_profile


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_profiles_cover_all_tuned_contexts(name):
    profile = load_profile(name)
    assert set(profile) == {"full", "random"}
    for contexts in profile.values():
        assert set(contexts) == {"hexagonal", "mst", "rng"}
    assert profile_path(name).is_file()


def test_library_and_publication_initialization_remain_distinct():
    library, paper = load_profile("library"), load_profile("publication")
    for sampling, topology in (("full", "rng"), ("random", "mst"), ("random", "rng")):
        assert library[sampling][topology]["initialization_method"] == "pca"
        assert paper[sampling][topology]["initialization_method"] == "random"
    for sampling in paper:
        for topology in paper[sampling]:
            a, b = dict(library[sampling][topology]), dict(paper[sampling][topology])
            a.pop("initialization_method")
            b.pop("initialization_method")
            assert a == b


def test_variants_remain_separate():
    variant = load_profile("publication-rng-random")
    assert variant["random"]["mst"]["initialization_method"] == "pca"
    assert variant["random"]["rng"]["initialization_method"] == "random"
    control = load_profile("publication-rng-mst-config")
    for sampling in ("full", "random"):
        assert control[sampling]["rng"] == control[sampling]["mst"]


def test_profile_resolution_normalizes_names_and_returns_independent_values():
    result = resolve_profile_defaults(" RANDOM ", " RNG ", "publication")
    assert result["enable_momentum"] is True
    assert result["initial_momentum"] == pytest.approx(0.5272565822346229)
    result["initial_momentum"] = 123
    assert resolve_profile_defaults("random", "rng", "publication")["initial_momentum"] != 123
    raw = load_profile("publication")
    raw["full"]["rng"].clear()
    assert load_profile("publication")["full"]["rng"]
    assert resolve_profile_defaults("hdsssom", "grid", "publication") == {}


def test_invalid_profiles_fail_clearly():
    with pytest.raises(ValueError, match="Unknown FloatSOM profile"):
        load_profile("../publication")
    with pytest.raises(ValueError, match="must be numeric"):
        validate_profile({"full": {"rng": {"initial_radius": True}}})
    with pytest.raises(ValueError, match="Unsupported contextual"):
        validate_profile({"full": {"rng": {"typo": 1}}})
