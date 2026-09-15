import json

import pytest

from enzymeopt.config import ExperimentConfig


def test_default_config_matches_baseline_plan() -> None:
    config = ExperimentConfig()

    assert config.true_km == 1.0
    assert config.true_vmax == 1.0
    assert config.noise_std == 0.05
    assert config.substrate_min == 0.05
    assert config.substrate_max == 20.0
    assert config.candidate_count == 200
    assert config.measurement_budgets == (3, 4, 6, 8, 12, 16, 24)
    assert config.replicates == 1_000
    assert config.strategies == ("random", "log_spaced", "d_optimal")


def test_config_json_round_trip_is_lossless() -> None:
    original = ExperimentConfig(seed=20260915, replicates=25, strategies=("random",))

    serialized = json.loads(json.dumps(original.to_dict()))

    assert ExperimentConfig.from_dict(serialized) == original


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"true_km": 0.0}, "true_km"),
        ({"true_vmax": float("inf")}, "true_vmax"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"substrate_min": 2.0, "substrate_max": 1.0}, "substrate_min"),
        ({"candidate_count": 1}, "candidate_count"),
        ({"measurement_budgets": (2, 3)}, "measurement budget"),
        ({"measurement_budgets": (3, 3)}, "strictly increasing"),
        ({"replicates": 0}, "replicates"),
        ({"strategies": ("unknown",)}, "unknown strategies"),
        ({"seed": -1}, "seed"),
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**changes)


def test_config_requires_unambiguous_tuple_collections() -> None:
    with pytest.raises(TypeError, match="measurement_budgets must be a tuple"):
        ExperimentConfig(measurement_budgets=[3, 4])  # type: ignore[arg-type]
