import numpy as np
import pytest

from enzymeopt.randomness import SeedManager, derive_seed, make_rng
from enzymeopt.simulation import simulate_dataset
from enzymeopt.config import ExperimentConfig
from enzymeopt.experiment import run_experiment


def test_same_master_seed_and_labels_reproduce_sequence() -> None:
    first = make_rng(42, "replicate", 3, "random", "noise").normal(size=20)
    second = make_rng(42, "replicate", 3, "random", "noise").normal(size=20)

    np.testing.assert_array_equal(first, second)


def test_distinct_labels_produce_distinct_streams() -> None:
    random_values = make_rng(42, "replicate", 3, "random").integers(0, 2**32, size=20)
    optimal_values = make_rng(42, "replicate", 3, "d_optimal").integers(0, 2**32, size=20)

    assert not np.array_equal(random_values, optimal_values)


def test_strategy_iteration_order_does_not_change_streams() -> None:
    manager = SeedManager(20260915)

    forward = {
        strategy: manager.rng_for("replicate", 7, strategy, "noise").normal(size=10)
        for strategy in ("random", "log_spaced", "d_optimal")
    }
    reverse = {
        strategy: manager.rng_for("replicate", 7, strategy, "noise").normal(size=10)
        for strategy in reversed(("random", "log_spaced", "d_optimal"))
    }

    for strategy in forward:
        np.testing.assert_array_equal(forward[strategy], reverse[strategy])


def test_label_types_cannot_collide() -> None:
    assert derive_seed(1, 7) != derive_seed(1, "7")


@pytest.mark.parametrize("seed", [-1, 2**128, True])
def test_invalid_master_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="master_seed"):
        SeedManager(seed)  # type: ignore[arg-type]


def test_invalid_seed_label_is_rejected() -> None:
    with pytest.raises(TypeError, match="strings or integers"):
        derive_seed(1, 1.5)  # type: ignore[arg-type]


def test_synthetic_dataset_is_reproducible_from_named_seed() -> None:
    concentrations = np.geomspace(0.05, 20.0, 25)
    first = simulate_dataset(
        concentrations,
        true_km=1.0,
        true_vmax=1.0,
        noise_std=0.05,
        rng=make_rng(99, "replicate", 4, "random", "noise"),
    )
    second = simulate_dataset(
        concentrations,
        true_km=1.0,
        true_vmax=1.0,
        noise_std=0.05,
        rng=make_rng(99, "replicate", 4, "random", "noise"),
    )

    np.testing.assert_array_equal(first.observed_rates, second.observed_rates)


def test_experiment_order_independence_and_paired_noise():
    config = ExperimentConfig(seed=42)
    forward = {s: run_experiment(config, strategy=s, measurement_budget=6)
               for s in config.strategies}
    reverse = {s: run_experiment(config, strategy=s, measurement_budget=6)
               for s in reversed(config.strategies)}
    assert forward == reverse
    residual_noise = []
    for result in forward.values():
        assert result.succeeded, result.message
        residual_noise.append([s.observation.rate - s.observation.expected_rate
                               for s in result.steps])
    np.testing.assert_allclose(residual_noise, np.tile(residual_noise[0], (3, 1)), atol=1e-15)
