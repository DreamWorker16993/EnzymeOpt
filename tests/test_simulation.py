import json

import numpy as np
import pytest

from enzymeopt.randomness import make_rng
from enzymeopt.results import Observation
from enzymeopt.simulation import SyntheticDataset, simulate_dataset


def test_zero_noise_returns_exact_model_values() -> None:
    dataset = simulate_dataset(
        [0.0, 1.0, 9.0],
        true_km=1.0,
        true_vmax=2.0,
        noise_std=0.0,
    )

    np.testing.assert_allclose(dataset.expected_rates, [0.0, 1.0, 1.8])
    np.testing.assert_array_equal(dataset.observed_rates, dataset.expected_rates)


def test_zero_noise_does_not_consume_optional_rng() -> None:
    used_rng = make_rng(10, "simulation")
    untouched_rng = make_rng(10, "simulation")

    simulate_dataset(
        [1.0],
        true_km=1.0,
        true_vmax=1.0,
        noise_std=0.0,
        rng=used_rng,
    )

    np.testing.assert_array_equal(used_rng.normal(size=10), untouched_rng.normal(size=10))


def test_positive_noise_requires_explicit_generator() -> None:
    with pytest.raises(TypeError, match="explicit numpy.random.Generator"):
        simulate_dataset(
            [1.0],
            true_km=1.0,
            true_vmax=1.0,
            noise_std=0.1,
        )


def test_gaussian_noise_has_expected_empirical_moments() -> None:
    sigma = 2.0
    dataset = simulate_dataset(
        np.zeros(100_000),
        true_km=1.0,
        true_vmax=1.0,
        noise_std=sigma,
        rng=make_rng(20260916, "noise-moments"),
    )
    noise = dataset.observed_rates - dataset.expected_rates

    assert float(np.mean(noise)) == pytest.approx(0.0, abs=0.02)
    assert float(np.std(noise, ddof=0)) == pytest.approx(sigma, rel=0.01)


def test_negative_noisy_rates_are_preserved() -> None:
    dataset = simulate_dataset(
        np.zeros(200),
        true_km=1.0,
        true_vmax=1.0,
        noise_std=1.0,
        rng=make_rng(5, "negative-rates"),
    )

    assert np.any(dataset.observed_rates < 0.0)


def test_repeated_concentrations_create_independent_measurements() -> None:
    dataset = simulate_dataset(
        [1.0, 1.0, 1.0],
        true_km=1.0,
        true_vmax=1.0,
        noise_std=0.2,
        rng=make_rng(8, "replicates"),
    )

    np.testing.assert_array_equal(dataset.concentrations, [1.0, 1.0, 1.0])
    assert np.unique(dataset.observed_rates).size == 3


def test_dataset_keeps_immutable_copies_and_metadata() -> None:
    concentrations = np.array([0.5, 1.0])
    dataset = simulate_dataset(
        concentrations,
        true_km=1.5,
        true_vmax=2.5,
        noise_std=0.0,
    )
    concentrations[0] = 99.0

    np.testing.assert_array_equal(dataset.concentrations, [0.5, 1.0])
    assert dataset.true_km == 1.5
    assert dataset.true_vmax == 2.5
    assert dataset.noise_std == 0.0
    with pytest.raises(ValueError, match="read-only"):
        dataset.observed_rates[0] = 10.0


def test_dataset_json_round_trip_is_lossless() -> None:
    original = simulate_dataset(
        [0.1, 1.0],
        true_km=1.0,
        true_vmax=2.0,
        noise_std=0.1,
        rng=make_rng(22, "round-trip"),
    )

    restored = SyntheticDataset.from_dict(json.loads(json.dumps(original.to_dict())))

    np.testing.assert_array_equal(restored.concentrations, original.concentrations)
    np.testing.assert_array_equal(restored.expected_rates, original.expected_rates)
    np.testing.assert_array_equal(restored.observed_rates, original.observed_rates)
    assert restored.true_km == original.true_km
    assert restored.true_vmax == original.true_vmax
    assert restored.noise_std == original.noise_std


def test_dataset_rejects_inconsistent_array_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        SyntheticDataset(
            concentrations=np.array([0.1, 1.0]),
            expected_rates=np.array([0.1]),
            observed_rates=np.array([0.1, 0.5]),
            true_km=1.0,
            true_vmax=1.0,
            noise_std=0.0,
        )


def test_dataset_converts_to_observation_records() -> None:
    dataset = simulate_dataset(
        [0.0, 1.0],
        true_km=1.0,
        true_vmax=2.0,
        noise_std=0.0,
    )

    assert dataset.observations() == (
        Observation(0.0, 0.0, 0.0),
        Observation(1.0, 1.0, 1.0),
    )


@pytest.mark.parametrize(
    "concentrations",
    [
        [],
        1.0,
        [[1.0]],
        [-1.0],
        [float("nan")],
        [float("inf")],
        [True],
        ["1.0"],
        ["bad"],
    ],
)
def test_invalid_concentration_vectors_are_rejected(concentrations: object) -> None:
    with pytest.raises(ValueError, match="concentrations"):
        simulate_dataset(
            concentrations,  # type: ignore[arg-type]
            true_km=1.0,
            true_vmax=1.0,
            noise_std=0.0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("true_km", 0.0),
        ("true_km", float("nan")),
        ("true_vmax", -1.0),
        ("true_vmax", float("inf")),
        ("noise_std", -0.1),
        ("noise_std", True),
    ],
)
def test_invalid_generation_parameters_are_rejected(name: str, value: object) -> None:
    parameters = {
        "true_km": 1.0,
        "true_vmax": 1.0,
        "noise_std": 0.0,
        name: value,
    }

    with pytest.raises(ValueError, match=name):
        simulate_dataset([1.0], **parameters)  # type: ignore[arg-type]
