import json

import numpy as np
import pytest

from enzymeopt.fitting import FitOptions, fit_michaelis_menten
from enzymeopt.model import michaelis_menten
from enzymeopt.randomness import make_rng
from enzymeopt.results import FitResult
from enzymeopt.simulation import simulate_dataset


def test_noiseless_well_designed_data_recovers_true_parameters() -> None:
    concentrations = np.geomspace(0.05, 20.0, 20)
    rates = michaelis_menten(concentrations, km=1.7, vmax=2.4)

    result = fit_michaelis_menten(concentrations, rates)

    assert result.converged
    assert result.km == pytest.approx(1.7, rel=1.0e-7)
    assert result.vmax == pytest.approx(2.4, rel=1.0e-7)
    assert result.residual_sum_squares == pytest.approx(0.0, abs=1.0e-18)
    assert result.jacobian_rank == 2
    assert result.covariance is not None
    assert result.km_interval is not None
    assert result.vmax_interval is not None


def test_low_noise_data_recovers_parameters_with_reasonable_error() -> None:
    dataset = simulate_dataset(
        np.geomspace(0.05, 20.0, 50),
        true_km=1.0,
        true_vmax=2.0,
        noise_std=0.02,
        rng=make_rng(20260916, "low-noise-fit"),
    )

    result = fit_michaelis_menten(dataset.concentrations, dataset.observed_rates)

    assert result.converged
    assert result.km == pytest.approx(1.0, rel=0.1)
    assert result.vmax == pytest.approx(2.0, rel=0.05)
    assert result.km_interval is not None
    assert result.vmax_interval is not None


def test_covariance_and_intervals_are_well_formed() -> None:
    dataset = simulate_dataset(
        np.geomspace(0.05, 20.0, 30),
        true_km=1.0,
        true_vmax=2.0,
        noise_std=0.05,
        rng=make_rng(30, "uncertainty"),
    )

    result = fit_michaelis_menten(dataset.concentrations, dataset.observed_rates)

    covariance = np.asarray(result.covariance)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.diag(covariance) >= 0.0)
    assert result.km_interval is not None
    assert result.vmax_interval is not None
    assert result.km_interval.lower < result.km < result.km_interval.upper
    assert result.vmax_interval.lower < result.vmax < result.vmax_interval.upper
    assert result.km_interval.level == 0.95


def test_fitted_parameters_are_positive_with_negative_observations() -> None:
    concentrations = np.geomspace(0.05, 20.0, 20)
    rates = np.linspace(-0.2, 1.0, concentrations.size)

    result = fit_michaelis_menten(concentrations, rates)

    assert result.converged
    assert result.km is not None and result.km > 0.0
    assert result.vmax is not None and result.vmax > 0.0


def test_two_measurements_do_not_produce_misleading_intervals() -> None:
    concentrations = np.array([0.5, 2.0])
    rates = michaelis_menten(concentrations, km=1.0, vmax=2.0)

    result = fit_michaelis_menten(concentrations, rates)

    assert result.converged
    assert result.km_interval is None
    assert result.vmax_interval is None
    assert result.covariance is None
    assert "no residual degrees of freedom" in result.message


def test_rank_deficient_design_does_not_produce_intervals() -> None:
    concentrations = np.zeros(5)
    rates = np.zeros(5)

    result = fit_michaelis_menten(concentrations, rates)

    assert result.converged
    assert result.jacobian_rank == 0
    assert result.jacobian_condition_number is None
    assert result.km_interval is None
    assert result.vmax_interval is None
    assert result.covariance is None
    assert "rank-deficient" in result.message


def test_condition_threshold_can_suppress_intervals() -> None:
    concentrations = np.geomspace(1.0e-6, 1.0e-5, 10)
    rates = michaelis_menten(concentrations, km=1.0, vmax=2.0)

    result = fit_michaelis_menten(
        concentrations,
        rates,
        options=FitOptions(condition_number_threshold=10.0),
    )

    assert result.converged
    assert result.jacobian_condition_number is not None
    assert result.jacobian_condition_number > 10.0
    assert result.km_interval is None
    assert "ill-conditioned" in result.message


def test_same_inputs_produce_identical_fit() -> None:
    concentrations = np.geomspace(0.05, 20.0, 20)
    rates = michaelis_menten(concentrations, km=0.8, vmax=1.4)

    first = fit_michaelis_menten(concentrations, rates)
    second = fit_michaelis_menten(concentrations, rates)

    assert first == second


def test_explicit_initial_values_are_supported() -> None:
    concentrations = np.geomspace(0.05, 20.0, 20)
    rates = michaelis_menten(concentrations, km=1.0, vmax=2.0)

    result = fit_michaelis_menten(
        concentrations,
        rates,
        initial_km=10.0,
        initial_vmax=0.5,
    )

    assert result.converged
    assert result.km == pytest.approx(1.0, rel=1.0e-7)
    assert result.vmax == pytest.approx(2.0, rel=1.0e-7)


def test_fit_result_diagnostics_survive_json_round_trip() -> None:
    concentrations = np.geomspace(0.1, 10.0, 10)
    rates = michaelis_menten(concentrations, km=1.0, vmax=2.0)
    original = fit_michaelis_menten(concentrations, rates)

    restored = FitResult.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original


@pytest.mark.parametrize(
    ("concentrations", "rates", "message"),
    [
        ([1.0], [0.5], "at least two"),
        ([1.0, 2.0], [0.5], "equal lengths"),
        ([-1.0, 2.0], [0.5, 1.0], "non-negative"),
        ([1.0, float("nan")], [0.5, 1.0], "concentrations"),
        ([1.0, 2.0], [0.5, float("inf")], "observed_rates"),
    ],
)
def test_invalid_fit_data_is_rejected(
    concentrations: object,
    rates: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_michaelis_menten(concentrations, rates)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["initial_km", "initial_vmax"])
def test_invalid_initial_values_are_rejected(name: str) -> None:
    arguments = {name: 0.0}

    with pytest.raises(ValueError, match=name):
        fit_michaelis_menten([1.0, 2.0], [0.5, 1.0], **arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"confidence_level": 1.0},
        {"condition_number_threshold": 1.0},
        {"max_evaluations": 0},
    ],
)
def test_invalid_fit_options_are_rejected(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FitOptions(**arguments)  # type: ignore[arg-type]


def test_optimizer_exception_is_returned_as_structured_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr("enzymeopt.fitting.optimize.least_squares", fail)

    result = fit_michaelis_menten([1.0, 2.0], [0.5, 1.0])

    assert not result.converged
    assert result.km is None
    assert result.vmax is None
    assert "synthetic optimizer failure" in result.message
