import json

import pytest

from enzymeopt.results import (
    ConfidenceInterval,
    ExperimentResult,
    ExperimentStep,
    FitResult,
    Observation,
)


def test_experiment_result_json_round_trip_is_lossless() -> None:
    result = ExperimentResult(
        replicate=4,
        strategy="d_optimal",
        seed=123,
        steps=(
            ExperimentStep(
                measurement_number=1,
                observation=Observation(concentration=0.05, rate=-0.01, expected_rate=0.02),
                fit=FitResult(
                    converged=True,
                    km=1.1,
                    vmax=0.9,
                    km_interval=ConfidenceInterval(0.8, 1.4),
                    vmax_interval=ConfidenceInterval(0.7, 1.1),
                    residual_sum_squares=0.003,
                ),
            ),
        ),
        succeeded=True,
    )

    serialized = json.loads(json.dumps(result.to_dict()))

    assert ExperimentResult.from_dict(serialized) == result


def test_observation_preserves_negative_rate() -> None:
    observation = Observation(concentration=0.01, rate=-0.2)

    assert observation.rate == -0.2


def test_converged_fit_requires_both_positive_estimates() -> None:
    with pytest.raises(ValueError, match="require both"):
        FitResult(converged=True, km=1.0)
    with pytest.raises(ValueError, match="greater than"):
        FitResult(converged=True, km=0.0, vmax=1.0)


def test_failed_fit_cannot_contain_estimates() -> None:
    with pytest.raises(ValueError, match="failed fits"):
        FitResult(converged=False, km=1.0, vmax=1.0, message="did not converge")


def test_experiment_steps_must_be_consecutive() -> None:
    step = ExperimentStep(2, Observation(1.0, 0.5))

    with pytest.raises(ValueError, match="consecutive"):
        ExperimentResult(0, "random", 1, (step,), True)


def test_result_records_reject_unstructured_nested_values() -> None:
    with pytest.raises(TypeError, match="Observation"):
        ExperimentStep(1, {"concentration": 1.0, "rate": 0.5})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ConfidenceInterval"):
        FitResult(
            converged=True,
            km=1.0,
            vmax=1.0,
            km_interval={"lower": 0.8, "upper": 1.2},  # type: ignore[arg-type]
        )


def test_confidence_interval_validates_bounds_and_level() -> None:
    with pytest.raises(ValueError, match="lower"):
        ConfidenceInterval(2.0, 1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ConfidenceInterval(1.0, 2.0, level=1.0)
