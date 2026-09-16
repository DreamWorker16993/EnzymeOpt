import numpy as np
import pytest

from enzymeopt.model import (
    michaelis_menten,
    michaelis_menten_jacobian,
    michaelis_menten_log_jacobian,
    parameters_from_log_space,
    parameters_to_log_space,
)


def test_model_matches_known_michaelis_menten_values() -> None:
    substrate = np.array([0.0, 2.0, 18.0])

    rates = michaelis_menten(substrate, km=2.0, vmax=10.0)

    np.testing.assert_allclose(rates, [0.0, 5.0, 9.0])


def test_rate_approaches_vmax_at_high_substrate() -> None:
    rate = michaelis_menten(1.0e12, km=1.0, vmax=3.0)

    assert rate < 3.0
    assert rate == pytest.approx(3.0, rel=2.0e-12)


def test_scalar_and_array_return_shapes_are_documented() -> None:
    scalar_rate = michaelis_menten(1.0, km=1.0, vmax=2.0)
    array_rate = michaelis_menten([[0.0, 1.0], [2.0, 3.0]], km=1.0, vmax=2.0)
    scalar_jacobian = michaelis_menten_jacobian(1.0, km=1.0, vmax=2.0)
    array_jacobian = michaelis_menten_jacobian([0.0, 1.0], km=1.0, vmax=2.0)

    assert isinstance(scalar_rate, float)
    assert isinstance(array_rate, np.ndarray)
    assert array_rate.shape == (2, 2)
    assert scalar_jacobian.shape == (2,)
    assert array_jacobian.shape == (2, 2)


@pytest.mark.parametrize(
    "substrate",
    [-1.0, [0.0, -0.1], float("nan"), float("inf"), True, [False, True], "1.0"],
)
def test_invalid_substrate_is_rejected(substrate: object) -> None:
    with pytest.raises(ValueError, match="substrate"):
        michaelis_menten(substrate, km=1.0, vmax=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("km", 0.0),
        ("km", -1.0),
        ("km", float("nan")),
        ("vmax", 0.0),
        ("vmax", float("inf")),
        ("vmax", True),
    ],
)
def test_nonphysical_parameters_are_rejected(name: str, value: object) -> None:
    parameters = {"km": 1.0, "vmax": 1.0, name: value}

    with pytest.raises(ValueError, match=name):
        michaelis_menten(1.0, **parameters)  # type: ignore[arg-type]


def test_natural_parameter_jacobian_matches_finite_differences() -> None:
    substrate = np.array([0.0, 0.1, 1.0, 10.0])
    km = 1.7
    vmax = 2.3
    step = 1.0e-6

    km_difference = (
        michaelis_menten(substrate, km + step, vmax)
        - michaelis_menten(substrate, km - step, vmax)
    ) / (2.0 * step)
    vmax_difference = (
        michaelis_menten(substrate, km, vmax + step)
        - michaelis_menten(substrate, km, vmax - step)
    ) / (2.0 * step)
    finite_difference = np.column_stack((km_difference, vmax_difference))

    np.testing.assert_allclose(
        michaelis_menten_jacobian(substrate, km, vmax),
        finite_difference,
        rtol=1.0e-7,
        atol=1.0e-9,
    )


def test_log_parameter_jacobian_matches_finite_differences() -> None:
    substrate = np.array([0.0, 0.1, 1.0, 10.0])
    log_km, log_vmax = parameters_to_log_space(1.7, 2.3)
    step = 1.0e-6

    def evaluate(first: float, second: float) -> np.ndarray:
        km, vmax = parameters_from_log_space(first, second)
        return np.asarray(michaelis_menten(substrate, km, vmax))

    km_difference = (
        evaluate(log_km + step, log_vmax) - evaluate(log_km - step, log_vmax)
    ) / (2.0 * step)
    vmax_difference = (
        evaluate(log_km, log_vmax + step) - evaluate(log_km, log_vmax - step)
    ) / (2.0 * step)
    finite_difference = np.column_stack((km_difference, vmax_difference))

    np.testing.assert_allclose(
        michaelis_menten_log_jacobian(substrate, log_km, log_vmax),
        finite_difference,
        rtol=1.0e-7,
        atol=1.0e-9,
    )


def test_parameter_transform_round_trip() -> None:
    log_parameters = parameters_to_log_space(km=0.25, vmax=4.5)

    assert parameters_from_log_space(*log_parameters) == pytest.approx((0.25, 4.5))


@pytest.mark.parametrize(
    "log_parameters",
    [
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (1_000.0, 0.0),
        (-1_000.0, 0.0),
    ],
)
def test_invalid_log_parameters_are_rejected(log_parameters: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="log"):
        parameters_from_log_space(*log_parameters)


def test_model_remains_stable_for_extreme_finite_concentrations() -> None:
    substrate = np.array([0.0, np.finfo(float).tiny, np.finfo(float).max])

    rates = michaelis_menten(substrate, km=np.finfo(float).max, vmax=1.0)
    jacobian = michaelis_menten_jacobian(
        substrate,
        km=np.finfo(float).max,
        vmax=1.0,
    )

    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(jacobian))
    assert rates[-1] == pytest.approx(0.5)
