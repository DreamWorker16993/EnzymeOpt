"""Michaelis-Menten model, parameter transforms, and sensitivities."""

from __future__ import annotations

from math import exp, isfinite, log
from numbers import Real
from typing import TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]


@overload
def michaelis_menten(substrate: Real, km: Real, vmax: Real) -> float: ...


@overload
def michaelis_menten(substrate: ArrayLike, km: Real, vmax: Real) -> FloatArray: ...


def michaelis_menten(
    substrate: ArrayLike,
    km: Real,
    vmax: Real,
) -> float | FloatArray:
    """Evaluate ``Vmax * S / (KM + S)`` for non-negative substrate.

    A scalar substrate produces a Python ``float``. Array-like input produces a
    NumPy array with the same shape. ``km`` and ``vmax`` must be positive and
    finite.
    """

    concentrations, scalar_input = _validate_substrate(substrate)
    km_value = _validate_positive_parameter("km", km)
    vmax_value = _validate_positive_parameter("vmax", vmax)
    fraction = _saturation_fraction(concentrations, km_value)
    rates = vmax_value * fraction
    return _scalar_or_array(rates, scalar_input)


def michaelis_menten_jacobian(
    substrate: ArrayLike,
    km: Real,
    vmax: Real,
) -> FloatArray:
    """Return derivatives with respect to ``(km, vmax)``.

    The last output axis contains ``d(rate)/d(km)`` followed by
    ``d(rate)/d(vmax)``. Therefore scalar input has shape ``(2,)`` and input
    with shape ``(...)`` produces output with shape ``(..., 2)``.
    """

    concentrations, _ = _validate_substrate(substrate)
    km_value = _validate_positive_parameter("km", km)
    vmax_value = _validate_positive_parameter("vmax", vmax)
    fraction = _saturation_fraction(concentrations, km_value)

    derivative_km = -vmax_value * fraction * (1.0 - fraction) / km_value
    derivative_vmax = fraction
    return np.stack((derivative_km, derivative_vmax), axis=-1)


def michaelis_menten_log_jacobian(
    substrate: ArrayLike,
    log_km: Real,
    log_vmax: Real,
) -> FloatArray:
    """Return derivatives with respect to ``(log(km), log(vmax))``."""

    km, vmax = parameters_from_log_space(log_km, log_vmax)
    concentrations, _ = _validate_substrate(substrate)
    fraction = _saturation_fraction(concentrations, km)

    derivative_log_km = -vmax * fraction * (1.0 - fraction)
    derivative_log_vmax = vmax * fraction
    return np.stack((derivative_log_km, derivative_log_vmax), axis=-1)


def parameters_to_log_space(km: Real, vmax: Real) -> tuple[float, float]:
    """Transform positive natural parameters to ``(log(km), log(vmax))``."""

    km_value = _validate_positive_parameter("km", km)
    vmax_value = _validate_positive_parameter("vmax", vmax)
    return log(km_value), log(vmax_value)


def parameters_from_log_space(log_km: Real, log_vmax: Real) -> tuple[float, float]:
    """Transform finite log parameters to positive natural parameters."""

    log_km_value = _validate_finite_scalar("log_km", log_km)
    log_vmax_value = _validate_finite_scalar("log_vmax", log_vmax)
    try:
        km = exp(log_km_value)
        vmax = exp(log_vmax_value)
    except OverflowError as error:
        raise ValueError("log parameters must exponentiate to finite values") from error
    if not isfinite(km) or not isfinite(vmax) or km <= 0.0 or vmax <= 0.0:
        raise ValueError("log parameters must exponentiate to positive finite values")
    return km, vmax


def _validate_substrate(substrate: ArrayLike) -> tuple[FloatArray, bool]:
    raw = np.asarray(substrate)
    if np.issubdtype(raw.dtype, np.bool_) or raw.dtype.kind in "SU":
        raise ValueError("substrate must contain finite non-negative numbers")
    try:
        concentrations = np.asarray(substrate, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("substrate must contain finite non-negative numbers") from error
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations < 0.0):
        raise ValueError("substrate must contain finite non-negative numbers")
    return concentrations, concentrations.ndim == 0


def _validate_positive_parameter(name: str, value: Real) -> float:
    result = _validate_finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validate_finite_scalar(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _saturation_fraction(substrate: FloatArray, km: float) -> FloatArray:
    fraction = np.zeros_like(substrate, dtype=np.float64)
    positive = substrate > 0.0
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        fraction[positive] = 1.0 / (1.0 + km / substrate[positive])
    return fraction


def _scalar_or_array(values: FloatArray, scalar_input: bool) -> float | FloatArray:
    if scalar_input:
        return values.item()
    return values
