"""Nonlinear least-squares fitting for Michaelis-Menten observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from enzymeopt.model import (
    michaelis_menten,
    michaelis_menten_log_jacobian,
    parameters_from_log_space,
    parameters_to_log_space,
)
from enzymeopt.results import ConfidenceInterval, FitResult


FloatArray = NDArray[np.float64]
_LOG_PARAMETER_LIMIT = 700.0


@dataclass(frozen=True, slots=True)
class FitOptions:
    """Deterministic numerical and uncertainty settings for a fit."""

    confidence_level: float = 0.95
    condition_number_threshold: float = 1.0e12
    max_evaluations: int = 1_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, Real)
            or not isfinite(self.confidence_level)
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be a finite number between 0 and 1")
        if (
            isinstance(self.condition_number_threshold, bool)
            or not isinstance(self.condition_number_threshold, Real)
            or not isfinite(self.condition_number_threshold)
            or self.condition_number_threshold <= 1.0
        ):
            raise ValueError("condition_number_threshold must be finite and greater than 1")
        if (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, Integral)
            or self.max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer")


def fit_michaelis_menten(
    concentrations: ArrayLike,
    observed_rates: ArrayLike,
    *,
    initial_km: Real | None = None,
    initial_vmax: Real | None = None,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit ``km`` and ``vmax`` in log space using nonlinear least squares.

    Invalid inputs raise ``ValueError``. Numerical optimizer failures are
    returned as ``FitResult(converged=False, ...)`` so Monte Carlo experiments
    can retain and count them.
    """

    substrate = _numeric_vector("concentrations", concentrations, nonnegative=True)
    rates = _numeric_vector("observed_rates", observed_rates)
    if substrate.size != rates.size:
        raise ValueError("concentrations and observed_rates must have equal lengths")
    if substrate.size < 2:
        raise ValueError("at least two measurements are required to fit two parameters")

    settings = FitOptions() if options is None else options
    if not isinstance(settings, FitOptions):
        raise TypeError("options must be a FitOptions instance")
    start_km, start_vmax = _initial_parameters(
        substrate,
        rates,
        initial_km=initial_km,
        initial_vmax=initial_vmax,
    )
    initial_log_parameters = np.asarray(
        parameters_to_log_space(start_km, start_vmax),
        dtype=np.float64,
    )

    def residual_function(log_parameters: FloatArray) -> FloatArray:
        km, vmax = parameters_from_log_space(*log_parameters)
        predicted = michaelis_menten(substrate, km=km, vmax=vmax)
        return np.asarray(predicted, dtype=np.float64) - rates

    def jacobian_function(log_parameters: FloatArray) -> FloatArray:
        return michaelis_menten_log_jacobian(substrate, *log_parameters)

    try:
        optimized = optimize.least_squares(
            residual_function,
            initial_log_parameters,
            jac=jacobian_function,
            bounds=(-_LOG_PARAMETER_LIMIT, _LOG_PARAMETER_LIMIT),
            max_nfev=int(settings.max_evaluations),
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return FitResult(converged=False, message=f"optimizer failed: {error}")

    residuals = np.asarray(optimized.fun, dtype=np.float64)
    residual_sum_squares = float(residuals @ residuals)
    evaluations = int(optimized.nfev)
    if not optimized.success:
        return FitResult(
            converged=False,
            residual_sum_squares=residual_sum_squares,
            residuals=_vector_tuple(residuals),
            evaluations=evaluations,
            message=f"optimizer did not converge: {optimized.message}",
        )

    try:
        km, vmax = parameters_from_log_space(*optimized.x)
    except ValueError as error:
        return FitResult(
            converged=False,
            residual_sum_squares=residual_sum_squares,
            residuals=_vector_tuple(residuals),
            evaluations=evaluations,
            message=f"optimizer returned invalid parameters: {error}",
        )

    jacobian = np.asarray(optimized.jac, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition_number = float(np.linalg.cond(jacobian)) if rank == 2 else None
    uncertainty, diagnostic = _estimate_uncertainty(
        km=km,
        vmax=vmax,
        residual_sum_squares=residual_sum_squares,
        jacobian=jacobian,
        active_bounds=bool(np.any(optimized.active_mask)),
        confidence_level=float(settings.confidence_level),
        condition_number=condition_number,
        condition_number_threshold=float(settings.condition_number_threshold),
    )

    return FitResult(
        converged=True,
        km=km,
        vmax=vmax,
        km_interval=uncertainty.km_interval,
        vmax_interval=uncertainty.vmax_interval,
        residual_sum_squares=residual_sum_squares,
        residuals=_vector_tuple(residuals),
        jacobian=_matrix_tuple(jacobian),
        covariance=uncertainty.covariance,
        jacobian_rank=rank,
        jacobian_condition_number=condition_number,
        evaluations=evaluations,
        message=diagnostic,
    )


@dataclass(frozen=True, slots=True)
class _Uncertainty:
    covariance: tuple[tuple[float, float], tuple[float, float]] | None = None
    km_interval: ConfidenceInterval | None = None
    vmax_interval: ConfidenceInterval | None = None


def _estimate_uncertainty(
    *,
    km: float,
    vmax: float,
    residual_sum_squares: float,
    jacobian: FloatArray,
    active_bounds: bool,
    confidence_level: float,
    condition_number: float | None,
    condition_number_threshold: float,
) -> tuple[_Uncertainty, str]:
    degrees_of_freedom = jacobian.shape[0] - 2
    if degrees_of_freedom <= 0:
        return _Uncertainty(), "converged; confidence intervals unavailable: no residual degrees of freedom"
    if np.linalg.matrix_rank(jacobian) < 2:
        return _Uncertainty(), "converged; confidence intervals unavailable: rank-deficient Jacobian"
    if condition_number is None or condition_number > condition_number_threshold:
        return _Uncertainty(), "converged; confidence intervals unavailable: ill-conditioned Jacobian"
    if active_bounds:
        return _Uncertainty(), "converged; confidence intervals unavailable: parameter reached a boundary"

    variance = residual_sum_squares / degrees_of_freedom
    try:
        covariance_log = variance * np.linalg.inv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        return _Uncertainty(), "converged; confidence intervals unavailable: singular information matrix"
    transform = np.diag([km, vmax])
    covariance = transform @ covariance_log @ transform
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0.0):
        return _Uncertainty(), "converged; confidence intervals unavailable: invalid covariance"

    critical_value = float(stats.t.ppf((1.0 + confidence_level) / 2.0, degrees_of_freedom))
    standard_errors = np.sqrt(np.diag(covariance))
    intervals = tuple(
        ConfidenceInterval(
            lower=float(estimate - critical_value * standard_error),
            upper=float(estimate + critical_value * standard_error),
            level=confidence_level,
        )
        for estimate, standard_error in zip((km, vmax), standard_errors, strict=True)
    )
    return (
        _Uncertainty(
            covariance=_matrix_tuple(covariance),
            km_interval=intervals[0],
            vmax_interval=intervals[1],
        ),
        "converged",
    )


def _initial_parameters(
    concentrations: FloatArray,
    rates: FloatArray,
    *,
    initial_km: Real | None,
    initial_vmax: Real | None,
) -> tuple[float, float]:
    positive_concentrations = concentrations[concentrations > 0.0]
    if positive_concentrations.size:
        default_km = float(
            np.exp(
                (log(float(np.min(positive_concentrations))) + log(float(np.max(positive_concentrations))))
                / 2.0
            )
        )
    else:
        default_km = 1.0

    maximum_rate = float(np.max(rates))
    if maximum_rate > 0.0:
        default_vmax = 1.2 * maximum_rate
    else:
        default_vmax = max(float(np.max(np.abs(rates))), 1.0)

    km = default_km if initial_km is None else _positive_scalar("initial_km", initial_km)
    vmax = default_vmax if initial_vmax is None else _positive_scalar("initial_vmax", initial_vmax)
    return km, vmax


def _numeric_vector(name: str, values: ArrayLike, *, nonnegative: bool = False) -> FloatArray:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.bool_) or raw.dtype.kind in "SU":
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must contain non-negative values")
    return result


def _positive_scalar(name: str, value: Real) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _vector_tuple(values: FloatArray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _matrix_tuple(values: FloatArray) -> tuple[tuple[float, float], ...]:
    return tuple((float(row[0]), float(row[1])) for row in values)
