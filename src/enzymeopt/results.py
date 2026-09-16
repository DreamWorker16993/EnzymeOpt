"""Serializable result records shared across EnzymeOpt workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True, slots=True)
class Observation:
    """One substrate-concentration measurement."""

    concentration: float
    rate: float
    expected_rate: float | None = None

    def __post_init__(self) -> None:
        _require_finite("concentration", self.concentration, minimum=0.0)
        _require_finite("rate", self.rate)
        if self.expected_rate is not None:
            _require_finite("expected_rate", self.expected_rate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A closed confidence interval for one fitted parameter."""

    lower: float
    upper: float
    level: float = 0.95

    def __post_init__(self) -> None:
        _require_finite("lower", self.lower)
        _require_finite("upper", self.upper)
        _require_finite("level", self.level)
        if self.lower > self.upper:
            raise ValueError("confidence interval lower must not exceed upper")
        if not 0.0 < self.level < 1.0:
            raise ValueError("confidence interval level must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceInterval:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class FitResult:
    """Parameter estimates, uncertainty diagnostics, or a fitting failure.

    ``residuals`` are predicted minus observed rates. ``jacobian`` contains
    derivatives of those residuals with respect to ``(log(km), log(vmax))``.
    ``covariance`` is reported in the natural ``(km, vmax)`` parameter space.
    """

    converged: bool
    km: float | None = None
    vmax: float | None = None
    km_interval: ConfidenceInterval | None = None
    vmax_interval: ConfidenceInterval | None = None
    residual_sum_squares: float | None = None
    residuals: tuple[float, ...] | None = None
    jacobian: tuple[tuple[float, float], ...] | None = None
    covariance: tuple[tuple[float, float], tuple[float, float]] | None = None
    jacobian_rank: int | None = None
    jacobian_condition_number: float | None = None
    evaluations: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool")
        if self.converged and (self.km is None or self.vmax is None):
            raise ValueError("converged fits require both km and vmax")
        if not self.converged and any(
            value is not None
            for value in (
                self.km,
                self.vmax,
                self.km_interval,
                self.vmax_interval,
                self.covariance,
            )
        ):
            raise ValueError("failed fits must not contain estimates, intervals, or covariance")
        for name in ("km", "vmax"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value, minimum=0.0, strict_minimum=True)
        for name in ("km_interval", "vmax_interval"):
            interval = getattr(self, name)
            if interval is not None and not isinstance(interval, ConfidenceInterval):
                raise TypeError(f"{name} must be a ConfidenceInterval")
        if self.residual_sum_squares is not None:
            _require_finite("residual_sum_squares", self.residual_sum_squares, minimum=0.0)
        if self.residuals is not None:
            _require_numeric_tuple("residuals", self.residuals)
        if self.jacobian is not None:
            if not isinstance(self.jacobian, tuple) or any(
                not isinstance(row, tuple) or len(row) != 2 for row in self.jacobian
            ):
                raise TypeError("jacobian must be a tuple of two-column tuples")
            for row in self.jacobian:
                _require_numeric_tuple("jacobian row", row)
        if self.covariance is not None:
            if (
                not isinstance(self.covariance, tuple)
                or len(self.covariance) != 2
                or any(not isinstance(row, tuple) or len(row) != 2 for row in self.covariance)
            ):
                raise TypeError("covariance must be a 2 by 2 tuple")
            for row in self.covariance:
                _require_numeric_tuple("covariance row", row)
        if self.jacobian_rank is not None:
            if (
                isinstance(self.jacobian_rank, bool)
                or not isinstance(self.jacobian_rank, Integral)
                or not 0 <= self.jacobian_rank <= 2
            ):
                raise ValueError("jacobian_rank must be an integer between 0 and 2")
        if self.jacobian_condition_number is not None:
            _require_finite(
                "jacobian_condition_number",
                self.jacobian_condition_number,
                minimum=0.0,
                strict_minimum=True,
            )
        if self.evaluations is not None:
            if (
                isinstance(self.evaluations, bool)
                or not isinstance(self.evaluations, Integral)
                or self.evaluations < 1
            ):
                raise ValueError("evaluations must be a positive integer")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitResult:
        values = dict(data)
        for name in ("km_interval", "vmax_interval"):
            if values.get(name) is not None:
                values[name] = ConfidenceInterval.from_dict(values[name])
        if values.get("residuals") is not None:
            values["residuals"] = tuple(values["residuals"])
        if values.get("jacobian") is not None:
            values["jacobian"] = tuple(tuple(row) for row in values["jacobian"])
        if values.get("covariance") is not None:
            values["covariance"] = tuple(tuple(row) for row in values["covariance"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ExperimentStep:
    """One measurement and its optional updated fit."""

    measurement_number: int
    observation: Observation
    fit: FitResult | None = None
    selection_message: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.measurement_number, bool)
            or not isinstance(self.measurement_number, Integral)
            or self.measurement_number < 1
        ):
            raise ValueError("measurement_number must be a positive integer")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        if self.fit is not None and not isinstance(self.fit, FitResult):
            raise TypeError("fit must be a FitResult")
        if not isinstance(self.selection_message, str):
            raise TypeError("selection_message must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_number": int(self.measurement_number),
            "observation": self.observation.to_dict(),
            "fit": None if self.fit is None else self.fit.to_dict(),
            "selection_message": self.selection_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentStep:
        values = dict(data)
        values["observation"] = Observation.from_dict(values["observation"])
        if values.get("fit") is not None:
            values["fit"] = FitResult.from_dict(values["fit"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """The complete trace and status of one strategy replicate."""

    replicate: int
    strategy: str
    seed: int
    steps: tuple[ExperimentStep, ...]
    succeeded: bool
    message: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, Integral) or self.replicate < 0:
            raise ValueError("replicate must be a non-negative integer")
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("strategy must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**128:
            raise ValueError("seed must be an integer in the range [0, 2**128)")
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple")
        if not all(isinstance(step, ExperimentStep) for step in self.steps):
            raise TypeError("steps must contain only ExperimentStep values")
        expected_numbers = tuple(range(1, len(self.steps) + 1))
        actual_numbers = tuple(int(step.measurement_number) for step in self.steps)
        if actual_numbers != expected_numbers:
            raise ValueError("experiment step numbers must be consecutive and start at 1")
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a bool")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicate": int(self.replicate),
            "strategy": self.strategy,
            "seed": int(self.seed),
            "steps": [step.to_dict() for step in self.steps],
            "succeeded": self.succeeded,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        values = dict(data)
        values["steps"] = tuple(ExperimentStep.from_dict(step) for step in values["steps"])
        return cls(**values)


def _require_finite(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = value <= minimum if strict_minimum else value < minimum
        if invalid:
            comparison = "greater than" if strict_minimum else "greater than or equal to"
            raise ValueError(f"{name} must be {comparison} {minimum}")


def _require_numeric_tuple(name: str, values: object) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _require_finite(name, value)
