"""Validated configuration for EnzymeOpt experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete, serializable inputs for a first-release experiment."""

    SUPPORTED_STRATEGIES: ClassVar[tuple[str, ...]] = (
        "random",
        "log_spaced",
        "d_optimal",
    )
    MINIMUM_BUDGET: ClassVar[int] = 3

    true_km: float = 1.0
    true_vmax: float = 1.0
    noise_std: float = 0.05
    substrate_min: float = 0.05
    substrate_max: float = 20.0
    candidate_count: int = 200
    measurement_budgets: tuple[int, ...] = (3, 4, 6, 8, 12, 16, 24)
    replicates: int = 1_000
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("true_km", "true_vmax", "substrate_min", "substrate_max"):
            _require_positive_finite(name, getattr(self, name))
        _require_nonnegative_finite("noise_std", self.noise_std)

        if self.substrate_min >= self.substrate_max:
            raise ValueError("substrate_min must be less than substrate_max")
        _require_integer_at_least("candidate_count", self.candidate_count, 2)
        _require_integer_at_least("replicates", self.replicates, 1)
        _require_seed(self.seed)

        budgets = _require_tuple("measurement_budgets", self.measurement_budgets)
        if not budgets:
            raise ValueError("measurement_budgets must not be empty")
        for budget in budgets:
            _require_integer_at_least("measurement budget", budget, self.MINIMUM_BUDGET)
        if tuple(sorted(set(budgets))) != budgets:
            raise ValueError("measurement_budgets must be strictly increasing and unique")

        strategies = _require_tuple("strategies", self.strategies)
        if not strategies:
            raise ValueError("strategies must not be empty")
        unknown = tuple(strategy for strategy in strategies if strategy not in self.SUPPORTED_STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {unknown}")
        if len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this configuration."""

        data = asdict(self)
        data["measurement_budgets"] = list(self.measurement_budgets)
        data["strategies"] = list(self.strategies)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Restore and validate a configuration from serialized data."""

        values = dict(data)
        if "measurement_budgets" in values:
            values["measurement_budgets"] = tuple(values["measurement_budgets"])
        if "strategies" in values:
            values["strategies"] = tuple(values["strategies"])
        return cls(**values)


def _require_tuple(name: str, value: object) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    return value


def _require_positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _require_nonnegative_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _require_integer_at_least(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _require_seed(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0 or value >= 2**128:
        raise ValueError("seed must be an integer in the range [0, 2**128)")
