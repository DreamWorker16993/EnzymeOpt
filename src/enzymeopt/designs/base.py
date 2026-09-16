"""Shared interface and validation for concentration design strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DesignState:
    """Validated state presented to a strategy before its next selection."""

    substrate_min: float
    substrate_max: float
    measurement_budget: int
    selected_concentrations: tuple[float, ...] = ()
    candidate_concentrations: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        lower = _positive_finite("substrate_min", self.substrate_min)
        upper = _positive_finite("substrate_max", self.substrate_max)
        if lower >= upper:
            raise ValueError("substrate_min must be less than substrate_max")
        if (
            isinstance(self.measurement_budget, bool)
            or not isinstance(self.measurement_budget, Integral)
            or self.measurement_budget < 2
        ):
            raise ValueError("measurement_budget must be an integer greater than or equal to 2")
        selected = _concentration_tuple(
            "selected_concentrations",
            self.selected_concentrations,
            lower,
            upper,
        )
        if len(selected) > self.measurement_budget:
            raise ValueError("selected_concentrations cannot exceed measurement_budget")
        candidates = _concentration_tuple(
            "candidate_concentrations",
            self.candidate_concentrations,
            lower,
            upper,
        )

        object.__setattr__(self, "substrate_min", lower)
        object.__setattr__(self, "substrate_max", upper)
        object.__setattr__(self, "measurement_budget", int(self.measurement_budget))
        object.__setattr__(self, "selected_concentrations", selected)
        object.__setattr__(self, "candidate_concentrations", candidates)

    @property
    def is_complete(self) -> bool:
        return len(self.selected_concentrations) == self.measurement_budget

    @property
    def next_measurement_number(self) -> int:
        if self.is_complete:
            raise ValueError("the design already contains its full measurement budget")
        return len(self.selected_concentrations) + 1

    def add(self, concentration: Real) -> DesignState:
        """Return a new state with one validated concentration appended."""

        if self.is_complete:
            raise ValueError("cannot add a concentration to a complete design")
        selected = self.selected_concentrations + (
            _positive_finite("concentration", concentration),
        )
        return replace(self, selected_concentrations=selected)


class ConcentrationStrategy(ABC):
    """Interface implemented by static and adaptive concentration strategies."""

    name: str

    @abstractmethod
    def select_next(self, state: DesignState, *, rng: Generator | None = None) -> float:
        """Select one concentration for the next measurement."""


def generate_design(
    strategy: ConcentrationStrategy,
    *,
    substrate_min: Real,
    substrate_max: Real,
    measurement_budget: int,
    selected_concentrations: ArrayLike = (),
    candidate_concentrations: ArrayLike = (),
    rng: Generator | None = None,
) -> FloatArray:
    """Run a strategy until its requested measurement budget is complete."""

    if not isinstance(strategy, ConcentrationStrategy):
        raise TypeError("strategy must implement ConcentrationStrategy")
    state = DesignState(
        substrate_min=substrate_min,
        substrate_max=substrate_max,
        measurement_budget=measurement_budget,
        selected_concentrations=_array_to_tuple(
            "selected_concentrations",
            selected_concentrations,
        ),
        candidate_concentrations=_array_to_tuple(
            "candidate_concentrations",
            candidate_concentrations,
        ),
    )
    while not state.is_complete:
        state = state.add(strategy.select_next(state, rng=rng))

    design = np.asarray(state.selected_concentrations, dtype=np.float64)
    design.flags.writeable = False
    return design


def _array_to_tuple(name: str, values: ArrayLike) -> tuple[float, ...]:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.bool_) or raw.dtype.kind in "SU":
        raise ValueError(f"{name} must be a one-dimensional numeric collection")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric collection") from error
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional collection")
    return tuple(float(value) for value in array)


def _concentration_tuple(
    name: str,
    values: tuple[float, ...],
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    validated = tuple(_positive_finite(name, value) for value in values)
    if any(value < lower or value > upper for value in validated):
        raise ValueError(f"{name} must stay within the substrate range")
    return validated


def _positive_finite(name: str, value: Real) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)
