"""Synthetic Michaelis-Menten dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from enzymeopt.model import michaelis_menten
from enzymeopt.results import Observation


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """A synthetic dataset together with all generation metadata.

    Array fields are stored as immutable, one-dimensional ``float64`` copies so
    that later mutation of caller-owned inputs cannot change a recorded run.
    """

    concentrations: FloatArray
    expected_rates: FloatArray
    observed_rates: FloatArray
    true_km: float
    true_vmax: float
    noise_std: float

    def __post_init__(self) -> None:
        arrays = {
            "concentrations": _immutable_vector("concentrations", self.concentrations),
            "expected_rates": _immutable_vector("expected_rates", self.expected_rates),
            "observed_rates": _immutable_vector("observed_rates", self.observed_rates),
        }
        length = arrays["concentrations"].size
        if length == 0:
            raise ValueError("synthetic datasets must contain at least one measurement")
        if any(values.size != length for values in arrays.values()):
            raise ValueError("dataset arrays must have equal lengths")
        if np.any(arrays["concentrations"] < 0.0):
            raise ValueError("concentrations must be non-negative")

        true_km = _positive_finite("true_km", self.true_km)
        true_vmax = _positive_finite("true_vmax", self.true_vmax)
        noise_std = _nonnegative_finite("noise_std", self.noise_std)

        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "true_km", true_km)
        object.__setattr__(self, "true_vmax", true_vmax)
        object.__setattr__(self, "noise_std", noise_std)

    def observations(self) -> tuple[Observation, ...]:
        """Return the dataset as scalar observation records."""

        return tuple(
            Observation(
                concentration=float(concentration),
                rate=float(observed),
                expected_rate=float(expected),
            )
            for concentration, expected, observed in zip(
                self.concentrations,
                self.expected_rates,
                self.observed_rates,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation including all metadata."""

        return {
            "concentrations": self.concentrations.tolist(),
            "expected_rates": self.expected_rates.tolist(),
            "observed_rates": self.observed_rates.tolist(),
            "true_km": self.true_km,
            "true_vmax": self.true_vmax,
            "noise_std": self.noise_std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyntheticDataset:
        """Restore and validate a dataset from serialized data."""

        return cls(**data)


def simulate_dataset(
    concentrations: ArrayLike,
    *,
    true_km: Real,
    true_vmax: Real,
    noise_std: Real,
    rng: Generator | None = None,
) -> SyntheticDataset:
    """Generate one independent measurement per supplied concentration.

    Repeated concentrations represent replicate measurements. Positive noise
    requires an explicit NumPy ``Generator``. Zero noise is deterministic and
    intentionally does not consume random values from an optional generator.
    """

    concentration_values = _concentration_vector(concentrations)
    km = _positive_finite("true_km", true_km)
    vmax = _positive_finite("true_vmax", true_vmax)
    sigma = _nonnegative_finite("noise_std", noise_std)

    expected = np.asarray(
        michaelis_menten(concentration_values, km=km, vmax=vmax),
        dtype=np.float64,
    )
    if sigma == 0.0:
        observed = expected.copy()
    else:
        if not isinstance(rng, Generator):
            raise TypeError("positive noise_std requires an explicit numpy.random.Generator")
        observed = expected + rng.normal(loc=0.0, scale=sigma, size=expected.shape)

    return SyntheticDataset(
        concentrations=concentration_values,
        expected_rates=expected,
        observed_rates=observed,
        true_km=km,
        true_vmax=vmax,
        noise_std=sigma,
    )


def _concentration_vector(concentrations: ArrayLike) -> FloatArray:
    raw = np.asarray(concentrations)
    if np.issubdtype(raw.dtype, np.bool_) or raw.dtype.kind in "SU":
        raise ValueError("concentrations must be a one-dimensional numeric array")
    try:
        values = np.asarray(concentrations, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("concentrations must be a one-dimensional numeric array") from error
    if values.ndim != 1:
        raise ValueError("concentrations must be a one-dimensional numeric array")
    if values.size == 0:
        raise ValueError("concentrations must contain at least one value")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("concentrations must contain finite non-negative values")
    return values.copy()


def _immutable_vector(name: str, values: ArrayLike) -> FloatArray:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.bool_) or raw.dtype.kind in "SU":
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result.flags.writeable = False
    return result


def _positive_finite(name: str, value: Real) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _nonnegative_finite(name: str, value: Real) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _finite_scalar(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)
