"""Deterministic log-spaced substrate-concentration design."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from enzymeopt.designs.base import ConcentrationStrategy, DesignState


class LogSpacedDesign(ConcentrationStrategy):
    """Select the next point from a budget-specific geometric grid."""

    name = "log_spaced"

    def select_next(self, state: DesignState, *, rng: Generator | None = None) -> float:
        del rng
        if state.is_complete:
            raise ValueError("cannot select from a complete design")
        design = np.geomspace(
            state.substrate_min,
            state.substrate_max,
            state.measurement_budget,
        )
        selected = np.asarray(state.selected_concentrations, dtype=np.float64)
        if selected.size and not np.allclose(selected, design[: selected.size], rtol=1.0e-12, atol=0.0):
            raise ValueError(
                "selected_concentrations must be a prefix of the budget-specific log-spaced design"
            )
        return float(design[len(state.selected_concentrations)])
