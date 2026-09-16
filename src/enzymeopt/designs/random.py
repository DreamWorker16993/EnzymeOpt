"""Random substrate-concentration design."""

from __future__ import annotations

from math import exp, log

from numpy.random import Generator

from enzymeopt.designs.base import ConcentrationStrategy, DesignState


class RandomDesign(ConcentrationStrategy):
    """Sample independently and uniformly in log-concentration space."""

    name = "random"

    def select_next(self, state: DesignState, *, rng: Generator | None = None) -> float:
        if state.is_complete:
            raise ValueError("cannot select from a complete design")
        if not isinstance(rng, Generator):
            raise TypeError("RandomDesign requires an explicit numpy.random.Generator")
        log_concentration = rng.uniform(log(state.substrate_min), log(state.substrate_max))
        return exp(float(log_concentration))
