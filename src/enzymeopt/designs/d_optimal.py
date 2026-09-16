"""One-step locally D-optimal selection at a supplied parameter estimate."""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from enzymeopt.designs.base import ConcentrationStrategy, DesignState
from enzymeopt.information import fisher_information, information_logdet


@dataclass(frozen=True)
class DOptimalDesign(ConcentrationStrategy):
    """Score candidates using the current estimate, without knowing true values.

    Construct a new instance after each refit. Measurement and refitting belong
    to the experiment runner; generate_design alone keeps this estimate fixed.
    Candidate concentrations are required explicitly. Exact ties choose the
    lowest concentration, independently of candidate order. Repeats are allowed.
    """

    km: float
    vmax: float
    noise_std: float = 1.0
    name = "d_optimal"

    def __post_init__(self) -> None:
        fisher_information([], km=self.km, vmax=self.vmax, noise_std=self.noise_std)

    def current_information(self, state: DesignState) -> np.ndarray:
        return fisher_information(state.selected_concentrations, km=self.km,
                                  vmax=self.vmax, noise_std=self.noise_std)

    def score_candidates(self, state: DesignState) -> tuple[float, ...]:
        """Return augmented log determinants in the original candidate order.

        -inf marks singular or numerically unresolved augmented designs. A
        singular current design is acceptable if an added point resolves it.
        """
        if state.is_complete:
            raise ValueError("cannot select from a complete design")
        if not state.candidate_concentrations:
            raise ValueError("D-optimal selection requires candidate concentrations")
        current = self.current_information(state)
        scores = []
        for candidate in state.candidate_concentrations:
            added = fisher_information([candidate], km=self.km, vmax=self.vmax,
                                       noise_std=self.noise_std)
            scores.append(information_logdet(current + added))
        return tuple(scores)

    def select_next(self, state: DesignState, *, rng: Generator | None = None) -> float:
        scores = self.score_candidates(state)
        best = max(scores)
        if not np.isfinite(best):
            raise ValueError("all augmented designs are singular or ill-conditioned; "
                             "provide informative initial measurements or revise the range")
        return min(candidate for candidate, score in zip(state.candidate_concentrations, scores)
                   if score == best)
