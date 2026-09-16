"""Sequential measure, refit, and select loop for one synthetic replicate."""

from numbers import Integral

import numpy as np

from enzymeopt.config import ExperimentConfig
from enzymeopt.designs import DOptimalDesign, DesignState, LogSpacedDesign, RandomDesign
from enzymeopt.fitting import FitOptions, fit_michaelis_menten
from enzymeopt.randomness import SeedManager
from enzymeopt.results import ExperimentResult, ExperimentStep
from enzymeopt.simulation import simulate_dataset


def run_experiment(config: ExperimentConfig, *, strategy: str, measurement_budget: int,
                   replicate: int = 0, fit_options: FitOptions | None = None) -> ExperimentResult:
    """Run a configured budget, retaining every measurement and fit.

    Random and adaptive designs share three exploration measurements. Log-spaced
    uses its budget-specific full grid. Fits start after three measurements;
    any failed fit stops the run for all strategies. Selection failures stop
    before taking another measurement. Success means budget reached and final
    fit converged, not that confidence intervals are necessarily available.

    Noise is paired across strategies within a replicate and budget. Selection
    has its own strategy-labelled stream. Truth enters only the simulator.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if strategy not in config.strategies:
        raise ValueError("strategy must be enabled in config")
    if (isinstance(measurement_budget, bool) or not isinstance(measurement_budget, Integral)
            or measurement_budget not in config.measurement_budgets):
        raise ValueError("measurement_budget must be configured")
    if (isinstance(replicate, bool) or not isinstance(replicate, Integral)
            or not 0 <= replicate < config.replicates):
        raise ValueError("replicate must be within the configured replicate count")
    if fit_options is not None and not isinstance(fit_options, FitOptions):
        raise TypeError("fit_options must be FitOptions")
    seeds = SeedManager(config.seed)
    labels = ("experiment-v1", "replicate", int(replicate), "budget", int(measurement_budget))
    noise = seeds.rng_for(*labels, "noise")
    selection = seeds.rng_for(*labels, strategy, "selection")
    initial = np.geomspace(config.substrate_min, config.substrate_max, 3)
    candidates = tuple(np.geomspace(config.substrate_min, config.substrate_max,
                                   config.candidate_count))
    steps = []
    concentrations = []
    rates = []

    def finish(succeeded: bool, message: str) -> ExperimentResult:
        return ExperimentResult(replicate=int(replicate), strategy=strategy,
                                seed=int(config.seed), steps=tuple(steps),
                                succeeded=succeeded, message=message)

    for index in range(measurement_budget):
        state = DesignState(config.substrate_min, config.substrate_max,
                            measurement_budget, tuple(concentrations), candidates)
        diagnostic = "initial exploration"
        try:
            if strategy == "log_spaced":
                concentration = LogSpacedDesign().select_next(state)
                diagnostic = "budget-specific log-spaced grid"
            elif index < 3:
                concentration = float(initial[index])
            elif strategy == "random":
                concentration = RandomDesign().select_next(state, rng=selection)
                diagnostic = "log-uniform random selection"
            else:
                previous = steps[-1].fit
                design = DOptimalDesign(previous.km, previous.vmax,
                                        noise_std=config.noise_std or 1.0)
                concentration = design.select_next(state)
                diagnostic = "local D-optimal selection from previous fit"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            return finish(False, f"selection failed before measurement {index + 1}: {error}")

        observation = simulate_dataset(
            [concentration], true_km=config.true_km, true_vmax=config.true_vmax,
            noise_std=config.noise_std, rng=noise,
        ).observations()[0]
        concentrations.append(concentration)
        rates.append(observation.rate)
        fit = None
        if index >= 2:
            fit = fit_michaelis_menten(concentrations, rates, options=fit_options)
        steps.append(ExperimentStep(index + 1, observation, fit,
                                    selection_message=diagnostic))
        if fit is not None and not fit.converged:
            return finish(False, f"fit failed at measurement {index + 1}: {fit.message}")
    return finish(True, "measurement budget completed; final fit converged")
