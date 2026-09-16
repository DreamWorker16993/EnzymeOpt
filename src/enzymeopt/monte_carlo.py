"""Reproducible multi-strategy, multi-budget experiments."""

from dataclasses import dataclass

from enzymeopt.config import ExperimentConfig
from enzymeopt.experiment import run_experiment
from enzymeopt.fitting import FitOptions
from enzymeopt.metrics import experiment_metrics, summarize_metrics


@dataclass(frozen=True)
class MonteCarloResult:
    """Configuration, per-replicate metrics, and group summaries.

    Traces remain available from run_experiment; this aggregate keeps scalar
    records, including failed runs, to limit memory consumption.
    """

    config: ExperimentConfig
    fit_options: FitOptions
    records: tuple[dict, ...]
    summaries: tuple[dict, ...]


def run_monte_carlo(config: ExperimentConfig, *, fit_options: FitOptions | None = None,
                    progress=None) -> MonteCarloResult:
    """Run all configured combinations; optionally call progress(done, total).

    Seeds are derived by the experiment runner. Programming errors propagate;
    structured fitting/selection failures remain explicit records.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError('config must be an ExperimentConfig')
    options = FitOptions() if fit_options is None else fit_options
    if not isinstance(options, FitOptions):
        raise TypeError('fit_options must be FitOptions')
    records = []
    total = len(config.strategies) * len(config.measurement_budgets) * config.replicates
    for budget in config.measurement_budgets:
        for replicate in range(config.replicates):
            for strategy in config.strategies:
                result = run_experiment(config, strategy=strategy, measurement_budget=budget,
                                        replicate=replicate, fit_options=options)
                records.append(experiment_metrics(result, true_km=config.true_km,
                               true_vmax=config.true_vmax, measurement_budget=budget))
                if progress is not None:
                    progress(len(records), total)
    records.sort(key=lambda r: (r['strategy'], r['measurement_budget'], r['replicate']))
    return MonteCarloResult(config, options, tuple(records), summarize_metrics(records))
