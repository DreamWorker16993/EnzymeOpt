from dataclasses import replace

from enzymeopt.config import ExperimentConfig
from enzymeopt.monte_carlo import run_monte_carlo
from enzymeopt.results import ExperimentResult


def test_small_monte_carlo_is_reproducible_and_order_independent():
    config = ExperimentConfig(replicates=2, measurement_budgets=(3, 4), noise_std=0)
    ticks = []
    first = run_monte_carlo(config, progress=lambda done, total: ticks.append((done, total)))
    second = run_monte_carlo(config)
    reverse = run_monte_carlo(replace(config, strategies=tuple(reversed(config.strategies))))
    assert first == second
    assert first.records == reverse.records
    assert first.summaries == reverse.summaries
    assert len(first.records) == 12
    assert len(first.summaries) == 6
    assert ticks[-1] == (12, 12)
    assert all(r['measurement_count'] == r['measurement_budget'] for r in first.records)
    assert all(r['succeeded'] for r in first.records)


def test_failed_experiments_are_retained(monkeypatch):
    monkeypatch.setattr('enzymeopt.monte_carlo.run_experiment',
        lambda config, **kw: ExperimentResult(kw['replicate'], kw['strategy'], config.seed,
                                               (), False, 'injected failure'))
    result = run_monte_carlo(ExperimentConfig(replicates=2, measurement_budgets=(3,)))
    assert len(result.records) == 6
    assert all(r['km_relative_error'] is None for r in result.records)
    assert all(s['failures'] == 2 for s in result.summaries)
