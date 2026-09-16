import json

import numpy as np
import pytest

from enzymeopt.config import ExperimentConfig
from enzymeopt.designs import DOptimalDesign, DesignState
from enzymeopt.experiment import run_experiment
from enzymeopt.results import ExperimentResult, FitResult


@pytest.mark.parametrize('strategy', ExperimentConfig.SUPPORTED_STRATEGIES)
@pytest.mark.parametrize('budget', ExperimentConfig().measurement_budgets)
def test_experiment_budget_and_noiseless_recovery(strategy, budget):
    result = run_experiment(ExperimentConfig(noise_std=0), strategy=strategy,
                            measurement_budget=budget)
    assert result.succeeded, result.message
    assert len(result.steps) == budget
    assert [s.measurement_number for s in result.steps] == list(range(1, budget + 1))
    assert all(s.fit is None for s in result.steps[:2])
    assert all(s.fit.converged for s in result.steps[2:])
    assert result.steps[-1].fit.km == pytest.approx(1, rel=1e-5)
    assert result.steps[-1].fit.vmax == pytest.approx(1, rel=1e-5)
    if strategy == 'log_spaced':
        np.testing.assert_allclose([s.observation.concentration for s in result.steps],
                                    np.geomspace(.05, 20, budget))
    else:
        np.testing.assert_allclose([s.observation.concentration for s in result.steps[:3]],
                                    [.05, 1, 20])


def test_d_optimal_uses_each_previous_fit():
    config = ExperimentConfig(seed=123)
    result = run_experiment(config, strategy='d_optimal', measurement_budget=8)
    assert result.succeeded, result.message
    for i in range(3, 8):
        fit = result.steps[i-1].fit
        state = DesignState(.05, 20, 8,
            tuple(s.observation.concentration for s in result.steps[:i]),
            tuple(np.geomspace(.05, 20, 200)))
        expected = DOptimalDesign(fit.km, fit.vmax, .05).select_next(state)
        assert result.steps[i].observation.concentration == expected
        assert len(result.steps[i].fit.residuals) == i + 1
    assert ExperimentResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result


@pytest.mark.parametrize('strategy', ExperimentConfig.SUPPORTED_STRATEGIES)
def test_failed_fit_preserves_partial_trace(monkeypatch, strategy):
    monkeypatch.setattr('enzymeopt.experiment.fit_michaelis_menten',
                        lambda *args, **kwargs: FitResult(False, message='test failure'))
    result = run_experiment(ExperimentConfig(), strategy=strategy, measurement_budget=8)
    assert not result.succeeded
    assert len(result.steps) == 3
    assert not result.steps[-1].fit.converged
    assert 'measurement 3' in result.message


def test_failed_selection_preserves_trace(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('singular candidate designs')
    monkeypatch.setattr(DOptimalDesign, 'select_next', fail)
    result = run_experiment(ExperimentConfig(), strategy='d_optimal', measurement_budget=8)
    assert not result.succeeded
    assert len(result.steps) == 3
    assert 'selection failed before measurement 4' in result.message


@pytest.mark.parametrize('kwargs', [{'strategy': 'bad'}, {'measurement_budget': 5},
                                    {'replicate': -1}, {'replicate': True}])
def test_invalid_experiment_inputs(kwargs):
    args = dict(strategy='random', measurement_budget=8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(), **args)
