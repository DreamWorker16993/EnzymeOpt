import pytest

from enzymeopt.metrics import experiment_metrics, summarize_metrics
from enzymeopt.results import ExperimentResult, ExperimentStep, FitResult, Observation, ConfidenceInterval


def record(estimate=1.2, success=True, interval=True):
    fit = FitResult(True, km=estimate, vmax=2,
                    km_interval=ConfidenceInterval(.8, 1.4) if interval else None)
    result = ExperimentResult(0, 'random', 0,
                              tuple(ExperimentStep(i, Observation(1, 1), fit) for i in range(1, 4)),
                              success)
    return experiment_metrics(result, true_km=1, true_vmax=2, measurement_budget=3)


def test_hand_calculated_metrics():
    row = record()
    assert row['km_relative_error'] == pytest.approx(.2)
    assert row['vmax_relative_error'] == 0
    assert row['km_ci_width'] == pytest.approx(.6)
    assert row['km_relative_ci_width'] == pytest.approx(.6)
    assert row['km_covered'] is True
    assert row['vmax_covered'] is None


def test_failed_runs_do_not_reuse_partial_fits():
    row = record(success=False)
    assert row['km_relative_error'] is None
    assert row['km_ci_width'] is None
    assert row['km_covered'] is None


def test_summary_denominators_and_quantiles():
    summary, = summarize_metrics([record(1), record(2, interval=False), record(success=False)])
    assert summary['success_rate'] == pytest.approx(2/3)
    assert summary['km_relative_error']['mean'] == .5
    assert summary['km_relative_error']['median'] == .5
    assert summary['km_relative_error']['q25'] == .25
    assert summary['km_relative_error']['q75'] == .75
    assert summary['km_relative_error']['q95'] == .95
    assert summary['km_relative_error']['count'] == 2
    assert summary['km_coverage'] == 1
    assert summary['km_ci_availability'] == pytest.approx(1/3)
    assert summary['km_covered_fraction_all'] == pytest.approx(1/3)


def test_all_failed_group_and_empty_input():
    summary, = summarize_metrics([record(success=False)])
    assert summary['km_relative_error']['mean'] is None
    assert summary['km_coverage'] is None
    assert summary['success_rate'] == 0
    assert summarize_metrics([]) == ()
