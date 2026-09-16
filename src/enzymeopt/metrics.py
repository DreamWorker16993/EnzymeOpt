"""Replicate metrics and explicitly denominated Monte Carlo summaries."""

import math

import numpy as np

from enzymeopt.results import ExperimentResult


def experiment_metrics(result: ExperimentResult, *, true_km: float, true_vmax: float,
                       measurement_budget: int) -> dict:
    """Missing/failed estimates remain None, never zero or stale partial fits."""
    if any(not math.isfinite(v) or v <= 0 for v in (true_km, true_vmax)):
        raise ValueError("parameter truth must be positive and finite")
    fit = result.steps[-1].fit if result.steps else None
    success = bool(result.succeeded and len(result.steps) == measurement_budget
                   and fit is not None and fit.converged)
    row = dict(strategy=result.strategy, replicate=result.replicate, seed=result.seed,
               measurement_budget=measurement_budget, measurement_count=len(result.steps),
               succeeded=success, message=result.message)
    for name, truth in (("km", true_km), ("vmax", true_vmax)):
        estimate = getattr(fit, name) if success else None
        interval = getattr(fit, name + "_interval") if success else None
        error = abs(estimate - truth) / truth if estimate is not None else None
        width = interval.upper - interval.lower if interval is not None else None
        relative_width = width / truth if width is not None else None
        for suffix, value in (("relative_error", error), ("ci_width", width),
                              ("relative_ci_width", relative_width)):
            row[name + "_" + suffix] = value if value is not None and math.isfinite(value) else None
        row[name + "_covered"] = (interval.lower <= truth <= interval.upper
                                   if interval is not None else None)
    return row


def _distribution(values: list) -> dict:
    available = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not available:
        return dict(count=0, mean=None, median=None, q25=None, q75=None, q95=None)
    # Divide before summing to avoid overflow from large finite outliers.
    quantiles = np.quantile(available, [.25, .5, .75, .95])
    return dict(count=len(available), mean=math.fsum(v / len(available) for v in available),
                median=float(quantiles[1]), q25=float(quantiles[0]),
                q75=float(quantiles[2]), q95=float(quantiles[3]))


def summarize_metrics(rows: list[dict]) -> tuple[dict, ...]:
    """Group by strategy/budget, sorting groups independently of execution order.

    Errors/widths describe available completed fits; counts expose exclusions.
    Coverage is conditional on available CIs. Availability and success rates use
    all attempted replicates. covered_fraction_all treats missing CIs as misses.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row['strategy'], row['measurement_budget']), []).append(row)
    summaries = []
    for (strategy, budget), group in sorted(groups.items()):
        n = len(group)
        success = sum(r['succeeded'] for r in group)
        summary = dict(strategy=strategy, measurement_budget=budget, attempts=n,
                       successes=success, failures=n-success, success_rate=success/n,
                       measurement_count=_distribution([r['measurement_count'] for r in group]))
        for name in ('km', 'vmax'):
            for metric in ('relative_error', 'ci_width', 'relative_ci_width'):
                key = name + '_' + metric
                summary[key] = _distribution([r[key] for r in group])
            covered = [r[name + '_covered'] for r in group if r[name + '_covered'] is not None]
            summary[name + '_ci_available_count'] = len(covered)
            summary[name + '_ci_availability'] = len(covered)/n
            summary[name + '_coverage'] = sum(covered)/len(covered) if covered else None
            summary[name + '_covered_fraction_all'] = sum(covered)/n
        summaries.append(summary)
    return tuple(summaries)
