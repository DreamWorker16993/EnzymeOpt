# Milestone 9 development comparison

KM=Vmax=1; Gaussian noise sigma=0.05; range [0.05,20]; 200 candidates;
seed=20260916; 50 replicates per strategy/budget; 450 experiments total.
Random and D-optimal share initial points (0.05,1,20). Log-spaced uses a full
budget-specific grid. Noise is paired across strategies. Budgets have independent
streams. Random here means random selection after three exploration measurements.

Errors are median absolute relative errors. Widths are median full 95% CI widths
divided by truth, not half widths.

| Budget | Strategy | KM error | Vmax error | KM CI width | Vmax CI width | KM coverage | Vmax coverage |
|---|---|---|---|---|---|---|---|
| 4 | random | 14.55% | 3.50% | 161.72% | 38.01% | 92% | 98% |
| 4 | log-spaced | 17.56% | 4.29% | 159.02% | 37.28% | 92% | 92% |
| 4 | D-optimal | 15.00% | 2.93% | 155.99% | 31.20% | 94% | 96% |
| 8 | random | 11.44% | 2.52% | 71.94% | 20.42% | 94% | 96% |
| 8 | log-spaced | 11.20% | 3.66% | 74.76% | 19.16% | 98% | 90% |
| 8 | D-optimal | 8.24% | 2.63% | 59.42% | 13.89% | 94% | 90% |
| 16 | random | 6.75% | 2.00% | 47.08% | 13.03% | 96% | 96% |
| 16 | log-spaced | 9.33% | 1.93% | 49.14% | 12.77% | 100% | 94% |
| 16 | D-optimal | 4.56% | 1.30% | 37.07% | 8.62% | 98% | 96% |

Every group completed its budget with 100% success and CI availability. D-optimal
had smaller median CI widths, but random had slightly smaller KM error at n=4
and Vmax error at n=8. Fifty replicates in one scenario do not establish universal
superiority or statistical significance. CI coverage is imprecisely estimated.
Formal >=1,000-replicate benchmarking and file export remain Milestone 10 work.

## Reproduction

```python
from enzymeopt.config import ExperimentConfig
from enzymeopt.monte_carlo import run_monte_carlo
result = run_monte_carlo(ExperimentConfig(
    seed=20260916, replicates=50, measurement_budgets=(4, 8, 16)))
print(result.summaries)
```

Records retain failures and their reasons. Error/width distributions use available
completed fits and report their count, mean, median, quartiles and 95th percentile.
Missing values remain None. Coverage uses available intervals; availability and
success rates use all attempts. covered_fraction_all counts missing CIs as misses.
Measurement counts include partial failed runs. Tests verify hand calculations,
failure retention and order independence, not a predetermined winning strategy.
