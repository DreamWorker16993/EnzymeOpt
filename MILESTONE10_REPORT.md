# Formal baseline benchmark

Completed: 1,000 replicates per strategy and budget, 21,000 total experiments.
Configuration: `configs/baseline.toml`, seed 20260916, KM=Vmax=1,
Gaussian sigma=0.05, range [0.05,20], 200 D-optimal candidates.
Runtime: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, EnzymeOpt 0.1.0.

Each cell below reports median absolute relative error: KM / Vmax.

| Measurements | Random | Log-spaced | D-optimal |
|---|---|---|---|
| 3 | 15.65% / 3.75% | 15.65% / 3.75% | 15.65% / 3.75% |
| 4 | 14.24% / 3.76% | 17.15% / 3.88% | 13.22% / 3.02% |
| 6 | 12.59% / 3.42% | 12.20% / 3.02% | 11.03% / 2.44% |
| 8 | 10.87% / 2.82% | 11.37% / 3.15% | 9.11% / 2.02% |
| 12 | 9.28% / 2.45% | 9.43% / 2.42% | 7.16% / 1.69% |
| 16 | 7.92% / 2.15% | 8.19% / 2.22% | 5.69% / 1.39% |
| 24 | 6.35% / 1.74% | 6.43% / 1.75% | 4.76% / 1.17% |

All 21 groups had 1,000 completed runs and zero recorded failures. These are
descriptive baseline results, not proof of superiority for other parameter/noise
settings. Random and D-optimal share three initial exploration points; log-spaced
uses the full budget-specific grid. At budget 3 the designs and paired noise agree.

The local output directory `outputs/baseline` contains `result.json`,
`records.csv`, `summaries.csv`, `config.json`, and `metadata.json`. Full summaries
include error distributions, CI widths, availability, and coverage. Outputs are
ignored by Git; this report and the configuration are tracked.

Validation: result reloaded successfully; 21,000 unique strategy/budget/replicate
keys; every group has 1,000 attempts; saved summaries equal summaries recomputed
from all records; CSV counts match JSON counts. The effective configuration equals
the checked-in baseline. Fixed-seed end-to-end byte reproducibility is covered by
the smaller test run rather than repeating the full formal benchmark.

Run with `enzymeopt --config configs/baseline.toml --output outputs/new-baseline`.
