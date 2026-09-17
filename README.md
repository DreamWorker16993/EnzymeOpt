# EnzymeOpt

EnzymeOpt is a Python project for comparing experimental-design strategies for Michaelis–Menten parameter estimation. Its goal is to estimate `KM` and `Vmax` with as few substrate-concentration measurements as possible.

The first release provides:

- reproducible synthetic enzyme-kinetics data with Gaussian noise;
- nonlinear least-squares estimation of `KM` and `Vmax`;
- random, log-spaced, and sequential locally D-optimal concentration designs;
- Monte Carlo comparisons using parameter error, confidence-interval quality, and measurement count.

Web interfaces and neural-network models are outside the first-release scope. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete milestone plan and mathematical assumptions.

## Requirements

- Python 3.10 or newer

Runtime dependencies are NumPy and SciPy. Test tooling is kept in the optional `dev` dependency group.

## Development installation

Create and activate a virtual environment, then install the project with its development dependencies:

```shell
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run tests

```shell
python -m pytest
```

## Run the baseline benchmark

The checked-in configuration runs 1,000 Monte Carlo replicates for every strategy
and measurement budget:

```shell
enzymeopt --config configs/baseline.toml --output outputs/baseline
```

For a quick development run, override the replicate count and seed:

```shell
enzymeopt --config configs/baseline.toml --output outputs/smoke --replicates 2 --seed 123
```

The equivalent repository wrapper is `python scripts/run_benchmark.py`. It uses
the formal baseline configuration and writes to `outputs/baseline`. A non-empty
output directory is rejected unless `--overwrite` is explicitly supplied.

Each output directory contains:

- `result.json`: lossless configuration, fit options, replicate records, and summaries;
- `records.csv`: one row per strategy, budget, and replicate;
- `summaries.csv`: flattened aggregate metrics;
- `config.json`: the effective configuration after command-line overrides;
- `metadata.json`: status, seed, schema, package/runtime versions, and record counts.

Use `enzymeopt.results_io.load_monte_carlo_result` to restore `result.json` without
losing types. CSV files are intended for inspection and analysis. Generated outputs
are ignored by Git. The seed alone is insufficient to identify a run: retain the
effective configuration, fit options, package version, strategy, budget, and
replicate index recorded in the output.

## Metrics and interpretation

Relative error is `abs(estimate - truth) / truth`. CI width is the full upper-minus-
lower width; relative CI width divides it by the true parameter. Error and width
distributions report count, mean, median, quartiles, and 95th percentile over
available completed fits. Failure and availability counts expose excluded values.
Coverage is reported conditional on an available CI, alongside CI availability and
the fraction covered over all attempts. Measurement count includes partial failed runs.

See [MILESTONE9_REPORT.md](MILESTONE9_REPORT.md) for the initial 50-replicate
comparison and [MILESTONE10_REPORT.md](MILESTONE10_REPORT.md) for the completed
1,000-replicate formal baseline. The baseline output can be reproduced locally.

## Run a sequential experiment

```python
from enzymeopt.config import ExperimentConfig
from enzymeopt.experiment import run_experiment

config = ExperimentConfig(seed=42)
result = run_experiment(config, strategy="d_optimal", measurement_budget=8)
for step in result.steps:
    if step.fit is not None:
        print(step.measurement_number, step.observation.concentration,
              step.fit.km, step.fit.vmax, step.fit.message)
print(result.succeeded, result.message)
```

Random and D-optimal start at the lower bound, geometric midpoint, and upper
bound. Log-spaced uses its full budget-specific grid. Fitting starts after
measurement 3 and uses all observations after every subsequent measurement.
D-optimal refreshes its parameter estimates from the immediately preceding fit.
Simulation truth is not passed to the fitter or selection strategy.

A failed fit stops any strategy with its observations and failed fit retained.
A failed selection stops before the next measurement. A completed run with a
converged final fit is successful even if its confidence intervals are unavailable;
inspect the fit diagnostics separately. Each step records its selection method.

The result's seed is the master seed. Reproduce a run using the original config,
strategy, replicate index, budget, and fit options. Noise streams are paired across
strategies for a given replicate and budget; random selection has a separate stream.
Budgets have independent streams, so separate budget runs are not trajectory prefixes.
The runner does not aggregate replicates or apply a precision-based stopping rule.

## Use real enzyme measurements interactively

Use this command when you have measured initial rates from a real enzyme assay.
Choose the concentration range that is feasible for your assay; all entered concentrations
must be inside that range. The rate values must use one consistent unit.

If no name is supplied, EnzymeOpt creates a separate folder using the current date
and the next available experiment number, such as
`outputs/2026-09-17-experiment-001`.

```powershell
enzymeopt interactive --min-concentration 0.05 --max-concentration 20
```

You can instead give the experiment a memorable name. EnzymeOpt saves it under
`outputs/NAME` and checks that the name has not already been used before asking for
measurements.

```powershell
enzymeopt interactive --name real-enzyme-01 --min-concentration 0.05 --max-concentration 20
```

Names may contain letters, numbers, hyphens, and underscores. To store results
somewhere else, use `--output PATH` instead of `--name`. Reusing a non-empty result
folder requires an explicit `--overwrite`.

Enter each measured pair as `concentration rate`, for example `0.1 0.074`.
After the third pair, EnzymeOpt refits the Michaelis–Menten model and prints a locally
D-optimal next concentration. Measure at that concentration, enter the new pair, and
repeat. Enter `report` when you want to stop: it prints the `Vmax` and `KM` estimates
and writes `report.md`, `report.json`, `raw_data.csv`, `fit_curve.csv`, and
`fit_curve.png` to the output directory. The plot overlays your raw measurements with
the fitted curve. The report includes Vmax and KM point estimates, local standard
errors, 95% confidence intervals, and confidence-interval widths. For real data,
these uncertainty values describe estimation error; relative error needs an external
reference value. Enter `help` for a reminder or `quit` to end without a report.

## Mathematical and simulation assumptions

Test runs allocate a unique `.pytest-run-*` directory in the project instead of
reusing the system `pytest-of-*` directory. This avoids Windows permission errors
when tests are run under different accounts or execution environments. An explicit
`--basetemp` still takes precedence. Pytest's shared cache is disabled, so
last-failed/cache-based reruns are unavailable. Scratch directories are Git-ignored
and retained for inspection; they can be removed after their test runs finish.

The response model is `v(S) = Vmax*S/(KM+S) + epsilon`, with independent additive,
homoscedastic Gaussian noise. Negative simulated rates are retained. Parameters are
fit by nonlinear least squares in log-parameter space. Reported 95% intervals use a
local Jacobian approximation and Student-t critical value; they may be unreliable for
very small, rank-deficient, ill-conditioned, or boundary fits, in which case the
software records the diagnostic and omits the interval. Random sampling is uniform
in log concentration. D-optimal selection is sequential and local to the current
parameter estimate and finite candidate grid.

## Local D-optimal selection

For independent Gaussian noise with standard deviation `sigma`, the information
matrix in natural `(KM, Vmax)` coordinates is `sum(g(S) g(S).T) / sigma**2`,
where `g` is the model's parameter gradient. `DOptimalDesign` scores each candidate
using the log determinant of the accumulated information after adding that point.
Larger determinants correspond to smaller approximate joint uncertainty regions.
This is a greedy, local criterion conditional on the supplied parameter estimate.

```python
import numpy as np
from enzymeopt.designs import DesignState, DOptimalDesign

state = DesignState(
    substrate_min=0.05,
    substrate_max=20.0,
    measurement_budget=8,
    selected_concentrations=(0.05, 1.0, 20.0),
    candidate_concentrations=tuple(np.geomspace(0.05, 20.0, 200)),
)
# Supply current fitted estimates, not simulation truth.
strategy = DOptimalDesign(km=1.2, vmax=0.9, noise_std=0.05)
next_concentration = strategy.select_next(state)
scores = strategy.score_candidates(state)
```

After measuring and refitting, construct a new strategy with the updated estimates.
The selection class does not itself simulate observations or fit parameters;
`generate_design` therefore uses a fixed estimate throughout its loop. Candidates
must be provided explicitly. Repeated concentrations are allowed, exact ties choose
the lowest concentration, and no random numbers are consumed.

Singular or numerically unresolved information gives a score of negative infinity.
If every augmented design is unresolved, selection raises a diagnostic error rather
than adding an undocumented prior or regularization. The log determinant is computed
from scaled eigenvalues to avoid determinant overflow. For noise-free simulations,
use `noise_std=1` for design ranking: zero-variance Fisher information is undefined,
but a shared positive variance does not change candidate ordering.
