# EnzymeOpt

EnzymeOpt is a Python project for comparing experimental-design strategies for Michaelis–Menten parameter estimation. Its goal is to estimate `KM` and `Vmax` with as few substrate-concentration measurements as possible.

The first release will provide:

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

## Current status

Milestones 1–8 provide configuration, reproducible simulation, nonlinear fitting,
random, log-spaced, and local D-optimal selection, and a sequential experiment
runner. Monte Carlo aggregation is still planned.

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
