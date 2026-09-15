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

The repository currently contains the tested project skeleton. Scientific functionality will be added milestone by milestone.
