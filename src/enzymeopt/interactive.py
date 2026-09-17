"""Interactive fitting and adaptive concentration selection for real experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Callable

import numpy as np

from enzymeopt.designs import DOptimalDesign, DesignState
from enzymeopt.fitting import fit_michaelis_menten
from enzymeopt.model import michaelis_menten
from enzymeopt.results import FitResult


@dataclass(frozen=True, slots=True)
class InteractiveOptions:
    """Bounds and output settings for a real-data interactive session."""

    substrate_min: float = 0.05
    substrate_max: float = 20.0
    measurement_budget: int = 24
    noise_std: float = 1.0
    candidate_count: int = 200
    output: Path = Path("outputs/interactive-report")
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("substrate_min", "substrate_max", "noise_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.substrate_min >= self.substrate_max:
            raise ValueError("substrate_min must be less than substrate_max")
        for name, minimum in (("measurement_budget", 3), ("candidate_count", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
        if not isinstance(self.output, Path):
            raise TypeError("output must be a Path")


@dataclass(frozen=True, slots=True)
class InteractiveUpdate:
    """Fit status and optional next concentration after one entered measurement."""

    fit: FitResult | None
    suggestion: float | None
    message: str


class InteractiveSession:
    """Accumulate real observations, refit, and make local D-optimal suggestions."""

    def __init__(self, options: InteractiveOptions | None = None) -> None:
        self.options = InteractiveOptions() if options is None else options
        if not isinstance(self.options, InteractiveOptions):
            raise TypeError("options must be an InteractiveOptions instance")
        self._concentrations: list[float] = []
        self._rates: list[float] = []
        self._fit: FitResult | None = None

    @property
    def concentrations(self) -> tuple[float, ...]:
        return tuple(self._concentrations)

    @property
    def rates(self) -> tuple[float, ...]:
        return tuple(self._rates)

    @property
    def fit(self) -> FitResult | None:
        return self._fit

    def add_measurement(self, concentration: Real, rate: Real) -> InteractiveUpdate:
        """Record one measured pair, refit it, and propose the next concentration."""

        concentration_value = _finite_number("concentration", concentration)
        rate_value = _finite_number("rate", rate)
        if not self.options.substrate_min <= concentration_value <= self.options.substrate_max:
            raise ValueError(
                f"concentration must be within [{self.options.substrate_min:g}, {self.options.substrate_max:g}]"
            )
        if len(self._concentrations) >= self.options.measurement_budget:
            raise ValueError("measurement budget reached; enter report to generate the results")
        self._concentrations.append(concentration_value)
        self._rates.append(rate_value)
        if len(self._concentrations) < 3:
            return InteractiveUpdate(None, None, "Recorded. Enter at least 3 measurements before requesting a recommendation.")

        self._fit = fit_michaelis_menten(self._concentrations, self._rates)
        if not self._fit.converged:
            return InteractiveUpdate(self._fit, None, f"Fit did not converge: {self._fit.message}")
        if len(self._concentrations) == self.options.measurement_budget:
            return InteractiveUpdate(self._fit, None, "Measurement budget reached. Enter report to generate the results.")
        try:
            suggestion = self._suggest_next()
        except ValueError as error:
            return InteractiveUpdate(self._fit, None, f"Fit updated, but no recommendation is available: {error}")
        return InteractiveUpdate(self._fit, suggestion, "Fit updated.")

    def write_report(self) -> Path:
        """Write report JSON/Markdown, entered measurements, and a fitted-curve PNG."""

        if self._fit is None or not self._fit.converged:
            raise ValueError("at least 3 measurements with a converged fit are required to create a report")
        destination = self.options.output
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(f"output is not a directory: {destination}")
        if destination.exists() and any(destination.iterdir()) and not self.options.overwrite:
            raise FileExistsError(f"output directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)

        curve_x = np.geomspace(self.options.substrate_min, self.options.substrate_max, 300)
        curve_y = np.asarray(michaelis_menten(curve_x, km=self._fit.km, vmax=self._fit.vmax), dtype=float)
        _write_raw_data(destination / "raw_data.csv", self._concentrations, self._rates)
        _write_curve(destination / "fit_curve.csv", curve_x, curve_y)
        _write_plot(destination / "fit_curve.png", self._concentrations, self._rates, curve_x, curve_y)
        report = {
            "measurement_count": len(self._concentrations),
            "fit": self._fit.to_dict(),
            "concentration_range": [self.options.substrate_min, self.options.substrate_max],
            "files": {"raw_data": "raw_data.csv", "fit_curve": "fit_curve.csv", "plot": "fit_curve.png"},
        }
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (destination / "report.md").write_text(_markdown_report(report, self._fit), encoding="utf-8")
        return destination

    def _suggest_next(self) -> float:
        assert self._fit is not None and self._fit.km is not None and self._fit.vmax is not None
        state = DesignState(
            substrate_min=self.options.substrate_min,
            substrate_max=self.options.substrate_max,
            measurement_budget=self.options.measurement_budget,
            selected_concentrations=tuple(self._concentrations),
            candidate_concentrations=tuple(np.geomspace(self.options.substrate_min, self.options.substrate_max, self.options.candidate_count)),
        )
        return DOptimalDesign(self._fit.km, self._fit.vmax, self.options.noise_std).select_next(state)


def run_interactive_session(
    options: InteractiveOptions,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run a terminal dialogue for real measurements; ``report`` ends it."""

    session = InteractiveSession(options)
    output_fn("Enter `concentration rate` pairs. Commands: report, help, quit.")
    output_fn(f"Allowed concentration range: {options.substrate_min:g} to {options.substrate_max:g}.")
    while True:
        try:
            line = input_fn("measurement> ").strip()
        except EOFError:
            output_fn("Session ended without writing a report.")
            return 0
        command = line.lower()
        if command == "quit":
            output_fn("Session ended without writing a report.")
            return 0
        if command == "help":
            output_fn("Enter two finite numbers separated by spaces, for example: 1.0 0.52. Enter report when finished.")
            continue
        if command == "report":
            try:
                destination = session.write_report()
            except (ValueError, FileExistsError, NotADirectoryError) as error:
                output_fn(f"Cannot write report: {error}")
                continue
            fit = session.fit
            assert fit is not None and fit.km is not None and fit.vmax is not None
            output_fn(f"Vmax estimate: {fit.vmax:.6g}")
            output_fn(f"KM estimate: {fit.km:.6g}")
            output_fn(f"Report written to: {destination.resolve()}")
            return 0
        try:
            concentration, rate = _parse_measurement(line)
            update = session.add_measurement(concentration, rate)
        except ValueError as error:
            output_fn(f"Invalid input: {error}")
            continue
        output_fn(update.message)
        if update.suggestion is not None:
            output_fn(f"Suggested next concentration: {update.suggestion:.6g}")


def _parse_measurement(line: str) -> tuple[float, float]:
    fields = line.split()
    if len(fields) != 2:
        raise ValueError("enter exactly two values: concentration rate")
    try:
        return float(fields[0]), float(fields[1])
    except ValueError as error:
        raise ValueError("concentration and rate must be numbers") from error


def _finite_number(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _write_raw_data(path: Path, concentrations: list[float], rates: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("concentration", "rate"))
        writer.writerows(zip(concentrations, rates, strict=True))


def _write_curve(path: Path, concentrations: np.ndarray, rates: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("concentration", "fitted_rate"))
        writer.writerows(zip(concentrations, rates, strict=True))


def _write_plot(path: Path, raw_x: list[float], raw_y: list[float], curve_x: np.ndarray, curve_y: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(raw_x, raw_y, color="tab:orange", label="Measured initial rates", zorder=2)
    axis.plot(curve_x, curve_y, color="tab:blue", label="Michaelis–Menten fit")
    axis.set_xscale("log")
    axis.set_xlabel("Substrate concentration")
    axis.set_ylabel("Initial rate")
    axis.set_title("EnzymeOpt fit")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _markdown_report(report: dict[str, object], fit: FitResult) -> str:
    assert fit.km is not None and fit.vmax is not None
    lines = [
        "# EnzymeOpt experimental report",
        "",
        f"Measurements: {report['measurement_count']}",
        f"Vmax estimate: {fit.vmax:.8g}",
        f"KM estimate: {fit.km:.8g}",
        f"Fit status: {fit.message}",
    ]
    for label, interval in (("Vmax", fit.vmax_interval), ("KM", fit.km_interval)):
        if interval is None:
            lines.append(f"{label} 95% confidence interval: unavailable")
        else:
            lines.append(f"{label} 95% confidence interval: [{interval.lower:.8g}, {interval.upper:.8g}]")
    lines.extend(["", "Files: raw_data.csv, fit_curve.csv, fit_curve.png, report.json.", ""])
    return "\n".join(lines)
