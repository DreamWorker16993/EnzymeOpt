import json

import numpy as np
import pytest

from enzymeopt.interactive import InteractiveOptions, InteractiveSession, run_interactive_session
from enzymeopt.model import michaelis_menten


def _session(tmp_path) -> InteractiveSession:
    return InteractiveSession(
        InteractiveOptions(
            substrate_min=0.05,
            substrate_max=20.0,
            measurement_budget=8,
            noise_std=0.1,
            output=tmp_path / "report",
        )
    )


def test_real_measurements_produce_d_optimal_suggestion(tmp_path) -> None:
    session = _session(tmp_path)
    concentrations = [0.05, 1.0, 20.0]
    rates = michaelis_menten(concentrations, km=1.0, vmax=2.0)

    for concentration, rate in zip(concentrations, rates, strict=True):
        update = session.add_measurement(concentration, rate)

    assert update.fit is not None and update.fit.converged
    assert update.suggestion is not None
    assert 0.05 <= update.suggestion <= 20.0
    assert update.fit.km == pytest.approx(1.0)
    assert update.fit.vmax == pytest.approx(2.0)


def test_report_contains_raw_data_curve_and_plot(tmp_path) -> None:
    session = _session(tmp_path)
    concentrations = [0.05, 0.4, 1.0, 4.0, 20.0]
    rates = michaelis_menten(concentrations, km=1.0, vmax=2.0)
    for concentration, rate in zip(concentrations, rates, strict=True):
        session.add_measurement(concentration, rate)

    destination = session.write_report()

    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert report["measurement_count"] == 5
    assert report["fit"]["km"] == pytest.approx(1.0)
    assert report["parameter_uncertainty"]["Vmax"]["standard_error"] == pytest.approx(0.0)
    assert report["parameter_uncertainty"]["KM"]["confidence_interval_width"] == pytest.approx(0.0)
    markdown = (destination / "report.md").read_text(encoding="utf-8")
    assert "Vmax standard error" in markdown
    assert "KM 95% confidence interval width" in markdown
    assert len((destination / "raw_data.csv").read_text(encoding="utf-8").splitlines()) == 6
    assert len((destination / "fit_curve.csv").read_text(encoding="utf-8").splitlines()) == 301
    assert (destination / "fit_curve.png").read_bytes().startswith(b"\x89PNG")


def test_interactive_dialogue_writes_report(tmp_path) -> None:
    values = iter(["0.05 0.0952381", "1 1", "20 1.9047619", "report"])
    messages: list[str] = []
    options = InteractiveOptions(output=tmp_path / "report", measurement_budget=8)

    status = run_interactive_session(options, input_fn=lambda _: next(values), output_fn=messages.append)

    assert status == 0
    assert any("Results will be saved to" in message for message in messages)
    assert any("Suggested next concentration" in message for message in messages)
    assert any("Vmax estimate" in message for message in messages)
    assert any("KM standard error" in message for message in messages)
    assert (tmp_path / "report" / "report.md").exists()


def test_invalid_measurements_and_report_without_fit_are_rejected(tmp_path) -> None:
    session = _session(tmp_path)
    with pytest.raises(ValueError, match="within"):
        session.add_measurement(21.0, 1.0)
    with pytest.raises(ValueError, match="at least 3"):
        session.write_report()

    with pytest.raises(ValueError, match="positive finite"):
        InteractiveOptions(noise_std=0.0)
