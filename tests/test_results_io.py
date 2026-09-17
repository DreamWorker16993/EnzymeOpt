import csv
import json

import pytest

from enzymeopt.config import ExperimentConfig
from enzymeopt.monte_carlo import run_monte_carlo
from enzymeopt.results_io import load_monte_carlo_result, save_monte_carlo_result


def test_saved_result_round_trip_and_audit_files(tmp_path) -> None:
    result = run_monte_carlo(
        ExperimentConfig(replicates=2, measurement_budgets=(3,), seed=17)
    )
    output = save_monte_carlo_result(result, tmp_path / "run")

    assert load_monte_carlo_result(output) == result
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "metadata.json",
        "records.csv",
        "result.json",
        "summaries.csv",
    }
    with (output / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    assert metadata["status"] == "completed"
    assert metadata["seed"] == 17
    assert metadata["record_count"] == 6
    assert metadata["package_version"] == "0.1.0"
    with (output / "records.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 6
    assert "km_relative_error" in records[0]


def test_save_refuses_nonempty_output_without_overwrite(tmp_path) -> None:
    result = run_monte_carlo(ExperimentConfig(replicates=1, measurement_budgets=(3,)))
    output = tmp_path / "run"
    save_monte_carlo_result(result, output)

    with pytest.raises(FileExistsError, match="not empty"):
        save_monte_carlo_result(result, output)
    save_monte_carlo_result(result, output, overwrite=True)


def test_load_rejects_unknown_schema(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "result.json").write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_monte_carlo_result(output)
