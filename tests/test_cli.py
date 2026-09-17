import json
from pathlib import Path

import pytest

from enzymeopt.cli import load_toml_config, main
from enzymeopt.results_io import load_monte_carlo_result


def test_baseline_config_loads() -> None:
    config = load_toml_config("configs/baseline.toml")

    assert config.replicates == 1_000
    assert config.seed == 20260916
    assert config.measurement_budgets == (3, 4, 6, 8, 12, 16, 24)


def test_cli_runs_with_overrides_and_saves_results(tmp_path) -> None:
    output = tmp_path / "cli-run"

    status = main(
        [
            "--config",
            "configs/baseline.toml",
            "--output",
            str(output),
            "--replicates",
            "2",
            "--seed",
            "91",
            "--quiet",
        ]
    )

    result = load_monte_carlo_result(output)
    assert status == 0
    assert result.config.replicates == 2
    assert result.config.seed == 91
    assert len(result.records) == 42
    with (output / "config.json").open(encoding="utf-8") as stream:
        assert json.load(stream)["seed"] == 91


def test_invalid_toml_shape_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("seed = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment"):
        load_toml_config(path)


def test_cli_rejects_existing_output_before_computation(tmp_path, monkeypatch):
    (tmp_path / 'existing.txt').write_text('keep', encoding='utf-8')
    def unexpected(*args, **kwargs):
        pytest.fail('experiment should not start for an occupied output directory')
    monkeypatch.setattr('enzymeopt.cli.run_monte_carlo', unexpected)
    with pytest.raises(FileExistsError):
        main(['--config', 'configs/baseline.toml', '--output', str(tmp_path)])
    assert (tmp_path / 'existing.txt').read_text() == 'keep'


def test_interactive_command_builds_real_data_options(tmp_path, monkeypatch) -> None:
    seen = {}

    def fake_session(options):
        seen["options"] = options
        return 0

    monkeypatch.setattr("enzymeopt.cli.run_interactive_session", fake_session)
    status = main([
        "interactive", "--min-concentration", "0.1", "--max-concentration", "12",
        "--max-measurements", "9", "--noise-std", "0.2", "--output", str(tmp_path / "real"),
    ])

    assert status == 0
    assert seen["options"].substrate_min == 0.1
    assert seen["options"].measurement_budget == 9


def test_interactive_name_creates_its_own_result_directory(monkeypatch) -> None:
    seen = {}

    def fake_session(options):
        seen["options"] = options
        return 0

    monkeypatch.setattr("enzymeopt.cli.run_interactive_session", fake_session)
    assert main(["interactive", "--name", "enzyme_run_01"]) == 0
    assert seen["options"].output == Path("outputs/enzyme_run_01")


def test_interactive_rejects_duplicate_name_before_session(tmp_path, monkeypatch) -> None:
    output = tmp_path / "used-name"
    output.mkdir()
    (output / "report.json").write_text("{}", encoding="utf-8")

    def unexpected(*args, **kwargs):
        pytest.fail("interactive session should not start for a used result name")

    monkeypatch.setattr("enzymeopt.cli.run_interactive_session", unexpected)
    with pytest.raises(SystemExit, match="2"):
        main(["interactive", "--output", str(output)])


@pytest.mark.parametrize("name", ["../escape", "has spaces", "name.with.dot", ""])
def test_interactive_rejects_unsafe_experiment_names(name) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["interactive", "--name", name])
