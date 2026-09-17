from enzymeopt.cli import load_toml_config
from enzymeopt.monte_carlo import run_monte_carlo
from enzymeopt.results_io import load_monte_carlo_result, save_monte_carlo_result


def test_fixed_seed_end_to_end_reproduction(tmp_path) -> None:
    baseline = load_toml_config("configs/baseline.toml")
    config = baseline.from_dict(
        {**baseline.to_dict(), "replicates": 2, "measurement_budgets": [3, 4]}
    )

    first = run_monte_carlo(config)
    second = run_monte_carlo(config)
    first_path = save_monte_carlo_result(first, tmp_path / "first")
    second_path = save_monte_carlo_result(second, tmp_path / "second")

    assert first == second
    assert load_monte_carlo_result(first_path) == load_monte_carlo_result(second_path)
    for name in ("result.json", "config.json", "metadata.json", "records.csv", "summaries.csv"):
        assert (first_path / name).read_bytes() == (second_path / name).read_bytes()
