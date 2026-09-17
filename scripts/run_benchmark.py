"""Run the checked-in baseline benchmark from the repository root."""

from pathlib import Path

from enzymeopt.cli import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(
        main(
            [
                "--config",
                str(root / "configs" / "baseline.toml"),
                "--output",
                str(root / "outputs" / "baseline"),
            ]
        )
    )
