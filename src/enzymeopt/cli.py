"""Command-line entry point for reproducible EnzymeOpt benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from enzymeopt.config import ExperimentConfig
from enzymeopt.monte_carlo import run_monte_carlo
from enzymeopt.results_io import save_monte_carlo_result


def load_toml_config(path: str | Path) -> ExperimentConfig:
    """Load an ``[experiment]`` table as a validated configuration."""

    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) != {"experiment"} or not isinstance(document["experiment"], dict):
        raise ValueError("configuration must contain only an [experiment] table")
    return ExperimentConfig.from_dict(document["experiment"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzymeopt",
        description="Run a reproducible Michaelis-Menten design benchmark.",
    )
    parser.add_argument("--config", required=True, type=Path, help="TOML configuration file")
    parser.add_argument("--output", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--seed", type=int, help="override the configured master seed")
    parser.add_argument("--replicates", type=int, help="override the configured replicate count")
    parser.add_argument("--overwrite", action="store_true", help="replace known output files")
    parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_toml_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.replicates is not None:
        config = replace(config, replicates=args.replicates)
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"output is not a directory: {args.output}")
        if any(args.output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {args.output}")

    last_percent = -1

    def report(done: int, total: int) -> None:
        nonlocal last_percent
        percent = done * 100 // total
        if not args.quiet and (percent // 10 > last_percent // 10 or done == total):
            print(f"progress: {done}/{total} ({percent}%)", flush=True)
        last_percent = percent

    result = run_monte_carlo(config, progress=report)
    destination = save_monte_carlo_result(result, args.output, overwrite=args.overwrite)
    if not args.quiet:
        print(f"completed: {len(result.records)} experiments")
        print(f"results: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
