"""Command-line entry point for reproducible EnzymeOpt benchmarks."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from enzymeopt.config import ExperimentConfig
from enzymeopt.interactive import InteractiveOptions, run_interactive_session
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


def build_interactive_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzymeopt interactive",
        description="Fit real Michaelis-Menten measurements and suggest the next concentration.",
    )
    parser.add_argument("--min-concentration", type=float, default=0.05)
    parser.add_argument("--max-concentration", type=float, default=20.0)
    parser.add_argument("--max-measurements", type=int, default=24)
    parser.add_argument("--noise-std", type=float, default=1.0, help="positive noise scale used for D-optimal ranking")
    parser.add_argument("--candidate-count", type=int, default=200)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--name",
        type=_experiment_name,
        help="experiment name; results are saved under outputs/NAME",
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="explicit result directory (advanced alternative to --name)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _experiment_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise argparse.ArgumentTypeError(
            "name must be 1-80 letters, numbers, hyphens, or underscores"
        )
    return value


def _check_interactive_output(parser: argparse.ArgumentParser, output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        parser.error(f"result path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        parser.error(
            f"experiment result already exists: {output}; choose another --name "
            "or explicitly use --overwrite"
        )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "interactive":
        parser = build_interactive_parser()
        args = parser.parse_args(values[1:])
        output = args.output if args.output is not None else Path("outputs") / args.name
        _check_interactive_output(parser, output, args.overwrite)
        return run_interactive_session(
            InteractiveOptions(
                substrate_min=args.min_concentration,
                substrate_max=args.max_concentration,
                measurement_budget=args.max_measurements,
                noise_std=args.noise_std,
                candidate_count=args.candidate_count,
                output=output,
                overwrite=args.overwrite,
            )
        )
    args = build_parser().parse_args(values)
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
