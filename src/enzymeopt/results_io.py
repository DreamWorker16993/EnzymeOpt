"""Persistent, auditable output for Monte Carlo runs."""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from enzymeopt.config import ExperimentConfig
from enzymeopt.fitting import FitOptions
from enzymeopt.monte_carlo import MonteCarloResult


SCHEMA_VERSION = 1


def save_monte_carlo_result(
    result: MonteCarloResult,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save lossless JSON, tabular CSV, configuration, and runtime metadata."""

    if not isinstance(result, MonteCarloResult):
        raise TypeError("result must be a MonteCarloResult")
    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": result.config.to_dict(),
        "fit_options": asdict(result.fit_options),
        "records": list(result.records),
        "summaries": list(result.summaries),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "package": "enzymeopt",
        "package_version": version("enzymeopt"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "seed": result.config.seed,
        "replicates": result.config.replicates,
        "record_count": len(result.records),
        "summary_count": len(result.summaries),
    }
    _write_json(destination / "result.json", payload)
    _write_json(destination / "config.json", result.config.to_dict())
    _write_json(destination / "metadata.json", metadata)
    _write_csv(destination / "records.csv", result.records)
    _write_csv(
        destination / "summaries.csv",
        tuple(_flatten(summary) for summary in result.summaries),
    )
    return destination


def load_monte_carlo_result(output_directory: str | Path) -> MonteCarloResult:
    """Load and validate the lossless result representation."""

    source = Path(output_directory) / "result.json"
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported result schema version")
    return MonteCarloResult(
        config=ExperimentConfig.from_dict(payload["config"]),
        fit_options=FitOptions(**payload["fit_options"]),
        records=tuple(payload["records"]),
        summaries=tuple(payload["summaries"]),
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: tuple[dict, ...]) -> None:
    flattened = tuple(_flatten(row) for row in rows)
    fields = sorted({field for row in flattened for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, name))
        else:
            flattened[name] = item
    return flattened
