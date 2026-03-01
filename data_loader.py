#!/usr/bin/env python3
"""Data loading and domain configuration for concrete optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class VariableBounds:
    cement_min: float = 250.0
    cement_max: float = 500.0
    scm_min: float = 0.0
    scm_max: float = 200.0
    water_min: float = 120.0
    water_max: float = 220.0
    aggregates_min: float = 1200.0
    aggregates_max: float = 2000.0
    temp_min: float = 15.0
    temp_max: float = 40.0
    rh_min: float = 40.0
    rh_max: float = 100.0
    curing_days_min: float = 1.0
    curing_days_max: float = 14.0


@dataclass(frozen=True)
class EngineeringConstraints:
    required_strength_mpa: float = 30.0
    wc_ratio_min: float = 0.30
    wc_ratio_max: float = 0.60


@dataclass(frozen=True)
class OptimizationConfig:
    population_size: int = 80
    generations: int = 60
    seed: int = 42


def load_optional_data(path: str | None) -> pd.DataFrame | None:
    """Load optional input data for future calibration workflows."""
    if not path:
        return None
    data_path = Path(path)
    if not data_path.exists():
        return None
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    return None


def bounds_as_arrays(bounds: VariableBounds) -> tuple[list[float], list[float]]:
    xl = [
        bounds.cement_min,
        bounds.scm_min,
        bounds.water_min,
        bounds.aggregates_min,
        bounds.temp_min,
        bounds.rh_min,
        bounds.curing_days_min,
    ]
    xu = [
        bounds.cement_max,
        bounds.scm_max,
        bounds.water_max,
        bounds.aggregates_max,
        bounds.temp_max,
        bounds.rh_max,
        bounds.curing_days_max,
    ]
    return xl, xu


def variable_names() -> list[str]:
    return [
        "cement_kg_m3",
        "scm_kg_m3",
        "water_kg_m3",
        "aggregates_kg_m3",
        "ambient_temp_c",
        "relative_humidity_pct",
        "curing_days",
    ]


def to_record(x: list[float] | tuple[float, ...] | Dict[str, float]) -> Dict[str, float]:
    if isinstance(x, dict):
        return {k: float(v) for k, v in x.items()}
    names = variable_names()
    return {name: float(value) for name, value in zip(names, x)}
