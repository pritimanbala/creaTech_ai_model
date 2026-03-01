#!/usr/bin/env python3
"""Evaluation and post-processing utilities for optimization outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_loader import variable_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionResult:
    cheapest_idx: int | None
    fastest_idx: int | None
    balanced_idx: int | None


def normalize_series(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        logger.warning("Cost/time collapsed to identical values: min=max=%.6f", vmin)
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def diversity_metric_spacing(F: np.ndarray) -> float:
    if len(F) < 2:
        return 0.0
    sorted_idx = np.argsort(F[:, 0])
    front = F[sorted_idx]
    d = np.sqrt(np.sum(np.diff(front, axis=0) ** 2, axis=1))
    if len(d) == 0:
        return 0.0
    return float(np.std(d) / (np.mean(d) + 1e-12))


def correlation_cost_time(F: np.ndarray) -> float:
    if len(F) < 2:
        return 0.0
    return float(np.corrcoef(F[:, 0], F[:, 1])[0, 1])


def objective_conflict(correlation: float) -> bool:
    return correlation > 0.2


def feasible_mask(G: np.ndarray) -> np.ndarray:
    return np.all(G <= 0.0, axis=1)


def select_representative_solutions(F: np.ndarray, G: np.ndarray) -> SelectionResult:
    feasible = feasible_mask(G)
    idx = np.where(feasible)[0]
    if len(idx) == 0:
        logger.warning("No feasible solutions found. Constraint space may be too tight.")
        return SelectionResult(None, None, None)

    feasible_F = F[idx]

    cheapest_local = int(np.argmin(feasible_F[:, 0]))
    fastest_local = int(np.argmin(feasible_F[:, 1]))

    cost_norm = normalize_series(feasible_F[:, 0])
    time_norm = normalize_series(feasible_F[:, 1])
    balanced_score = 0.5 * cost_norm + 0.5 * time_norm
    balanced_local = int(np.argmin(balanced_score))

    cheapest_idx = int(idx[cheapest_local])
    fastest_idx = int(idx[fastest_local])
    balanced_idx = int(idx[balanced_local])

    logger.info("Cheapest solution selected because cost = %.4f is minimum.", F[cheapest_idx, 0])
    logger.info("Fastest solution selected because time = %.4f is minimum.", F[fastest_idx, 1])
    logger.info(
        "Balanced solution selected based on normalized tradeoff score = %.6f.",
        float(balanced_score[balanced_local]),
    )

    if cheapest_idx == fastest_idx == balanced_idx:
        logger.info("Fastest, cheapest, and balanced are identical due to aligned objectives and constraints.")

    return SelectionResult(cheapest_idx, fastest_idx, balanced_idx)


def build_decision_table(X: np.ndarray, F: np.ndarray, G: np.ndarray, strengths: np.ndarray) -> pd.DataFrame:
    cols = variable_names() + ["cost", "time_days", "predicted_strength_mpa"]
    base = np.column_stack([X, F, strengths.reshape(-1, 1)])
    df = pd.DataFrame(base, columns=cols)
    for i in range(G.shape[1]):
        df[f"constraint_g{i+1}"] = G[:, i]
    df["is_feasible"] = np.all(G <= 0.0, axis=1)
    return df


def summarize_constraints(G: np.ndarray) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for i in range(G.shape[1]):
        gv = G[:, i]
        summary[f"g{i+1}_mean"] = float(np.mean(gv))
        summary[f"g{i+1}_max"] = float(np.max(gv))
        summary[f"g{i+1}_min"] = float(np.min(gv))
    return summary
