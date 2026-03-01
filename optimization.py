#!/usr/bin/env python3
"""NSGA-II optimization module with full industrial logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from data_loader import EngineeringConstraints, VariableBounds, bounds_as_arrays, to_record, variable_names
from models import CombinedStrengthPredictor, compute_cost, compute_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    objective_scatter_path: Path
    pareto_front_path: Path


class ConcreteOptimizationProblem(Problem):
    def __init__(
        self,
        predictor: CombinedStrengthPredictor,
        bounds: VariableBounds,
        constraints: EngineeringConstraints,
    ):
        self.predictor = predictor
        self.constraints = constraints
        xl, xu = bounds_as_arrays(bounds)
        super().__init__(n_var=7, n_obj=2, n_constr=2, xl=np.array(xl), xu=np.array(xu))

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        F = []
        G = []
        strengths = []
        for x in X:
            rec = to_record(x.tolist())
            strength = self.predictor.predict(rec)
            cost = compute_cost(rec)
            time_days = compute_time(rec)
            wc_ratio = rec["water_kg_m3"] / max(rec["cement_kg_m3"], 1e-9)

            g_strength = self.constraints.required_strength_mpa - strength
            g_wc = max(0.0, self.constraints.wc_ratio_min - wc_ratio) + max(0.0, wc_ratio - self.constraints.wc_ratio_max)

            F.append([cost, time_days])
            G.append([g_strength, g_wc])
            strengths.append(strength)

        out["F"] = np.asarray(F, dtype=float)
        out["G"] = np.asarray(G, dtype=float)
        out["strengths"] = np.asarray(strengths, dtype=float)


class OptimizationLoggerCallback(Callback):
    def __init__(self):
        super().__init__()

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")
        X = pop.get("X")
        feasible = np.all(G <= 0.0, axis=1)
        feasible_count = int(np.sum(feasible))
        infeasible_count = int(len(feasible) - feasible_count)

        logger.info("Generation %d", gen)
        logger.info("Population initialized/updated with %d individuals", len(X))
        logger.info("Feasible=%d Infeasible=%d", feasible_count, infeasible_count)
        logger.info("Objective values (cost,time): %s", np.array2string(F, precision=3, threshold=20))
        logger.info("Constraint violations: %s", np.array2string(G, precision=3, threshold=20))

        nd_idx = algorithm.opt.get("index") if hasattr(algorithm.opt, "get") else None
        if nd_idx is None or len(nd_idx) == 0:
            pf = algorithm.opt.get("F")
        else:
            pf = F[nd_idx]
        if pf is not None:
            logger.info("Pareto front snapshot at generation %d: %s", gen, np.array2string(np.asarray(pf), precision=3, threshold=20))


def run_nsga2(
    predictor: CombinedStrengthPredictor,
    bounds: VariableBounds,
    constraints: EngineeringConstraints,
    population_size: int,
    generations: int,
    seed: int,
):
    problem = ConcreteOptimizationProblem(predictor=predictor, bounds=bounds, constraints=constraints)

    algorithm = NSGA2(pop_size=population_size)
    logger.info("Initializing NSGA-II with pop_size=%d generations=%d seed=%d", population_size, generations, seed)

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=seed,
        verbose=False,
        callback=OptimizationLoggerCallback(),
        save_history=True,
    )
    return result


def log_solution(label: str, idx: int, X: np.ndarray, F: np.ndarray, G: np.ndarray, strengths: np.ndarray):
    rec = to_record(X[idx].tolist())
    logger.info("%s solution index=%d", label, idx)
    logger.info(
        "%s values: Cement=%.3f Water=%.3f SCM=%.3f Aggregates=%.3f Temp=%.3f RH=%.3f CuringDays=%.3f",
        label,
        rec["cement_kg_m3"],
        rec["water_kg_m3"],
        rec["scm_kg_m3"],
        rec["aggregates_kg_m3"],
        rec["ambient_temp_c"],
        rec["relative_humidity_pct"],
        rec["curing_days"],
    )
    logger.info(
        "%s objectives: Cost=%.4f Time=%.4f Strength=%.4f ConstraintMargins=%s",
        label,
        F[idx, 0],
        F[idx, 1],
        strengths[idx],
        np.array2string(G[idx], precision=6),
    )


def save_plots(F: np.ndarray, out_dir: Path) -> RunArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    scatter = out_dir / "objective_scatter.png"
    pareto = out_dir / "pareto_front.png"

    plt.figure(figsize=(8, 6))
    plt.scatter(F[:, 0], F[:, 1], alpha=0.7)
    plt.xlabel("Total Cost")
    plt.ylabel("Time to Strength (days)")
    plt.title("Objective Scatter Plot")
    plt.tight_layout()
    plt.savefig(scatter)
    plt.close()

    sorted_idx = np.argsort(F[:, 0])
    pf = F[sorted_idx]
    plt.figure(figsize=(8, 6))
    plt.plot(pf[:, 0], pf[:, 1], marker="o")
    plt.xlabel("Total Cost")
    plt.ylabel("Time to Strength (days)")
    plt.title("Pareto Front")
    plt.tight_layout()
    plt.savefig(pareto)
    plt.close()

    logger.info("Saved objective scatter plot to %s", scatter)
    logger.info("Saved Pareto front plot to %s", pareto)
    return RunArtifacts(objective_scatter_path=scatter, pareto_front_path=pareto)


def log_decision_table(df):
    logger.info("Decision variable table columns: %s", list(df.columns))
    logger.info("Decision variable table sample:\n%s", df.head(10).to_string(index=False))
    logger.info("Constraint violation summary: feasible=%d infeasible=%d", int(df["is_feasible"].sum()), int((~df["is_feasible"]).sum()))
