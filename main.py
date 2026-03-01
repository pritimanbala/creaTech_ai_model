#!/usr/bin/env python3
"""Entry point for industrial concrete multi-objective optimization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data_loader import EngineeringConstraints, OptimizationConfig, VariableBounds, load_optional_data
from evaluation import (
    build_decision_table,
    correlation_cost_time,
    diversity_metric_spacing,
    objective_conflict,
    select_representative_solutions,
    summarize_constraints,
)
from models import CombinedStrengthPredictor, StrengthModelA, StrengthModelB
from optimization import log_decision_table, log_solution, run_nsga2, save_plots


def configure_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSGA-II concrete production optimization")
    parser.add_argument("--required-strength", type=float, default=30.0)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None, help="Optional calibration dataset path")
    parser.add_argument("--output-dir", type=str, default="artifacts_optimization")
    parser.add_argument("--log-file", type=str, default="optimization_log.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(Path(args.log_file))
    logger = logging.getLogger(__name__)

    logger.info("Starting optimization workflow")
    optional_data = load_optional_data(args.data)
    if optional_data is not None:
        logger.info("Optional calibration dataset loaded: shape=%s", optional_data.shape)

    bounds = VariableBounds()
    constraints = EngineeringConstraints(required_strength_mpa=float(args.required_strength))
    cfg = OptimizationConfig(population_size=args.population, generations=args.generations, seed=args.seed)

    predictor = CombinedStrengthPredictor(model_a=StrengthModelA(), model_b=StrengthModelB(), weight_a=0.5, weight_b=0.5)

    result = run_nsga2(
        predictor=predictor,
        bounds=bounds,
        constraints=constraints,
        population_size=cfg.population_size,
        generations=cfg.generations,
        seed=cfg.seed,
    )

    X = result.pop.get("X")
    F = result.pop.get("F")
    G = result.pop.get("G")
    strengths = result.pop.get("strengths")

    decision_df = build_decision_table(X, F, G, strengths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_path = out_dir / "decision_table.csv"
    decision_df.to_csv(decision_path, index=False)
    logger.info("Saved decision variable table to %s", decision_path)
    log_decision_table(decision_df)

    constraint_summary = summarize_constraints(G)
    logger.info("Constraint violation summary values: %s", json.dumps(constraint_summary, indent=2))

    corr = correlation_cost_time(F)
    conflict = objective_conflict(corr)
    diversity = diversity_metric_spacing(F)
    logger.info("Correlation(cost,time)=%.6f", corr)
    logger.info("Objectives conflicting? %s", conflict)
    logger.info("Pareto diversity metric (spacing CV)=%.6f", diversity)

    if not conflict:
        logger.info("Diagnostics: objectives are weakly conflicting or aligned under current constraints.")

    selections = select_representative_solutions(F, G)
    if selections.cheapest_idx is None:
        logger.warning("No feasible solutions available. Suggestion: relax strength threshold or widen W/C limits.")
    else:
        log_solution("Cheapest", selections.cheapest_idx, X, F, G, strengths)
        log_solution("Fastest", selections.fastest_idx, X, F, G, strengths)
        log_solution("Balanced", selections.balanced_idx, X, F, G, strengths)

    plot_artifacts = save_plots(F, out_dir)

    summary = {
        "decision_table": str(decision_path),
        "pareto_plot": str(plot_artifacts.pareto_front_path),
        "objective_scatter": str(plot_artifacts.objective_scatter_path),
        "correlation_cost_time": corr,
        "objectives_conflicting": conflict,
        "diversity_metric": diversity,
        "constraint_summary": constraint_summary,
        "selected_indices": {
            "cheapest": selections.cheapest_idx,
            "fastest": selections.fastest_idx,
            "balanced": selections.balanced_idx,
        },
    }
    (out_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved optimization summary to %s", out_dir / "optimization_summary.json")
    logger.info("Optimization workflow completed")


if __name__ == "__main__":
    main()
