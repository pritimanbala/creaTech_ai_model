#!/usr/bin/env python3
"""Production planning and optimization for precast yard operations.

This module extends strength modeling into order-level planning.
Key capabilities:
- Estimate *actual minimum curing time* to reach required strength (no 48h cap).
- Convert hour/day values safely and consistently.
- Model throughput constraints: daily mold capacity + yard space limits.
- Produce three actionable pathways: fastest, balanced, cheapest.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from concrete_strength_dataset_model import (
    MIX_FEATURE_COLUMNS,
    load_early_age_strength_predictor,
    train_strength_regressor_from_csv,
)


# -------------------------
# Domain Configuration
# -------------------------


@dataclass(frozen=True)
class CostConfig:
    cement_rate_per_kg: float = 7.0
    flyash_rate_per_kg: float = 3.0
    admixture_rate_per_kg: float = 50.0
    energy_rate_per_degree_hour: float = 2.0
    yard_holding_rate_per_mold_day: float = 1000.0
    base_labor_cost_per_day: float = 5000.0
    labor_reduction_per_automation_level: float = 0.10


@dataclass(frozen=True)
class PlantConfig:
    molds_per_day_capacity: int
    yard_size_available: int


@dataclass(frozen=True)
class OrderRequest:
    units_ordered: int
    mold_volume_m3: float
    mold_area_m2: float
    required_strength_mpa: float
    completion_days_target: float
    location: str
    order_date: date


@dataclass(frozen=True)
class ClimateRecord:
    ambient_temp_c: float
    humidity_pct: float


@dataclass(frozen=True)
class ProcessPolicy:
    name: str
    steam_temp_c: float
    automation_level: float
    curing_duration_hours_per_day: float


# -------------------------
# Utility / conversions
# -------------------------


def hours_to_days(hours: float) -> float:
    return float(hours) / 24.0


def days_to_hours(days: float) -> float:
    return float(days) * 24.0


# -------------------------
# Strength-time estimation
# -------------------------


def resolve_climate(location: str, order_date: date, climate_csv: str | None = None) -> ClimateRecord:
    """Resolve climate from optional dataset; otherwise use robust defaults.

    Expected optional CSV columns: location, date, ambient_temp_c, humidity_pct
    where date is ISO format (YYYY-MM-DD).
    """
    if climate_csv and Path(climate_csv).exists():
        climate_df = pd.read_csv(climate_csv)
        expected = {"location", "date", "ambient_temp_c", "humidity_pct"}
        if expected.issubset(set(climate_df.columns)):
            date_str = order_date.isoformat()
            subset = climate_df[
                (climate_df["location"].astype(str).str.lower() == str(location).lower())
                & (climate_df["date"].astype(str) == date_str)
            ]
            if not subset.empty:
                row = subset.iloc[0]
                return ClimateRecord(
                    ambient_temp_c=float(row["ambient_temp_c"]),
                    humidity_pct=float(row["humidity_pct"]),
                )

    return ClimateRecord(ambient_temp_c=25.0, humidity_pct=60.0)


def compute_equivalent_age_factor(policy: ProcessPolicy, climate: ClimateRecord) -> float:
    """Convert actual time into equivalent-age multiplier.

    The factor accelerates/slows effective hydration age based on process + climate.
    It is intentionally bounded to maintain numerical stability.
    """
    steam_term = 1.0 + 0.012 * max(0.0, policy.steam_temp_c - 20.0)
    curing_term = 0.8 + 0.2 * np.clip(policy.curing_duration_hours_per_day / 24.0, 0.0, 1.0)
    temp_term = 1.0 + 0.01 * (climate.ambient_temp_c - 20.0)
    humidity_term = 1.0 - 0.002 * max(0.0, climate.humidity_pct - 60.0)

    raw = steam_term * curing_term * temp_term * humidity_term
    return float(np.clip(raw, 0.5, 3.0))


def predict_strength_at_age_days(
    strength_model: Any,
    mix_features: Mapping[str, float],
    age_days: float,
) -> float:
    """Predict strength at a specific age (days) for a given mix."""
    row = {col: float(mix_features[col]) for col in MIX_FEATURE_COLUMNS}
    row["age_day"] = float(age_days)
    pred = np.asarray(strength_model.base_model.predict(pd.DataFrame([row])), dtype=float).reshape(-1)
    return float(pred[0])


def find_min_time_to_strength_days(
    strength_model: Any,
    mix_features: Mapping[str, float],
    required_strength_mpa: float,
    eq_age_factor: float,
    max_search_days: float = 60.0,
    tol_days: float = 1e-3,
) -> float:
    """Find minimum *actual* days required to achieve strength.

    Uses bisection over actual age while querying model at equivalent age:
    equivalent_age = actual_age * eq_age_factor

    No hard 48-hour stress cap is applied; search extends to `max_search_days`.
    """
    def strength_at_actual_day(d: float) -> float:
        return predict_strength_at_age_days(
            strength_model=strength_model,
            mix_features=mix_features,
            age_days=max(0.01, d * eq_age_factor),
        )

    lo, hi = 0.0, 1.0
    while hi <= max_search_days and strength_at_actual_day(hi) < required_strength_mpa:
        hi *= 2.0

    if hi > max_search_days:
        hi = max_search_days
        if strength_at_actual_day(hi) < required_strength_mpa:
            return float(max_search_days)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if strength_at_actual_day(mid) >= required_strength_mpa:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol_days:
            break

    return float(hi)


# -------------------------
# Throughput-constrained planning
# -------------------------


def simulate_order_completion_days(
    units_ordered: int,
    curing_days_per_unit: float,
    molds_per_day_capacity: int,
    yard_size_available: int,
) -> float:
    """Simulate completion with daily start limit + yard occupancy limit.

    If each mold needs several days in curing, occupied yard slots block new starts
    until earlier molds are released. This captures the bottleneck behavior requested.
    """
    if units_ordered <= 0:
        return 0.0
    if molds_per_day_capacity <= 0:
        raise ValueError("molds_per_day_capacity must be > 0")
    if yard_size_available <= 0:
        raise ValueError("yard_size_available must be > 0")

    pending = int(units_ordered)
    in_progress_completion_heap: List[float] = []
    current_day = 0.0
    last_completion = 0.0

    while pending > 0 or in_progress_completion_heap:
        while in_progress_completion_heap and in_progress_completion_heap[0] <= current_day + 1e-12:
            heapq.heappop(in_progress_completion_heap)

        available_slots = yard_size_available - len(in_progress_completion_heap)
        starts_today = max(0, min(pending, molds_per_day_capacity, available_slots))

        for _ in range(starts_today):
            completion_t = current_day + curing_days_per_unit
            heapq.heappush(in_progress_completion_heap, completion_t)
            last_completion = max(last_completion, completion_t)

        pending -= starts_today

        if pending <= 0 and in_progress_completion_heap:
            current_day = min(in_progress_completion_heap)
        else:
            current_day += 1.0

    return float(last_completion)


# -------------------------
# Cost & scenario evaluation
# -------------------------


def compute_total_cost(
    mix_features: Mapping[str, float],
    policy: ProcessPolicy,
    order: OrderRequest,
    completion_days: float,
    cost_cfg: CostConfig,
) -> float:
    """Compute total order cost for a plan."""
    cement_total_kg = float(mix_features["cement_kg"]) * order.mold_volume_m3 * order.units_ordered
    flyash_total_kg = float(mix_features["fly_ash_kg"]) * order.mold_volume_m3 * order.units_ordered
    admixture_total_kg = float(mix_features["superplasticizer_kg"]) * order.mold_volume_m3 * order.units_ordered

    material_cost = (
        cement_total_kg * cost_cfg.cement_rate_per_kg
        + flyash_total_kg * cost_cfg.flyash_rate_per_kg
        + admixture_total_kg * cost_cfg.admixture_rate_per_kg
    )

    energy_cost = (
        policy.steam_temp_c
        * policy.curing_duration_hours_per_day
        * completion_days
        * cost_cfg.energy_rate_per_degree_hour
    )

    yard_cost = completion_days * order.units_ordered * cost_cfg.yard_holding_rate_per_mold_day

    labor_multiplier = max(0.0, 1.0 - cost_cfg.labor_reduction_per_automation_level * policy.automation_level)
    labor_cost = completion_days * cost_cfg.base_labor_cost_per_day * labor_multiplier

    return float(material_cost + energy_cost + yard_cost + labor_cost)


def evaluate_policy(
    strength_model: Any,
    mix_features: Mapping[str, float],
    order: OrderRequest,
    plant: PlantConfig,
    policy: ProcessPolicy,
    climate: ClimateRecord,
    cost_cfg: CostConfig,
) -> Dict[str, float | str]:
    """Evaluate one process policy and return KPI bundle."""
    eq_factor = compute_equivalent_age_factor(policy, climate)
    curing_days_per_unit = find_min_time_to_strength_days(
        strength_model=strength_model,
        mix_features=mix_features,
        required_strength_mpa=order.required_strength_mpa,
        eq_age_factor=eq_factor,
    )

    total_completion_days = simulate_order_completion_days(
        units_ordered=order.units_ordered,
        curing_days_per_unit=curing_days_per_unit,
        molds_per_day_capacity=plant.molds_per_day_capacity,
        yard_size_available=plant.yard_size_available,
    )

    total_cost = compute_total_cost(
        mix_features=mix_features,
        policy=policy,
        order=order,
        completion_days=total_completion_days,
        cost_cfg=cost_cfg,
    )

    completion_deviation_days = total_completion_days - order.completion_days_target

    return {
        "policy_name": policy.name,
        "steam_temp_c": float(policy.steam_temp_c),
        "automation_level": float(policy.automation_level),
        "curing_duration_hours_per_day": float(policy.curing_duration_hours_per_day),
        "equivalent_age_factor": float(eq_factor),
        "min_curing_days_per_unit": float(curing_days_per_unit),
        "total_completion_days": float(total_completion_days),
        "completion_deviation_days": float(completion_deviation_days),
        "total_cost": float(total_cost),
    }


def recommend_three_paths(
    strength_model: Any,
    mix_features: Mapping[str, float],
    order: OrderRequest,
    plant: PlantConfig,
    climate: ClimateRecord,
    cost_cfg: CostConfig | None = None,
) -> Dict[str, Dict[str, float | str]]:
    """Return fastest, balanced, and cheapest pathways with transparent metrics."""
    cfg = cost_cfg or CostConfig()

    candidate_policies = [
        ProcessPolicy("fast-track", steam_temp_c=80.0, automation_level=4.0, curing_duration_hours_per_day=20.0),
        ProcessPolicy("balanced", steam_temp_c=60.0, automation_level=2.0, curing_duration_hours_per_day=16.0),
        ProcessPolicy("cost-saver", steam_temp_c=35.0, automation_level=1.0, curing_duration_hours_per_day=10.0),
    ]

    evaluated = [
        evaluate_policy(
            strength_model=strength_model,
            mix_features=mix_features,
            order=order,
            plant=plant,
            policy=policy,
            climate=climate,
            cost_cfg=cfg,
        )
        for policy in candidate_policies
    ]

    fastest = min(evaluated, key=lambda x: x["total_completion_days"])
    cheapest = min(evaluated, key=lambda x: x["total_cost"])

    def balanced_score(item: Mapping[str, float | str]) -> float:
        completion = float(item["total_completion_days"])
        cost = float(item["total_cost"])
        min_completion = min(float(e["total_completion_days"]) for e in evaluated)
        max_completion = max(float(e["total_completion_days"]) for e in evaluated)
        min_cost = min(float(e["total_cost"]) for e in evaluated)
        max_cost = max(float(e["total_cost"]) for e in evaluated)

        norm_completion = 0.0 if max_completion == min_completion else (completion - min_completion) / (max_completion - min_completion)
        norm_cost = 0.0 if max_cost == min_cost else (cost - min_cost) / (max_cost - min_cost)
        return 0.5 * norm_completion + 0.5 * norm_cost

    balanced = min(evaluated, key=balanced_score)

    return {
        "fastest_path": dict(fastest),
        "balanced_path": dict(balanced),
        "cheapest_path": dict(cheapest),
    }


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Order-level precast planning with model-driven curing time.")
    parser.add_argument("--data", default="concrete_data.csv", help="Dataset path (concrete_data.csv schema).")
    parser.add_argument("--model-path", default="artifacts/concrete_strength_age_model.joblib", help="Trained model path.")
    parser.add_argument("--retrain-model", action="store_true", help="Retrain model from --data before planning.")

    parser.add_argument("--units-ordered", type=int, required=True)
    parser.add_argument("--mold-volume", type=float, required=True, help="m^3 per mold")
    parser.add_argument("--mold-area", type=float, required=True, help="m^2 per mold")
    parser.add_argument("--required-strength", type=float, required=True, help="MPa")
    parser.add_argument("--completion-days", type=float, required=True)
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--yard-size", type=int, required=True)
    parser.add_argument("--molds-per-day", type=int, required=True)
    parser.add_argument("--climate-csv", type=str, default=None, help="Optional climate lookup CSV")

    parser.add_argument(
        "--mix-json",
        type=str,
        required=True,
        help="JSON string with mix features using keys: cement_kg, blast_furnace_slag_kg, fly_ash_kg, water_kg, superplasticizer_kg, coarse_aggregate_kg, fine_aggregate_kg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    if args.retrain_model or not model_path.exists():
        artifacts = train_strength_regressor_from_csv(args.data, output_model_path=model_path)
        print(f"Trained model -> {artifacts.model_path}")

    strength_model = load_early_age_strength_predictor(model_path)

    order_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
    order = OrderRequest(
        units_ordered=args.units_ordered,
        mold_volume_m3=args.mold_volume,
        mold_area_m2=args.mold_area,
        required_strength_mpa=args.required_strength,
        completion_days_target=args.completion_days,
        location=args.location,
        order_date=order_dt,
    )
    plant = PlantConfig(molds_per_day_capacity=args.molds_per_day, yard_size_available=args.yard_size)

    mix_features = json.loads(args.mix_json)
    missing = [c for c in MIX_FEATURE_COLUMNS if c not in mix_features]
    if missing:
        raise ValueError(f"mix_json missing keys: {missing}")

    climate = resolve_climate(args.location, order_dt, climate_csv=args.climate_csv)

    recommendations = recommend_three_paths(
        strength_model=strength_model,
        mix_features=mix_features,
        order=order,
        plant=plant,
        climate=climate,
    )

    output = {
        "input_summary": {
            "units_ordered": order.units_ordered,
            "completion_days_target": order.completion_days_target,
            "required_strength_mpa": order.required_strength_mpa,
            "location": order.location,
            "order_date": order.order_date.isoformat(),
            "molds_per_day_capacity": plant.molds_per_day_capacity,
            "yard_size_available": plant.yard_size_available,
            "climate": {
                "ambient_temp_c": climate.ambient_temp_c,
                "humidity_pct": climate.humidity_pct,
            },
            "hour_day_conversion_note": "All optimization time is in days. Where hourly settings exist (e.g., curing_duration_hours_per_day), conversion uses day = 24 hours.",
        },
        "recommended_paths": recommendations,
        "pathway_logic": [
            "1) Estimated minimum curing days per unit from model using equivalent-age factor.",
            "2) Simulated order completion under molds/day and yard-space constraints.",
            "3) Computed total cost from materials + energy + yard holding + labor automation effect.",
            "4) Selected fastest, balanced, and cheapest among policy candidates.",
        ],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
