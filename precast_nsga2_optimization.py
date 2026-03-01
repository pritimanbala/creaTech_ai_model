#!/usr/bin/env python3
"""Production planning and optimization for precast yard operations."""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

from concrete_strength_dataset_model import (
    MIX_FEATURE_COLUMNS,
    load_early_age_strength_predictor,
    train_strength_regressor_from_file,
)


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
    latitude: float
    longitude: float
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


@dataclass(frozen=True)
class WeatherApiConfig:
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    forecast_days: int = 4
    timeout_seconds: float = 20.0


def hours_to_days(hours: float) -> float:
    return float(hours) / 24.0


def days_to_hours(days: float) -> float:
    return float(days) * 24.0


def resolve_climate_from_api(
    latitude: float,
    longitude: float,
    order_date: date,
    weather_cfg: WeatherApiConfig | None = None,
) -> ClimateRecord:
    """Fetch 4-day weather forecast and return mean temp/humidity."""
    cfg = weather_cfg or WeatherApiConfig()

    start_date = order_date
    end_date = start_date + timedelta(days=cfg.forecast_days - 1)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean",
        "timezone": "auto",
    }

    response = requests.get(cfg.base_url, params=params, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    temps = daily.get("temperature_2m_mean") or []
    humidity = daily.get("relative_humidity_2m_mean") or []

    if len(temps) < cfg.forecast_days or len(humidity) < cfg.forecast_days:
        raise ValueError("Weather API returned incomplete 4-day climate data")

    return ClimateRecord(
        ambient_temp_c=float(np.mean(temps[: cfg.forecast_days])),
        humidity_pct=float(np.mean(humidity[: cfg.forecast_days])),
    )


def compute_equivalent_age_factor(policy: ProcessPolicy, climate: ClimateRecord) -> float:
    steam_term = 1.0 + 0.012 * max(0.0, policy.steam_temp_c - 20.0)
    curing_term = 0.8 + 0.2 * np.clip(policy.curing_duration_hours_per_day / 24.0, 0.0, 1.0)
    temp_term = 1.0 + 0.01 * (climate.ambient_temp_c - 20.0)
    humidity_term = 1.0 - 0.002 * max(0.0, climate.humidity_pct - 60.0)

    raw = steam_term * curing_term * temp_term * humidity_term
    return float(np.clip(raw, 0.5, 3.0))


def predict_strength_at_age_days(strength_model: Any, mix_features: Mapping[str, float], age_days: float) -> float:
    row = {col: float(mix_features[col]) for col in MIX_FEATURE_COLUMNS}
    row["age_day"] = float(age_days)
    pred = np.asarray(strength_model.base_model.predict(pd.DataFrame([row])), dtype=float).reshape(-1)
    return float(pred[0])


def find_min_time_to_strength_days(
    strength_model: Any,
    mix_features: Mapping[str, float],
    required_strength_mpa: float,
    eq_age_factor: float,
    max_search_days: float = 120.0,
    tol_days: float = 1e-3,
) -> float:
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

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if strength_at_actual_day(mid) >= required_strength_mpa:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol_days:
            break

    return float(hi)


def simulate_order_completion_days(
    units_ordered: int,
    curing_days_per_unit: float,
    molds_per_day_capacity: int,
    yard_size_available: int,
) -> float:
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


def compute_total_cost(
    mix_features: Mapping[str, float],
    policy: ProcessPolicy,
    order: OrderRequest,
    completion_days: float,
    cost_cfg: CostConfig,
) -> float:
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
    total_cost = compute_total_cost(mix_features, policy, order, total_completion_days, cost_cfg)
    completion_deviation_days = total_completion_days - order.completion_days_target

    return {
        "policy_name": policy.name,
        "steam_temp_c": float(policy.steam_temp_c),
        "automation_level": float(policy.automation_level),
        "curing_duration_hours_per_day": float(policy.curing_duration_hours_per_day),
        "equivalent_age_factor": float(eq_factor),
        "min_curing_days_per_unit": float(curing_days_per_unit),
        "min_curing_hours_per_unit": float(days_to_hours(curing_days_per_unit)),
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
    cfg = cost_cfg or CostConfig()
    candidate_policies = [
        ProcessPolicy("fast-track", steam_temp_c=80.0, automation_level=4.0, curing_duration_hours_per_day=20.0),
        ProcessPolicy("balanced", steam_temp_c=60.0, automation_level=2.0, curing_duration_hours_per_day=16.0),
        ProcessPolicy("cost-saver", steam_temp_c=35.0, automation_level=1.0, curing_duration_hours_per_day=10.0),
    ]

    evaluated = [
        evaluate_policy(strength_model, mix_features, order, plant, policy, climate, cfg)
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


def ensure_trained_model(data_path: str | Path, model_path: str | Path) -> Path:
    """Train once if model is missing; otherwise reuse existing artifact."""
    model_file = Path(model_path)
    if not model_file.exists():
        train_strength_regressor_from_file(data_path=data_path, output_model_path=model_file)
    return model_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Order-level precast planning with model-driven curing time.")
    parser.add_argument("--data", default="concrete_data.xlsx", help="Dataset path (.xlsx/.csv) used for first-time training only.")
    parser.add_argument("--model-path", default="artifacts/concrete_strength_age_model.joblib", help="Trained model path.")

    parser.add_argument("--units-ordered", type=int, required=True)
    parser.add_argument("--mold-volume", type=float, required=True, help="m^3 per mold")
    parser.add_argument("--mold-area", type=float, required=True, help="m^2 per mold")
    parser.add_argument("--required-strength", type=float, required=True, help="MPa")
    parser.add_argument("--completion-days", type=float, required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--yard-size", type=int, required=True)
    parser.add_argument("--molds-per-day", type=int, required=True)
    parser.add_argument(
        "--mix-json",
        type=str,
        required=True,
        help="JSON string with mix features keys: cement_kg, blast_furnace_slag_kg, fly_ash_kg, water_kg, superplasticizer_kg, coarse_aggregate_kg, fine_aggregate_kg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_file = ensure_trained_model(args.data, args.model_path)
    strength_model = load_early_age_strength_predictor(model_file)

    order_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
    order = OrderRequest(
        units_ordered=args.units_ordered,
        mold_volume_m3=args.mold_volume,
        mold_area_m2=args.mold_area,
        required_strength_mpa=args.required_strength,
        completion_days_target=args.completion_days,
        latitude=args.latitude,
        longitude=args.longitude,
        order_date=order_dt,
    )
    plant = PlantConfig(molds_per_day_capacity=args.molds_per_day, yard_size_available=args.yard_size)

    mix_features = json.loads(args.mix_json)
    missing = [c for c in MIX_FEATURE_COLUMNS if c not in mix_features]
    if missing:
        raise ValueError(f"mix_json missing keys: {missing}")

    climate = resolve_climate_from_api(order.latitude, order.longitude, order.order_date)
    recommendations = recommend_three_paths(strength_model, mix_features, order, plant, climate)

    output = {
        "input_summary": {
            "units_ordered": order.units_ordered,
            "completion_days_target": order.completion_days_target,
            "required_strength_mpa": order.required_strength_mpa,
            "latitude": order.latitude,
            "longitude": order.longitude,
            "order_date": order.order_date.isoformat(),
            "molds_per_day_capacity": plant.molds_per_day_capacity,
            "yard_size_available": plant.yard_size_available,
            "climate_4day_average": {
                "ambient_temp_c": climate.ambient_temp_c,
                "humidity_pct": climate.humidity_pct,
            },
        },
        "recommended_paths": recommendations,
        "pathway_logic": [
            "1) Model trained only on first run if artifact is absent, then reused.",
            "2) Pulled 4-day weather forecast using latitude/longitude and averaged temperature+humidity.",
            "3) Estimated minimum curing days from model and equivalent-age factor.",
            "4) Simulated throughput under molds/day and yard-size constraints and computed cost.",
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
