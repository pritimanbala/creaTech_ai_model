#!/usr/bin/env python3
"""Streamlit UI for precast order planning.

This app captures user inputs and displays three recommendation pathways:
- Fastest
- Balanced
- Cheapest
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from concrete_strength_dataset_model import MIX_FEATURE_COLUMNS, load_early_age_strength_predictor, train_strength_regressor_from_csv
from precast_nsga2_optimization import (
    ClimateRecord,
    CostConfig,
    OrderRequest,
    PlantConfig,
    recommend_three_paths,
    resolve_climate,
)

st.set_page_config(page_title="Precast Order Planner", page_icon="🏗️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_strength_model(data_path: str, model_path: str, retrain: bool):
    model_file = Path(model_path)
    if retrain or not model_file.exists():
        train_strength_regressor_from_csv(csv_path=data_path, output_model_path=model_file)
    return load_early_age_strength_predictor(model_file)


def build_mix_inputs() -> Dict[str, float]:
    st.subheader("Mix Design Inputs (kg/m³)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cement_kg = st.number_input("Cement", min_value=0.0, value=300.0, step=1.0)
        blast_furnace_slag_kg = st.number_input("Blast Furnace Slag", min_value=0.0, value=120.0, step=1.0)
    with c2:
        fly_ash_kg = st.number_input("Fly Ash", min_value=0.0, value=60.0, step=1.0)
        water_kg = st.number_input("Water", min_value=0.0, value=180.0, step=1.0)
    with c3:
        superplasticizer_kg = st.number_input("Superplasticizer", min_value=0.0, value=8.0, step=0.1)
        coarse_aggregate_kg = st.number_input("Coarse Aggregate", min_value=0.0, value=950.0, step=1.0)
    with c4:
        fine_aggregate_kg = st.number_input("Fine Aggregate", min_value=0.0, value=780.0, step=1.0)

    mix = {
        "cement_kg": cement_kg,
        "blast_furnace_slag_kg": blast_furnace_slag_kg,
        "fly_ash_kg": fly_ash_kg,
        "water_kg": water_kg,
        "superplasticizer_kg": superplasticizer_kg,
        "coarse_aggregate_kg": coarse_aggregate_kg,
        "fine_aggregate_kg": fine_aggregate_kg,
    }
    return mix


def render_results_table(result_map: Dict[str, Dict[str, float | str]]) -> None:
    rows = []
    for key, payload in result_map.items():
        rows.append(
            {
                "Path": key,
                "Policy": payload["policy_name"],
                "Steam Temp (°C)": payload["steam_temp_c"],
                "Automation Level": payload["automation_level"],
                "Curing Hours/Day": payload["curing_duration_hours_per_day"],
                "Equivalent Age Factor": payload["equivalent_age_factor"],
                "Min Curing Days/Unit": payload["min_curing_days_per_unit"],
                "Total Completion Days": payload["total_completion_days"],
                "Deviation vs Target (days)": payload["completion_deviation_days"],
                "Total Cost": payload["total_cost"],
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def main() -> None:
    st.title("🏗️ Precast Yard Order Planning")
    st.caption("Model-driven planning from concrete_data.csv with fastest, balanced, and cheapest pathways.")

    st.sidebar.header("Model Configuration")
    data_path = st.sidebar.text_input("Training data path", value="concrete_data.csv")
    model_path = st.sidebar.text_input("Model artifact path", value="artifacts/concrete_strength_age_model.joblib")
    retrain_model = st.sidebar.checkbox("Retrain model before run", value=False)

    st.subheader("Order Inputs")
    left, right = st.columns(2)

    with left:
        units_ordered = st.number_input("Units Ordered", min_value=1, value=20, step=1)
        mold_volume_m3 = st.number_input("Volume of Mold (m³)", min_value=0.01, value=1.0, step=0.01)
        mold_area_m2 = st.number_input("Area of Mold (m²)", min_value=0.01, value=2.2, step=0.01)
        completion_days_target = st.number_input("Days of Completion Target", min_value=0.1, value=4.0, step=0.1)
        required_strength_mpa = st.number_input("Required Strength (MPa)", min_value=0.1, value=9.0, step=0.1)

    with right:
        location = st.text_input("Location", value="chennai")
        order_date = st.date_input("Date (for climatic conditions)", value=date.today())
        yard_size_available = st.number_input("Yard Size Available (molds)", min_value=1, value=150, step=1)
        molds_per_day_capacity = st.number_input("Molds Produced per Day", min_value=1, value=50, step=1)
        climate_csv = st.text_input("Optional climate CSV", value="")

    mix_features = build_mix_inputs()

    if st.button("Generate Planning Pathways", type="primary"):
        missing_keys = [col for col in MIX_FEATURE_COLUMNS if col not in mix_features]
        if missing_keys:
            st.error(f"Missing mix fields: {missing_keys}")
            return

        try:
            with st.spinner("Loading/training model and computing recommendations..."):
                strength_model = get_strength_model(
                    data_path=data_path,
                    model_path=model_path,
                    retrain=retrain_model,
                )

                order = OrderRequest(
                    units_ordered=int(units_ordered),
                    mold_volume_m3=float(mold_volume_m3),
                    mold_area_m2=float(mold_area_m2),
                    required_strength_mpa=float(required_strength_mpa),
                    completion_days_target=float(completion_days_target),
                    location=str(location),
                    order_date=order_date,
                )
                plant = PlantConfig(
                    molds_per_day_capacity=int(molds_per_day_capacity),
                    yard_size_available=int(yard_size_available),
                )

                climate: ClimateRecord = resolve_climate(
                    location=location,
                    order_date=order_date,
                    climate_csv=climate_csv or None,
                )

                recommendations = recommend_three_paths(
                    strength_model=strength_model,
                    mix_features=mix_features,
                    order=order,
                    plant=plant,
                    climate=climate,
                    cost_cfg=CostConfig(),
                )

            st.success("Recommendations generated successfully.")
            st.markdown("### Results")
            render_results_table(recommendations)

            st.markdown("### Pathway Logic")
            st.markdown(
                """
1. Estimate minimum curing days per unit from the trained strength model.
2. Apply climate/process equivalent-age effect.
3. Simulate completion under molds/day and yard-space constraints.
4. Compute total cost and rank paths as fastest, balanced, and cheapest.
"""
            )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to generate planning pathways: {exc}")


if __name__ == "__main__":
    main()
