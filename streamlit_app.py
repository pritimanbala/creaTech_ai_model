
#!/usr/bin/env python3
"""Streamlit UI for precast order planning."""

from __future__ import annotations

from datetime import date
import logging
from typing import Dict

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# DEMO MODE — set DEMO_MODE = False to re-enable live model & API calls
# ---------------------------------------------------------------------------
DEMO_MODE = True

DEMO_CLIMATE = {
    "ambient_temp_c": 28.50,
    "humidity_pct": 72.30,
}


# ---------------------------------------------------------------------------
# Dynamic cost engine — all values respond to user inputs
# ---------------------------------------------------------------------------

def compute_demo_recommendations(
    units_ordered: int,
    mold_volume_m3: float,
    mold_area_m2: float,
    completion_days_target: float,
    required_strength_mpa: float,
    molds_per_day_capacity: int,
    yard_size_available: int,
    cement_kg: float,
    blast_furnace_slag_kg: float,
    fly_ash_kg: float,
    water_kg: float,
    superplasticizer_kg: float,
    coarse_aggregate_kg: float,
    fine_aggregate_kg: float,
) -> Dict[str, Dict[str, float | str]]:
    """
    Realistic cost formula that updates whenever any input changes.

    Material unit prices (Rs/kg) — roughly calibrated to Indian precast market:
      Cement              Rs 7.50/kg   — most sensitive input
      Blast Furnace Slag  Rs 4.20/kg
      Fly Ash             Rs 2.80/kg
      Water               Rs 0.05/kg
      Superplasticizer    Rs 85.0/kg   — expensive additive
      Coarse Aggregate    Rs 0.90/kg
      Fine Aggregate      Rs 0.80/kg
    """

    # Material cost per m3 of concrete
    material_cost_per_m3 = (
        cement_kg             * 7.50  +
        blast_furnace_slag_kg * 4.20  +
        fly_ash_kg            * 2.80  +
        water_kg              * 0.05  +
        superplasticizer_kg   * 85.00 +
        coarse_aggregate_kg   * 0.90  +
        fine_aggregate_kg     * 0.80
    )

    # Per-unit base cost
    volume_cost      = mold_volume_m3 * material_cost_per_m3
    area_cost        = mold_area_m2   * 420.0
    strength_premium = required_strength_mpa * 180.0

    base_unit_cost = volume_cost + area_cost + strength_premium

    # Mild economies of scale (max 15% saving at large volumes)
    scale_factor = max(0.85, 1.0 - (units_ordered - 1) * 0.0015)
    unit_cost    = base_unit_cost * scale_factor

    # Curing duration driven by w/c ratio and strength target
    wc_ratio         = water_kg / max(cement_kg, 1.0)
    base_curing_days = 1.0 + wc_ratio * 3.5 + required_strength_mpa * 0.05

    # Throughput batches
    effective_daily = min(molds_per_day_capacity, yard_size_available)
    batches_needed  = max(1.0, units_ordered / max(effective_daily, 1))

    # PATH A - Fastest (steam 65C, high automation)
    eq_age_a      = 2.85
    curing_days_a = round(base_curing_days / eq_age_a, 2)
    curing_hrs_a  = round(curing_days_a * 20.0, 1)
    completion_a  = round(curing_days_a + batches_needed * 0.60, 2)
    steam_cost_a  = mold_area_m2 * 18.0 * 65.0
    auto_cost_a   = units_ordered * 95.0
    total_cost_a  = round((unit_cost + steam_cost_a) * units_ordered + auto_cost_a, 2)
    deviation_a   = round(completion_a - completion_days_target, 2)

    # PATH B - Balanced (steam 50C, medium automation)
    eq_age_b      = 1.95
    curing_days_b = round(base_curing_days / eq_age_b, 2)
    curing_hrs_b  = round(curing_days_b * 14.0, 1)
    completion_b  = round(curing_days_b + batches_needed * 0.80, 2)
    steam_cost_b  = mold_area_m2 * 14.0 * 50.0
    auto_cost_b   = units_ordered * 55.0
    total_cost_b  = round((unit_cost + steam_cost_b) * units_ordered + auto_cost_b, 2)
    deviation_b   = round(completion_b - completion_days_target, 2)

    # PATH C - Cheapest (ambient, no steam, low automation)
    eq_age_c      = 1.00
    curing_days_c = round(base_curing_days / eq_age_c, 2)
    curing_hrs_c  = round(curing_days_c * 10.0, 1)
    completion_c  = round(curing_days_c + batches_needed * 1.10, 2)
    total_cost_c  = round(unit_cost * units_ordered * 0.72, 2)
    deviation_c   = round(completion_c - completion_days_target, 2)

    return {
        "Path A - Fastest": {
            "policy_name":                   "Accelerated Steam Curing",
            "steam_temp_c":                  65.0,
            "automation_level":              "High",
            "curing_duration_hours_per_day": 20.0,
            "equivalent_age_factor":         eq_age_a,
            "min_curing_days_per_unit":      curing_days_a,
            "min_curing_hours_per_unit":     curing_hrs_a,
            "total_completion_days":         completion_a,
            "completion_deviation_days":     deviation_a,
            "total_cost":                    total_cost_a,
        },
        "Path B - Balanced": {
            "policy_name":                   "Moderate Steam + Ambient",
            "steam_temp_c":                  50.0,
            "automation_level":              "Medium",
            "curing_duration_hours_per_day": 14.0,
            "equivalent_age_factor":         eq_age_b,
            "min_curing_days_per_unit":      curing_days_b,
            "min_curing_hours_per_unit":     curing_hrs_b,
            "total_completion_days":         completion_b,
            "completion_deviation_days":     deviation_b,
            "total_cost":                    total_cost_b,
        },
        "Path C - Cheapest": {
            "policy_name":                   "Ambient Curing (No Steam)",
            "steam_temp_c":                  0.0,
            "automation_level":              "Low",
            "curing_duration_hours_per_day": 10.0,
            "equivalent_age_factor":         eq_age_c,
            "min_curing_days_per_unit":      curing_days_c,
            "min_curing_hours_per_unit":     curing_hrs_c,
            "total_completion_days":         completion_c,
            "completion_deviation_days":     deviation_c,
            "total_cost":                    total_cost_c,
        },
    }


st.set_page_config(page_title="Precast Order Planner", page_icon="🏗️", layout="wide")
logger = logging.getLogger(__name__)


def append_web_log(message: str) -> None:
    if "web_logs" not in st.session_state:
        st.session_state["web_logs"] = []
    st.session_state["web_logs"].append(message)
    logger.info(message)


@st.cache_resource(show_spinner=False)
def get_strength_model(data_path: str, model_path: str):
    if DEMO_MODE:
        return None
    from concrete_strength_dataset_model import load_early_age_strength_predictor
    from precast_nsga2_optimization import ensure_trained_model
    append_web_log(f"[MODEL] Checking model artifact at: {model_path}")
    model_file = ensure_trained_model(data_path=data_path, model_path=model_path)
    append_web_log(f"[MODEL] Loading model from: {model_file}")
    return load_early_age_strength_predictor(model_file)


def build_mix_inputs() -> Dict[str, float]:
    st.subheader("Mix Design Inputs (kg/m3)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cement_kg             = st.number_input("Cement",            min_value=0.0, value=300.0, step=1.0)
        blast_furnace_slag_kg = st.number_input("Blast Furnace Slag", min_value=0.0, value=120.0, step=1.0)
    with c2:
        fly_ash_kg  = st.number_input("Fly Ash", min_value=0.0, value=60.0,  step=1.0)
        water_kg    = st.number_input("Water",   min_value=0.0, value=180.0, step=1.0)
    with c3:
        superplasticizer_kg = st.number_input("Superplasticizer", min_value=0.0, value=8.0,   step=0.1)
        coarse_aggregate_kg = st.number_input("Coarse Aggregate", min_value=0.0, value=950.0, step=1.0)
    with c4:
        fine_aggregate_kg = st.number_input("Fine Aggregate", min_value=0.0, value=780.0, step=1.0)

    return {
        "cement_kg":             cement_kg,
        "blast_furnace_slag_kg": blast_furnace_slag_kg,
        "fly_ash_kg":            fly_ash_kg,
        "water_kg":              water_kg,
        "superplasticizer_kg":   superplasticizer_kg,
        "coarse_aggregate_kg":   coarse_aggregate_kg,
        "fine_aggregate_kg":     fine_aggregate_kg,
    }


def render_results_table(result_map: Dict[str, Dict[str, float | str]]) -> pd.DataFrame:
    rows = []
    for key, payload in result_map.items():
        rows.append({
            "Path":                       key,
            "Policy":                     payload["policy_name"],
            "Steam Temp (C)":             payload["steam_temp_c"],
            "Automation Level":           payload["automation_level"],
            "Curing Hours/Day":           payload["curing_duration_hours_per_day"],
            "Equivalent Age Factor":      payload["equivalent_age_factor"],
            "Min Curing Days/Unit":       payload["min_curing_days_per_unit"],
            "Min Curing Hours/Unit":      payload["min_curing_hours_per_unit"],
            "Total Completion Days":      payload["total_completion_days"],
            "Deviation vs Target (days)": payload["completion_deviation_days"],
            "Total Cost (Rs)":            payload["total_cost"],
        })

    df = pd.DataFrame(rows)

    def colour_deviation(val):
        return "color: green" if val <= 0 else "color: red"

    styled = df.style.map(colour_deviation, subset=["Deviation vs Target (days)"])
    st.dataframe(styled, use_container_width=True)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    st.title("Precast Yard Order Planning Dashboard")

    # if DEMO_MODE:
    #     st.info(
    #         "Demo Mode — costs and schedule update live as you change any input above. "
    #         "Set DEMO_MODE = False to enable live model and weather API.",
    #         icon="🏗️",
    #     )
    # else:
    #     st.caption("Model is trained once from concrete_data.xlsx, then loaded from saved artifact in future runs.")

    # st.sidebar.header("Model Configuration")
    # data_path  = st.sidebar.text_input("Initial training data path", value="concrete_data.xlsx")
    # model_path = st.sidebar.text_input("Model artifact path", value="artifacts/concrete_strength_age_model.joblib")
    data_path  = "concrete_data.xlsx"
    model_path = "artifacts/concrete_strength_age_model.joblib"

    st.subheader("Order Inputs")
    left, right = st.columns(2)

    with left:
        units_ordered          = st.number_input("Units Ordered",             min_value=1,    value=20,   step=1)
        mold_volume_m3         = st.number_input("Volume of Mold (m3)",       min_value=0.01, value=1.0,  step=0.01)
        mold_area_m2           = st.number_input("Area of Mold (m2)",         min_value=0.01, value=2.2,  step=0.01)
        completion_days_target = st.number_input("Days of Completion Target", min_value=0.1,  value=4.0,  step=0.1)
        required_strength_mpa  = st.number_input("Required Strength (MPa)",   min_value=0.1,  value=9.0,  step=0.1)

    with right:
        latitude               = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=13.0827, step=0.0001)
        longitude              = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.2707, step=0.0001)
        order_date             = st.date_input("Date (for 4-day weather average)", value=date.today())
        yard_size_available    = st.number_input("Yard Size Available (molds)", min_value=1, value=150, step=1)
        molds_per_day_capacity = st.number_input("Molds Produced per Day",      min_value=1, value=50,  step=1)

    mix_features = build_mix_inputs()

    # In demo mode: recompute and redraw on every input change (no button needed)
    if DEMO_MODE:
        live_recs = compute_demo_recommendations(
            units_ordered=int(units_ordered),
            mold_volume_m3=float(mold_volume_m3),
            mold_area_m2=float(mold_area_m2),
            completion_days_target=float(completion_days_target),
            required_strength_mpa=float(required_strength_mpa),
            molds_per_day_capacity=int(molds_per_day_capacity),
            yard_size_available=int(yard_size_available),
            **mix_features,
        )

        st.markdown("---")
        st.markdown("### Live Results  *(updates as you change inputs)*")
        live_df = render_results_table(live_recs)

        st.markdown("### Climate (Demo Average)")
        m1, m2 = st.columns(2)
        m1.metric("Mean Temperature (C)", f"{DEMO_CLIMATE['ambient_temp_c']:.2f}")
        m2.metric("Mean Humidity (%)",     f"{DEMO_CLIMATE['humidity_pct']:.2f}")

        st.markdown("### Dashboard Views")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Total Completion Days**")
            st.bar_chart(live_df.set_index("Path")[["Total Completion Days"]])
        with c2:
            st.markdown("**Total Cost (Rs)**")
            st.bar_chart(live_df.set_index("Path")[["Total Cost (Rs)"]])
        with c3:
            st.markdown("**Deviation vs Target (days)**")
            st.bar_chart(live_df.set_index("Path")[["Deviation vs Target (days)"]])

    # Process console + generate button (always shown; runs full pipeline in live mode)
    st.markdown("---")
    st.markdown("### Process Console")
    log_placeholder = st.empty()

    if st.session_state.get("web_logs"):
        log_placeholder.code("\n".join(st.session_state["web_logs"]), language="text")

    if st.button("Generate Planning Pathways", type="primary"):
        st.session_state["web_logs"] = []
        try:
            with st.spinner("Loading model and fetching 4-day weather..."):
                append_web_log("[START] Beginning planning pipeline")

                if DEMO_MODE:
                    append_web_log("[MODEL] Demo mode — skipping model training")
                    append_web_log(
                        f"[WEATHER] Demo climate: "
                        f"Avg Temp={DEMO_CLIMATE['ambient_temp_c']:.2f}C  "
                        f"Avg Humidity={DEMO_CLIMATE['humidity_pct']:.2f}%"
                    )
                    append_web_log("[PLANNER] Evaluating policies and computing pathways")
                    recommendations = compute_demo_recommendations(
                        units_ordered=int(units_ordered),
                        mold_volume_m3=float(mold_volume_m3),
                        mold_area_m2=float(mold_area_m2),
                        completion_days_target=float(completion_days_target),
                        required_strength_mpa=float(required_strength_mpa),
                        molds_per_day_capacity=int(molds_per_day_capacity),
                        yard_size_available=int(yard_size_available),
                        **mix_features,
                    )
                    climate_temp = DEMO_CLIMATE["ambient_temp_c"]
                    climate_hum  = DEMO_CLIMATE["humidity_pct"]
                    append_web_log("[DONE] Recommendations generated")

                else:
                    strength_model = get_strength_model(data_path=data_path, model_path=model_path)
                    append_web_log("[MODEL] Model ready")
                    from precast_nsga2_optimization import (
                        ClimateRecord, CostConfig, OrderRequest, PlantConfig,
                        recommend_three_paths, resolve_climate_from_api,
                    )
                    order = OrderRequest(
                        units_ordered=int(units_ordered),
                        mold_volume_m3=float(mold_volume_m3),
                        mold_area_m2=float(mold_area_m2),
                        required_strength_mpa=float(required_strength_mpa),
                        completion_days_target=float(completion_days_target),
                        latitude=float(latitude),
                        longitude=float(longitude),
                        order_date=order_date,
                    )
                    plant = PlantConfig(
                        molds_per_day_capacity=int(molds_per_day_capacity),
                        yard_size_available=int(yard_size_available),
                    )
                    append_web_log(
                        f"[WEATHER] Fetching 4-day weather for "
                        f"lat={order.latitude}, lon={order.longitude}, date={order.order_date}"
                    )
                    climate      = resolve_climate_from_api(order.latitude, order.longitude, order.order_date)
                    climate_temp = climate.ambient_temp_c
                    climate_hum  = climate.humidity_pct
                    append_web_log(f"[WEATHER] Avg Temp={climate_temp:.2f}C  Avg Humidity={climate_hum:.2f}%")
                    append_web_log("[PLANNER] Evaluating policies and computing pathways")
                    recommendations = recommend_three_paths(
                        strength_model=strength_model,
                        mix_features=mix_features,
                        order=order,
                        plant=plant,
                        climate=climate,
                        cost_cfg=CostConfig(),
                    )
                    append_web_log("[DONE] Recommendations generated")

            log_placeholder.code(
                "\n".join(st.session_state.get("web_logs", [])) or "No logs yet.",
                language="text",
            )
            st.success("Recommendations generated successfully.")
            st.markdown("### Final Results")
            render_results_table(recommendations)

            st.markdown("### Climate Used (4-day Average)")
            m1, m2 = st.columns(2)
            m1.metric("Mean Temperature (C)", f"{climate_temp:.2f}")
            m2.metric("Mean Humidity (%)",     f"{climate_hum:.2f}")

        except Exception as exc:
            append_web_log(f"[ERROR] {exc}")
            log_placeholder.code("\n".join(st.session_state.get("web_logs", [])), language="text")
            st.error(f"Unable to generate planning pathways: {exc}")

    st.markdown("### Pathway Logic")
    st.markdown(
        """
1. **Train once** - model artifact is reused on all subsequent runs.
2. **Weather fetch** - 4-day forecast averaged for temperature and humidity.
3. **Equivalent-age factor** - adjusts curing time for steam temperature.
4. **Throughput simulation** - daily mold production and yard-occupancy limits applied.
5. **Three pathways reported** - Fastest (lowest days), Balanced (days near target, moderate cost), Cheapest (lowest cost).
"""
    )


if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """Streamlit UI for precast order planning."""

# from __future__ import annotations

# from datetime import date
# import logging
# from typing import Dict

# import pandas as pd
# import streamlit as st

# from concrete_strength_dataset_model import MIX_FEATURE_COLUMNS, load_early_age_strength_predictor
# from precast_nsga2_optimization import (
#     ClimateRecord,
#     CostConfig,
#     OrderRequest,
#     PlantConfig,
#     ensure_trained_model,
#     recommend_three_paths,
#     resolve_climate_from_api,
# )

# st.set_page_config(page_title="Precast Order Planner", page_icon="🏗️", layout="wide")
# logger = logging.getLogger(__name__)




# def append_web_log(message: str) -> None:
#     if "web_logs" not in st.session_state:
#         st.session_state["web_logs"] = []
#     st.session_state["web_logs"].append(message)
#     logger.info(message)

# @st.cache_resource(show_spinner=False)
# def get_strength_model(data_path: str, model_path: str):
#     append_web_log(f"[MODEL] Checking model artifact at: {model_path}")
#     model_file = ensure_trained_model(data_path=data_path, model_path=model_path)
#     append_web_log(f"[MODEL] Loading model from: {model_file}")
#     return load_early_age_strength_predictor(model_file)


# def build_mix_inputs() -> Dict[str, float]:
#     st.subheader("Mix Design Inputs (kg/m³)")
#     c1, c2, c3, c4 = st.columns(4)
#     with c1:
#         cement_kg = st.number_input("Cement", min_value=0.0, value=300.0, step=1.0)
#         blast_furnace_slag_kg = st.number_input("Blast Furnace Slag", min_value=0.0, value=120.0, step=1.0)
#     with c2:
#         fly_ash_kg = st.number_input("Fly Ash", min_value=0.0, value=60.0, step=1.0)
#         water_kg = st.number_input("Water", min_value=0.0, value=180.0, step=1.0)
#     with c3:
#         superplasticizer_kg = st.number_input("Superplasticizer", min_value=0.0, value=8.0, step=0.1)
#         coarse_aggregate_kg = st.number_input("Coarse Aggregate", min_value=0.0, value=950.0, step=1.0)
#     with c4:
#         fine_aggregate_kg = st.number_input("Fine Aggregate", min_value=0.0, value=780.0, step=1.0)

#     return {
#         "cement_kg": cement_kg,
#         "blast_furnace_slag_kg": blast_furnace_slag_kg,
#         "fly_ash_kg": fly_ash_kg,
#         "water_kg": water_kg,
#         "superplasticizer_kg": superplasticizer_kg,
#         "coarse_aggregate_kg": coarse_aggregate_kg,
#         "fine_aggregate_kg": fine_aggregate_kg,
#     }


# def render_results_table(result_map: Dict[str, Dict[str, float | str]]) -> pd.DataFrame:
#     rows = []
#     for key, payload in result_map.items():
#         rows.append(
#             {
#                 "Path": key,
#                 "Policy": payload["policy_name"],
#                 "Steam Temp (°C)": payload["steam_temp_c"],
#                 "Automation Level": payload["automation_level"],
#                 "Curing Hours/Day": payload["curing_duration_hours_per_day"],
#                 "Equivalent Age Factor": payload["equivalent_age_factor"],
#                 "Min Curing Days/Unit": payload["min_curing_days_per_unit"],
#                 "Min Curing Hours/Unit": payload["min_curing_hours_per_unit"],
#                 "Total Completion Days": payload["total_completion_days"],
#                 "Deviation vs Target (days)": payload["completion_deviation_days"],
#                 "Total Cost": payload["total_cost"],
#             }
#         )

#     df = pd.DataFrame(rows)
#     st.dataframe(df, use_container_width=True)
#     return df


# def main() -> None:
#     logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
#     st.title("Precast Yard Order Planning Dashboard")
#     st.caption("Model is trained once from concrete_data.xlsx, then loaded from saved artifact in future runs.")

#     st.sidebar.header("Model Configuration")
#     data_path = st.sidebar.text_input("Initial training data path", value="concrete_data.xlsx")
#     model_path = st.sidebar.text_input("Model artifact path", value="artifacts/concrete_strength_age_model.joblib")

#     st.subheader("Order Inputs")
#     left, right = st.columns(2)

#     with left:
#         units_ordered = st.number_input("Units Ordered", min_value=1, value=20, step=1)
#         mold_volume_m3 = st.number_input("Volume of Mold (m³)", min_value=0.01, value=1.0, step=0.01)
#         mold_area_m2 = st.number_input("Area of Mold (m²)", min_value=0.01, value=2.2, step=0.01)
#         completion_days_target = st.number_input("Days of Completion Target", min_value=0.1, value=4.0, step=0.1)
#         required_strength_mpa = st.number_input("Required Strength (MPa)", min_value=0.1, value=9.0, step=0.1)

#     with right:
#         latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=13.0827, step=0.0001)
#         longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.2707, step=0.0001)
#         order_date = st.date_input("Date (for 4-day weather average)", value=date.today())
#         yard_size_available = st.number_input("Yard Size Available (molds)", min_value=1, value=150, step=1)
#         molds_per_day_capacity = st.number_input("Molds Produced per Day", min_value=1, value=50, step=1)

#     mix_features = build_mix_inputs()

#     st.markdown("### Process Console")
#     log_placeholder = st.empty()

#     if st.session_state.get("web_logs"):
#         log_placeholder.code("\n".join(st.session_state.get("web_logs", [])), language="text")

#     if st.button("Generate Planning Pathways", type="primary"):
#         st.session_state["web_logs"] = []
#         try:
#             with st.spinner("Loading model and fetching 4-day weather..."):
#                 append_web_log("[START] Beginning planning pipeline")
#                 strength_model = get_strength_model(data_path=data_path, model_path=model_path)
#                 append_web_log("[MODEL] Model ready")
#                 order = OrderRequest(
#                     units_ordered=int(units_ordered),
#                     mold_volume_m3=float(mold_volume_m3),
#                     mold_area_m2=float(mold_area_m2),
#                     required_strength_mpa=float(required_strength_mpa),
#                     completion_days_target=float(completion_days_target),
#                     latitude=float(latitude),
#                     longitude=float(longitude),
#                     order_date=order_date,
#                 )
#                 plant = PlantConfig(
#                     molds_per_day_capacity=int(molds_per_day_capacity),
#                     yard_size_available=int(yard_size_available),
#                 )
#                 append_web_log(f"[WEATHER] Fetching 4-day weather for lat={order.latitude}, lon={order.longitude}, date={order.order_date}")
#                 climate: ClimateRecord = resolve_climate_from_api(order.latitude, order.longitude, order.order_date)
#                 append_web_log(f"[WEATHER] Avg Temp={climate.ambient_temp_c:.2f}C Avg Humidity={climate.humidity_pct:.2f}%")

#                 append_web_log("[PLANNER] Evaluating policies and computing pathways")
#                 recommendations = recommend_three_paths(
#                     strength_model=strength_model,
#                     mix_features=mix_features,
#                     order=order,
#                     plant=plant,
#                     climate=climate,
#                     cost_cfg=CostConfig(),
#                 )
#                 append_web_log("[DONE] Recommendations generated")

#             log_placeholder.code("\n".join(st.session_state.get("web_logs", [])) or "No logs yet.", language="text")
#             st.success("Recommendations generated successfully.")
#             st.markdown("### Results")
#             df = render_results_table(recommendations)

#             st.markdown("### Climate Used (4-day Average)")
#             st.metric("Mean Temperature (°C)", f"{climate.ambient_temp_c:.2f}")
#             st.metric("Mean Humidity (%)", f"{climate.humidity_pct:.2f}")

#             st.markdown("### Dashboard Views")
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.bar_chart(df.set_index("Path")[["Total Completion Days"]])
#             with c2:
#                 st.bar_chart(df.set_index("Path")[["Total Cost"]])

#             st.markdown("### Pathway Logic")
#             st.markdown(
#                 """
# 1. Train model only on first run if artifact is missing, then always reuse the saved model.
# 2. Fetch weather forecast for 4 days from latitude/longitude and compute average temperature and humidity.
# 3. Estimate minimum curing time from model and convert using equivalent-age factor.
# 4. Simulate throughput with daily mold production and yard occupancy limits.
# 5. Report fastest, balanced, and cheapest pathways.
# """
#             )
#         except Exception as exc:  # noqa: BLE001
#             append_web_log(f"[ERROR] {exc}")
#             log_placeholder.code("\n".join(st.session_state.get("web_logs", [])), language="text")
#             st.error(f"Unable to generate planning pathways: {exc}")


# if __name__ == "__main__":
#     main()





