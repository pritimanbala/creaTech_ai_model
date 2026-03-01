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

# Three hardcoded pathways that look realistic and show clear trade-offs.
# Keys must match render_results_table expectations exactly.
DEMO_RECOMMENDATIONS: Dict[str, Dict[str, float | str]] = {
    "Path A – Fastest": {
        "policy_name": "Accelerated Steam Curing",
        "steam_temp_c": 65.0,
        "automation_level": "High",
        "curing_duration_hours_per_day": 20.0,
        "equivalent_age_factor": 2.85,
        "min_curing_days_per_unit": 1.0,
        "min_curing_hours_per_unit": 20.0,
        "total_completion_days": 2.4,
        "completion_deviation_days": -1.6,   # finishes 1.6 days ahead of target
        "total_cost": 48_200.00,
    },
    "Path B – Balanced": {
        "policy_name": "Moderate Steam + Ambient",
        "steam_temp_c": 50.0,
        "automation_level": "Medium",
        "curing_duration_hours_per_day": 14.0,
        "equivalent_age_factor": 1.95,
        "min_curing_days_per_unit": 1.5,
        "min_curing_hours_per_unit": 21.0,
        "total_completion_days": 3.8,
        "completion_deviation_days": -0.2,   # just under target
        "total_cost": 33_750.00,
    },
    "Path C – Cheapest": {
        "policy_name": "Ambient Curing (No Steam)",
        "steam_temp_c": 0.0,
        "automation_level": "Low",
        "curing_duration_hours_per_day": 10.0,
        "equivalent_age_factor": 1.00,
        "min_curing_days_per_unit": 3.0,
        "min_curing_hours_per_unit": 30.0,
        "total_completion_days": 6.1,
        "completion_deviation_days": +2.1,   # 2.1 days over target
        "total_cost": 19_400.00,
    },
}

# ---------------------------------------------------------------------------

if not DEMO_MODE:
    from concrete_strength_dataset_model import MIX_FEATURE_COLUMNS, load_early_age_strength_predictor
    from precast_nsga2_optimization import (
        ClimateRecord,
        CostConfig,
        OrderRequest,
        PlantConfig,
        ensure_trained_model,
        recommend_three_paths,
        resolve_climate_from_api,
    )

st.set_page_config(page_title="Precast Order Planner", page_icon="🏗️", layout="wide")
logger = logging.getLogger(__name__)


def append_web_log(message: str) -> None:
    if "web_logs" not in st.session_state:
        st.session_state["web_logs"] = []
    st.session_state["web_logs"].append(message)
    logger.info(message)


# ── Always define this function, but guard the internals ──────────────────
@st.cache_resource(show_spinner=False)
def get_strength_model(data_path: str, model_path: str):
    if DEMO_MODE:
        return None  # Not used in demo mode, return a no-op
    from concrete_strength_dataset_model import load_early_age_strength_predictor
    from precast_nsga2_optimization import ensure_trained_model
    append_web_log(f"[MODEL] Checking model artifact at: {model_path}")
    model_file = ensure_trained_model(data_path=data_path, model_path=model_path)
    append_web_log(f"[MODEL] Loading model from: {model_file}")
    return load_early_age_strength_predictor(model_file)

def build_mix_inputs() -> Dict[str, float]:
    st.subheader("Mix Design Inputs (kg/m³)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cement_kg             = st.number_input("Cement",           min_value=0.0, value=300.0, step=1.0)
        blast_furnace_slag_kg = st.number_input("Blast Furnace Slag", min_value=0.0, value=120.0, step=1.0)
    with c2:
        fly_ash_kg  = st.number_input("Fly Ash", min_value=0.0, value=60.0,  step=1.0)
        water_kg    = st.number_input("Water",   min_value=0.0, value=180.0, step=1.0)
    with c3:
        superplasticizer_kg = st.number_input("Superplasticizer",  min_value=0.0, value=8.0,   step=0.1)
        coarse_aggregate_kg = st.number_input("Coarse Aggregate",  min_value=0.0, value=950.0, step=1.0)
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
        rows.append(
            {
                "Path":                        key,
                "Policy":                      payload["policy_name"],
                "Steam Temp (°C)":             payload["steam_temp_c"],
                "Automation Level":            payload["automation_level"],
                "Curing Hours/Day":            payload["curing_duration_hours_per_day"],
                "Equivalent Age Factor":       payload["equivalent_age_factor"],
                "Min Curing Days/Unit":        payload["min_curing_days_per_unit"],
                "Min Curing Hours/Unit":       payload["min_curing_hours_per_unit"],
                "Total Completion Days":       payload["total_completion_days"],
                "Deviation vs Target (days)":  payload["completion_deviation_days"],
                "Total Cost":                  payload["total_cost"],
            }
        )

    df = pd.DataFrame(rows)

    # Colour-code deviation column: green = ahead, red = over target
    def colour_deviation(val):
        colour = "color: green" if val <= 0 else "color: red"
        return colour

    styled = df.style.map(colour_deviation, subset=["Deviation vs Target (days)"])
    st.dataframe(styled, use_container_width=True)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    st.title("Precast Yard Order Planning Dashboard")

    # if DEMO_MODE:
    #     st.info(
    #         "🟡 **Demo Mode** — results are representative hardcoded values. "
    #         "Set `DEMO_MODE = False` in the source to enable live model & weather API.",
    #         icon="🏗️",
    #     )
    # else:
    #     st.caption("Model is trained once from concrete_data.xlsx, then loaded from saved artifact in future runs.")

    st.sidebar.header("Model Configuration (Ignore this in the web version)")
    data_path  = st.sidebar.text_input("Initial training data path", value="concrete_data.xlsx")
    model_path = st.sidebar.text_input("Model artifact path", value="artifacts/concrete_strength_age_model.joblib")

    st.subheader("Order Inputs")
    left, right = st.columns(2)

    with left:
        units_ordered          = st.number_input("Units Ordered",              min_value=1,    value=20,    step=1)
        mold_volume_m3         = st.number_input("Volume of Mold (m³)",        min_value=0.01, value=1.0,   step=0.01)
        mold_area_m2           = st.number_input("Area of Mold (m²)",          min_value=0.01, value=2.2,   step=0.01)
        completion_days_target = st.number_input("Days of Completion Target",  min_value=0.1,  value=4.0,   step=0.1)
        required_strength_mpa  = st.number_input("Required Strength (MPa)",    min_value=0.1,  value=9.0,   step=0.1)

    with right:
        latitude            = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=13.0827, step=0.0001)
        longitude           = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.2707, step=0.0001)
        order_date          = st.date_input("Date (for 4-day weather average)", value=date.today())
        yard_size_available = st.number_input("Yard Size Available (molds)", min_value=1, value=150, step=1)
        molds_per_day_capacity = st.number_input("Molds Produced per Day",   min_value=1, value=50,  step=1)

    mix_features = build_mix_inputs()

    st.markdown("### Process Console")
    log_placeholder = st.empty()

    if st.session_state.get("web_logs"):
        log_placeholder.code("\n".join(st.session_state.get("web_logs", [])), language="text")

    if st.button("Generate Planning Pathways", type="primary"):
        st.session_state["web_logs"] = []
        try:
            with st.spinner("Loading model and fetching 4-day weather..."):
                append_web_log("[START] Beginning planning pipeline")

                if DEMO_MODE:
                    # ── DEMO BRANCH ───────────────────────────────────────────
                    append_web_log("[MODEL] Demo mode — skipping model training")
                    append_web_log(
                        f"[WEATHER] Demo climate: "
                        f"Avg Temp={DEMO_CLIMATE['ambient_temp_c']:.2f}°C  "
                        f"Avg Humidity={DEMO_CLIMATE['humidity_pct']:.2f}%"
                    )
                    append_web_log("[PLANNER] Evaluating policies and computing pathways (demo data)")
                    recommendations = DEMO_RECOMMENDATIONS
                    climate_temp    = DEMO_CLIMATE["ambient_temp_c"]
                    climate_hum     = DEMO_CLIMATE["humidity_pct"]
                    append_web_log("[DONE] Demo recommendations loaded")
                    # ─────────────────────────────────────────────────────────
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
                    climate = resolve_climate_from_api(order.latitude, order.longitude, order.order_date)
                    climate_temp = climate.ambient_temp_c
                    climate_hum  = climate.humidity_pct
                    append_web_log(
                        f"[WEATHER] Avg Temp={climate_temp:.2f}°C  Avg Humidity={climate_hum:.2f}%"
                    )
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

            st.markdown("### Results")
            df = render_results_table(recommendations)

            # ── Climate metrics ────────────────────────────────────────────────
            st.markdown("### Climate Used (4-day Average)")
            m1, m2 = st.columns(2)
            m1.metric("Mean Temperature (°C)", f"{climate_temp:.2f}")
            m2.metric("Mean Humidity (%)",      f"{climate_hum:.2f}")

            # ── Charts ─────────────────────────────────────────────────────────
            st.markdown("### Dashboard Views")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Total Completion Days**")
                st.bar_chart(df.set_index("Path")[["Total Completion Days"]])
            with c2:
                st.markdown("**Total Cost (₹)**")
                st.bar_chart(df.set_index("Path")[["Total Cost"]])
            with c3:
                st.markdown("**Deviation vs Target (days)**")
                st.bar_chart(df.set_index("Path")[["Deviation vs Target (days)"]])

            # ── Legend ─────────────────────────────────────────────────────────
            st.markdown("### Pathway Logic")
            st.markdown(
                """
1. **Train once** – model artifact is reused on all subsequent runs.
2. **Weather fetch** – 4-day forecast averaged for temperature & humidity.
3. **Equivalent-age factor** – adjusts curing time for steam temperature.
4. **Throughput simulation** – daily mold production and yard-occupancy limits applied.
5. **Three pathways reported** – Fastest (lowest days), Balanced (days ≈ target, moderate cost), Cheapest (lowest cost).
"""
            )

        except Exception as exc:  # noqa: BLE001
            append_web_log(f"[ERROR] {exc}")
            log_placeholder.code("\n".join(st.session_state.get("web_logs", [])), language="text")
            st.error(f"Unable to generate planning pathways: {exc}")


if __name__ == "__main__":
    main()