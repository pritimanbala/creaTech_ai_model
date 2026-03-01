# Precast Order Planning Dashboard

This project provides a production planning system for precast operations using a trained concrete strength model.

## Key behavior now

1. **First run training only**
   - The app/CLI trains a model from `concrete_data.xlsx` (or `.csv`) only when model artifact does not exist.
   - Next runs load the saved model directly (no retraining during normal user flow).

2. **Model-driven curing time**
   - Finds actual minimum curing time needed to reach required strength.
   - No 48-hour hard cap.
   - Uses strict hour/day conversion (`1 day = 24 hours`).

3. **Weather from latitude/longitude**
   - Takes latitude + longitude as input.
   - Calls Open-Meteo API for next 4 days from selected date.
   - Uses average temperature and humidity from those 4 days.

4. **Yard-aware scheduling**
   - Uses molds/day + yard-size occupancy constraints.
   - Captures waiting behavior when yard is full.

5. **Three pathways output**
   - Fastest
   - Balanced
   - Cheapest

## Dataset schema expected

- Cement (component 1)(kg in a m^3 mixture)
- Blast Furnace Slag (component 2)(kg in a m^3 mixture)
- Fly Ash (component 3)(kg in a m^3 mixture)
- Water  (component 4)(kg in a m^3 mixture)
- Superplasticizer (component 5)(kg in a m^3 mixture)
- Coarse Aggregate  (component 6)(kg in a m^3 mixture)
- Fine Aggregate (component 7)(kg in a m^3 mixture)
- Age (day)
- Concrete compressive strength(MPa)

## Streamlit app

```bash
streamlit run streamlit_app.py
```

In the dashboard, provide:
- units ordered
- mold volume
- mold area
- completion days
- required strength
- latitude and longitude
- order date
- yard size available
- molds produced per day
- mix design values

The dashboard displays:
- 4-day average weather used
- pathway comparison table
- completion-time chart
- cost chart
- planning logic

## CLI usage

```bash
python precast_nsga2_optimization.py \
  --data concrete_data.xlsx \
  --model-path artifacts/concrete_strength_age_model.joblib \
  --units-ordered 20 \
  --mold-volume 1.0 \
  --mold-area 2.2 \
  --required-strength 9 \
  --completion-days 4 \
  --latitude 13.0827 \
  --longitude 80.2707 \
  --date 2026-03-01 \
  --yard-size 150 \
  --molds-per-day 50 \
  --mix-json '{"cement_kg":300,"blast_furnace_slag_kg":120,"fly_ash_kg":60,"water_kg":180,"superplasticizer_kg":8,"coarse_aggregate_kg":950,"fine_aggregate_kg":780}'
```

The command prints JSON with full input summary, climate averages, and all three recommended pathways.
