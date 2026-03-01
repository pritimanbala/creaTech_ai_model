# Precast Order Planning with `concrete_data.csv`

This repository now supports model-driven order planning for precast production.

## What the system can do

1. Train a concrete strength model from `concrete_data.csv`.
2. Estimate the **actual minimum curing time** to reach a required strength (not capped at 48 hours).
3. Respect day/hour conversion explicitly (`1 day = 24 hours`).
4. Simulate production with real constraints:
   - molds produced per day
   - total yard space available (concurrent molds in curing)
5. Return **three decision pathways** for an order:
   - fastest path
   - balanced speed/cost path
   - cheapest path

## Dataset schema expected

- Cement (component 1)(kg in a m^3 mixture)
- Blast Furnace Slag (component 2)(kg in a m^3 mixture)
- Fly Ash (component 3)(kg in a m^3 mixture)
- Water  (component 4)(kg in a m^3 mixture)
- Superplasticizer (component 5)(kg in a m^3 mixture)
- Coarse Aggregate  (component 6)(kg in a m^3 mixture)
- Fine Aggregate (component 7)(kg in a m^3 mixture)
- Age (day)
- Concrete compressive strength(MPa, megapascals)

## Run planning

```bash
python precast_nsga2_optimization.py \
  --data concrete_data.csv \
  --model-path artifacts/concrete_strength_age_model.joblib \
  --retrain-model \
  --units-ordered 20 \
  --mold-volume 1.0 \
  --mold-area 2.2 \
  --required-strength 9 \
  --completion-days 4 \
  --location chennai \
  --date 2026-03-01 \
  --yard-size 150 \
  --molds-per-day 50 \
  --mix-json '{"cement_kg":300,"blast_furnace_slag_kg":120,"fly_ash_kg":60,"water_kg":180,"superplasticizer_kg":8,"coarse_aggregate_kg":950,"fine_aggregate_kg":780}'
```

Optional:

- `--climate-csv climate_lookup.csv` with columns: `location,date,ambient_temp_c,humidity_pct`

## Output

JSON output includes:
- full input summary
- recommended paths (`fastest_path`, `balanced_path`, `cheapest_path`)
- each path's curing time, completion time, deviation vs target, and total cost
- pathway logic used by the planner

## Streamlit Web Application

A UI is available to enter the order inputs and get recommended pathways.

Run:

```bash
streamlit run streamlit_app.py
```

In the app, users can provide:
- units ordered
- mold volume
- mold area
- completion days target
- location
- yard size available
- date (for climate)
- molds per day capacity
- mix design variables

Output includes:
- fastest path
- balanced path
- cheapest path
- completion deviation and total cost for each pathway
