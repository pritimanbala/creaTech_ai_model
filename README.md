# Industrial Concrete Multi-Objective Optimization (NSGA-II)

This project now includes a clean architecture for true multi-objective optimization:

- **f1:** Minimize total cost
- **f2:** Minimize time to reach required strength
- Subject to:
  - Predicted strength ≥ required strength
  - Water/Cement ratio within engineering limits

## Module Structure

- `data_loader.py` - bounds/config/data utilities
- `models.py` - Strength Model A, Strength Model B, combined predictor, cost/time models
- `optimization.py` - pymoo NSGA-II problem/callback/plotting/logging
- `evaluation.py` - feasible filtering, balanced selection, diagnostics
- `main.py` - orchestration and outputs

## Engineering Bounds

- Cement: 250–500 kg/m³
- Water: 120–220 kg/m³
- SCM: 0–200 kg/m³
- Aggregates: 1200–2000 kg/m³
- Temperature: 15–40°C
- RH: 40–100%
- Curing duration: 1–14 days
- W/C ratio: 0.30–0.60

## Run

```bash
python main.py --required-strength 30 --population 80 --generations 60 --seed 42
```

## Outputs

- `optimization_log.txt` (full generation-by-generation logs)
- `artifacts_optimization/decision_table.csv`
- `artifacts_optimization/pareto_front.png`
- `artifacts_optimization/objective_scatter.png`
- `artifacts_optimization/optimization_summary.json`

## Logging Transparency

The optimizer logs:

- population initialization and generation progression
- objective values and constraint violations
- feasible vs infeasible counts
- Pareto front snapshots by generation
- cheapest / fastest / balanced solution selection reasons
- exact variable values and constraint margins for selected solutions
- diagnostics: cost-time correlation, objective conflict, diversity metric

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```
