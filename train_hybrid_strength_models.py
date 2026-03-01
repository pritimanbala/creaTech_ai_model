#!/usr/bin/env python3
"""Train hybrid Physics + ML models for early-age concrete strength prediction.

This script computes the Nurse-Saul maturity index and trains separate
XGBoost regressors for predicting concrete strength at 8h, 12h, and 24h.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

TARGET_COLUMNS = ["strength_8h", "strength_12h", "strength_24h"]

logger = logging.getLogger(__name__)


def parse_profile(values: object) -> np.ndarray:
    """Parse temperature or time profile values into a numeric array."""
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        return np.array(values, dtype=float)

    if pd.isna(values):
        return np.array([], dtype=float)

    if isinstance(values, str):
        cleaned = values.strip().replace(";", ",")
        if not cleaned:
            return np.array([], dtype=float)
        return np.array([float(item.strip()) for item in cleaned.split(",") if item.strip()], dtype=float)

    return np.array([float(values)], dtype=float)


def compute_nurse_saul_maturity(
    temperature_profile: object,
    datum_temperature: float,
    delta_t: float = 1.0,
    delta_t_profile: object | None = None,
) -> float:
    """Compute Nurse-Saul maturity index M = Σ(T - T0)Δt."""
    temps = parse_profile(temperature_profile)
    if temps.size == 0:
        return np.nan

    if delta_t_profile is not None and not pd.isna(delta_t_profile):
        dt_values = parse_profile(delta_t_profile)
        if dt_values.size == 1:
            dt_values = np.repeat(dt_values, temps.size)
        if dt_values.size != temps.size:
            raise ValueError(
                f"Mismatch in profile lengths: temperatures={temps.size}, delta_t={dt_values.size}."
            )
    else:
        dt_values = np.full(temps.shape, float(delta_t), dtype=float)

    return float(np.sum((temps - datum_temperature) * dt_values))


def build_preprocessor(features: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessing transformer with imputation + categorical encoding."""
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Create a preprocessing + XGBoost pipeline."""
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def run_grid_search(model_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int) -> GridSearchCV:
    """Run hyperparameter tuning via grid search cross-validation."""
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )
    logger.info("Running grid search with %d folds for %d rows", cv_folds, len(X_train))
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute RMSE, MAE, and R² on test data."""
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def get_feature_importance(best_model: Pipeline) -> pd.DataFrame:
    """Extract feature importance after preprocessing expansion."""
    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_shap_summary(best_model: Pipeline, X_train: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """Compute SHAP mean absolute values per transformed feature."""
    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    X_sample = X_train.sample(min(sample_size, len(X_train)), random_state=42)
    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    return shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def train_target_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    output_dir: Path,
    test_size: float,
    cv_folds: int,
) -> Dict[str, float]:
    """Train, tune, evaluate, and save artifacts for one target variable."""
    logger.info("Preparing target model training for %s", target_column)
    local_df = df.dropna(subset=[target_column]).copy()

    X = local_df[feature_columns]
    y = local_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    logger.info("Split sizes for %s -> train=%d test=%d", target_column, len(X_train), len(X_test))
    model_pipeline = build_model_pipeline(preprocessor)
    tuned = run_grid_search(model_pipeline, X_train, y_train, cv_folds)
    best_model = tuned.best_estimator_

    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info("Metrics for %s: RMSE=%.4f MAE=%.4f R2=%.4f", target_column, metrics["rmse"], metrics["mae"], metrics["r2"])
    metrics["cv_best_rmse"] = float(-tuned.best_score_)

    model_path = output_dir / f"{target_column}_xgb_pipeline.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Saved model for %s -> %s", target_column, model_path)

    importance_df = get_feature_importance(best_model)
    importance_df.to_csv(output_dir / f"{target_column}_feature_importance.csv", index=False)

    shap_df = compute_shap_summary(best_model, X_train)
    shap_df.to_csv(output_dir / f"{target_column}_shap_summary.csv", index=False)

    return metrics


def identify_feature_columns(df: pd.DataFrame, target_columns: Iterable[str], excluded: Iterable[str]) -> List[str]:
    """Infer feature columns by removing targets and excluded source columns."""
    exclude = set(target_columns) | set(excluded)
    return [col for col in df.columns if col not in exclude]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("Starting hybrid model training script")
    parser = argparse.ArgumentParser(description="Train hybrid Physics + ML early-age concrete strength models.")
    parser.add_argument("--data", required=True, help="Path to input CSV dataset.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to save model artifacts.")
    parser.add_argument("--temp-profile-column", default="temperature_profile", help="Column containing temperature profile.")
    parser.add_argument(
        "--dt-profile-column",
        default="time_interval_profile",
        help="Optional column with profile of interval durations matching temperature profile.",
    )
    parser.add_argument("--dt-default-hours", type=float, default=1.0, help="Default Δt in hours if no profile is provided.")
    parser.add_argument("--datum-temperature", type=float, default=-10.0, help="Nurse-Saul datum temperature T0 (°C).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout split ratio for evaluation.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds for grid search.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training dataset from %s", args.data)
    df = pd.read_csv(args.data)
    logger.info("Dataset loaded with shape %s", df.shape)

    dt_col_exists = args.dt_profile_column in df.columns
    logger.info("Computing Nurse-Saul maturity index")
    df["maturity_index"] = df.apply(
        lambda row: compute_nurse_saul_maturity(
            temperature_profile=row[args.temp_profile_column],
            datum_temperature=args.datum_temperature,
            delta_t=args.dt_default_hours,
            delta_t_profile=row[args.dt_profile_column] if dt_col_exists else None,
        ),
        axis=1,
    )

    dropped_source_columns = [args.temp_profile_column]
    if dt_col_exists:
        dropped_source_columns.append(args.dt_profile_column)

    feature_columns = identify_feature_columns(df, TARGET_COLUMNS, dropped_source_columns)
    logger.info("Identified %d feature columns", len(feature_columns))

    summary: Dict[str, Dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        if target not in df.columns:
            print(f"Skipping {target} because it is not present in dataset.")
            continue

        print(f"Training model for target: {target}")
        metrics = train_target_model(
            df=df,
            feature_columns=feature_columns,
            target_column=target,
            output_dir=output_dir,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
        )
        summary[target] = metrics

    logger.info("Writing metrics summary to %s", output_dir / "metrics_summary.json")
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete. Metrics summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
