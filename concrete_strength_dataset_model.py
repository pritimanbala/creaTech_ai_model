#!/usr/bin/env python3
"""Utilities to train and serve an early-age concrete strength predictor.

This module is tailored for datasets with columns like:
- Cement (component 1)(kg in a m^3 mixture)
- Blast Furnace Slag (component 2)(kg in a m^3 mixture)
- Fly Ash (component 3)(kg in a m^3 mixture)
- Water  (component 4)(kg in a m^3 mixture)
- Superplasticizer (component 5)(kg in a m^3 mixture)
- Coarse Aggregate  (component 6)(kg in a m^3 mixture)
- Fine Aggregate (component 7)(kg in a m^3 mixture)
- Age (day)
- Concrete compressive strength(MPa, megapascals)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW_TO_STANDARD_COLUMNS: Dict[str, str] = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement_kg",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast_furnace_slag_kg",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash_kg",
    "Water  (component 4)(kg in a m^3 mixture)": "water_kg",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer_kg",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_aggregate_kg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_aggregate_kg",
    "Age (day)": "age_day",
    "Concrete compressive strength(MPa, megapascals)": "concrete_strength_mpa",
}

MIX_FEATURE_COLUMNS: List[str] = [
    "cement_kg",
    "blast_furnace_slag_kg",
    "fly_ash_kg",
    "water_kg",
    "superplasticizer_kg",
    "coarse_aggregate_kg",
    "fine_aggregate_kg",
]

MODEL_FEATURE_COLUMNS: List[str] = MIX_FEATURE_COLUMNS + ["age_day"]
TARGET_COLUMN = "concrete_strength_mpa"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path


class EarlyAgeStrengthPredictor:
    """Adapter exposing `predict(input_dataframe)` -> [8h, 16h, 24h] strengths.

    The wrapped base model must predict concrete strength from the 8 canonical mix
    columns + age in days.
    """

    def __init__(self, base_model: Any):
        self.base_model = base_model

    def _prepare_features(self, input_dataframe: pd.DataFrame, age_days: float) -> pd.DataFrame:
        missing = [col for col in MIX_FEATURE_COLUMNS if col not in input_dataframe.columns]
        if missing:
            raise ValueError(f"Input dataframe missing required mix columns: {missing}")

        features = input_dataframe[MIX_FEATURE_COLUMNS].copy()
        features["age_day"] = float(age_days)
        return features

    def predict(self, input_dataframe: pd.DataFrame) -> np.ndarray:
        """Predict strengths at 8h, 16h, and 24h for each row in input_dataframe."""
        predictions = []
        for age in (8.0 / 24.0, 16.0 / 24.0, 1.0):
            features = self._prepare_features(input_dataframe, age_days=age)
            pred = np.asarray(self.base_model.predict(features), dtype=float).reshape(-1)
            predictions.append(pred)

        stacked = np.column_stack(predictions)
        return stacked


def load_and_standardize_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load concrete CSV/XLSX and normalize headers into internal snake_case columns."""
    data_path = Path(data_path)
    logger.info("Loading dataset from %s", data_path)
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Unsupported dataset format. Use .csv, .xlsx, or .xls")

    logger.info("Loaded dataset with shape %s", df.shape)
    missing_raw = [c for c in RAW_TO_STANDARD_COLUMNS if c not in df.columns]
    if missing_raw:
        raise ValueError(
            "CSV does not contain required headers. Missing: "
            + ", ".join(missing_raw)
        )

    standardized = df.rename(columns=RAW_TO_STANDARD_COLUMNS).copy()
    standardized = standardized[list(RAW_TO_STANDARD_COLUMNS.values())]
    return standardized


def train_strength_regressor_from_file(
    data_path: str | Path,
    output_model_path: str | Path = "artifacts/concrete_strength_age_model.joblib",
) -> TrainingArtifacts:
    """Train a concrete-strength regressor from concrete_data file.

    The trained model learns strength as a function of mix proportions + age_day.
    """
    output_model_path = Path(output_model_path)
    logger.info("Starting model training from %s", data_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_standardize_dataset(data_path)
    logger.info("Standardized dataset rows=%d columns=%d", len(df), len(df.columns))

    X = df[MODEL_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MODEL_FEATURE_COLUMNS),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    logger.info("Fitting training pipeline")
    pipeline.fit(X, y)

    joblib.dump(pipeline, output_model_path)
    logger.info("Saved trained model to %s", output_model_path)

    metrics_path = output_model_path.with_suffix(".metrics.txt")
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("trained_rows=" + str(len(df)) + "\n")
        f.write("feature_columns=" + ",".join(MODEL_FEATURE_COLUMNS) + "\n")

    return TrainingArtifacts(model_path=output_model_path, metrics_path=metrics_path)


def train_strength_regressor_from_csv(
    csv_path: str | Path,
    output_model_path: str | Path = "artifacts/concrete_strength_age_model.joblib",
) -> TrainingArtifacts:
    """Backward-compatible wrapper for CSV-specific naming."""
    return train_strength_regressor_from_file(
        data_path=csv_path,
        output_model_path=output_model_path,
    )


def load_early_age_strength_predictor(model_path: str | Path) -> EarlyAgeStrengthPredictor:
    """Load persisted base model and wrap it for 8h/16h/24h prediction API."""
    logger.info("Loading trained model from %s", model_path)
    base_model = joblib.load(model_path)
    return EarlyAgeStrengthPredictor(base_model=base_model)


__all__ = [
    "RAW_TO_STANDARD_COLUMNS",
    "MIX_FEATURE_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "EarlyAgeStrengthPredictor",
    "TrainingArtifacts",
    "load_and_standardize_dataset",
    "train_strength_regressor_from_file",
    "train_strength_regressor_from_csv",
    "load_early_age_strength_predictor",
]
