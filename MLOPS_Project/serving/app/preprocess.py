"""Apply the same batch preprocessing as Kedro preprocess_batch (inference only)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from app.encode import encode_cat_variables

COLUMN_RENAME_MAP = {
    "carID": "car_id",
    "Brand": "brand",
    "fuelType": "fuel_type",
    "previousOwners": "previous_owners",
    "engineSize": "engine_size",
    "paintQuality%": "paint_quality_pct",
    "hasDamage": "has_damage",
}

MEAN_IMPUTATION_COLS = ["mpg", "paint_quality_pct"]
MEDIAN_IMPUTATION_COLS = ["tax", "mileage"]
MODE_IMPUTATION_COLS = [
    "previous_owners",
    "transmission",
    "year",
    "fuel_type",
    "engine_size",
    "has_damage",
]
COLS_OHE = ["transmission", "fuel_type", "previous_owners"]


def _artifacts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "artifacts"


def load_artifact_pickle(name: str) -> dict:
    path = _artifacts_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy from kedro-carprice-prediction/data/04_feature/ "
            "after running production_full_train_process."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_RENAME_MAP)


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "car_id" not in df.columns:
        df["car_id"] = 0
    else:
        df["car_id"] = pd.to_numeric(df["car_id"], errors="coerce").fillna(0).astype(int)
    numeric_cols = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engine_size",
        "previous_owners",
        "paint_quality_pct",
        "has_damage",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["brand", "model", "transmission", "fuel_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def apply_simple_impute(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["model"].fillna("Unknown")
    df[MEAN_IMPUTATION_COLS] = df[MEAN_IMPUTATION_COLS].fillna(stats["means"])
    df[MEDIAN_IMPUTATION_COLS] = df[MEDIAN_IMPUTATION_COLS].fillna(stats["medians"])
    df[MODE_IMPUTATION_COLS] = df[MODE_IMPUTATION_COLS].fillna(stats["modes"])
    return df.dropna(subset=["brand"])


def create_new_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = 2022 - df["year"]
    df["mileage_per_litre"] = (df["mileage"] / df["engine_size"]).where(
        df["engine_size"] != 0, 0
    )
    df["efficiency_ratio"] = (df["mpg"] / df["engine_size"]).where(
        df["engine_size"] != 0, 0
    )
    return df


def apply_fitted_transforms(batch_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = batch_df.copy()
    te = artifacts["target_encoder"]
    df[te.cols] = te.transform(df[te.cols])
    df, _ = encode_cat_variables(df, ohe_cols=COLS_OHE, ohe_encoder=artifacts["ohe_encoder"])
    df[artifacts["cols_to_scale"]] = artifacts["scaler"].transform(df[artifacts["cols_to_scale"]])
    return df


def preprocess_raw_cars(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw car rows into model-ready features."""
    imputation_stats = load_artifact_pickle("imputation_stats.pkl")
    preprocessing_artifacts = load_artifact_pickle("preprocessing_artifacts.pkl")

    df = rename_columns(raw_df)
    df = fix_data_types(df)
    df = apply_simple_impute(df, imputation_stats)
    df = create_new_variables(df)
    df = apply_fitted_transforms(df, preprocessing_artifacts)

    target_col = preprocessing_artifacts.get("target_col", "price")
    drop_cols = [c for c in [target_col] if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")
