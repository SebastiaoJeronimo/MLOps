"""
This is a boilerplate pipeline 'preprocess_train'
generated using Kedro 1.3.1
"""


"""
Node functions for the preprocess_train pipeline: outlier removal, fitted
imputation, feature creation, and fitted encoding/scaling. Everything that
FITS statistics lives here -- preprocess_batch only ever imports the
apply-only counterparts (apply_imputation, apply_fitted_transforms).
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from kedro_carprice_prediction.data_prep import scaleNumVariables, encodeCatVariables

from kedro_carprice_prediction.feature_store.manager import FeatureStoreManager
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# Initialize Kedro Config
conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

# --- Imputation column groups ---
MEAN_IMPUTATION_COLS = ["mpg", "paint_quality_pct"]
MEDIAN_IMPUTATION_COLS = ["tax", "mileage"]
MODE_IMPUTATION_COLS = ["previous_owners", "transmission", "year", "fuel_type", "engine_size", "has_damage"]
KNN_IMPUTE_COLS = ["tax", "mpg"]
KNN_NUMERIC_COLS = ["mileage", "engine_size", "tax", "mpg"]
KNN_CATEGORICAL_COLS = ["model", "transmission", "fuel_type", "year"]

# --- Encoding column groups ---
COL_TARGET = ["model", "brand"]
COLS_OHE = ["transmission", "fuel_type", "previous_owners"]

_SCALERS = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}



def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop physically-implausible rows from TRAIN. Never applied to val or
    batch -- you don't get to discard a row you need to score/evaluate."""
    before = len(df)
    mask = (
        (df["mileage"].between(0, 170000) | df["mileage"].isna())
        & (df["tax"].between(0, 300) | df["tax"].isna())
        & (df["mpg"].between(0, 150) | df["mpg"].isna())
        & (df["engine_size"].between(0, 5) | df["engine_size"].isna())
    )
    df = df[mask].copy()
    logger.info(f"Outlier treatment: {before - len(df)} training rows removed.")
    return df


def fit_simple_impute_train(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Mean/mode imputation, fit on TRAIN only, applied to both train and
    val. No KNN, no per-model logic -- simple stats only."""
    stats = {
        "means": train_df[MEAN_IMPUTATION_COLS].mean(),
        "medians": train_df[MEDIAN_IMPUTATION_COLS].median(),
        "modes": train_df[MODE_IMPUTATION_COLS].mode().iloc[0],
    }

    train_imputed = _apply_simple_impute(train_df, stats)
    val_imputed = _apply_simple_impute(val_df, stats)
    return train_imputed, val_imputed, stats


def _apply_simple_impute(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["model"].fillna("Unknown")
    logging.getLogger(__name__).info(stats.keys())
    
    
    df[MEAN_IMPUTATION_COLS] = (df[MEAN_IMPUTATION_COLS].fillna(stats["means"]))
    df[MEDIAN_IMPUTATION_COLS] = (df[MEDIAN_IMPUTATION_COLS].fillna(stats["medians"]))
    df[MODE_IMPUTATION_COLS] = (df[MODE_IMPUTATION_COLS].fillna(stats["modes"]))
    
    return df.dropna(subset=["brand"])


def create_new_variables(df: pd.DataFrame) -> pd.DataFrame:
    """age, mileage_per_litre, efficiency_ratio. Pure row-wise function of
    each car's own attributes -- no fitting, so it's fine to apply to train,
    val, and batch alike."""
    df = df.copy()
    df["age"] = 2022 - df["year"]
    df["mileage_per_litre"] = (df["mileage"] / df["engine_size"]).where(df["engine_size"] != 0, 0)
    df["efficiency_ratio"] = (df["mpg"] / df["engine_size"]).where(df["engine_size"] != 0, 0)
    return df


def push_engineered_to_fs(df: pd.DataFrame, group_name: str, group_description: list):
    """Push TRAIN engineered features (post create_new_variables, pre
    encode/scale) as their own feature group. Encoded/scaled values are
    intentionally not pushed -- those are fit per training run, not
    stable/reusable feature-store content."""
    manager = FeatureStoreManager(credentials=credentials)

    
    return manager.upload(
        dataframe=df, 
        name=group_name,
        description=group_description, 
        version="1",
        primary_key=["car_id"]
    )


def fit_transform_train(train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str, scaler_method: str = "Robust"):
    """Fit target-encoder, OHE-encoder, and scaler on TRAIN only; transform
    both train and val. Returns transformed frames + an artifacts bundle
    for apply_fitted_transforms to reuse on a batch later."""
    train_df, val_df = train_df.copy(), val_df.copy()
    y_train = train_df[target_col]

    te = ce.TargetEncoder(cols=COL_TARGET, smoothing=1)
    train_df[COL_TARGET] = te.fit_transform(train_df[COL_TARGET], y_train)
    val_df[COL_TARGET] = te.transform(val_df[COL_TARGET])

    train_encoded, ohe_encoder = encodeCatVariables(train_df, ohe_cols=COLS_OHE)
    val_encoded, _ = encodeCatVariables(val_df, ohe_cols=COLS_OHE, ohe_encoder=ohe_encoder)

    cols_to_scale = [
        c for c in train_encoded.columns
        if c not in COLS_OHE + ["has_damage", target_col, "car_id"]
        and pd.api.types.is_numeric_dtype(train_encoded[c])
    ]

    scaler = _SCALERS[scaler_method]()
    train_encoded[cols_to_scale] = scaler.fit_transform(train_encoded[cols_to_scale])
    val_encoded[cols_to_scale] = scaler.transform(val_encoded[cols_to_scale])

    artifacts = {
        "target_encoder": te, "ohe_encoder": ohe_encoder, "scaler": scaler,
        "cols_to_scale": cols_to_scale, "target_col": target_col,
    }
    return train_encoded, val_encoded, artifacts


def apply_fitted_transforms(batch_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply train-fitted target-encoder/OHE/scaler to a new batch --
    never refits. Used by preprocess_batch, defined here since this is
    where the fitting (and therefore the matching apply logic) lives."""
    df = batch_df.copy()
    df[artifacts["target_encoder"].cols] = artifacts["target_encoder"].transform(
        df[artifacts["target_encoder"].cols]
    )
    df, _ = encodeCatVariables(df, ohe_cols=COLS_OHE, ohe_encoder=artifacts["ohe_encoder"])
    df[artifacts["cols_to_scale"]] = artifacts["scaler"].transform(df[artifacts["cols_to_scale"]])
    return df

def preprocess_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_name: str,
    group_description: list,
    target_col: str,
    scaler_method: str = "Robust",
):
    """Orchestrates the full preprocess_train flow. Handles train/val with
    the correct asymmetry:
      - remove_outliers: TRAIN ONLY -- val must stay representative of
        real-world data; you don't get to discard rows you need to evaluate.
      - fit_simple_impute_train: FIT on train, APPLIED to both.
      - create_new_variables: row-wise, no fitting -- safe to apply to
        train and val independently.
      - push_engineered_to_fs: TRAIN ONLY -- val is an evaluation holdout,
        not production-like data; it has no business in the feature store.
      - fit_transform_train: FIT on train, APPLIED to both.

    Returns:
        train_features, val_features, imputation_stats,
        preprocessing_artifacts, feature_group_metadata
    """
    # 1. Outlier removal -- train only
    train_no_outliers = remove_outliers(train_df)

    # 2. Imputation -- fit on train, apply to both
    train_imputed, val_imputed, imputation_stats = fit_simple_impute_train(
        train_no_outliers, val_df
    )

    # Drop remaining NaNs
    train_before = len(train_imputed)
    val_before = len(val_imputed)

    train_imputed = train_imputed.dropna().reset_index(drop=True)
    val_imputed = val_imputed.dropna().reset_index(drop=True)

    logger.info(
        "Dropped %d training rows with remaining NaNs.",
        train_before - len(train_imputed),
    )
    logger.info(
        "Dropped %d validation rows with remaining NaNs.",
        val_before - len(val_imputed),
    )
    # 3. Feature creation -- row-wise, applied to each independently
    train_fe = create_new_variables(train_imputed)
    val_fe = create_new_variables(val_imputed)

    # 4. Push engineered features to the Feature Store -- train only
    #feature_group_metadata = push_engineered_to_fs(train_fe, group_name, group_description)

    # 5. Encode/scale -- fit on train, transform both
    train_features, val_features, preprocessing_artifacts = fit_transform_train(
        train_fe, val_fe, target_col, scaler_method
    )


    return (
        train_features,
        val_features,
        imputation_stats,
        preprocessing_artifacts
        #feature_group_metadata,
    )



def split_features_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split preprocessed train and validation datasets into features (X)
    and target (y).

    Args:
        train_df: Fully preprocessed training dataframe.
        val_df: Fully preprocessed validation dataframe.
        target_col: Name of the target column.

    Returns:
        X_train, y_train, X_val, y_val
    """
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[[target_col]]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[[target_col]]

    return X_train, y_train, X_val, y_val