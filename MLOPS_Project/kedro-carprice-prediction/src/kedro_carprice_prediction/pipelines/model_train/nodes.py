"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 1.3.1
"""
"""
Node functions for the model_train pipeline. model_train fits + evaluates
+ logs metrics inside a NESTED mlflow run (kedro-mlflow's hook already has
an outer run open for this pipeline -- nested=True creates a proper child
run rather than conflicting with it). log_and_register_champion re-opens
that exact run by run_id to attach the model artifact and set the alias,
so everything for one training attempt lives in a single mlflow run even
though it's split across two Kedro nodes.
"""
import logging
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline as SklearnPipeline

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    "linear": lambda params: LinearRegression(**(params or {})),
}

try:
    from xgboost import XGBRegressor
    _MODEL_REGISTRY["xgboost"] = lambda params: XGBRegressor(**(params or {}))
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
    _MODEL_REGISTRY["lightgbm"] = lambda params: LGBMRegressor(**(params or {}))
except ImportError:
    pass

_AUTOLOG_FN = {
    "linear": mlflow.sklearn.autolog,
    "xgboost": mlflow.xgboost.autolog,
    "lightgbm": mlflow.lightgbm.autolog,
}
_MLFLOW_LOG_MODEL = {
    "linear": mlflow.sklearn.log_model,
    "xgboost": mlflow.xgboost.log_model,
    "lightgbm": mlflow.lightgbm.log_model,
}


def model_train(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    parameters: Dict[str, Any],
    best_columns: list = None,
    best_params: dict = None,  # <-- Optuna output, takes priority over params:model_params
):
    """Fits the configured model family, evaluates on the validation
    split, and logs metrics inside one nested mlflow run.

    Hyperparameter resolution order (highest priority first):
      1. best_params -- output of the Optuna hyperparameter_tuning pipeline,
         persisted as best_params.json and wired in from the catalog.
      2. params:model_params -- static fallback defined in parameters.yml,
         used when running model_train standalone without a prior HPO study.

    This means the same node works both with and without Optuna -- just wire
    best_params="best_params" in pipeline.py when HPO has run, or omit it
    (defaulting to None) to fall back to the static params.
    """
    model_type = parameters["model_type"]
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown or unavailable model_type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    # Resolve hyperparameters -- Optuna output wins over static params
    resolved_params = best_params or parameters.get("model_params") or {}
    logger.info(
        "Training %s with params: %s (source: %s)",
        model_type,
        resolved_params,
        "optuna" if best_params else "parameters.yml",
    )

    autolog_fn = _AUTOLOG_FN.get(model_type, mlflow.sklearn.autolog)
    autolog_fn(
        log_model_signatures=False,
        log_input_examples=False,
        log_models=False,
        log_datasets=False,
    )

    if parameters.get("use_feature_selection") and best_columns:
        logger.info("Using feature selection: %d columns", len(best_columns))
        X_train = X_train[best_columns]
        X_val = X_val[best_columns]

    with mlflow.start_run(nested=True, run_name=f"{model_type}_champion_train") as run:

        model = _MODEL_REGISTRY[model_type](resolved_params)
        model.fit(X_train, np.ravel(y_train))

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        metrics = {
            "rmse_train": root_mean_squared_error(y_train, y_train_pred),
            "rmse": root_mean_squared_error(y_val, y_val_pred),
            "mae": mean_absolute_error(y_val, y_val_pred),
            "r2": r2_score(y_val, y_val_pred),
        }

        mlflow.log_metrics(metrics)
        # Log whichever params were actually used, so the run is self-contained
        mlflow.log_params(resolved_params)
        mlflow.set_tags({
            "model_family": model_type,
            "hpo": "optuna" if best_params else "none",
        })

        sample = X_train.iloc[:100]
        signature = infer_signature(sample, model.predict(sample))

        log_fn = _MLFLOW_LOG_MODEL[model_type]
        model_info = log_fn(
            model,
            name=model_type,
            signature=signature,
            input_example=X_train.iloc[[0]],
            registered_model_name=parameters["registered_model_name"],
        )

        client = mlflow.MlflowClient()
        champion_alias = parameters.get("champion_alias", "Champion")
        client.set_registered_model_alias(
            parameters["registered_model_name"],
            champion_alias,
            model_info.registered_model_version,
        )

        registration_metadata = {
            "model_type": model_type,
            "model_uri": model_info.model_uri,
            "version": model_info.registered_model_version,
            "registered_model_name": parameters["registered_model_name"],
            "champion_alias": champion_alias,
            "run_id": run.info.run_id,
            "hpo_params_used": resolved_params,
        }

        logger.info(
            "Logged %s Champion (v%s) in run %s | rmse=%.2f r2=%.4f",
            model_type,
            model_info.registered_model_version,
            run.info.run_id,
            metrics["rmse"],
            metrics["r2"],
        )

    return model, metrics, registration_metadata


def build_training_report(
    registration_metadata: dict, metrics: dict, X_train: pd.DataFrame, X_val: pd.DataFrame
) -> pd.DataFrame:
    """Summarize the run as a single-row DataFrame."""
    return pd.DataFrame([{
        **registration_metadata,
        "rmse": metrics["rmse"],
        "rmse_train": metrics["rmse_train"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
    }])