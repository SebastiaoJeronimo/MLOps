# pipelines/hyperparameter_tuning/nodes.py
import optuna
import mlflow
import numpy as np
import pandas as pd 
from sklearn.metrics import root_mean_squared_error

import logging
logger = logging.getLogger(__name__)

def run_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    n_trials: int = 50,
    study_name: str = "car_price_hpo",
) -> dict:
    """Run an Optuna hyperparameter search. Each trial is logged as a nested
    MLflow run (kedro-mlflow's hook has the outer run open already).
    Returns the best params dict -- this becomes the input to model_train's
    params:model_params, so no code changes needed there."""

    def objective(trial):
        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            }
        else:  
            return root_mean_squared_error(y_val, [y_train.mean()] * len(y_val))

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            from xgboost import XGBRegressor
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            rmse = root_mean_squared_error(y_val, model.predict(X_val))
            mlflow.log_metric("rmse", rmse)

        return rmse

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        load_if_exists=True,  # resume a prior study if it was interrupted
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_rmse = study.best_value
    logger.info(f"Best params (rmse={best_rmse:.2f}): {best_params}")
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_rmse", best_rmse)

    return best_params


def update_model_params(best_params: dict) -> dict:
    """Wrap best_params in the same structure model_train expects from
    params:model_params -- just a passthrough so the catalog can persist it."""
    return best_params
