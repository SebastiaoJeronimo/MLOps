"""Load the Champion model from MLflow Model Registry."""

from __future__ import annotations

import os

import mlflow


def load_champion_model(
    registered_model_name: str | None = None,
    champion_alias: str | None = None,
):
    name = registered_model_name or os.getenv("REGISTERED_MODEL_NAME", "car_price_model")
    alias = champion_alias or os.getenv("CHAMPION_ALIAS", "Champion")
    uri = f"models:/{name}@{alias}"
    return mlflow.pyfunc.load_model(uri)
