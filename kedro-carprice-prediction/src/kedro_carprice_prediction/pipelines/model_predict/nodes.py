import logging

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def load_champion_model(parameters: dict):
    """Load the Champion model from the MLflow Model Registry."""

    registered_model_name = parameters["registered_model_name"]
    champion_alias = parameters.get("champion_alias", "Champion")

    model_uri = f"models:/{registered_model_name}@{champion_alias}"

    logger.info("Loading model from %s", model_uri)

    return mlflow.pyfunc.load_model(model_uri)


def predict(
    model,
    preprocessed_test: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for the test set."""

    predictions = model.predict(preprocessed_test)

    return pd.DataFrame(
        {
            "prediction": predictions,
        },
        index=preprocessed_test.index,
    )