"""Nodes for the data_drifts pipeline.

This pipeline checks whether new data has drifted away from the data the model
was trained on. It splits the model-input data into a reference half and a
current half, runs an Evidently data drift report, and saves the HTML report
plus a small metrics file.
"""
import logging

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

logger = logging.getLogger(__name__)


def prepare_drift_data(model_input: pd.DataFrame) -> pd.DataFrame:
    """Drop datetime columns.

    Evidently 0.6.5 crashes on datetime columns with newer pandas, and we don't
    need dates to detect drift, so we simply remove them.
    """
    dt_cols = model_input.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    logger.info("Dropping datetime columns before drift check: %s", dt_cols)
    return model_input.drop(columns=dt_cols)


def split_reference_current(data: pd.DataFrame, split_frac: float, random_state: int):
    """Split the data into a reference half and a current half."""
    reference = data.sample(frac=split_frac, random_state=random_state)
    current = data.drop(reference.index)
    return reference, current


def evaluate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target: str,
    categorical_features: list,
    exclude_columns: list,
):
    """Run the Evidently drift report and return the HTML + the main metrics."""
    # tell Evidently the role of each column
    exclude = categorical_features + exclude_columns
    num_features = [c for c in reference.columns if c not in exclude]

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.categorical_features = categorical_features
    column_mapping.numerical_features = num_features

    # build and run the report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    # pull out the main numbers (same helper as in the notebook)
    result = report.as_dict()["metrics"][0]["result"]
    metrics = {
        "dataset_drift": result["dataset_drift"],
        "drifted_columns": result["number_of_drifted_columns"],
        "total_columns": result["number_of_columns"],
        "share_drifted": round(result["share_of_drifted_columns"], 3),
    }
    logger.info("Drift result: %s", metrics)

    # the catalog saves the HTML as a file and the metrics as JSON
    return report.get_html(), metrics
