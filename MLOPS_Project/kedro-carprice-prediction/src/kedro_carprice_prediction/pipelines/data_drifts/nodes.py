"""Nodes for the data_drifts pipeline.

This pipeline checks whether new data has drifted away from the data the model
was trained on. It splits the cleaned training data into a reference half and a
current half, runs an Evidently data drift report, and saves the HTML report
plus a small metrics file.
"""
import logging

import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)


def prepare_drift_data(model_input: pd.DataFrame) -> pd.DataFrame:
    """Drop datetime columns.

    Evidently crashes on datetime columns with newer pandas, and we don't
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
    exclude = set(exclude_columns)
    num_features = [c for c in reference.columns if c not in exclude and c not in categorical_features]
    drift_cols = categorical_features + num_features
    logger.info("Drift columns: %s (target=%s)", drift_cols, target)

    data_definition = DataDefinition(
        id_column="car_id" if "car_id" in reference.columns else None,
        categorical_columns=categorical_features,
        numerical_columns=num_features,
        regression=[Regression(target=target)],
    )
    reference_ds = Dataset.from_pandas(reference, data_definition=data_definition)
    current_ds = Dataset.from_pandas(current, data_definition=data_definition)

    report = Report([DataDriftPreset(columns=drift_cols)])
    snapshot = report.run(current_data=current_ds, reference_data=reference_ds)

    drift_summary = next(
        (
            m
            for m in snapshot.dict()["metrics"]
            if "DriftedColumnsCount" in m.get("metric_name", "")
        ),
        None,
    )
    if drift_summary is None:
        raise RuntimeError("DriftedColumnsCount metric not found in Evidently report")

    count = float(drift_summary["value"]["count"])
    share = float(drift_summary["value"]["share"])
    total = int(round(count / share)) if share else len(drift_cols)
    metrics = {
        "dataset_drift": share >= 0.5,
        "drifted_columns": int(count),
        "total_columns": total,
        "share_drifted": round(share, 3),
    }
    logger.info("Drift result: %s", metrics)

    return snapshot.get_html_str(as_iframe=False), metrics
