"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro_carprice_prediction.pipelines import (
    data_cleaning,
    data_quality,
    split_train,
    preprocess_train,
    preprocess_batch,
    model_train,
    model_predict
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    cleaning_pipeline = data_cleaning.create_pipeline()
    quality_pipeline = data_quality.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    preprocess_batch_pipeline = preprocess_batch.create_pipeline()
    model_train_pipeline = model_train.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()

    return {
        "data_quality": quality_pipeline,
        "data_cleaning": cleaning_pipeline,
        "split_pipeline": split_train_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "preprocess_batch": preprocess_batch_pipeline,
        "model_train": model_train_pipeline,
        "model_predict": model_predict_pipeline,
        "production_full_train_process" : cleaning_pipeline + split_train_pipeline + preprocess_train_pipeline + model_train_pipeline,
        "production_full_prediction_process" : cleaning_pipeline + preprocess_batch_pipeline + model_predict_pipeline,
    }
