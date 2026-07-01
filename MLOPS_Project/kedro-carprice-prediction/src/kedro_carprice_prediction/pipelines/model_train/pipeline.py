"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_train, log_and_register_champion, build_training_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(model_train,
             dict(X_train="X_train", X_val="X_val", y_train="y_train", y_val="y_val",
                  parameters="parameters", best_columns="params:best_columns"),
             ["trained_model", "val_metrics", "registration_metadata"], name="model_train_node"),
        node(build_training_report,
             dict(registration_metadata="registration_metadata", metrics="val_metrics",
                  X_train="X_train", X_val="X_val"),
             "model_train_report", name="build_training_report_node"),
    ])