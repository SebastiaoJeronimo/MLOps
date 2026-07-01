"""
This is a boilerplate pipeline 'model_selection'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_optuna_study, update_model_params


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_optuna_study,
            inputs=dict(
                X_train="X_train", y_train="y_train",
                X_val="X_val", y_val="y_val",
                model_type="params:model_type",
                n_trials="params:n_trials",
                study_name="params:study_name",
            ),
            outputs="best_params",
            name="run_optuna_study_node",
        ),
    ])
