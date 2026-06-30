"""
This is a boilerplate pipeline 'preprocess_batch'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import ( apply_simple_impute,
    create_new_variables, apply_fitted_transforms,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_simple_impute,
            inputs=dict(df="car_test_clean", stats="imputation_stats"),
            outputs="car_batch_imputed",
            name="apply_simple_impute_batch_node",
        ),
        node(create_new_variables, "car_batch_imputed", "car_batch_fe", name="create_new_variables_batch_node"),
        node(
            func=apply_fitted_transforms,
            inputs=dict(batch_df="car_batch_fe", artifacts="preprocessing_artifacts"),
            outputs="preprocessed_test",
            name="apply_fitted_transforms_batch_node",
        ),
    ])