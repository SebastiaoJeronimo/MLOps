from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_data, ingestion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_data,
                inputs=dict(
                    df="car_train_raw",
                    suite_name="params:suite_name",
                    feature_group="params:feature_group_name",
                    fail_on_error="params:fail_pipeline_on_error",
                ),
                outputs="data_quality_report",
                name="validate_data_node",
            ),
            node(
                func=ingestion,
                inputs=dict(
                    df="car_train_raw",
                    validation_report= "data_quality_report", 
                    group_name="params:feature_group_name",
                    group_description="params:feature_group_feature_descriptions",
                ),
                outputs="feature_group_metadata",
                name="upload_to_feature_store_node",
            ),
        ]
    )