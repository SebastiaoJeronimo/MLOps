from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_drift_data, split_reference_current, evaluate_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_drift_data,
                inputs=dict(model_input="car_train_clean"),
                outputs="drift_data",
                name="prepare_drift_data_node",
            ),
            node(
                func=split_reference_current,
                inputs=dict(
                    data="drift_data",
                    split_frac="params:drift_split_frac",
                    random_state="params:drift_random_state",
                ),
                outputs=["drift_reference", "drift_current"],
                name="split_reference_current_node",
            ),
            node(
                func=evaluate_drift,
                inputs=dict(
                    reference="drift_reference",
                    current="drift_current",
                    target="params:drift_target",
                    categorical_features="params:drift_categorical_features",
                    exclude_columns="params:drift_exclude_columns",
                ),
                outputs=["drift_report_html", "drift_metrics"],
                name="evaluate_drift_node",
            ),
        ]
    )
