"""
This is a boilerplate pipeline 'split_train'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_val


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_val,
            inputs=dict(df="car_train_clean", val_size="params:val_size", random_state="params:random_state"),
            outputs=["car_train_split", "car_val_split"],
            name="split_train_val_node",
        ),
    ])