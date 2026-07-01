"""
This is a boilerplate pipeline 'model_predict'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_champion_model, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_champion_model,
                inputs="parameters",
                outputs="champion_model",
                name="load_champion_model_node",
            ),
            node(
                func=predict,
                inputs=dict(
                    model="champion_model",
                    preprocessed_test="preprocessed_test",
                ),
                outputs="predictions",
                name="predict_node",
            ),
        ]
    )