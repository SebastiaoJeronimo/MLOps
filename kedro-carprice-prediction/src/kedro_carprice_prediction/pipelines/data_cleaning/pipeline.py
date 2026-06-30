"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(clean_data, "car_train_raw", "car_train_clean", name="basic_cleaning_train_node"),
            node(clean_data, "car_test_raw", "car_test_clean", name="basic_cleaning_test_node")
        ])
