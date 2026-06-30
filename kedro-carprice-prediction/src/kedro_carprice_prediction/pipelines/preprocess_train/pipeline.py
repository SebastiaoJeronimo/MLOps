"""
This is a boilerplate pipeline 'preprocess_train'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_train,
    split_features_target
)



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_train,
            inputs=dict(
                train_df="car_train_split",
                val_df="car_val_split",
                group_name="params:engineered_group_name",
                group_description="params:engineered_group_description",
                target_col="params:target_col",
                scaler_method="params:scaler_method",
            ),
            outputs=[
                "preprocessed_train",
                "preprocessed_val",
                "imputation_stats",
                "preprocessing_artifacts"
                #"engineered_feature_group_metadata",
            ],
            name="preprocess_train_node",
        ),
        node (
            func= split_features_target,
            inputs=dict(
                train_df ="preprocessed_train",
                val_df = "preprocessed_val",
                target_col = "params:target_col"
            ),
            outputs=[
                "X_train", 
                "y_train", 
                "X_val", 
                "y_val"
            ],
            name="split_modelinput") 
    ])