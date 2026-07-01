"""
This is a boilerplate pipeline 'preprocess_batch'
generated using Kedro 1.3.1
"""

"""
Node functions for the preprocess_batch pipeline.

basic_cleaning/standardize_types are reused from data_cleaning (no fitting,
safe to reuse as-is). apply_simple_impute/apply_fitted_transforms are
imported from preprocess_train, where the matching fit_* logic lives --
batch never fits anything itself, only applies what train already fit.
remove_outliers is deliberately NOT reused here -- you never drop a row
from a batch you need a prediction for.
"""
from kedro_carprice_prediction.pipelines.preprocess_train.nodes import (
    _apply_simple_impute as apply_simple_impute,
    create_new_variables,
    apply_fitted_transforms,
)