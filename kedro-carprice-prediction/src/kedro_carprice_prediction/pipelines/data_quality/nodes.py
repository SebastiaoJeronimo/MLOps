import logging
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import hopsworks
from hopsworks.client.exceptions import RestAPIError


# --- GREAT EXPECTATIONS 1.x IMPORTS ---
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnValuesToBeOfType,
    ExpectColumnMinToBeBetween,
    ExpectColumnDistinctValuesToBeInSet
)

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from kedro_carprice_prediction.feature_store.manager import FeatureStoreManager

# Initialize Kedro Config
conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

def build_car_quality_suite(context, suite_name:str, feature_group:str) -> ExpectationSuite:
    """Define the data contract for raw car listing data.

    Boundaries are based on profiling train.csv: real listings shouldn't have
    negative mileage/tax/mpg/engineSize/previousOwners, paintQuality% should be
    a real percentage, year should be plausible, and hasDamage should be binary.
    `mostly=` is used instead of hard 100% thresholds because some noise is
    expected in raw data -- the contract should catch *systemic* problems,
    not reject a batch over a handful of bad rows.
    """

    suite = gx.ExpectationSuite(name=suite_name)

    # ID column ensures entity integrity (has to uniquely identifie a car and cannot be null)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="carID"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="carID"))

    # In the EDA phase we noticed some numerical columns has negative values
    # Given the problem context this should not be possible.

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="year", min_value=1990, max_value=2026, mostly=0.99
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="mileage", min_value=0, max_value=None, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tax", min_value=0, max_value=None, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="mpg", min_value=0, max_value=None, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="engineSize", min_value=0, max_value=8.0, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="paintQuality%", min_value=0, max_value=100, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="previousOwners", min_value=0, max_value=15, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="hasDamage", value_set=[0, 1], mostly=0.95
        )
    )

    return context.suites.add(suite)


def validate_data(
    df: pd.DataFrame,
    suite_name: str,
    feature_group: str,
    fail_on_error: bool,):

    context = gx.get_context(mode="ephemeral")

    suite = build_car_quality_suite(
        context=context,
        suite_name=suite_name,
        feature_group=feature_group,
    )

    data_source = context.data_sources.add_pandas("validation_source")
    data_asset = data_source.add_dataframe_asset("validation_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("validation_batch")

    validation_def = gx.ValidationDefinition(
        name="car_data_validation", data=batch_definition, suite=suite,
    )
    results = validation_def.run(batch_parameters={"dataframe": df})
    
    def _parse_validation_results(validation_results) -> pd.DataFrame:
        validation_dict = (
            validation_results.to_json_dict()
            if hasattr(validation_results, "to_json_dict")
            else validation_results
        )
        results = validation_dict.get("results", [])
        rows = []
        for result in results:
            kwargs = result.get("expectation_config", {}).get("kwargs", {})
            res = result.get("result", {})
            rows.append(
                {
                    "success": result.get("success", ""),
                    "expectation_type": result.get("expectation_config", {}).get(
                        "expectation_type", ""
                    ),
                    "column": kwargs.get("column", ""),
                    "min_value": kwargs.get("min_value", ""),
                    "max_value": kwargs.get("max_value", ""),
                    "element_count": res.get("element_count", ""),
                    "unexpected_count": res.get("unexpected_count", ""),
                    "unexpected_percent": res.get("unexpected_percent", ""),
                }
            )
        return pd.DataFrame(rows)

    
    results_df = _parse_validation_results(results)

    if results.success:
        logger.info("Data quality validation PASSED.")
    else:
        logger.error("Data quality validation FAILED. See results table for details.")
        if fail_on_error:
            raise ValueError(
                "Data quality validation failed -- halting pipeline before "
                "data_cleaning. See validation results node output for details."
            )

    return results_df

def ingestion(
    df: pd.DataFrame,
    validation_report: dict,  # not used directly, but forces node ordering in the DAG
    group_name: str,
    group_description: list
):
    """Push validated data to the Feature Store. Only called if validate_data
    didn't raise (or, if fail_on_error=False, runs regardless but logs the report)."""

    # Feature-store-safe column names
    column_renames = {
        "paintQuality%": "paint_quality_pct"
    }

    # hopsworks does not acccept % in col names 
    df = df.rename(columns=column_renames)

    # hopsworks does not accept nan numpy type 
    df = df.replace({np.nan: None})
    
    feature_store = FeatureStoreManager(credentials=credentials)

    return feature_store.upload(
        dataframe=df,
        name=group_name,
        version="1",
        description=group_description,
        primary_key=["carid"]
    )
    

