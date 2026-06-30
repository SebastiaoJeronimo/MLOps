from __future__ import annotations
import pandas as pd
from .client import FeatureStoreClient
from hopsworks.client.exceptions import RestAPIError


import logging
logger = logging.getLogger(__name__)

class FeatureStoreManager:

    def __init__(self, credentials: dict):
        self.client = FeatureStoreClient(credentials)

    def get_or_create_feature_group(
        self,
        *,
        name: str,
        version: int,
        description: str,
        primary_key: list[str],
        online_enabled: bool = False,
    ):
        return self.client.feature_store.get_or_create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=primary_key,
            online_enabled=online_enabled,
            time_travel_format = "HUDI"

        )

    def upload(
        self,
        dataframe: pd.DataFrame,
        *,
        name: str,
        version: int,
        description: str,
        primary_key: list[str],
        feature_descriptions: list | None = None,
        overwrite: bool = False,
        compute_statistics: bool = True,
        online_enabled: bool = False,
    ):

        fg = self.get_or_create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=primary_key,
            online_enabled=online_enabled,
        )

        try:
            fg.insert(
                features=dataframe,
                overwrite=overwrite,
                write_options={"wait_for_job": True}
            )
        except RestAPIError as e:
            if "415" in str(e) or "Unsupported Media Type" in str(e):
                logger.warning(
                    f"Insert raised non-fatal 415 on job-execution launch "
                    f"for '{name}' v{version}: {e}"
                )
            else:
                raise

        if feature_descriptions:
            for feature in feature_descriptions:
                fg.update_feature_description(
                    feature["name"],
                    feature["description"],
                )

        if compute_statistics:
            fg.compute_statistics()

        return fg

    def get_feature_group(self, name: str, version: int):
        return self.client.feature_store.get_feature_group(
            name=name,
            version=version,
        )

    def read(self, name: str, version: int):
        fg = self.get_feature_group(name, version)
        return fg.read()