from hopsworks import login


class FeatureStoreClient:
    """Handles authentication with Hopsworks."""

    def __init__(self, credentials: dict):
        self._credentials = credentials
        self._project = None
        self._feature_store = None

    @property
    def project(self):
        if self._project is None:
            self._project = login(
                api_key_value=self._credentials["feature_store"]["FS_API_KEY"],
                project=self._credentials["feature_store"]["FS_PROJECT_NAME"],
            )
        return self._project

    @property
    def feature_store(self):
        if self._feature_store is None:
            self._feature_store = self.project.get_feature_store()

        return self._feature_store