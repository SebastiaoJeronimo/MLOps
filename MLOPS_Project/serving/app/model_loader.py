"""Load the Champion model from MLflow Model Registry."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow


def _registry_download_uri_to_mount_path(download_uri: str, mount_root: str) -> str | None:
    """Map MLflow file:.../mlartifacts/... URI to a path inside a mounted volume."""
    if not download_uri.startswith("file:"):
        return None

    path = download_uri.removeprefix("file:").replace("\\", "/")
    marker = "/mlartifacts/"
    idx = path.lower().find(marker)
    if idx < 0:
        return None

    suffix = path[idx + len(marker) :]
    local_path = Path(mount_root) / suffix
    if (local_path / "MLmodel").is_file():
        return str(local_path)
    return None


def load_champion_model(
    registered_model_name: str | None = None,
    champion_alias: str | None = None,
):
    """Load Champion from the registry.

    In Docker/K8s, mount the host ``mlartifacts`` folder at ``MLFLOW_ARTIFACT_MOUNT``
    (default ``/mlartifacts``). Registry metadata still comes from MLflow HTTP; model
    files are read from the mount because local ``file://`` paths are host-specific.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    mlflow.set_tracking_uri(tracking_uri)

    name = registered_model_name or os.getenv("REGISTERED_MODEL_NAME", "car_price_model")
    alias = champion_alias or os.getenv("CHAMPION_ALIAS", "Champion")
    artifact_mount = os.getenv("MLFLOW_ARTIFACT_MOUNT", "/mlartifacts")

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    download_uri = client.get_model_version_download_uri(name, mv.version)

    mounted_path = _registry_download_uri_to_mount_path(download_uri, artifact_mount)
    if mounted_path:
        return mlflow.pyfunc.load_model(mounted_path)

    mount = Path(artifact_mount)
    if not mount.is_dir():
        raise RuntimeError(
            f"Champion model files not found. Mount host mlartifacts into the container, e.g. "
            f"-v C:\\IMS\\MLOps\\MLOps\\MLOPS_Project\\mlartifacts:{artifact_mount}:ro "
            f"(download_uri={download_uri!r})"
        )

    raise RuntimeError(
        f"Champion artifacts missing under {artifact_mount}. "
        f"Expected MLmodel at mapped path for {download_uri!r}"
    )
