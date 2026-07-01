# MLOps Car Price Project

Predict used car prices with **Kedro** pipelines, **MLflow** model registry, and an optional **FastAPI** API.

## Repository layout

```text
MLOPS_Project/
├── mlflow.db                    # MLflow tracking database
├── mlartifacts/                 # Logged model files
├── kedro-carprice-prediction/   # Kedro app — run pipelines here
└── serving/                     # FastAPI + Docker API
```

## Prerequisites

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Raw data in `kedro-carprice-prediction/data/01_raw/train.csv` and `test.csv`
- Docker Desktop (only for HTTP serving in `serving/`)

## Quick start (Kedro + MLflow)

Open **two PowerShell terminals**. Use the same folder for both: `kedro-carprice-prediction/`.

**Terminal 1 — MLflow (keep running)**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv sync
uv run mlflow server --host 127.0.0.1 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db
```

Open **http://127.0.0.1:8080**

**Terminal 2 — Kedro pipelines**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
uv run kedro run --pipeline production_full_train_process
uv run kedro run --pipeline production_full_prediction_process
```

Predictions: `kedro-carprice-prediction/data/07_model_output/predictions.csv`

## Quick start (HTTP API)

After training (and with MLflow still running):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
docker build -t car-price-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:8080 car-price-api
```

API docs: **http://localhost:8000/docs**

## Full documentation

| Guide | Contents |
|-------|----------|
| [kedro-carprice-prediction/README.md](kedro-carprice-prediction/README.md) | Install, pipelines, step-by-step runs, troubleshooting |
| [serving/README.md](serving/README.md) | Docker build/run, `/predict` examples, optional Kubernetes |
