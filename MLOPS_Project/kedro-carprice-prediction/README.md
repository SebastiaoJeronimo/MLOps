# Kedro Car Price Prediction — Local Run Guide

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## What this project does

Predicts **used car price** from listing features (brand, model, year, mileage, etc.).

| Component | Role |
|-----------|------|
| **Kedro** | Modular ML pipelines (clean → train → predict) |
| **MLflow** | Experiment tracking + model registry (`car_price_model@Champion`) |
| **Hopsworks** | Optional feature store (`data_quality` pipeline) |

---

## Prerequisites

- Python 3.10+ (3.13 recommended)
- [uv](https://docs.astral.sh/uv/) package manager
- Raw data: `data/01_raw/train.csv` and `data/01_raw/test.csv`

---

## Install (first time)

From the repo root, go into this folder and sync dependencies:

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv sync
```

All pipeline dependencies are listed in `pyproject.toml`. If a package is still missing:

```powershell
uv add kedro-datasets rapidfuzz category-encoders mlflow xgboost lightgbm great-expectations kedro-mlflow optuna
```

---

## How the system works

```text
train.csv / test.csv
        ↓
   data_cleaning          ← fix typos, rename columns, bad values
        ↓
   split_pipeline         ← train/validation split (from train only)
        ↓
   preprocess_train       ← feature engineering, encoding, scaling
        ↓
   model_train            ← fit model → MLflow Champion
        ↓
   preprocess_batch       ← same transforms on test.csv
        ↓
   model_predict          ← predictions.csv
```

MLflow artifacts live in the parent folder: `../mlflow.db` and `../mlartifacts/`.

---

## Run locally (two terminals, one folder)

Use **`kedro-carprice-prediction/`** for both terminals (same `.venv`).

### Terminal 1 — MLflow server (keep open)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

Open: **http://127.0.0.1:8080**

Use `--host 0.0.0.0` so Docker can reach MLflow. Do **not** put `0.0.0.0` in Kedro's tracking URI.

### MLflow URLs — don't mix these up

| What | Value | When |
|------|--------|------|
| MLflow server `--host` | `0.0.0.0` | Terminal 1 — server bind address |
| MLflow `--allowed-hosts` | `localhost:8080,127.0.0.1:8080,host.docker.internal:8080` | Required for Docker API |
| Browser / MLflow UI | `http://127.0.0.1:8080` | Open in browser |
| Kedro `MLFLOW_TRACKING_URI` | `http://127.0.0.1:8080` | Terminal 2 — Kedro on the host |
| Docker `MLFLOW_TRACKING_URI` | `http://host.docker.internal:8080` | API container |

### Terminal 2 — Kedro pipelines

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
```

`MLFLOW_TRACKING_URI` is the address Kedro **connects to** — use `127.0.0.1`, not `0.0.0.0` (same as `mlflow.set_tracking_uri(...)` in `notebooks/03_MLflow_CarPrice.ipynb`).

### Do I need MLflow running?

**Yes — keep Terminal 1 (MLflow server) open for the whole workflow.**

| Step | MLflow required? | Why |
|------|------------------|-----|
| `production_full_train_process` | **Yes** | Logs the run and **registers** `car_price_model` with the **Champion** alias in the Model Registry |
| `production_full_prediction_process` | **Yes** | `model_predict` loads **`models:/car_price_model@Champion`** from the registry |
| Docker / Kubernetes API | **Yes** | FastAPI loads Champion at startup — `http://host.docker.internal:8080` in Docker/K8s pods |

The model files live under `../mlartifacts/`, but the **registry** (which version is Champion) is served by **MLflow on port 8080**. If MLflow is stopped, Kedro predict and the HTTP API cannot resolve `car_price_model@Champion`.

**Typical order:**

1. Start MLflow (Terminal 1) — leave it running  
2. Run train pipeline (Terminal 2) — creates Champion in registry  
3. Optionally run predict pipeline — still needs MLflow  
4. Copy `.pkl` files to `serving/artifacts/`  
5. Start Docker or Kubernetes API — **MLflow must still be running**

Verify Champion exists: http://127.0.0.1:8080 → **Models** → `car_price_model` → alias **Champion**

---

## Pipelines reference

| Pipeline | What it does |
|----------|----------------|
| `data_quality` | Great Expectations validation + optional Hopsworks upload |
| `data_cleaning` | Fix typos, bad values, standardize column names |
| `split_pipeline` | Split cleaned train data into train / validation |
| `preprocess_train` | Feature engineering, encoding, scaling → `X_train`, `y_train`, etc. |
| `model_train` | Train model, log metrics, register Champion in MLflow |
| `preprocess_batch` | Apply saved preprocessing to `test.csv` |
| `model_predict` | Load Champion model, write `predictions.csv` |
| `data_drifts` | Evidently drift report → `data/08_reporting/drift_report.html` |
| `production_full_train_process` | **All training steps in one command** |
| `production_full_prediction_process` | **All prediction steps in one command** |

List registered pipelines:

```powershell
uv run kedro registry list
```

---

## Option A — Run together (recommended)

### Train end-to-end

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
uv run kedro run --pipeline production_full_train_process
```

Runs: `data_cleaning` → `split_pipeline` → `preprocess_train` → `model_train`

### Predict end-to-end

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
uv run kedro run --pipeline production_full_prediction_process
```

Runs: `data_cleaning` → `preprocess_batch` → `model_predict`  
Output: `data/07_model_output/predictions.csv`

### Data drift

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
$env:PYTHONUTF8 = "1"
uv run kedro run --pipeline data_drifts
```

**Expected outputs** (pipeline succeeded if both exist and `drift_report.html` is non-empty):

| File | Path |
|------|------|
| Drift HTML report | `data/08_reporting/drift_report.html` |
| Drift metrics | `data/08_reporting/drift_metrics.json` |

```powershell
Test-Path .\data\08_reporting\drift_report.html
Get-Content .\data\08_reporting\drift_metrics.json
Start-Process .\data\08_reporting\drift_report.html
```

**MLflow must still be running** — the kedro-mlflow plugin connects at pipeline startup.

---

## Option B — Run separately (step by step)

Use this to debug or run only part of the flow.

| Step | Command | Needs MLflow? |
|------|---------|---------------|
| 0 | MLflow server in Terminal 1 | — |
| 1 | `uv run kedro run --pipeline data_cleaning` | No |
| 2 | `uv run kedro run --pipeline split_pipeline` | No |
| 3 | `uv run kedro run --pipeline preprocess_train` | No |
| 4 | `uv run kedro run --pipeline model_train` | **Yes** |
| 5 | `uv run kedro run --pipeline preprocess_batch` | No (needs step 3 artifacts) |
| 6 | `uv run kedro run --pipeline model_predict` | **Yes** |

Optional (feature store deliverable):

```powershell
uv run kedro run --pipeline data_quality
```

Requires `conf/local/credentials.yml` with Hopsworks API key.

---

## Notebook fallback

If `kedro run` fails (e.g. catalog issues), use notebooks with MLflow server running:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
uv run jupyter lab
```

Run in order:

1. `notebooks/01_Data_Unit_Tests_CarPrice.ipynb`
2. `notebooks/02_Feature_Store_CarPrice.ipynb` (optional — Hopsworks)
3. `notebooks/03_MLflow_CarPrice.ipynb`

---

## Data paths

| File | Path |
|------|------|
| Raw train | `data/01_raw/train.csv` |
| Raw test | `data/01_raw/test.csv` |
| Cleaned data | `data/02_intermediate/` |
| Model inputs | `data/05_model_input/` |
| Predictions | `data/07_model_output/predictions.csv` |
| **Drift report (verify pipeline)** | `data/08_reporting/drift_report.html` |
| Drift metrics | `data/08_reporting/drift_metrics.json` |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No such command 'registry'` | Run commands from `kedro-carprice-prediction/`, not parent `MLOPS_Project/` |
| `ModuleNotFoundError` | `uv add <package>` or `uv sync` |
| MLflow connection refused | Start Terminal 1 MLflow server first |
| Docker API times out loading Champion | Restart MLflow with `--serve-artifacts` and `--allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"` |
| K8s pods crash loading model | Same MLflow command — see **Deployment B** |
| `No module named 'kedro_datasets'` | `uv add kedro-datasets` |
| Hopsworks errors | Skip `data_quality` or add `conf/local/credentials.yml` |

---

## Deployment — HTTP API

**Prerequisite:** Run `production_full_train_process` first so **`car_price_model@Champion`** exists in MLflow. **Keep the MLflow server running** (Terminal 1) — the API loads the model from the **Model Registry**, not from a local `.pkl` file alone.

After training, copy preprocessing pickles for the API (one time):

```powershell
Copy-Item data\04_feature\imputation_stats.pkl ..\serving\artifacts\
Copy-Item data\04_feature\preprocessing_artifacts.pkl ..\serving\artifacts\
```

Keep the MLflow server running (Terminal 1) while the API container or Kubernetes pods start. If MLflow stops, `/ready` may fail because Champion cannot be loaded.

### Deployment A — Docker (Week 4)

Single container. Fastest way to demo `/predict` and Swagger.

Full steps: [../serving/README.md](../serving/README.md) **Option A**

```powershell
cd ..\serving
docker build -t car-price-api .
docker run -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:8080 `
  -v C:\IMS\MLOps\MLOps\MLOPS_Project\mlartifacts:/mlartifacts:ro `
  -v car_price_history:/data/predictions `
  car-price-api
```

### Deployment B — Kubernetes / KIND (Week 5)

**2 pods minimum** (each runs the full model). HPA scales 2–6 pods under load. Shared prediction history via hostPath volume.

Prerequisites: Docker Desktop, kind, kubectl.

**Terminal 1 — MLflow** (pods reach the host via `host.docker.internal:8080`):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

**Terminal 2 — KIND (copy-paste all):**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
docker build -t car-price-api-kubernetes:latest .
kind create cluster --config k8s/kind.yml --name car-price-cluster
kind load docker-image car-price-api-kubernetes:latest --name car-price-cluster
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json --patch-file k8s/metrics-server-patch.json
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/hpa.yml
kubectl get pods
kubectl port-forward service/car-price-api 8000:8000
```

Command reference: [../serving/README.md](../serving/README.md) **Option C**.

Open http://localhost:8000/docs — try `POST /predict/single` and `GET /predictions/history`.

---

## Project conventions

- Don't commit data to the repository (`data/` is gitignored).
- Keep credentials in `conf/local/` (not committed).
- See [Kedro documentation](https://docs.kedro.org) for framework details.

## Tests

```powershell
uv run pytest
```
