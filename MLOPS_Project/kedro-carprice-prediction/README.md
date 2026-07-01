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
uv run mlflow server --host 127.0.0.1 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db
```

Open: **http://127.0.0.1:8080**

### Terminal 2 — Kedro pipelines

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
```

`MLFLOW_TRACKING_URI` tells Kedro where the MLflow server is (same as `mlflow.set_tracking_uri(...)` in `notebooks/03_MLflow_CarPrice.ipynb`).

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

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No such command 'registry'` | Run commands from `kedro-carprice-prediction/`, not parent `MLOPS_Project/` |
| `ModuleNotFoundError` | `uv add <package>` or `uv sync` |
| MLflow connection refused | Start Terminal 1 MLflow server first |
| `No module named 'kedro_datasets'` | `uv add kedro-datasets` |
| Hopsworks errors | Skip `data_quality` or add `conf/local/credentials.yml` |

---

## HTTP API (FastAPI + Docker)

After `production_full_train_process`, copy preprocessing pickles for the API (one time):

```powershell
Copy-Item data\04_feature\imputation_stats.pkl ..\serving\artifacts\
Copy-Item data\04_feature\preprocessing_artifacts.pkl ..\serving\artifacts\
```

Full Docker and `/predict` instructions: [../serving/README.md](../serving/README.md)

Keep the MLflow server running (Terminal 1) while the API container starts.

---

## Project conventions

- Don't commit data to the repository (`data/` is gitignored).
- Keep credentials in `conf/local/` (not committed).
- See [Kedro documentation](https://docs.kedro.org) for framework details.

## Tests

```powershell
uv run pytest
```
