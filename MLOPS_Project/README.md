# MLOps Car Price Project

**Students:** Diogo Tibério (20250341), José Montez (20250351), Henrique Figueiredo (20250433), Sebastião Jerónimo (20240660)

Predict used car prices with **Kedro** pipelines, **MLflow** model registry, and an optional **FastAPI** API.

## Repository layout

```text
MLOPS_Project/
├── mlflow.db                    # MLflow tracking database
├── mlartifacts/                 # Logged model files
├── kedro-carprice-prediction/   # Kedro app — run pipelines here
└── serving/                     # FastAPI + Docker + Kubernetes
    └── postman/                 # Postman collection + environment
```

## Quick start — Kedro + MLflow

Open **two PowerShell terminals** in `kedro-carprice-prediction/`.

**Terminal 1 — MLflow:**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

Open UI: **http://127.0.0.1:8080**

**Terminal 2 — Kedro** (use `127.0.0.1` — Kedro runs on the host, not in Docker):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
$env:PYTHONUTF8 = "1"
uv sync
uv run kedro run --pipeline production_full_train_process
uv run kedro run --pipeline production_full_prediction_process
uv run kedro run --pipeline data_drifts
```

**Verify drift:** `Test-Path kedro-carprice-prediction\data\08_reporting\drift_report.html` should return `True`, then open that HTML file in a browser. Details: [HOW_TO_RUN.md](HOW_TO_RUN.md#verify-pipeline-outputs).

Full pipeline guide: [kedro-carprice-prediction/README.md](kedro-carprice-prediction/README.md)

### MLflow URLs

| Role | URL / host |
|------|------------|
| MLflow server `--host` | `0.0.0.0` |
| MLflow `--allowed-hosts` | `localhost:8080,127.0.0.1:8080,host.docker.internal:8080` |
| Kedro on host | `http://127.0.0.1:8080` |
| Docker API | `http://host.docker.internal:8080` |

Do not use `0.0.0.0` in `MLFLOW_TRACKING_URI` — it is only the server listen address.

### Important: keep MLflow running after pipelines

Training registers **`car_price_model@Champion`** in the Model Registry. Prediction (Kedro or HTTP API) **loads that Champion from MLflow** — it does not use a standalone model file on disk.

So for deployment:

1. Train with Kedro (Terminal 2) while MLflow runs (Terminal 1)  
2. **Leave MLflow running**  
3. Start Docker or Kubernetes — API still needs `MLFLOW_TRACKING_URI=http://host.docker.internal:8080`

## Deploy API — Docker (Week 4, simple)

Image: **`car-price-api`** (`latest`). Single container, quick demo.

See [serving/README.md](serving/README.md) **Option A**

## Deploy API — Kubernetes (Week 5, 2 pods + HPA)

**Requires:** Docker Desktop + **kind** — run `. .\k8s\setup-kind.ps1` from `serving/` (see [HOW_TO_RUN.md](HOW_TO_RUN.md#option-b--kubernetes-with-kind-week-5)).

Image: **`car-price-api-kubernetes`** (`latest`) — same Dockerfile, different tag so Docker and KIND images stay separate.

Same image on a local KIND cluster — minimum **2 replicas**, autoscaling up to 6.

**MLflow on the host** must use `--serve-artifacts` and `--allowed-hosts` (pods call `http://host.docker.internal:8080`):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

See [serving/README.md](serving/README.md) **Option C**

## Test API with Postman

Import **`serving/postman/Car-Price-API.postman_collection.json`** and **`serving/postman/Local.postman_environment.json`**, then select environment **Car Price API — Local**.
