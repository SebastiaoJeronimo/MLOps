# Car Price API — FastAPI + Docker

HTTP serving for the production model **`car_price_model@Champion`** from MLflow.

## Before you start

1. **Train the model** (if not done yet) — see [kedro-carprice-prediction/README.md](../kedro-carprice-prediction/README.md)
2. **MLflow server running** on port 8080
3. **Preprocessing artifacts** in `serving/artifacts/` (copy after training):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
Copy-Item ..\kedro-carprice-prediction\data\04_feature\imputation_stats.pkl artifacts\
Copy-Item ..\kedro-carprice-prediction\data\04_feature\preprocessing_artifacts.pkl artifacts\
```

### Terminal 1 — MLflow (from kedro folder)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 127.0.0.1 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db
```

---

## Option A — Docker (recommended)

**Terminal 2:**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
docker build -t car-price-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:8080 car-price-api
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/health | Liveness check |
| http://localhost:8000/ready | Model loaded |
| http://localhost:8000/docs | Swagger UI — try `POST /predict/single` |

### Example prediction (PowerShell)

```powershell
$body = @{
  brand = "Toyota"
  model = "Yaris"
  year = 2019
  transmission = "Manual"
  mileage = 25000
  fuelType = "Petrol"
  tax = 145
  mpg = 55
  engineSize = 1.5
  previousOwners = 1
  paintQuality = 85
  hasDamage = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict/single -Method POST -Body $body -ContentType "application/json"
```

---

## Option B — Local uvicorn (no Docker)

Useful if Docker is not available. MLflow must still be running.

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
pip install -r requirements.txt
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/docs

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Model loaded |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Batch: `{"cars": [{...}, {...}]}` |
| POST | `/predict/single` | One car in JSON body |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Connection refused to MLflow | Start MLflow in Terminal 1 first |
| Docker cannot reach MLflow | Use `host.docker.internal:8080` on Windows |
| `Missing ... imputation_stats.pkl` | Run train pipeline and copy `.pkl` files to `artifacts/` |
| Model load 404 | Confirm Champion exists at http://127.0.0.1:8080 → Models → `car_price_model` |

---

## Kubernetes (optional)

Manifests in `k8s/` (Week 5 lab pattern). Requires [kind](https://kind.sigs.k8s.io/) and kubectl.

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
docker build -t car-price-api:v1.0 .
kind load docker-image car-price-api:v1.0
kubectl apply -f k8s/
kubectl port-forward service/car-price-api 8000:8000
```

MLflow must be reachable from the cluster (adjust `MLFLOW_TRACKING_URI` in `k8s/deployment.yml` if needed).
