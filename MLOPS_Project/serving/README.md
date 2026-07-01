# Car Price API — FastAPI + Docker + Kubernetes

HTTP serving for the production model **`car_price_model@Champion`** from the **MLflow Model Registry**.

## Before you start

1. **Train the model** with Kedro — `production_full_train_process` (registers Champion in MLflow)
2. **MLflow server must be running** on port 8080 — required for train, predict, **and** this API (all load from the registry)
3. **Preprocessing artifacts** in `serving/artifacts/` (copy after training):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
Copy-Item ..\kedro-carprice-prediction\data\04_feature\imputation_stats.pkl artifacts\
Copy-Item ..\kedro-carprice-prediction\data\04_feature\preprocessing_artifacts.pkl artifacts\
```

### Terminal 1 — MLflow (from kedro folder)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

**Do not stop Terminal 1** when you move on to Kedro pipelines or Docker/Kubernetes. The API calls `models:/car_price_model@Champion` at startup — that URI is resolved by the MLflow server.

MLflow 3.x uses **localhost-only security middleware** by default. Without `--allowed-hosts`, Docker/Kubernetes requests from `host.docker.internal` are rejected (timeout or 403). Without `--serve-artifacts`, containers cannot download model files (`No such artifact: ''`). Restart MLflow with the full command above if you see those errors.

### MLflow URLs — don't mix these up

| What | Value | When |
|------|--------|------|
| MLflow server `--host` | `0.0.0.0` | Terminal 1 — Docker and KIND pods reach MLflow on Windows |
| MLflow `--serve-artifacts` | (flag) | Required for Docker and K8s — serve model files over HTTP |
| MLflow `--allowed-hosts` | `localhost:8080,127.0.0.1:8080,host.docker.internal:8080` | Required for Docker/K8s (MLflow 3.x security middleware) |
| Browser / MLflow UI | `http://127.0.0.1:8080` | Open in browser |
| Kedro `MLFLOW_TRACKING_URI` | `http://127.0.0.1:8080` | Kedro on the host |
| Docker / K8s `MLFLOW_TRACKING_URI` | `http://host.docker.internal:8080` | `docker run` or `k8s/deployment.yml` |
| `mlartifacts` volume mount | `-v .../mlartifacts:/mlartifacts:ro` | Champion model files (registry metadata still via HTTP) |

Never put `0.0.0.0` in `MLFLOW_TRACKING_URI` — clients cannot connect to that address.

---

## Which deployment should I use?

| | Option A — Docker (Week 4) | Option C — Kubernetes/KIND (Week 5) |
|--|------------------------------|-------------------------------------|
| **Complexity** | Low | Higher |
| **Containers** | 1 | **Min 2 pods** (each with full model) |
| **Scaling** | Manual | **HPA** (2–6 pods under load) |
| **Best for** | Quick demo, report baseline | Production-like PoC |

---

## Option A — Docker (Week 4)

**Terminal 2:**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
docker build -t car-price-api .
docker run -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:8080 `
  -v C:\IMS\MLOps\MLOps\MLOPS_Project\mlartifacts:/mlartifacts:ro `
  -v car_price_history:/data/predictions `
  car-price-api
```

The named volume `car_price_history` persists prediction request/response history across container restarts.

| URL | Purpose |
|-----|---------|
| http://localhost:8000/health | Liveness check |
| http://localhost:8000/ready | Model loaded |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/predictions/history | Past predictions |

### Postman

Import from **`postman/`** in this folder:

- `postman/Car-Price-API.postman_collection.json`
- `postman/Local.postman_environment.json`

Set environment **Car Price API — Local** (`baseUrl` = `http://localhost:8000`).

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
Invoke-RestMethod -Uri http://localhost:8000/predictions/history
```

---

## Option B — Local uvicorn (no Docker)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
pip install -r requirements.txt
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
$env:PREDICTION_HISTORY_DIR = ".\data\predictions"
New-Item -ItemType Directory -Force -Path $env:PREDICTION_HISTORY_DIR
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Option C — Kubernetes with KIND (Week 5)

**Prerequisites:** Docker Desktop **running**, [kind](https://kind.sigs.k8s.io/), kubectl.

### Install and enable kind (Windows)

```powershell
winget install --id Kubernetes.kind -e --accept-source-agreements --accept-package-agreements
```

**Every KIND session** — from `serving/`:

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
. .\k8s\setup-kind.ps1
```

The script finds `kind.exe` (winget portable install or existing PATH), installs if missing, and prints `kind version`.

Check without the script: `Get-Command kind` — if empty, run `setup-kind.ps1`.

Then verify:

```powershell
docker info
kind version
kubectl version --client
```

**Important:** If `kind version` fails, **do not** run `kubectl apply` — you will get `Error from server (NotFound): the server could not find the requested resource` because no cluster exists yet.

Each pod runs **one container with the full FastAPI app and Champion model**. With `minReplicas: 2`, you always have **at least 2 identical inference containers**. HPA can scale up to 6 under CPU load.

### Step 0 — MLflow on the host (Terminal 1)

Kubernetes pods use `MLFLOW_TRACKING_URI=http://host.docker.internal:8080` (see `k8s/deployment.yml`). **Keep MLflow running** on the host with the same flags as Docker:

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

| Flag | Why pods need it |
|------|------------------|
| `--host 0.0.0.0` | Accept connections from KIND nodes |
| `--allowed-hosts ...host.docker.internal:8080` | Avoid **403** (`Rejected request with invalid Host header`) |
| `--serve-artifacts` | Avoid **`No such artifact: ''`** when loading Champion |

Verify Champion: http://127.0.0.1:8080 → **Models** → `car_price_model` → **Champion**

### Step 1 — Terminal 2: full KIND deploy

Edit `k8s/kind.yml` `extraMounts.hostPath` if your `mlartifacts` path differs.

#### What each command does

| Command | What it does |
|---------|----------------|
| `. .\k8s\setup-kind.ps1` | Installs kind if missing, finds `kind.exe`, adds to PATH |
| `kind create cluster ...` | Local Kubernetes cluster in Docker (`k8s/kind.yml`: 1 control-plane + 2 workers, `mlartifacts` mount) |
| `kind load docker-image ...` | Imports locally built image into the cluster |
| `kubectl apply -f k8s/deployment.yml` | **2 pods** with FastAPI + MLflow env vars |
| `kubectl apply -f k8s/service.yml` | Service `car-price-api` on port 8000 |
| `kubectl apply -f k8s/hpa.yml` | HPA: **2–6 pods** on CPU (needs metrics-server) |
| `kubectl port-forward ...` | **localhost:8000** — keep terminal open |

#### Copy-paste all (Terminal 2)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving

# Setup kind (install if missing, add to PATH)
. .\k8s\setup-kind.ps1
docker info

# Build image and create cluster
docker build -t car-price-api-kubernetes:latest .

kind create cluster --config k8s/kind.yml --name car-price-cluster
kind load docker-image car-price-api-kubernetes:latest --name car-price-cluster

# Confirm cluster before kubectl
kind get clusters
kubectl cluster-info

# Metrics-server (needed for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json --patch-file k8s/metrics-server-patch.json

# Deploy API (2 pods + HPA)
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/hpa.yml
kubectl get pods

# Open API on localhost (leave this running)
kubectl port-forward service/car-price-api 8000:8000
```

**After it runs:** http://localhost:8000/docs — `Invoke-RestMethod http://localhost:8000/health`

**metrics-server + patch:** only needed for **HPA**. Skip those lines and `hpa.yml` if you only want 2 fixed pods.

### Cleanup

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
. .\k8s\setup-kind.ps1
kind delete cluster --name car-price-cluster
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Model loaded |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Batch: `{"cars": [{...}]}` |
| POST | `/predict/single` | One car in JSON body |
| GET | `/predictions/history` | Recent requests + responses (`?limit=50`) |
| GET | `/predictions/history/{id}` | Single history record |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `kind` is not recognized | From `serving/`: `. .\k8s\setup-kind.ps1` |
| `kubectl` … `could not find the requested resource` | No cluster running — run `kind create cluster` first; check `kind get clusters` |
| `/ready` returns 503 / model not loaded | **Start MLflow first** — API loads `car_price_model@Champion` from the registry |
| Connection refused to MLflow | Start MLflow in Terminal 1 and keep it running through train + serving |
| Docker MLflow timeout / startup failed | Restart MLflow with `--allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"` |
| MLflow log: `Rejected request with invalid Host header: host.docker.internal` | Same — add `--allowed-hosts` as above |
| Docker error: `No such artifact: ''` or missing `MLmodel` | Mount `mlartifacts` at `/mlartifacts` and rebuild the API image |
| Docker cannot reach MLflow | Use `host.docker.internal:8080` on Windows |
| K8s pods `CrashLoopBackOff` / `/ready` 503 | Same MLflow flags as Docker — `--serve-artifacts` + `--allowed-hosts`; check `kubectl logs deployment/car-price-api` |
| K8s MLflow 403 or timeout | Restart MLflow with full command in **Option C Step 0** |
| KIND pods ImagePullBackOff | Run `kind load docker-image car-price-api-kubernetes:latest --name car-price-cluster` |
| HPA shows `<unknown>` | Install metrics-server and run `kubectl patch ... --patch-file k8s/metrics-server-patch.json` |
| Empty prediction history in K8s | Predict first via port-forward, then `GET /predictions/history` |
| `Missing ... imputation_stats.pkl` | Run train pipeline and copy `.pkl` files to `artifacts/` |

---

## Report: Docker vs Kubernetes

**Docker:** Single container, fastest to demo, named volume for history.

**Kubernetes:** Same image, **minimum 2 pods** each loading the full model, health probes, HPA autoscaling, shared hostPath history — mirrors Week 5 production patterns at the cost of kind/kubectl setup.
