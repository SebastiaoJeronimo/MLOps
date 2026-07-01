# MLOps Car Price Project — How to Run

**Students:** Diogo Tibério (20250341), José Montez (20250351), Henrique Figueiredo (20250433), Sebastião Jerónimo (20240660)

Predict used car prices with **Kedro** pipelines, **MLflow** model registry, and a **FastAPI** API (Docker or Kubernetes).

## Repository layout

```text
MLOPS_Project/
├── mlflow.db                    # MLflow tracking database
├── mlartifacts/                 # Logged model files
├── kedro-carprice-prediction/   # Kedro app — run pipelines here
└── serving/                     # FastAPI + Docker + Kubernetes
    └── postman/                 # Postman collection + environment (API requests)
```

## Prerequisites

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Raw data in `kedro-carprice-prediction/data/01_raw/train.csv` and `test.csv`
- Docker Desktop (for HTTP serving)
- For Kubernetes: [kind](https://kind.sigs.k8s.io/) and kubectl

---

## Step 1 — Kedro + MLflow (two terminals)

Open **two PowerShell terminals**. Use the same folder for both: `kedro-carprice-prediction/`.

**Terminal 1 — MLflow (keep running the whole session)**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv sync
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

Open **http://127.0.0.1:8080** in your browser (UI always uses localhost).

MLflow 3.x blocks Docker by default (`localhost-only` security middleware). **`--allowed-hosts`** is required so `host.docker.internal` can load Champion. If you already started MLflow without it, press **Ctrl+C** and restart with the command above.

### MLflow URLs — don't mix these up

`0.0.0.0` is where the **server listens**. Clients connect to a real address — never use `0.0.0.0` in `MLFLOW_TRACKING_URI`.

| What | Value | When |
|------|--------|------|
| MLflow server `--host` | `0.0.0.0` | Terminal 1 — listen on all interfaces so **Docker** can reach MLflow |
| MLflow `--allowed-hosts` | `localhost:8080,127.0.0.1:8080,host.docker.internal:8080` | Required for Docker — without this, MLflow returns **403** |
| MLflow `--serve-artifacts` | (flag, no value) | Required for Docker — serves model files over HTTP to the container |
| Browser / MLflow UI | `http://127.0.0.1:8080` | Open in browser |
| Kedro `MLFLOW_TRACKING_URI` | `http://127.0.0.1:8080` | Terminal 2 — Kedro runs **on the host** |
| Docker `MLFLOW_TRACKING_URI` | `http://host.docker.internal:8080` | `docker run -e ...` — API runs **inside the container** |

**Terminal 2 — Kedro pipelines**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
$env:PYTHONUTF8 = "1"
uv run kedro run --pipeline production_full_train_process
uv run kedro run --pipeline production_full_prediction_process
uv run kedro run --pipeline data_drifts
```

`$env:PYTHONUTF8 = "1"` avoids Windows encoding errors when Kedro/MLflow write UTF-8 output.

**Optional — hyperparameter tuning:** run `uv run kedro run --pipeline model_selection_pipeline` before train to overwrite `data/06_models/best_params.json` with Optuna results. If that file is missing, train falls back to defaults in `parameters_model_train.yml` (an empty `{}` is shipped in the repo for first-run convenience).

**Note:** Run Kedro from `kedro-carprice-prediction/`, not the parent `MLOPS_Project/` folder — otherwise `uv run kedro` fails with `program not found`.

### Data drift (`data_drifts`)

Run **after** `production_full_train_process` (needs `car_train_clean` from cleaning). Uses [Evidently](https://docs.evidentlyai.com/) to compare a reference half vs a current half of the cleaned training data.

```powershell
uv run kedro run --pipeline data_drifts
```

### Verify pipeline outputs

After all three pipelines finish, check these **expected outputs**:

| Pipeline | Expected file | What success looks like |
|----------|---------------|-------------------------|
| `production_full_train_process` | MLflow UI → `car_price_model` @ **Champion** | Model registered in registry |
| `production_full_prediction_process` | `data/07_model_output/predictions.csv` | CSV with prediction rows |
| `data_drifts` | `data/08_reporting/drift_report.html` | HTML report exists (typically ~1–5 MB) |
| `data_drifts` | `data/08_reporting/drift_metrics.json` | JSON with `dataset_drift`, `drifted_columns`, etc. |

**PowerShell — confirm drift pipeline worked:**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction

# File must exist and be non-empty
Test-Path .\data\08_reporting\drift_report.html
(Get-Item .\data\08_reporting\drift_report.html).Length -gt 0

# Quick view of summary metrics
Get-Content .\data\08_reporting\drift_metrics.json

# Open the HTML report in your default browser
Start-Process .\data\08_reporting\drift_report.html
```

If `Test-Path` returns `False` or the file size is `0`, the `data_drifts` pipeline did not complete successfully — check the Kedro log for errors (often encoding on Windows; keep `$env:PYTHONUTF8 = "1"` set).

Config: `conf/base/parameters_data_drifts.yml`

### Keep MLflow running

**Do not stop Terminal 1** after the pipelines. Train registers **`car_price_model@Champion`** in the Model Registry. Predict, Docker, and Kubernetes all load that Champion from MLflow on port 8080.

Verify: http://127.0.0.1:8080 → **Models** → `car_price_model` → alias **Champion**

---

## Step 2 — Copy preprocessing artifacts (one time, before API)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
Copy-Item ..\kedro-carprice-prediction\data\04_feature\imputation_stats.pkl artifacts\
Copy-Item ..\kedro-carprice-prediction\data\04_feature\preprocessing_artifacts.pkl artifacts\
```

---

## Step 3 — Deploy the HTTP API

Choose **Docker** (simple) or **Kubernetes/KIND** (Week 5, min 2 pods + HPA). MLflow (Terminal 1) must still be running.

| | Docker (Week 4) | Kubernetes / KIND (Week 5) |
|--|-----------------|---------------------------|
| Complexity | Low | Higher |
| Containers | 1 | Min **2 pods** (each with full model) |
| Scaling | Manual | HPA (2–6 pods under load) |

---

### Option A — Docker (Week 4)

**Terminal 2 or 3:**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
docker build -t car-price-api .
docker run -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:8080 `
  -v C:\IMS\MLOps\MLOps\MLOPS_Project\mlartifacts:/mlartifacts:ro `
  -v car_price_history:/data/predictions `
  car-price-api
```

API docs: **http://localhost:8000/docs**

Mount **`mlartifacts`** into the container — Champion files live on the host disk; MLflow only provides registry metadata over HTTP.

---

### Option B — Kubernetes with KIND (Week 5)

**Prerequisites:** Docker Desktop **running**, **kind**, and **kubectl**.

#### Install and enable kind (Windows)

`kubectl` usually comes with **Docker Desktop**. **kind** is installed via winget as a **portable** app (not added to system PATH automatically).

**One-time install** (if needed):

```powershell
winget install --id Kubernetes.kind -e --accept-source-agreements --accept-package-agreements
```

**Every KIND session** — run from `serving/` (installs if missing, finds `kind.exe`, adds to PATH, prints version):

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
. .\k8s\setup-kind.ps1
```

Or check manually without the script:

```powershell
Get-Command kind -ErrorAction SilentlyContinue
# if empty, kind is not on PATH yet — run setup-kind.ps1
```

Then verify Docker and the cluster tool:

```powershell
docker info
kind version
kubectl version --client
kind get clusters
```

| Check | Expected |
|-------|----------|
| `docker info` | No error — Docker Desktop is running |
| `kind version` | Prints kind version (e.g. `kind v0.32.0`) |
| `kubectl version --client` | Prints client version |
| `kind get clusters` | `No kind clusters found.` **before** first deploy — that is OK |

**Do not run `kubectl apply` until `kind create cluster` succeeds.** If kind is missing, every `kubectl` command fails with:

```text
Error from server (NotFound): the server could not find the requested resource
```

**Terminal 1 — MLflow (same as Docker — keep running)**

Pods load Champion from the host MLflow server (`host.docker.internal:8080` in `k8s/deployment.yml`). Start MLflow with **`--serve-artifacts`** and **`--allowed-hosts`**:

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\kedro-carprice-prediction
uv run mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root ../mlartifacts --backend-store-uri sqlite:///../mlflow.db --serve-artifacts --allowed-hosts "localhost:8080,127.0.0.1:8080,host.docker.internal:8080"
```

| Flag | Why Kubernetes needs it |
|------|-------------------------|
| `--host 0.0.0.0` | Listen on all interfaces |
| `--allowed-hosts ...host.docker.internal:8080` | Pods call MLflow with that Host header (avoids **403**) |
| `--serve-artifacts` | Pods download model files over HTTP (avoids **`No such artifact: ''`**) |

**Terminal 2 — full KIND deploy**

**Before you start:**
- Terminal 1 — MLflow still running (command above)
- Preprocessing artifacts in `serving/artifacts/` (see Step 2)
- Edit `k8s/kind.yml` `extraMounts.hostPath` if your repo path is not `C:/IMS/MLOps/MLOps/MLOPS_Project/mlartifacts`

#### What each command does

| Command | What it does |
|---------|----------------|
| `. .\k8s\setup-kind.ps1` | Installs kind via winget if missing, finds `kind.exe`, adds to PATH |
| `kind create cluster --config k8s/kind.yml --name car-price-cluster` | **Creates a local Kubernetes cluster** inside Docker (1 control-plane + 2 worker nodes). `k8s/kind.yml` mounts host `mlartifacts` at `/mlartifacts` on each node. |
| `kind load docker-image car-price-api-kubernetes:latest --name car-price-cluster` | **Copies your Docker image into the cluster.** KIND nodes cannot pull from Docker Hub — only locally built + loaded images. |
| `kubectl apply -f .../components.yaml` | Installs **metrics-server** so HPA can read CPU (optional if you skip HPA). |
| `kubectl patch ... metrics-server-patch.json` | Fixes metrics-server on **local KIND** (`--kubelet-insecure-tls`). Only needed with HPA. |
| `kubectl apply -f k8s/deployment.yml` | **Starts the API pods** — Deployment with **2 replicas** (`car-price-api-kubernetes`). |
| `kubectl apply -f k8s/service.yml` | **Service** `car-price-api` on port 8000 → pods with `app=car-price-api`. |
| `kubectl apply -f k8s/hpa.yml` | **HPA** — min **2** pods, scales up to **6** at 75% CPU. |
| `kubectl get pods` | Wait until **2× `Running`** / `READY 1/1`. |
| `kubectl port-forward service/car-price-api 8000:8000` | Maps **localhost:8000** to the cluster. **Leave this terminal open** while testing. |

#### Copy-paste all (Terminal 2)

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
. .\k8s\setup-kind.ps1
docker info
docker build -t car-price-api-kubernetes:latest .
kind create cluster --config k8s/kind.yml --name car-price-cluster
kind load docker-image car-price-api-kubernetes:latest --name car-price-cluster
kind get clusters
kubectl cluster-info
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json --patch-file k8s/metrics-server-patch.json
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/hpa.yml
kubectl get pods
kubectl port-forward service/car-price-api 8000:8000
```

**After it runs:** open **http://localhost:8000/docs** — test with `Invoke-RestMethod http://localhost:8000/health`

If `kind create cluster` fails with *already exists*: `. .\k8s\setup-kind.ps1` then `kind delete cluster --name car-price-cluster` and retry.

**metrics-server patch — do you need it?** Only if you use **HPA** (`k8s/hpa.yml`). **Skip both** `metrics-server` lines and `hpa.yml` if you only want 2 fixed pods.

One-line patch alternative:

```powershell
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

| Problem | Fix |
|---------|-----|
| `kind` is not recognized | From `serving/`: `. .\k8s\setup-kind.ps1` |
| `kubectl` … `could not find the requested resource` | **No cluster** — `kind create cluster` must succeed first; check `kind get clusters` |
| `kind create cluster` hangs or fails | Docker Desktop running? (`docker info`) |
| Pods `CrashLoopBackOff` | `kubectl logs deployment/car-price-api` — usually MLflow not running or missing `--serve-artifacts` / `--allowed-hosts` |

API docs: **http://localhost:8000/docs**

**Cleanup**

```powershell
cd C:\IMS\MLOps\MLOps\MLOPS_Project\serving
. .\k8s\setup-kind.ps1
kind delete cluster --name car-price-cluster
```

---

## Step 4 — API requests (copy-paste)

Works the same for **Docker** (`localhost:8000`) and **Kubernetes** (after `kubectl port-forward`).

### Postman

Ready-made requests are in **`serving/postman/`**:

| File | Purpose |
|------|---------|
| `Car-Price-API.postman_collection.json` | All API endpoints with example bodies |
| `Local.postman_environment.json` | `baseUrl` = `http://localhost:8000` |

**Import in Postman:** File → Import → select both files → choose environment **Car Price API — Local**.

**`baseUrl` = `http://localhost:8000`** — no trailing slash. If you use `http://localhost:8000/`, requests become `//health` and return **404**.

For KIND, run `kubectl port-forward service/car-price-api 8000:8000` before sending requests.

For `GET /predictions/history/{id}`, copy an `id` from a history response into the `recordId` environment variable.

### Health checks

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/ready
```

### Single prediction — `POST /predict/single`

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

### Batch prediction — `POST /predict`

```powershell
$body = @{
  cars = @(
    @{
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
    },
    @{
      brand = "BMW"
      model = "3 Series"
      year = 2018
      transmission = "Automatic"
      mileage = 45000
      fuelType = "Diesel"
      tax = 150
      mpg = 50
      engineSize = 2.0
      previousOwners = 2
      paintQuality = 75
      hasDamage = 0
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -Body $body -ContentType "application/json"
```

### Prediction history — `GET /predictions/history`

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predictions/history
Invoke-RestMethod -Uri "http://localhost:8000/predictions/history?limit=10"
```

### Single history record — `GET /predictions/history/{id}`

Replace `{id}` with an ID from the history response:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predictions/history/YOUR_ID_HERE
```

### curl (Git Bash / WSL)

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"brand":"Toyota","model":"Yaris","year":2019,"transmission":"Manual","mileage":25000,"fuelType":"Petrol","tax":145,"mpg":55,"engineSize":1.5,"previousOwners":1,"paintQuality":85,"hasDamage":0}'

curl http://localhost:8000/predictions/history
```

---

## Endpoints summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Model loaded from MLflow registry |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Batch: `{"cars": [{...}, {...}]}` |
| POST | `/predict/single` | One car in JSON body |
| GET | `/predictions/history` | Recent requests + responses (`?limit=50`) |
| GET | `/predictions/history/{id}` | Single history record |

Postman files: **`serving/postman/`** (`Car-Price-API.postman_collection.json`, `Local.postman_environment.json`).

---

## Full documentation

| Guide | Contents |
|-------|----------|
| [README.md](README.md) | Project overview and quick start |
| [kedro-carprice-prediction/README.md](kedro-carprice-prediction/README.md) | Install, pipelines, step-by-step runs, troubleshooting |
| [serving/README.md](serving/README.md) | Docker, uvicorn, KIND details, troubleshooting |
