# Enterprise Cloud MLOps: Automated Asset Analytics & Risk Pipeline
[![MLOps CI/CD Pipeline](https://github.com)](https://github.com)

---

### 🚀 Project Status: Active Development & Applied Cloud Learning
> **Note to Reviewers & Hiring Teams:** This repository serves as a live, end-to-end technical lab demonstrating my transition into enterprise cloud data systems.
>
> Having completed **90% of my GitHub Foundations**, I am currently utilizing this framework to practically apply **Microsoft Azure Machine Learning (DP-100)** and **Azure Databricks / Delta Lake** architectures to enterprise operations. The codebase is under active development as I continuously integrate production-grade MLOps features.

---

## 📌 Executive Summary & Business Context
Industrial, mining, and insurance operations across high-capital sectors incur significant financial losses due to unscheduled asset down-time. This project implements an enterprise-grade, end-to-end Machine Learning Operations (MLOps) architecture designed to transition infrastructure maintenance from expensive reactive fixing to automated, data-driven predictive optimization.

## 🛠️ Tech Stack & Infrastructure Layout
* **Data Architecture:** Azure Blob Storage Gen2 (Data Lake), Azure Databricks, PySpark, Delta Lake.
* **Model Governance & Lifecycle:** MLflow (Experiment Tracking, Model Registry).
* **Production Deployment:** Azure Machine Learning Workspace, Azure Managed Online Endpoints.
* **DevOps & Automation:** GitHub Actions (CI/CD Automated Testing Framework), PyTest, Flake8.

## 📁 Repository Structure
```text
enterprise-mlops/
├── .github/workflows/
│   └── mlops_ci_cd.yml       <- Automated GitHub Actions CI/CD script
├── src/
│   ├── etl_pipeline.py       <- Production PySpark cleaning script
│   ├── train.py              <- MLflow training & registry script
│   └── deploy.py             <- Azure ML API deployment script
├── tests/
│   └── test_pipeline_logic.py <- Automated PyTest quality verification script
├── scripts/
│   ├── generate_mock_data.py <- Synthetic data generator engine
│   └── test_endpoint.py      <- Live HTTP REST API verification script
├── data/                     <- Generated locally; not committed (see .gitignore)
├── models/                   <- Generated locally; not committed (see .gitignore)
├── .flake8                   <- Lint configuration
├── requirements.txt          <- Python package dependencies list
└── README.md                 <- Enterprise consulting overview documentation
```

## 🏗️ Technical Pipeline Breakdown

### 1. Reproducible Data Generation (`scripts/generate_mock_data.py`)
To maintain complete environmental reproducibility without compromising sensitive data privacy rules, this framework utilizes a mathematical simulator. It generates thousands of telemetry profiles containing machine operating characteristics (vibration, heat, cycle metrics) alongside failure vectors across distinct regional deployment zones. Roughly one in six simulated assets is given an escalating pre-failure signature, and a small amount of sensor dropout is injected deliberately so the ETL stage has real cleaning work to do.

```bash
python scripts/generate_mock_data.py --assets 400 --readings-per-asset 40 --seed 42
```

### 2. Scalable Data Ingestion & ETL (`src/etl_pipeline.py`)
* **Objective:** Extract raw, messy cloud storage files and prepare structured features.
* **Implementation:** A scalable PySpark script designed to run on **Databricks**, handling random null fields via per-asset-type baseline imputation. Engineers rolling risk flags (short-window rolling averages of vibration/heat that flag early degradation) and writes outputs to optimized, ACID-compliant **Delta Lake** tables partitioned by region.
* **Local development note:** When PySpark isn't available (a laptop, or a GitHub Actions runner with no Spark cluster), the script automatically falls back to an equivalent pandas implementation and writes Parquet instead of Delta. The feature-engineering *rules* are identical in both paths — the pandas path exists purely so the logic is testable everywhere.

```bash
python src/etl_pipeline.py --input data/raw/asset_telemetry.csv --output data/processed/asset_features
```

### 3. Experiment Tracking & Model Registry (`src/train.py`)
* **Objective:** Train a supervised classifier that predicts `failure_within_7_days` from the engineered risk features, with a fully auditable experiment trail.
* **Implementation:** A scikit-learn pipeline (one-hot encoding + Random Forest or Gradient Boosting) logs parameters, metrics (accuracy, precision, recall, F1, ROC-AUC), and the fitted model artifact to **MLflow**, and registers the model in the MLflow Model Registry under `asset-failure-risk-classifier`.
* **Local development note:** If no MLflow tracking server is configured, the script falls back to writing the model (`joblib`) and run metadata (`json`) to the local `models/` directory, so the pipeline is still fully runnable end-to-end offline.

```bash
python src/train.py --features data/processed/asset_features/asset_features.parquet --model-type random_forest
```

### 4. Production Deployment (`src/deploy.py`)
* **Objective:** Serve the registered model behind a low-latency, autoscaling REST API.
* **Implementation:** Uses the `azure-ai-ml` SDK to provision an **Azure Machine Learning Managed Online Endpoint**, deploy the registered model to it, and route live traffic to the new deployment. Requires an authenticated Azure session (`az login`) and a target workspace — this is the one script in the repo that is not runnable offline, by design.

```bash
python src/deploy.py \
  --model-name asset-failure-risk-classifier --model-version 1 \
  --endpoint-name asset-risk-endpoint \
  --subscription-id <sub-id> --resource-group <rg> --workspace-name <ws>
```

### 5. Post-Deployment Verification (`scripts/test_endpoint.py`)
Sends sample scoring requests to the live endpoint and validates the HTTP response shape — the final gate in the CI/CD pipeline before a deployment is considered healthy.

```bash
python scripts/test_endpoint.py --url <scoring-uri> --api-key <endpoint-key>
```

### 6. Automated Quality Verification (`tests/test_pipeline_logic.py`)
PyTest suite covering the data generator (schema, reproducibility, label validity, injected missingness) and the ETL feature-engineering logic (null imputation correctness, non-mutation of inputs, risk-flag behavior on a synthetic high-risk vs. baseline asset). Runs against the pandas implementations directly, so it needs no Spark cluster and no cloud credentials.

```bash
pytest tests/ -v
```

### 7. CI/CD Automation (`.github/workflows/mlops_ci_cd.yml`)
On every push/PR to `main`:
1. **Lint & Unit Test** — `flake8` + `pytest`.
2. **End-to-End Smoke Test** — regenerates mock data, runs the ETL (pandas fallback), trains a model, and uploads the trained artifact.
3. **Deploy** (main branch only) — authenticates to Azure, deploys the registered model to the managed online endpoint, then runs the live endpoint smoke test.

Steps 1–2 run on every PR with no cloud credentials required. Step 3 requires the following repository secrets to be configured: `AZURE_CREDENTIALS`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME`, `ENDPOINT_SCORING_URL`, `ENDPOINT_API_KEY`.

## ⚡ Getting Started

```bash
git clone https://github.com/<your-username>/enterprise-mlops.git
cd enterprise-mlops
pip install -r requirements.txt

# 1. Generate synthetic telemetry data
python scripts/generate_mock_data.py

# 2. Clean & engineer features
python src/etl_pipeline.py

# 3. Train and evaluate the model
python src/train.py

# 4. Run the test suite
pytest tests/ -v
```

Steps 1–3 run fully offline on any machine with the Python dependencies installed — PySpark, Delta Lake, and MLflow are used automatically when present, and the pipeline degrades gracefully to pandas/joblib when they're not, which is how this project stays demoable outside an actual Databricks/Azure ML workspace.

## 🗺️ Roadmap
- [ ] Wire up an actual Azure Databricks workspace and run `etl_pipeline.py` against a real Delta Lake table.
- [ ] Stand up an MLflow tracking server (Azure ML–hosted) instead of the local fallback.
- [ ] Add model drift monitoring on the deployed endpoint.
- [ ] Add a `bicep`/Terraform module to provision the Azure ML workspace and endpoint infrastructure as code.
