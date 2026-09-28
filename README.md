# Jambi Wind Correction MLOps

> MLOps project for short-term wind speed forecast correction in Jambi, Indonesia, using ECMWF IFS forecasts and machine learning.

## Overview

This project develops an end-to-end MLOps workflow for improving short-term wind speed forecasts in Jambi, Indonesia.

Instead of predicting wind speed entirely from scratch, the project applies **residual correction** to ECMWF IFS forecasts.

```text
residual = reference_wind_speed - forecast_wind_speed

predicted_residual = ML(features)

corrected_wind = forecast_wind + predicted_residual
```

The current forecast horizons are:

```text
+6h
+12h
+18h
+24h
```

The project is developed incrementally through the course assignments.

```text
LK01 → MLOps system design
LK02 → repository and reproducible environment
LK03 → dynamic data-pipeline design and technical preview
LK04 → dynamic ingestion and automated preprocessing
```

---

## Data Sources

### Operational Forecast

Forecast data is obtained from:

**ECMWF IFS via Open-Meteo Single Runs API**

Project location:

```text
Location  : Jambi, Indonesia
Latitude  : -1.633333
Longitude : 103.650000
Timezone  : Asia/Jakarta
```

Meteorological variables currently collected include:

- `wind_speed_10m`
- `wind_direction_10m`
- `wind_gusts_10m`
- `temperature_2m`
- `relative_humidity_2m`
- `pressure_msl`
- `cloud_cover`
- `precipitation`

Each forecast snapshot also stores:

- `run_time`
- `valid_time`
- `retrieved_at`
- `source`

### Reference

The current proof-of-concept uses **ECMWF IFS Analysis** as the forecast reference.

> ECMWF IFS Analysis is used as a project reference and is **not treated as independent observational ground truth**.

---

## Initial Proof-of-Concept

The initial experiment used ECMWF forecast runs from:

```text
15 May 2026 → 15 August 2026
```

with initialization cycles:

```text
00 UTC
06 UTC
12 UTC
18 UTC
```

Initial data results:

```text
Forecast runs requested : 372
Successful runs         : 371
Ingestion success rate  : 99.73%

Forecast-reference rows : 1484
Valid critical rows     : 1460
Critical completeness   : 98.38%
Duplicate pairs         : 0
```

The initial residual-correction experiment evaluated:

- Linear Regression
- Random Forest
- HistGradientBoosting

A temporal split was used instead of random shuffling.

The initial Random Forest proof-of-concept produced an average MAE improvement of approximately **12.42%** compared with the raw ECMWF forecast.

These results are still proof-of-concept results and are not final production model performance.

---

# LK04 — Dynamic Data Pipeline

LK04 implements dynamic forecast ingestion and automated preprocessing.

The implemented flow is:

```text
ECMWF IFS / Open-Meteo
          ↓
src/ingest_data.py
          ↓
Raw Forecast Snapshot
          ↓
src/preprocess.py
          ↓
Processed Dataset
          ↓
Data Quality Report
```

---

## Dynamic Data Ingestion

Dynamic ingestion is implemented in:

```text
src/ingest_data.py
```

### Latest Available Forecast

Run:

```bash
python src/ingest_data.py
```

The script automatically:

1. determines candidate ECMWF forecast runs,
2. checks the newest candidate,
3. falls back to an earlier run when necessary,
4. handles transient connection errors with retry logic,
5. validates the API response,
6. stores the resulting raw snapshot.

ECMWF forecast cycles used by the project are:

```text
00 UTC
06 UTC
12 UTC
18 UTC
```

### Deterministic Historical Run

A specific forecast run can also be requested:

```bash
python src/ingest_data.py \
  --run-time 2026-08-18T00:00
```

This mode is useful for:

- reproducible testing,
- debugging,
- historical validation,
- and periodic-ingestion simulation.

### Raw Snapshot Policy

Raw files are stored using the forecast initialization time.

Example:

```text
data/raw/single_runs/
├── 20260818T00.csv
├── 20260818T06.csv
└── ...
```

If a snapshot already exists in a persistent environment, the script safely skips destructive replacement.

---

## Automated Preprocessing

Automated preprocessing is implemented in:

```text
src/preprocess.py
```

### Process Latest Raw Snapshot

```bash
python src/preprocess.py
```

### Process Specific Raw Snapshot

```bash
python src/preprocess.py \
  --input data/raw/single_runs/20260818T00.csv
```

The preprocessing pipeline performs:

```text
Raw Data
   ↓
Schema Validation
   ↓
Datatype Normalization
   ↓
Timestamp Validation
   ↓
Lead-Time Calculation
   ↓
Select +6 / +12 / +18 / +24
   ↓
Duplicate Handling
   ↓
Critical Missing-Value Handling
   ↓
Physical-Range Validation
   ↓
Cyclical Feature Engineering
   ↓
Processed Dataset
   ↓
Quality Report
```

### Data Quality Rules

Current validation includes:

```text
wind_speed_10m >= 0

0 <= wind_direction_10m <= 360

0 <= relative_humidity_2m <= 100

0 <= cloud_cover <= 100

precipitation >= 0
```

Critical wind variables are explicitly checked for missing values, while additional meteorological variables are validated according to their applicable quality rules.

---

## Feature Engineering

Wind direction is encoded using:

```text
wind_direction_sin
wind_direction_cos
```

This preserves the circular relationship between directions such as `359°` and `1°`.

Temporal cyclical features include:

```text
hour_sin
hour_cos
day_of_year_sin
day_of_year_cos
```

Canonical timestamps are stored in UTC.

Local temporal features are derived using:

```text
Asia/Jakarta
```

---

## Pipeline Outputs

Generated processed datasets are written to:

```text
data/processed/
```

Example:

```text
forecast_20260818T00_processed.csv
```

Data-quality reports are written to:

```text
data/reports/
```

Example:

```text
forecast_20260818T00_quality.csv
```

The quality report records:

- raw row count,
- selected forecast horizons,
- invalid timestamps,
- duplicate rows,
- missing critical rows,
- physically invalid rows,
- processed rows,
- completeness percentage.

A validated historical execution produced:

```text
Raw rows               : 30
Target lead rows       : 4
Duplicate rows         : 0
Missing critical rows  : 0
Invalid physical rows  : 0
Processed rows         : 4
Completeness           : 100.00%
```

---

## Raw Sample

Operational datasets are excluded from normal Git tracking.

A representative raw sample is included for reproducibility:

```text
data/raw/samples/forecast_sample.csv
```

Generated operational files remain ignored:

```text
data/raw/single_runs/
data/interim/
data/processed/
data/reports/
```

Formal dataset versioning with **DVC** is planned for a later project stage.

---

## GitHub Actions Automation

LK04 includes:

```text
.github/workflows/lk04-data-pipeline.yml
```

The workflow supports:

- Pull Request validation,
- manual execution using `workflow_dispatch`,
- scheduled execution every three hours,
- Python 3.11 setup,
- dependency installation,
- script validation,
- dynamic ingestion,
- automated preprocessing,
- generated-output inspection,
- artifact upload.

Pipeline flow:

```text
GitHub Actions
      ↓
Install Environment
      ↓
Validate Python Scripts
      ↓
Dynamic Ingestion
      ↓
Automated Preprocessing
      ↓
Upload Pipeline Artifact
```

The three-hour schedule represents a **polling interval**, not the ECMWF model-generation frequency.

The ingestion script independently determines the latest available ECMWF forecast run.

### Automation Validation

The workflow has been successfully validated through:

```text
Pull Request execution        ✅
Manual execution from main    ✅
Dynamic data ingestion        ✅
Automated preprocessing       ✅
Artifact generation           ✅
```

A workflow execution generates an artifact named approximately:

```text
lk04-data-pipeline-<run_id>
```

containing generated raw, processed, and quality-report files.

> GitHub Actions artifacts are execution outputs and are not a replacement for formal DVC dataset versioning.

---

## Repository Structure

```text
MLOps-Jambi-Wind-Correction/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── .github/
│   └── workflows/
│       └── lk04-data-pipeline.yml
│
├── config/
│   └── project.yaml
│
├── data/
│   ├── raw/
│   │   └── samples/
│   │       └── forecast_sample.csv
│   ├── interim/
│   ├── processed/
│   └── reports/
│
├── docs/
├── models/
├── notebooks/
│
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   │
│   ├── data/
│   │   ├── build_poc_dataset.py
│   │   ├── lk03_pipeline_preview.py
│   │   ├── test_reference.py
│   │   └── test_single_run.py
│   │
│   ├── features/
│   ├── models/
│   └── monitoring/
│
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Development Environment

The project uses:

- Python 3.11
- GitHub Codespaces
- pandas
- NumPy
- Requests
- scikit-learn
- GitHub Actions

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Validate the environment:

```bash
python --version
python -m pip check
```

Validate LK04 Python scripts:

```bash
python -m py_compile \
  src/ingest_data.py \
  src/preprocess.py
```

Code quality was also validated using Ruff:

```bash
ruff check \
  src/ingest_data.py \
  src/preprocess.py
```

---

## Git Workflow

The repository follows a lightweight GitHub Flow.

```text
main
  ↓
feature branch
  ↓
implementation
  ↓
local validation
  ↓
push
  ↓
Pull Request
  ↓
GitHub Actions validation
  ↓
merge to main
```

LK04 was developed using:

```text
feat/lk04-dynamic-ingestion-preprocessing
```

and merged into `main` after the automated PR workflow passed.

---

## Current Development Status

| Stage | Status |
|---|---|
| Project formulation | ✅ Completed |
| Forecast data validation | ✅ Completed |
| Reference data validation | ✅ Completed |
| Forecast-reference PoC | ✅ Completed |
| Residual-correction PoC | ✅ Completed |
| Reproducible repository environment | ✅ Completed |
| LK03 data-pipeline preview | ✅ Completed |
| Dynamic data ingestion | ✅ Implemented |
| Automated preprocessing | ✅ Implemented |
| Raw sample | ✅ Implemented |
| Data-quality reporting | ✅ Implemented |
| Retry and fallback handling | ✅ Implemented |
| GitHub Actions automation | ✅ Implemented |
| Pull Request CI validation | ✅ Passed |
| Manual main-branch workflow | ✅ Passed |
| DVC data versioning | ⏳ Planned |
| MLflow tracking | ⏳ Planned |
| Apache Airflow orchestration | ⏳ Planned |
| Production model training | ⏳ Planned |
| Model serving | ⏳ Planned |
| Drift monitoring | ⏳ Planned |
| Continuous training | ⏳ Planned |

---

## Planned MLOps Stack

| Component | Technology |
|---|---|
| Source Control | GitHub |
| Development Environment | GitHub Codespaces |
| Data Processing | pandas / NumPy |
| Machine Learning | scikit-learn |
| Lightweight Automation | GitHub Actions |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Workflow Orchestration | Apache Airflow |
| Model Serving | FastAPI |
| Containerization | Docker |
| Drift Monitoring | Evidently / custom monitoring |
| Metrics | Prometheus |
| Dashboard | Grafana |

GitHub Actions currently handles the lightweight LK04 data-pipeline automation.

Apache Airflow remains planned for future workflows involving more complex dependencies such as training, evaluation, deployment, monitoring, and retraining.

---

## Continuous Training Direction

The future continuous-training design follows a champion-challenger strategy.

```text
Current Champion
       ↓
Monitor Performance / Drift
       ↓
Retraining Trigger
       ↓
Train Challenger
       ↓
Evaluate
       ↓
Better?
 ┌─────┴─────┐
Yes          No
 ↓            ↓
Promote     Keep Champion
```

Retraining automation is not part of LK04 and remains a future project stage.

---

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

## Disclaimer

ECMWF IFS Analysis is currently used as an initial forecast reference and is **not treated as independent observational ground truth**.

The architecture and technology stack will continue to evolve as later MLOps stages are implemented.