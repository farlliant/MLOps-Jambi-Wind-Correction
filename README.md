# Jambi Wind Correction MLOps

MLOps project for short-term wind speed forecast correction in Jambi, Indonesia, using ECMWF IFS forecasts and machine learning.

## Overview

Raw forecasts are obtained from **ECMWF IFS** through the Open-Meteo Single Runs API, while **ECMWF IFS Analysis** is used as the initial reference.

Machine learning target:

```text
residual = reference_wind_speed - forecast_wind_speed
```

Corrected forecast:

```text
corrected_wind = forecast_wind + predicted_residual
```

Forecast lead times:

- +6 hours
- +12 hours
- +18 hours
- +24 hours

## Project Structure

```text
jambi-wind-correction-mlops/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── src/
│   ├── data/
│   │   ├── test_single_run.py
│   │   ├── test_reference.py
│   │   └── build_poc_dataset.py
│   └── models/
│       └── sanity_residual_correction.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets and reports are excluded from Git and will later be managed using a dedicated data versioning mechanism.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run PoC

```powershell
python .\src\data\test_single_run.py
python .\src\data\test_reference.py
python .\src\data\build_poc_dataset.py
python .\src\models\sanity_residual_correction.py
```

## Initial PoC Results

Data period:

```text
15 May 2026 - 15 August 2026
```

Data validation results:

```text
Forecast runs requested : 372
Successful runs         : 371
Ingestion success rate  : 99.73%

Forecast-reference rows : 1484
Valid critical rows     : 1460
Critical completeness   : 98.38%
Duplicate pairs         : 0
```

The temporal sanity test showed a positive correction signal across all evaluated forecast lead times, with Random Forest producing the lowest MAE in the initial experiment.

These results are still considered **proof-of-concept results**, not final production model performance.

## MLOps Roadmap

Planned components:

- DVC for data versioning
- MLflow for experiment tracking and model registry
- Apache Airflow for orchestration
- FastAPI for model serving
- Docker for containerization
- GitHub Actions for CI/CD
- Evidently for drift monitoring
- Prometheus and Grafana for system monitoring
- Continuous training with champion-challenger validation

## Disclaimer

ECMWF IFS Analysis is used as an initial reference and is not treated as independent observational ground truth.
