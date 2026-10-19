
"""LK06 - Jambi wind forecast residual correction with MLflow 3.x."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model as log_sklearn_model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/versioned/lk06_training_dataset.csv")
REPORT_DIR = Path("data/reports/lk06")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "LK06-Jambi-Wind-Residual-Correction"

RANDOM_SEED = 42

# Only forecast features. Never include reference or residual.
FEATURES = [
    "lead_hours",
    "forecast_wind_speed_10m",
    "forecast_wind_gusts_10m",
    "forecast_temperature_2m",
    "forecast_relative_humidity_2m",
    "forecast_pressure_msl",
    "forecast_cloud_cover",
    "forecast_precipitation",
    "wind_direction_sin",
    "wind_direction_cos",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

# Three hyperparameter experiments required by LK06.
CONFIGS = [
    {
        "name": "RF-01",
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_leaf": 2,
    },
    {
        "name": "RF-02",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 2,
    },
    {
        "name": "RF-03",
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 1,
    },
]

# Specific types found in our own RandomForest model.
# Do not trust these types for arbitrary third-party models.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error."""
    return float(
        np.sqrt(mean_squared_error(actual, predicted))
    )


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_and_validate():
    """Read the DVC dataset and validate its schema and target."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset DVC tidak ditemukan: {DATA_PATH}"
        )

    dataset_sha256 = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    df = pd.read_csv(DATA_PATH)

    required_columns = set(FEATURES) | {
        "run_time",
        "valid_time",
        "reference_wind_speed_10m",
        "residual",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Kolom dataset tidak lengkap: {missing_columns}"
        )

    # Normalize timestamps.
    for column in ["run_time", "valid_time"]:
        df[column] = pd.to_datetime(
            df[column],
            utc=True,
            errors="raise",
        )

    # Convert numeric columns.
    for column in [
        *FEATURES,
        "reference_wind_speed_10m",
        "residual",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df.empty:
        raise ValueError("Dataset kosong")

    # Validate natural forecast keys.
    if df.duplicated(
        ["run_time", "valid_time"]
    ).any():
        raise ValueError(
            "Terdapat duplikasi pasangan forecast"
        )

    critical_columns = [
        "forecast_wind_speed_10m",
        "reference_wind_speed_10m",
        "residual",
    ]

    if df[critical_columns].isna().any().any():
        raise ValueError(
            "Label atau forecast wind speed mengandung NaN"
        )

    wind_columns = [
        "forecast_wind_speed_10m",
        "reference_wind_speed_10m",
    ]

    if (df[wind_columns] < 0).any().any():
        raise ValueError(
            "Nilai kecepatan angin negatif"
        )

    # Validate regression target.
    expected_residual = (
        df["reference_wind_speed_10m"]
        - df["forecast_wind_speed_10m"]
    )

    if not np.allclose(
        expected_residual,
        df["residual"],
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(
            "Residual tidak konsisten dengan reference - forecast"
        )

    if np.isinf(
        df[FEATURES].to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Terdapat nilai infinity pada fitur"
        )

    # Sort chronologically.
    df = df.sort_values(
        ["valid_time", "run_time"]
    ).reset_index(drop=True)

    return df, dataset_sha256


# ============================================================
# TEMPORAL TRAIN-VALIDATION SPLIT
# ============================================================

def chronological_split(df):
    """
    Split based on unique valid_time instead of random rows.

    Prevent the same reference timestamp from appearing in
    both training and validation datasets.
    """

    unique_times = (
        df["valid_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_times) < 10:
        raise ValueError(
            "Minimal 10 timestamp unik diperlukan"
        )

    cutoff_index = int(len(unique_times) * 0.80)
    cutoff = unique_times.iloc[cutoff_index]

    train = df[
        df["valid_time"] < cutoff
    ].copy()

    valid = df[
        df["valid_time"] >= cutoff
    ].copy()

    if train.empty or valid.empty:
        raise ValueError(
            "Train/validation split kosong"
        )

    overlapping_times = (
        set(train["valid_time"])
        & set(valid["valid_time"])
    )

    if overlapping_times:
        raise ValueError(
            "Data leakage: valid_time overlap"
        )

    if train[FEATURES].isna().all().any():
        raise ValueError(
            "Ada fitur yang seluruhnya kosong"
        )

    return train, valid, cutoff


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df, dataset_sha256 = load_and_validate()

    train, valid, cutoff = chronological_split(df)

    # Explicit float64 prevents MLflow integer schema warnings.
    x_train = train[FEATURES].astype("float64")
    x_valid = valid[FEATURES].astype("float64")

    y_train = train["residual"].to_numpy(
        dtype=float
    )

    reference = valid[
        "reference_wind_speed_10m"
    ].to_numpy(dtype=float)

    raw_forecast = valid[
        "forecast_wind_speed_10m"
    ].to_numpy(dtype=float)

    lead_valid = valid[
        "lead_hours"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 2. Validation baseline
    # --------------------------------------------------------

    baseline_mae = float(
        mean_absolute_error(
            reference,
            raw_forecast,
        )
    )

    baseline_rmse = calculate_rmse(
        reference,
        raw_forecast,
    )

    print("=" * 70)
    print("LK06 - JAMBI WIND RESIDUAL CORRECTION WITH MLFLOW")
    print("=" * 70)

    print("Training rows           :", len(train))
    print("Validation rows         :", len(valid))

    print(
        "Unique train timestamps :",
        train["valid_time"].nunique(),
    )

    print(
        "Unique valid timestamps :",
        valid["valid_time"].nunique(),
    )

    print(
        "Validation cutoff (UTC) :",
        cutoff.isoformat(),
    )

    print(
        f"Validation baseline MAE : "
        f"{baseline_mae:.4f} m/s"
    )

    print(
        f"Validation baseline RMSE: "
        f"{baseline_rmse:.4f} m/s"
    )

    # --------------------------------------------------------
    # 3. MLflow configuration
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # --------------------------------------------------------
    # 4. Execute three experiments
    # --------------------------------------------------------

    for config in CONFIGS:

        hyperparams = {
            key: value
            for key, value in config.items()
            if key != "name"
        }

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    **hyperparams,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ])

        print(
            f"\n[TRAINING] {config['name']} "
            f"| {hyperparams}"
        )

        with mlflow.start_run(
            run_name=config["name"]
        ) as run:

            # -----------------------------------------------
            # 4.1 Train model
            # -----------------------------------------------

            model.fit(
                x_train,
                y_train,
            )

            predicted_residual = model.predict(
                x_valid
            )

            # Forecast correction.
            corrected_forecast = np.maximum(
                0.0,
                raw_forecast + predicted_residual,
            )

            # -----------------------------------------------
            # 4.2 Evaluate model
            # -----------------------------------------------

            val_mae = float(
                mean_absolute_error(
                    reference,
                    corrected_forecast,
                )
            )

            val_rmse = calculate_rmse(
                reference,
                corrected_forecast,
            )

            val_bias = float(
                np.mean(
                    corrected_forecast - reference
                )
            )

            improvement_pct = (
                100.0
                * (baseline_mae - val_mae)
                / baseline_mae
                if baseline_mae > 0
                else 0.0
            )

            # -----------------------------------------------
            # 4.3 Log hyperparameters
            # -----------------------------------------------

            mlflow.log_params({
                **hyperparams,
                "random_state": RANDOM_SEED,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "feature_count": len(FEATURES),
                "split_method": "chronological_valid_time_80_20",
                "serialization_format": "skops",
            })

            mlflow.set_tags({
                "dataset_sha256": dataset_sha256,
                "forecast_source": "ECMWF IFS",
                "reference_source": "ECMWF IFS Analysis",
                "target_definition": "reference_minus_forecast",
            })

            # -----------------------------------------------
            # 4.4 Log evaluation metrics
            # -----------------------------------------------

            mlflow.log_metrics({
                "val_mae_ms": val_mae,
                "val_rmse_ms": val_rmse,
                "val_bias_ms": val_bias,
                "val_baseline_mae_ms": baseline_mae,
                "val_baseline_rmse_ms": baseline_rmse,
                "val_mae_improvement_pct": improvement_pct,
            })

            # Per-horizon evaluation.
            for lead in (6, 12, 18, 24):

                mask = lead_valid == lead

                if mask.any():

                    lead_mae = mean_absolute_error(
                        reference[mask],
                        corrected_forecast[mask],
                    )

                    mlflow.log_metric(
                        f"val_mae_lead_{lead}_ms",
                        float(lead_mae),
                    )

            # -----------------------------------------------
            # 4.5 Log model - MLflow 3.x
            # -----------------------------------------------

            example = x_train.head(3).copy()

            signature = infer_signature(
                example,
                model.predict(example),
            )

            model_info = log_sklearn_model(
                sk_model=model,
                name="residual_model",
                signature=signature,
                input_example=example,
                serialization_format="skops",
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

            # -----------------------------------------------
            # 4.6 Save prediction evidence
            # -----------------------------------------------

            predictions = valid[
                ["run_time", "valid_time", "lead_hours"]
            ].copy()

            predictions["reference_ms"] = reference

            predictions["raw_forecast_ms"] = raw_forecast

            predictions["predicted_residual_ms"] = (
                predicted_residual
            )

            predictions["corrected_forecast_ms"] = (
                corrected_forecast
            )

            prediction_file = (
                REPORT_DIR
                / f"predictions_{config['name']}.csv"
            )

            predictions.to_csv(
                prediction_file,
                index=False,
            )

            mlflow.log_artifact(
                str(prediction_file),
                artifact_path="evaluation",
            )

            # -----------------------------------------------
            # 4.7 Save experiment summary
            # -----------------------------------------------

            results.append({
                "name": config["name"],
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "n_estimators": hyperparams["n_estimators"],
                "max_depth": hyperparams["max_depth"],
                "min_samples_leaf": hyperparams["min_samples_leaf"],
                "val_mae_ms": val_mae,
                "val_rmse_ms": val_rmse,
                "val_bias_ms": val_bias,
                "val_mae_improvement_pct": improvement_pct,
            })

            print(
                f"[{config['name']}] "
                f"MAE={val_mae:.4f}, "
                f"RMSE={val_rmse:.4f} m/s"
            )

            print(
                f"[{config['name']}] "
                f"Model URI: {model_info.model_uri}"
            )

    # --------------------------------------------------------
    # 5. Compare experiments
    # --------------------------------------------------------

    if len(results) != 3:
        raise RuntimeError(
            "Tiga eksperimen belum selesai"
        )

    comparison = (
        pd.DataFrame(results)
        .sort_values(
            ["val_mae_ms", "val_rmse_ms"]
        )
        .reset_index(drop=True)
    )

    comparison_path = (
        REPORT_DIR / "experiment_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Select best model
    # --------------------------------------------------------

    best = comparison.iloc[0]

    metadata = {
        "best_run_name": str(best["name"]),
        "run_id": str(best["run_id"]),
        "model_uri": str(best["model_uri"]),
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "val_mae_ms": float(best["val_mae_ms"]),
        "val_rmse_ms": float(best["val_rmse_ms"]),
        "val_bias_ms": float(best["val_bias_ms"]),
        "val_mae_improvement_pct": float(
            best["val_mae_improvement_pct"]
        ),
        "validation_baseline_mae_ms": baseline_mae,
        "validation_baseline_rmse_ms": baseline_rmse,
        "validation_cutoff_utc": cutoff.isoformat(),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "dataset_sha256": dataset_sha256,
        "feature_columns": FEATURES,
    }

    metadata_path = (
        REPORT_DIR / "best_model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 7. Final report
    # --------------------------------------------------------

    print("\n=== PERBANDINGAN 3 EKSPERIMEN ===")

    print(
        comparison[
            [
                "name",
                "val_mae_ms",
                "val_rmse_ms",
                "val_bias_ms",
                "val_mae_improvement_pct",
            ]
        ].to_string(index=False)
    )

    print(
        "\nBEST RUN:",
        metadata["best_run_name"],
    )

    print(
        "BEST MODEL URI:",
        metadata["model_uri"],
    )

    print("Comparison:", comparison_path)
    print("Metadata:", metadata_path)

    print("\nTRAINING & MLFLOW TRACKING PASSED")


if __name__ == "__main__":
    main()
