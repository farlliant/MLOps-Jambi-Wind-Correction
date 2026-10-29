"""LK07 - Registered model inference using an MLflow alias."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Jambi-Wind-Residual-RandomForest"
METADATA_PATH = Path("data/reports/lk06/best_model_metadata.json")
REPORT_DIR = Path("data/reports/lk07")


def build_synthetic_input(feature_columns):
    """Build a new synthetic forecast input, not observed data."""

    run_time = pd.Timestamp("2026-10-29T06:00:00Z")
    valid_time = run_time + pd.Timedelta(hours=6)

    local_time = valid_time.tz_convert("Asia/Jakarta")
    hour = local_time.hour
    day = local_time.dayofyear

    direction_rad = np.deg2rad(135.0)

    values = {
        "lead_hours": 6.0,
        "forecast_wind_speed_10m": 2.40,
        "forecast_wind_gusts_10m": 4.10,
        "forecast_temperature_2m": 30.10,
        "forecast_relative_humidity_2m": 76.0,
        "forecast_pressure_msl": 1011.0,
        "forecast_cloud_cover": 65.0,
        "forecast_precipitation": 0.10,
        "wind_direction_sin": np.sin(direction_rad),
        "wind_direction_cos": np.cos(direction_rad),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_year_sin": np.sin(2 * np.pi * day / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day / 365.25),
    }

    if set(values) != set(feature_columns):
        raise ValueError("Input features do not match LK06 metadata")

    frame = pd.DataFrame(
        [{name: values[name] for name in feature_columns}]
    ).astype("float64")

    return frame, run_time, valid_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alias",
        choices=["champion", "challenger", "candidate"],
        default="champion",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    print("=" * 60)
    print("LK07 - MODEL REGISTRY INFERENCE TEST")
    print("=" * 60)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    features = metadata["feature_columns"]
    inputs, run_time, valid_time = build_synthetic_input(features)

    # Resolve and verify the actual registered model version.
    registered_version = client.get_model_version_by_alias(
        MODEL_NAME,
        args.alias,
    )

    if args.alias == "champion":
        if registered_version.current_stage != "Production":
            raise RuntimeError(
                "Champion must be in Production simulation stage"
            )

    model_uri = f"models:/{MODEL_NAME}@{args.alias}"

    print("\n[1] Registry")
    print("Model URI :", model_uri)
    print("Version   :", registered_version.version)
    print("Run ID    :", registered_version.run_id)
    print("Stage     :", registered_version.current_stage)

    # Load the model through the MLflow Registry.
    model = mlflow.pyfunc.load_model(model_uri)

    print("\n[2] Synthetic input")
    print("Forecast run (UTC):", run_time.isoformat())
    print("Valid time (UTC)  :", valid_time.isoformat())
    print("Input features    :", len(inputs.columns))

    # MLflow model predicts a residual, not the final wind speed.
    prediction = np.asarray(
        model.predict(inputs),
        dtype=float,
    ).reshape(-1)

    if prediction.size != 1 or not np.isfinite(prediction).all():
        raise ValueError("Invalid model prediction")

    raw_wind = float(
        inputs.iloc[0]["forecast_wind_speed_10m"]
    )
    residual = float(prediction[0])
    corrected_wind = max(0.0, raw_wind + residual)

    assert np.isfinite(corrected_wind)
    assert corrected_wind >= 0.0

    print("\n[3] Inference results")
    print(f"Raw forecast       : {raw_wind:.4f} m/s")
    print(f"Predicted residual : {residual:+.4f} m/s")
    print(f"Corrected forecast : {corrected_wind:.4f} m/s")

    # Save evidence without modifying the model or Registry.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"inference_{args.alias}.json"

    evidence = {
        "task": "LK07 Model Registry Inference",
        "input_type": "synthetic_forecast_example",
        "tracking_uri": TRACKING_URI,
        "model_uri": model_uri,
        "registered_model": MODEL_NAME,
        "model_version": str(registered_version.version),
        "model_alias": args.alias,
        "model_stage": registered_version.current_stage,
        "run_id": registered_version.run_id,
        "dataset_sha256": metadata["dataset_sha256"],
        "forecast_run_utc": run_time.isoformat(),
        "valid_time_utc": valid_time.isoformat(),
        "feature_count": len(features),
        "input_features": inputs.iloc[0].to_dict(),
        "raw_wind_ms": raw_wind,
        "predicted_residual_ms": residual,
        "corrected_wind_ms": corrected_wind,
        "actual_reference_available": False,
        "academic_simulation": True,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path.write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print("\nEvidence:", output_path)
    print("[PASS] REGISTERED MODEL INFERENCE VERIFIED")


if __name__ == "__main__":
    main()
