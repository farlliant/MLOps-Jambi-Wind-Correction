"""LK08 - Automated model evaluation before Registry."""

import json
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.lk08_policy import evaluate_metrics
from train import FEATURES, chronological_split, load_and_validate


POLICY_PATH = Path("config/lk08_validation.yaml")
HISTORY_PATH = Path("data/reports/lk06/best_model_metadata.json")
BEST_PATH = Path("data/reports/lk08/best_model_metadata.json")
OUTPUT_PATH = Path("data/reports/lk08/validation_decision.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def calculate_metrics(reference, raw, residual):
    corrected = np.maximum(0.0, raw + residual)

    return {
        "mae": float(mean_absolute_error(reference, corrected)),
        "rmse": float(
            np.sqrt(mean_squared_error(reference, corrected))
        ),
        "bias": float(np.mean(corrected - reference)),
    }


def calculate_per_horizon(
    reference, raw, residual, leads, required_leads
):
    corrected = np.maximum(0.0, raw + residual)
    result = {}

    for lead in required_leads:
        mask = leads == lead

        if not mask.any():
            raise ValueError(
                f"Validation tidak memiliki horizon +{lead}"
            )

        result[int(lead)] = float(
            mean_absolute_error(
                reference[mask],
                corrected[mask],
            )
        )

    return result


def predict_residual(model_uri, features, expected_rows):
    model = mlflow.pyfunc.load_model(model_uri)

    prediction = np.asarray(
        model.predict(features),
        dtype=float,
    ).reshape(-1)

    if len(prediction) != expected_rows:
        raise ValueError("Jumlah prediksi tidak sesuai")

    if not np.isfinite(prediction).all():
        raise ValueError("Prediksi memiliki NaN atau Infinity")

    return prediction


def assert_close(label, actual, expected, tolerance=1e-6):
    if not np.isfinite(actual) or not np.isfinite(expected):
        raise ValueError(f"{label} tidak finite")

    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(
            f"{label} tidak konsisten: "
            f"{actual} != {expected}"
        )


def evaluate():
    policy = yaml.safe_load(
        POLICY_PATH.read_text(encoding="utf-8")
    )

    best = read_json(BEST_PATH)
    historical = read_json(HISTORY_PATH)

    required_flags = [
        "require_mae_not_worse_than_raw",
        "require_mae_not_worse_than_reference",
        "require_rmse_not_worse_than_reference",
        "require_absolute_bias_not_worse_than_reference",
    ]

    for flag in required_flags:
        if policy["evaluation"].get(flag) is not True:
            raise RuntimeError(
                f"Validation policy tidak aktif: {flag}"
            )

    # A. DATASET INTEGRITY
    df, dataset_sha = load_and_validate()
    train_df, valid_df, cutoff = chronological_split(df)

    if dataset_sha != best["dataset_sha256"]:
        raise RuntimeError(
            "Dataset hash berbeda dari training"
        )

    if len(df) < policy["dataset"]["min_labeled_rows"]:
        raise RuntimeError("Dataset terlalu kecil")

    if len(train_df) != best["train_rows"]:
        raise RuntimeError("Jumlah training rows berbeda")

    if len(valid_df) != best["validation_rows"]:
        raise RuntimeError("Jumlah validation rows berbeda")

    if valid_df["valid_time"].nunique() < policy[
        "dataset"
    ]["min_unique_validation_times"]:
        raise RuntimeError("Jumlah waktu validasi tidak cukup")

    if set(train_df["valid_time"]) & set(valid_df["valid_time"]):
        raise RuntimeError("Temporal leakage terdeteksi")

    if best["feature_columns"] != FEATURES:
        raise RuntimeError("Feature schema berbeda")

    if best["validation_cutoff_utc"] != cutoff.isoformat():
        raise RuntimeError("Validation cutoff berbeda")

    required_leads = [
        int(h)
        for h in policy["dataset"]["required_lead_hours"]
    ]

    actual_leads = set(
        valid_df["lead_hours"].astype(int)
    )

    if not set(required_leads).issubset(actual_leads):
        raise RuntimeError("Horizon validasi tidak lengkap")

    # B. MLFLOW RUN INTEGRITY
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI belum diisi")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    run = client.get_run(best["run_id"])

    if run.info.status != "FINISHED":
        raise RuntimeError("MLflow run belum selesai")

    if run.data.tags.get("dataset_sha256") != dataset_sha:
        raise RuntimeError("MLflow dataset hash tidak cocok")

    for key in (
        "val_mae_ms",
        "val_rmse_ms",
        "val_bias_ms",
    ):
        if key not in run.data.metrics:
            raise RuntimeError(f"MLflow metric hilang: {key}")

        assert_close(
            key,
            float(best[key]),
            float(run.data.metrics[key]),
            tolerance=1e-8,
        )

    # C. RECOMPUTE MODEL METRICS
    features = valid_df[FEATURES].astype("float64")

    raw = valid_df[
        "forecast_wind_speed_10m"
    ].to_numpy(dtype=float)

    reference = valid_df[
        "reference_wind_speed_10m"
    ].to_numpy(dtype=float)

    leads = valid_df[
        "lead_hours"
    ].to_numpy(dtype=float)

    residual = predict_residual(
        best["model_uri"],
        features,
        len(valid_df),
    )

    candidate = calculate_metrics(
        reference, raw, residual
    )

    candidate_per_lead = calculate_per_horizon(
        reference,
        raw,
        residual,
        leads,
        required_leads,
    )

    for metric, key in (
        ("mae", "val_mae_ms"),
        ("rmse", "val_rmse_ms"),
        ("bias", "val_bias_ms"),
    ):
        assert_close(
            key,
            candidate[metric],
            float(best[key]),
        )

    for lead, value in candidate_per_lead.items():
        key = f"val_mae_lead_{lead}_ms"

        if key not in run.data.metrics:
            raise RuntimeError(
                f"MLflow metric hilang: {key}"
            )

        assert_close(
            key,
            value,
            float(run.data.metrics[key]),
        )

    raw_mae = float(
        mean_absolute_error(reference, raw)
    )

    assert_close(
        "Raw baseline MAE",
        raw_mae,
        float(best["validation_baseline_mae_ms"]),
    )

    # D. FIND CHAMPION
    model_name = policy["reference_model"]["model_name"]
    champion_alias = policy[
        "reference_model"
    ]["champion_alias"]

    versions = client.search_model_versions(
        f"name = '{model_name}'"
    )

    champion_version = next(
        (
            version
            for version in versions
            if champion_alias in (version.aliases or [])
        ),
        None,
    )

    if champion_version is not None:
        champion_uri = (
            f"models:/{model_name}@{champion_alias}"
        )

        champion_residual = predict_residual(
            champion_uri,
            features,
            len(valid_df),
        )

        comparator = calculate_metrics(
            reference,
            raw,
            champion_residual,
        )

        comparator_per_lead = calculate_per_horizon(
            reference,
            raw,
            champion_residual,
            leads,
            required_leads,
        )

        comparison_source = (
            "registered_champion_same_holdout"
        )

    else:
        # First-run bootstrap only:
        # historical results are permitted if data,
        # features and temporal split are identical.
        if dataset_sha != historical["dataset_sha256"]:
            raise RuntimeError(
                "Dataset berbeda dari LK06, tetapi "
                "remote champion belum tersedia"
            )

        if best["feature_columns"] != historical[
            "feature_columns"
        ]:
            raise RuntimeError(
                "Fitur tidak cocok dengan LK06"
            )

        if (
            len(train_df) != historical["train_rows"]
            or len(valid_df) != historical["validation_rows"]
            or cutoff.isoformat()
            != historical["validation_cutoff_utc"]
        ):
            raise RuntimeError(
                "Temporal split tidak cocok dengan LK06"
            )

        comparator = {
            "mae": float(historical["val_mae_ms"]),
            "rmse": float(historical["val_rmse_ms"]),
            "bias": float(historical["val_bias_ms"]),
        }

        comparator_per_lead = None

        comparison_source = (
            "historical_RF01_same_dataset"
        )

    # E. VALIDATION GATE
    checks = evaluate_metrics(
        **candidate,
        raw_mae=raw_mae,
        reference_mae=comparator["mae"],
        reference_rmse=comparator["rmse"],
        reference_abs_bias=abs(comparator["bias"]),
        unique_times=int(
            valid_df["valid_time"].nunique()
        ),
        lead_hours=list(actual_leads),
        min_unique_times=policy[
            "dataset"
        ]["min_unique_validation_times"],
        required_leads=required_leads,
        epsilon=policy[
            "evaluation"
        ]["numerical_tolerance_ms"],
        per_lead_mae=candidate_per_lead,
        reference_per_lead_mae=comparator_per_lead,
    )

    passed = all(checks.values())

    return {
        "status": "PASS" if passed else "REJECTED",
        "policy_version": policy["policy"]["version"],
        "run_id": best["run_id"],
        "model_uri": best["model_uri"],
        "dataset_sha256": dataset_sha,
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "comparison_source": comparison_source,
        "candidate_metrics": candidate,
        "raw_baseline_mae": raw_mae,
        "reference_metrics": comparator,
        "candidate_per_horizon_mae": candidate_per_lead,
        "per_horizon_comparison_available":
            comparator_per_lead is not None,
        "checks": checks,
    }


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        result = evaluate()

    except Exception as exc:
        result = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }

        OUTPUT_PATH.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print("[ERROR]", str(exc))
        raise

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))

    if result["status"] != "PASS":
        raise SystemExit(
            "LK08 model rejected; registration blocked"
        )


if __name__ == "__main__":
    main()
