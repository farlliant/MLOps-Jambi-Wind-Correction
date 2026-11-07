"""LK08: guarded automatic registration to MLflow Staging."""

import hashlib
import json
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import yaml
from mlflow import MlflowClient

from train import FEATURES, chronological_split, load_and_validate


POLICY_PATH = Path("config/lk08_validation.yaml")
GATE_PATH = Path("data/reports/lk08/validation_decision.json")
BEST_PATH = Path("data/reports/lk08/best_model_metadata.json")
OUTPUT_PATH = Path("data/reports/lk08/registration_result.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def smoke_test(model_uri, features):
    model = mlflow.pyfunc.load_model(model_uri)

    prediction = np.asarray(
        model.predict(features), dtype=float
    ).reshape(-1)

    if prediction.size != len(features):
        raise RuntimeError("Jumlah prediksi tidak sesuai")

    if not np.isfinite(prediction).all():
        raise RuntimeError("Prediksi model tidak finite")

    return float(prediction[0])


def register():
    policy = yaml.safe_load(
        POLICY_PATH.read_text(encoding="utf-8")
    )
    gate = read_json(GATE_PATH)
    best = read_json(BEST_PATH)

    # A. VALIDATION GATE
    if gate.get("status") != "PASS":
        raise RuntimeError("Validation gate tidak PASS")

    checks = gate.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise RuntimeError("Ada validation check yang gagal")

    # B. TRUSTED GITHUB EVENT
    git_sha = os.getenv("GITHUB_SHA", "")
    git_ref = os.getenv("GITHUB_REF", "")

    if not git_sha or git_ref != "refs/heads/main":
        raise RuntimeError(
            "Registrasi hanya diizinkan dari main"
        )

    if os.getenv("GITHUB_EVENT_NAME") != "push":
        raise RuntimeError(
            "Registrasi hanya untuk push ke main"
        )

    if gate.get("git_sha") != git_sha:
        raise RuntimeError("Git SHA tidak cocok")

    if (
        gate.get("run_id") != best["run_id"]
        or gate.get("model_uri") != best["model_uri"]
    ):
        raise RuntimeError("Identitas model tidak cocok")

    # C. DATASET INTEGRITY
    dataset_path = Path(policy["dataset"]["path"])

    actual_sha = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()

    if (
        actual_sha != gate["dataset_sha256"]
        or actual_sha != best["dataset_sha256"]
    ):
        raise RuntimeError("Dataset SHA-256 tidak cocok")

    df, verified_sha = load_and_validate()

    if verified_sha != actual_sha:
        raise RuntimeError("Validasi dataset gagal")

    _, valid_df, _ = chronological_split(df)

    if best["feature_columns"] != FEATURES:
        raise RuntimeError("Feature schema berbeda")

    features = valid_df[FEATURES].head(1).astype("float64")

    # D. MLFLOW INTEGRITY
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI belum diatur")

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    run = client.get_run(best["run_id"])

    if run.info.status != "FINISHED":
        raise RuntimeError("Training run belum FINISHED")

    if run.data.tags.get("dataset_sha256") != actual_sha:
        raise RuntimeError("MLflow dataset hash berbeda")

    for metric in ("val_mae_ms", "val_rmse_ms", "val_bias_ms"):
        logged = float(run.data.metrics[metric])
        reported = float(best[metric])

        if not np.isfinite(logged):
            raise RuntimeError(f"Metric tidak valid: {metric}")

        if abs(logged - reported) > 1e-8:
            raise RuntimeError(
                f"Metric tidak konsisten: {metric}"
            )

    # Pastikan model asli dapat di-load sebelum registrasi.
    smoke_test(best["model_uri"], features)

    # E. REGISTRY POLICY
    registry = policy["registry"]

    if registry["target_stage"] != "Staging":
        raise RuntimeError("Target stage harus Staging")

    if registry["auto_promote_champion"] is not False:
        raise RuntimeError(
            "Automatic champion promotion dilarang"
        )

    model_name = policy["reference_model"]["model_name"]
    alias = registry["candidate_alias"]
    version_policy = str(policy["policy"]["version"])

    existing = client.search_model_versions(
        f"name = '{model_name}'"
    )

    # Simpan kondisi champion sebelum registrasi.
    champion_before = {
        str(item.version)
        for item in existing
        if "champion" in (item.aliases or [])
    }

    # F. IDEMPOTENCY
    duplicate = next(
        (
            item for item in existing
            if item.tags.get("source_git_sha") == git_sha
            and item.tags.get("dataset_sha256") == actual_sha
            and item.tags.get("validation_policy") == version_policy
        ),
        None,
    )

    if duplicate is None:
        created = mlflow.register_model(
            model_uri=best["model_uri"],
            name=model_name,
        )

        version = str(created.version)
        action = "registered_new_version"

        tags = {
            "source_git_sha": git_sha,
            "dataset_sha256": actual_sha,
            "source_run_id": best["run_id"],
            "validation_status": "passed",
            "validation_policy": version_policy,
            "deployment_scope":
                policy["policy"]["deployment_scope"],
            "operational_approval": "not_approved",
        }

        for key, value in tags.items():
            client.set_model_version_tag(
                model_name, version, key, str(value)
            )

    else:
        version = str(duplicate.version)
        action = "existing_version_reused"

    # G. STAGING TRANSITION
    current = client.get_model_version(
        model_name, version
    )

    if current.current_stage != "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )

    client.set_registered_model_alias(
        model_name, alias, version
    )

    # H. VERIFY ACTUAL REGISTRY
    updated = client.get_model_version(
        model_name, version
    )

    if updated.current_stage != "Staging":
        raise RuntimeError("Model belum masuk Staging")

    alias_target = client.get_model_version_by_alias(
        model_name, alias
    )

    if str(alias_target.version) != version:
        raise RuntimeError("Alias Staging tidak cocok")

    # Pastikan alias champion tidak berubah.
    after_versions = client.search_model_versions(
        f"name = '{model_name}'"
    )

    champion_after = {
        str(item.version)
        for item in after_versions
        if "champion" in (item.aliases or [])
    }

    if champion_before != champion_after:
        raise RuntimeError(
            "Champion alias berubah tanpa izin"
        )

    registered_uri = f"models:/{model_name}/{version}"

    test_prediction = smoke_test(
        registered_uri, features
    )

    return {
        "status": "PASS",
        "action": action,
        "registered_model": model_name,
        "version": version,
        "stage": updated.current_stage,
        "candidate_alias": alias,
        "git_sha": git_sha,
        "dataset_sha256": actual_sha,
        "training_run_id": best["run_id"],
        "registered_model_uri": registered_uri,
        "test_predicted_residual_ms": test_prediction,
        "champion_alias_modified": False,
        "operational_approval": "not_approved",
    }


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    try:
        result = register()

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


if __name__ == "__main__":
    main()
