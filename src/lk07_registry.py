"""LK07 - Guarded MLflow Model Registry management.

Read-only:
    python src/lk07_registry.py verify

Registry changes require an explicit --apply flag:
    python src/lk07_registry.py sync --apply
    python src/lk07_registry.py staging --apply
    python src/lk07_registry.py production --apply

Production is an academic simulation, not deployment approval.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Jambi-Wind-Residual-RandomForest"

COMPARISON = Path(
    "data/reports/lk06/experiment_comparison.csv"
)
METADATA = Path(
    "data/reports/lk06/best_model_metadata.json"
)
TRAINING_DATA = Path(
    "data/versioned/lk06_training_dataset.csv"
)


def load_evidence(client):
    """Verify experiment results and dataset lineage."""

    with COMPARISON.open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {
            row["name"]: row
            for row in csv.DictReader(handle)
        }

    metadata = json.loads(
        METADATA.read_text(encoding="utf-8")
    )

    assert {"RF-01", "RF-02"}.issubset(rows)
    assert metadata["best_run_name"] == "RF-01"

    actual_hash = hashlib.sha256(
        TRAINING_DATA.read_bytes()
    ).hexdigest()

    assert actual_hash == metadata["dataset_sha256"], (
        "Training dataset hash mismatch."
    )

    for key in ("RF-01", "RF-02"):
        run = client.get_run(rows[key]["run_id"])

        assert run.info.status == "FINISHED"

        for metric in ("val_mae_ms", "val_rmse_ms"):
            assert abs(
                float(rows[key][metric])
                - run.data.metrics[metric]
            ) < 1e-9

    assert float(
        rows["RF-01"]["val_mae_ms"]
    ) < float(rows["RF-02"]["val_mae_ms"])

    assert float(
        rows["RF-01"]["val_rmse_ms"]
    ) < float(rows["RF-02"]["val_rmse_ms"])

    return rows, metadata


def validate_versions(client, rows, require_v2=True):
    """Ensure versions refer to the intended LK06 runs."""

    versions = {
        str(mv.version): mv
        for mv in client.search_model_versions(
            f"name = '{MODEL_NAME}'"
        )
    }

    if "1" not in versions:
        raise RuntimeError("LK06 V1 not found.")

    if require_v2 and "2" not in versions:
        raise RuntimeError("LK07 V2 not found.")

    if not set(versions).issubset({"1", "2"}):
        raise RuntimeError(
            "Unexpected Registry versions. Inspect manually."
        )

    for number, key in (
        ("1", "RF-01"),
        ("2", "RF-02"),
    ):
        if number not in versions:
            continue

        mv = versions[number]

        if (
            mv.run_id != rows[key]["run_id"]
            or mv.source != rows[key]["model_uri"]
        ):
            raise RuntimeError(
                f"Unexpected source for V{number}"
            )

    return versions


def verify(client, rows):
    """Read-only verification of the final LK07 Registry."""

    versions = validate_versions(client, rows)

    expected_aliases = {
        "candidate": "1",
        "champion": "1",
        "challenger": "2",
    }

    for alias, expected in expected_aliases.items():
        mv = client.get_model_version_by_alias(
            MODEL_NAME, alias
        )

        assert str(mv.version) == expected, (
            f"Alias {alias} points to the wrong version."
        )

    assert versions["1"].current_stage == "Production"
    assert versions["2"].current_stage == "None"

    assert (
        versions["1"].tags.get("deployment_scope")
        == "academic_simulation_only"
    )

    assert (
        versions["1"].tags.get("operational_approval")
        == "not_approved"
    )

    print("=" * 60)
    print("LK07 - MODEL REGISTRY VERIFICATION")
    print("=" * 60)

    print("Registered model:", MODEL_NAME)

    for number, key in (
        ("1", "RF-01"),
        ("2", "RF-02"),
    ):
        mv = client.get_model_version(
            MODEL_NAME, number
        )

        print(f"\nVersion {number}: {key}")
        print("Run ID:", mv.run_id)
        print("Stage:", mv.current_stage)
        print("MAE:", rows[key]["val_mae_ms"])
        print("RMSE:", rows[key]["val_rmse_ms"])

    print("\nVerified aliases:")
    for alias, number in expected_aliases.items():
        print(f"  @{alias} -> V{number}")

    print("\nDeployment: academic simulation only")
    print("Operational approval: NOT APPROVED")

    print("\n[PASS] LK07 REGISTRY VERIFICATION")


def sync(client, rows, metadata):
    """Register V2 if absent and synchronize metadata."""

    versions = validate_versions(
        client, rows, require_v2=False
    )

    if "2" not in versions:
        created = mlflow.register_model(
            model_uri=rows["RF-02"]["model_uri"],
            name=MODEL_NAME,
        )

        if str(created.version) != "2":
            raise RuntimeError(
                "Unexpected version number during registration."
            )

    validate_versions(client, rows)

    client.update_registered_model(
        MODEL_NAME,
        description=(
            "Random Forest residual correction for Jambi "
            "ECMWF forecasts. ECMWF IFS Analysis is a "
            "model-based reference, not independent ground truth. "
            "Registry lifecycle is an academic simulation."
        ),
    )

    for version, key, selection_status in (
        ("1", "RF-01", "candidate_champion"),
        ("2", "RF-02", "challenger_not_promoted"),
    ):
        row = rows[key]

        client.update_model_version(
            name=MODEL_NAME,
            version=version,
            description=(
                f"{key}: n_estimators={row['n_estimators']}, "
                f"max_depth={row['max_depth']}, "
                f"min_samples_leaf={row['min_samples_leaf']}. "
                f"Validation MAE={float(row['val_mae_ms']):.4f} "
                f"m/s, RMSE={float(row['val_rmse_ms']):.4f} m/s. "
                "Academic demonstration, not operational approval."
            ),
        )

        tags = {
            "source_experiment":
                "LK06-Jambi-Wind-Residual-Correction",
            "source_run": key,
            "selection_status": selection_status,
            "validation_mae_ms": row["val_mae_ms"],
            "validation_rmse_ms": row["val_rmse_ms"],
            "dataset_sha256": metadata["dataset_sha256"],
        }

        for tag, value in tags.items():
            client.set_model_version_tag(
                MODEL_NAME, version, tag, str(value)
            )

    client.set_registered_model_alias(
        MODEL_NAME, "candidate", "1"
    )
    client.set_registered_model_alias(
        MODEL_NAME, "challenger", "2"
    )

    print("[PASS] Registry metadata synchronized")
    print("Existing stages were not changed.")


def transition(client, rows, target):
    """Guarded academic lifecycle transition of V1."""

    versions = validate_versions(client, rows)

    if versions["2"].current_stage != "None":
        raise RuntimeError(
            "V2 should remain unpromoted."
        )

    current = versions["1"].current_stage

    if target == "Staging":
        if current == "Production":
            print(
                "[SKIP] V1 is already Production. "
                "Refusing to roll back."
            )
            return

        if current == "None":
            client.transition_model_version_stage(
                MODEL_NAME, "1", "Staging"
            )
        elif current != "Staging":
            raise RuntimeError(
                f"Unexpected stage: {current}"
            )

        client.set_model_version_tag(
            MODEL_NAME,
            "1",
            "lk07_lifecycle_status",
            "staging_simulation",
        )

    elif target == "Production":
        if current == "Staging":
            client.transition_model_version_stage(
                MODEL_NAME, "1", "Production"
            )
        elif current != "Production":
            raise RuntimeError(
                "V1 must pass through Staging first."
            )

        client.set_registered_model_alias(
            MODEL_NAME, "champion", "1"
        )

        tags = {
            "lk07_lifecycle_status": "production_simulation",
            "deployment_scope": "academic_simulation_only",
            "operational_approval": "not_approved",
            "promotion_reason":
                "V1 has lower validation MAE and RMSE than V2",
        }

        for key, value in tags.items():
            client.set_model_version_tag(
                MODEL_NAME, "1", key, value
            )

    updated = client.get_model_version(
        MODEL_NAME, "1"
    )

    assert updated.current_stage == target
    print(f"[PASS] V1 stage: {target}")
    print("Scope: academic simulation only")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "action",
        nargs="?",
        default="verify",
        choices=[
            "verify",
            "sync",
            "staging",
            "production",
        ],
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Required for Registry modifications",
    )

    args = parser.parse_args()

    if args.action != "verify" and not args.apply:
        parser.error(
            "Registry changes require --apply."
        )

    if args.action == "verify" and args.apply:
        parser.error(
            "Verification is read-only. Omit --apply."
        )

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()
    rows, metadata = load_evidence(client)

    if args.action == "verify":
        verify(client, rows)
    elif args.action == "sync":
        sync(client, rows, metadata)
    elif args.action == "staging":
        transition(client, rows, "Staging")
    elif args.action == "production":
        transition(client, rows, "Production")


if __name__ == "__main__":
    main()
