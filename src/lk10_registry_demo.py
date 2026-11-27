"""LK10 academic Production registry copy, without altering LK09's Staging model.

Usage:
  python -m src.lk10_registry_demo status
  python -m src.lk10_registry_demo prepare
  python -m src.lk10_registry_demo serve

Only 'prepare' writes to the remote Registry. The other modes are read-only.
"""

from __future__ import annotations

import argparse
import subprocess

from mlflow import MlflowClient

from src.lk10_model_ops import registry_identity

EXPECTED_SOURCE_VERSION = "2"
EXPECTED_RUN_ID = "007309b8db124732804cbdea3acda948"
DEST_SUFFIX = "-LK10-Demo"
SOURCE_TAG = "lk10_source_uri"
DEMO_ALIAS = "lk10-demo"


def get_state():
    source_name, alias, candidate = registry_identity()
    client = MlflowClient()
    if str(candidate.version) != EXPECTED_SOURCE_VERSION:
        raise SystemExit(
            f"[STOP] Alias {alias} now points to version {candidate.version}, "
            f"not the downloaded and tested version {EXPECTED_SOURCE_VERSION}."
        )
    if candidate.current_stage != "Staging":
        raise SystemExit("[STOP] Source version no longer has stage Staging.")
    if candidate.run_id != EXPECTED_RUN_ID:
        raise SystemExit("[STOP] Source run ID differs from the verified LK10 model.")

    source_uri = f"models:/{source_name}/{EXPECTED_SOURCE_VERSION}"
    dest_name = source_name + DEST_SUFFIX
    versions = list(client.search_model_versions(f"name='{dest_name}'"))
    matching = [
        v for v in versions
        if (v.tags or {}).get(SOURCE_TAG) == source_uri
        or v.source == source_uri
    ]
    if versions and not matching:
        raise SystemExit(
            "[STOP] LK10 demo Registry name already exists with a different source. "
            "No remote changes made."
        )
    if len(matching) > 1:
        raise SystemExit("[STOP] More than one matching demo version; review Registry manually.")
    return client, source_name, source_uri, dest_name, matching[0] if matching else None


def show_status():
    client, source_name, source_uri, dest_name, demo = get_state()
    print("[PASS] Verified source:", source_uri)
    print("Source lifecycle: Staging (unchanged)")
    print("Source run:", EXPECTED_RUN_ID)
    print("Academic demonstration Registry:", dest_name)
    if demo:
        current = client.get_model_version(dest_name, demo.version)
        print("Demo version:", current.version)
        print("Demo stage:", current.current_stage)
        print("Demo source:", current.source)
        print("Demo URI:", f"models:/{dest_name}/Production")
    else:
        print("Demo status: not created")


def prepare():
    client, source_name, source_uri, dest_name, demo = get_state()
    if demo is None:
        print("[INFO] Copying the verified source model to an isolated demo Registry...")
        demo = client.copy_model_version(source_uri, dest_name)
        print("[PASS] Copied as version", demo.version)
    else:
        print("[PASS] Existing copy found: version", demo.version)

    client.set_model_version_tag(dest_name, demo.version, SOURCE_TAG, source_uri)
    client.set_model_version_tag(dest_name, demo.version, "academic_simulation", "true")
    client.set_model_version_tag(dest_name, demo.version, "operational_approval", "not_approved")
    client.set_model_version_tag(dest_name, demo.version, "lk10_source_run_id", EXPECTED_RUN_ID)

    if demo.current_stage != "Production":
        client.transition_model_version_stage(
            dest_name, demo.version, "Production", archive_existing_versions=False
        )
    client.set_registered_model_alias(dest_name, DEMO_ALIAS, demo.version)

    after = client.get_model_version(dest_name, demo.version)
    original = client.get_model_version(source_name, EXPECTED_SOURCE_VERSION)
    assert after.current_stage == "Production"
    assert original.current_stage == "Staging"
    print("[PASS] Demo model:", dest_name)
    print("[PASS] Demo stage: Production (academic simulation only)")
    print("[PASS] Source model remains Staging")
    print("[PASS] Production URI:", f"models:/{dest_name}/Production")
    print("[NOTE] This is not approval for operational weather forecasting.")


def serve():
    client, _, _, dest_name, demo = get_state()
    if not demo or client.get_model_version(dest_name, demo.version).current_stage != "Production":
        raise SystemExit("[STOP] Run 'prepare' and verify Production before serving.")
    uri = f"models:/{dest_name}/Production"
    print("[INFO] Starting native MLflow REST serving from:", uri, flush=True)
    subprocess.run(
        ["mlflow", "models", "serve", "--model-uri", uri, "--port", "5002",
         "--host", "127.0.0.1", "--env-manager", "local"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "prepare", "serve"])
    args = parser.parse_args()
    {"status": show_status, "prepare": prepare, "serve": serve}[args.action]()


if __name__ == "__main__":
    main()
