"""LK10 model-registry checks and artifact preparation.

No remote Registry writes are made by this module. LK09 remains untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".lk10" / "model"


def load_env() -> None:
    """Load key=value settings from the local Git-ignored .env without echoing secrets."""
    env_file = ROOT / ".env"
    if not env_file.is_file():
        raise SystemExit("[FAIL] .env missing: copy .env.example and fill credentials")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)
    required = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    missing = [key for key in required if not os.environ.get(key) or "replace_with" in os.environ[key]]
    if missing:
        raise SystemExit(f"[FAIL] Credentials not configured: {', '.join(missing)}")


def registry_identity():
    load_env()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name = os.environ.get("MODEL_NAME", "Jambi-Wind-Residual-RandomForest")
    alias = os.environ.get("MODEL_ALIAS", "staging-candidate")
    version = MlflowClient().get_model_version_by_alias(name, alias)
    return name, alias, version


def cmd_status() -> None:
    name, alias, version = registry_identity()
    print("[PASS] DagsHub MLflow registry is reachable")
    print("Model:", name)
    print("Alias:", alias)
    print("Version:", version.version)
    print("Stage:", version.current_stage)
    print("Source run:", version.run_id)
    print("Model URI:", f"models:/{name}@{alias}")
    if version.current_stage != "Production":
        print("[NOTE] This is not Production. LK09 requires Staging: do not promote this version in place.")
        print("[NOTE] LK10 local serving and scaling can be tested without altering Registry lifecycle.")


def cmd_download() -> None:
    name, alias, version = registry_identity()
    uri = f"models:/{name}@{alias}"
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lk10-model-") as tmpdir:
        path = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmpdir))
        if not (path / "MLmodel").is_file():
            raise RuntimeError(f"MLmodel not found inside downloaded model: {path}")
        if CACHE.exists():
            shutil.rmtree(CACHE)
        shutil.copytree(path, CACHE)
    print("[PASS] Registered model downloaded as a local model directory")
    print("Model version:", version.version, "Stage:", version.current_stage)
    print("Model path:", CACHE)
    print("[PASS] MLmodel:", (CACHE / "MLmodel").is_file())
    print("[NOTE] Directory .lk10 is ignored by Git and should NOT be committed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "download"])
    args = parser.parse_args()
    {"status": cmd_status, "download": cmd_download}[args.action]()


if __name__ == "__main__":
    main()
