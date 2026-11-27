"""Smoke test for native MLflow Models REST serving and each Compose replica."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request

from train import FEATURES
from src.lk07_inference import build_synthetic_input


def request(url: str, method: str, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc


def wait_until_ready(base_url: str, seconds: int) -> None:
    end = time.monotonic() + seconds
    while True:
        try:
            status, _ = request(f"{base_url}/ping", "GET", timeout=3)
            if status == 200:
                return
        except (OSError, RuntimeError):
            pass
        if time.monotonic() >= end:
            raise RuntimeError(f"Endpoint not ready: {base_url}/ping")
        time.sleep(3)


def infer(base_url: str, wait_seconds: int) -> float:
    wait_until_ready(base_url, wait_seconds)
    frame, _, _ = build_synthetic_input(FEATURES)
    payload = {"dataframe_split": frame.to_dict(orient="split")}
    status, body = request(f"{base_url}/invocations", "POST", payload=payload, timeout=60)
    result = json.loads(body)
    raw_predictions = result.get("predictions") if isinstance(result, dict) else result
    if not isinstance(raw_predictions, list) or len(raw_predictions) != 1:
        raise RuntimeError(f"Unexpected MLflow response format: {body[:800]}")
    residual = float(raw_predictions[0])
    if not math.isfinite(residual):
        raise RuntimeError("Non-finite model response")
    raw_wind = float(frame.iloc[0]["forecast_wind_speed_10m"])
    corrected_wind = max(0.0, raw_wind + residual)
    print(json.dumps({
        "endpoint": base_url,
        "http_status": status,
        "model_type": "MLflow pyfunc residual regressor",
        "raw_wind_ms": raw_wind,
        "predicted_residual_ms": residual,
        "corrected_wind_ms": corrected_wind,
        "test_input": "synthetic",
    }, indent=2))
    return residual


def get_replica_port(index: int) -> str:
    result = subprocess.run(
        ["docker", "compose", "port", "--index", str(index), "model-serving", "8080"],
        check=True, capture_output=True, text=True,
    )
    address = result.stdout.strip().splitlines()[-1].strip()
    port = int(address.rsplit(":", 1)[-1])
    return f"http://127.0.0.1:{port}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--all-replicas", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.all_replicas:
        infer(args.url, args.wait_seconds)
        print("[PASS] MLflow Models REST inference")
        return

    predictions = []
    for index in (1, 2, 3):
        base = get_replica_port(index)
        print(f"--- MLflow replica {index} ---")
        predictions.append(infer(base, args.wait_seconds))
    if not math.isclose(predictions[0], predictions[1], rel_tol=1e-8, abs_tol=1e-8) or not math.isclose(predictions[0], predictions[2], rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError("Replicas returned different predictions for identical input")
    print("[PASS] Three MLflow serving replicas responded independently and consistently")


if __name__ == "__main__":
    main()
