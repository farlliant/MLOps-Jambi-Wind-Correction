from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

LATITUDE = -1.633333
LONGITUDE = 103.650000

SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "ecmwf_ifs"

FORECAST_VARIABLES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]

RUN_HOURS = [0, 6, 12, 18]

MAX_REQUEST_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 90

RAW_DIR = Path("data/raw/single_runs")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LK04 dynamic ECMWF IFS data ingestion."
    )

    parser.add_argument(
        "--run-time",
        type=str,
        default=None,
        help="Optional forecast run, example: 2026-08-18T00:00",
    )

    return parser.parse_args()


# ============================================================
# TIME HELPERS
# ============================================================

def parse_run_time(value: str) -> datetime:
    """Convert user-provided run time into UTC datetime."""
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


def candidate_runs(
    now: datetime | None = None,
    lookback_hours: int = 48,
):
    """Yield ECMWF run candidates from newest to oldest."""
    if now is None:
        now = datetime.now(timezone.utc)

    latest_run_hour = (now.hour // 6) * 6

    latest_candidate = now.replace(
        hour=latest_run_hour,
        minute=0,
        second=0,
        microsecond=0,
    )

    for offset in range(0, lookback_hours + 1, 6):
        yield latest_candidate - timedelta(hours=offset)


# ============================================================
# HTTP
# ============================================================

def request_forecast(run_time: datetime) -> dict | None:
    """Request one ECMWF IFS run with retry handling."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": FORECAST_MODEL,
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "hourly": ",".join(FORECAST_VARIABLES),
        "forecast_hours": 30,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(
                SINGLE_RUN_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in {400, 404, 422}:
                print(
                    f"[WARN] Run {run_time.isoformat()} "
                    "belum/tidak tersedia."
                )
                return None

            print(
                f"[WARN] HTTP {response.status_code} "
                f"attempt {attempt}/{MAX_REQUEST_ATTEMPTS}"
            )

        except (requests.RequestException, ValueError) as exc:
            print(
                f"[WARN] Request attempt "
                f"{attempt}/{MAX_REQUEST_ATTEMPTS} gagal: {exc}"
            )

        if attempt < MAX_REQUEST_ATTEMPTS:
            print(
                f"[INFO] Retry dalam {RETRY_DELAY_SECONDS} detik..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return None


# ============================================================
# RAW DATA CREATION
# ============================================================

def build_raw_dataframe(
    payload: dict,
    run_time: datetime,
) -> pd.DataFrame | None:
    """Convert API payload into raw tabular data."""
    hourly = payload.get("hourly")

    if not hourly:
        print("[WARN] Payload tidak memiliki hourly data.")
        return None

    missing_variables = [
        variable
        for variable in ["time", *FORECAST_VARIABLES]
        if variable not in hourly
    ]

    if missing_variables:
        print(
            "[WARN] Kolom API tidak lengkap:",
            ", ".join(missing_variables),
        )
        return None

    dataframe = pd.DataFrame(hourly)

    if dataframe.empty:
        print("[WARN] API menghasilkan DataFrame kosong.")
        return None

    retrieved_at = datetime.now(timezone.utc)

    dataframe["run_time"] = pd.Timestamp(run_time)
    dataframe["valid_time"] = pd.to_datetime(
        dataframe["time"],
        utc=True,
    )
    dataframe["retrieved_at"] = retrieved_at
    dataframe["source"] = "ECMWF IFS via Open-Meteo Single Runs API"

    return dataframe


def snapshot_path(run_time: datetime) -> Path:
    """Return immutable snapshot path for one forecast run."""
    return RAW_DIR / f"{run_time:%Y%m%dT%H}.csv"


def ingest_run(run_time: datetime) -> Path | None:
    """Fetch and persist one raw forecast run."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = snapshot_path(run_time)

    if output_path.exists():
        print(f"[SKIP] Snapshot sudah tersedia: {output_path}")
        return output_path

    print(f"[FETCH] {run_time.isoformat()}")

    payload = request_forecast(run_time)

    if payload is None:
        return None

    dataframe = build_raw_dataframe(payload, run_time)

    if dataframe is None:
        return None

    dataframe.to_csv(output_path, index=False)

    print(f"[PASS] Rows : {len(dataframe)}")
    print(f"[SAVE] {output_path}")

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run deterministic or dynamic ingestion."""
    args = parse_arguments()

    print("=" * 70)
    print("LK04 - DYNAMIC ECMWF IFS DATA INGESTION")
    print("=" * 70)

    if args.run_time:
        run_time = parse_run_time(args.run_time)

        output_path = ingest_run(run_time)

        if output_path is None:
            raise RuntimeError(
                f"Forecast run {run_time.isoformat()} gagal diambil."
            )

        print("\nINGESTION PASSED")
        return

    print("[INFO] Mencari latest available forecast run...")

    for run_time in candidate_runs():
        print(f"\n[TRY] {run_time.isoformat()}")

        output_path = ingest_run(run_time)

        if output_path is not None:
            print(f"\n[SELECTED] {run_time.isoformat()}")
            print("INGESTION PASSED")
            return

    raise RuntimeError(
        "Tidak menemukan forecast run yang tersedia "
        "dalam lookback window."
    )


if __name__ == "__main__":
    main()
