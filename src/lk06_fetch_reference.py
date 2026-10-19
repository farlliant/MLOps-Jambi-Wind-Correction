
"""LK06: Fetch ECMWF IFS Analysis reference for the DVC dataset."""

from pathlib import Path

import pandas as pd
import requests


FORECAST_PATH = Path("data/versioned/forecast_history.csv")
OUTPUT_PATH = Path("data/raw/reference/lk06_ecmwf_ifs_analysis.csv")

API_URL = "https://archive-api.open-meteo.com/v1/archive"
TARGET_LEADS = [6, 12, 18, 24]


def main():
    # 1. Read dataset versioned with DVC in LK05.
    forecast = pd.read_csv(FORECAST_PATH)

    forecast["run_time"] = pd.to_datetime(
        forecast["run_time"], utc=True, errors="raise"
    )
    forecast["valid_time"] = pd.to_datetime(
        forecast["valid_time"], utc=True, errors="raise"
    )

    # 2. Select the forecast horizons used in LK01.
    forecast["lead_hours"] = (
        forecast["valid_time"] - forecast["run_time"]
    ).dt.total_seconds() / 3600

    forecast = forecast[
        forecast["lead_hours"].isin(TARGET_LEADS)
    ].copy()

    if forecast.empty:
        raise ValueError("No forecast rows for target lead hours.")

    target_times = forecast["valid_time"].drop_duplicates()

    start_date = target_times.min().date().isoformat()
    end_date = target_times.max().date().isoformat()

    print("=== LK06 REFERENCE INGESTION ===")
    print("Forecast rows:", len(forecast))
    print("Unique valid times:", len(target_times))
    print("Period:", start_date, "to", end_date)

    # 3. Fetch reference using UTC timestamps.
    params = {
        "latitude": -1.633333333,
        "longitude": 103.65,
        "models": "ecmwf_ifs_analysis_long_window",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "wind_speed_10m",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }

    response = requests.get(API_URL, params=params, timeout=90)
    print("HTTP status:", response.status_code)
    response.raise_for_status()

    hourly = response.json().get("hourly")
    if not hourly:
        raise ValueError("Reference API returned no hourly data.")

    reference = pd.DataFrame(hourly)

    reference["valid_time"] = pd.to_datetime(
        reference["time"], utc=True, errors="raise"
    )
    reference["reference_wind_speed_10m"] = pd.to_numeric(
        reference["wind_speed_10m"], errors="coerce"
    )

    if reference["valid_time"].duplicated().any():
        raise ValueError("Duplicate reference timestamps found.")

    # 4. Save only references required by the DVC forecast.
    reference = reference[
        reference["valid_time"].isin(target_times)
    ][["valid_time", "reference_wind_speed_10m"]].copy()

    reference = reference.sort_values("valid_time")

    missing_times = set(target_times) - set(reference["valid_time"])

    if missing_times:
        raise ValueError(
            f"Missing reference timestamps: {len(missing_times)}"
        )

    if reference["reference_wind_speed_10m"].isna().any():
        raise ValueError("Reference wind speed contains missing values.")

    if (reference["reference_wind_speed_10m"] < 0).any():
        raise ValueError("Reference wind speed contains invalid values.")

    # 5. Persist the validated reference dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(OUTPUT_PATH, index=False)

    print("Matched reference rows:", len(reference))
    print("Missing values: 0")
    print("Saved:", OUTPUT_PATH)
    print("REFERENCE INGESTION PASSED")


if __name__ == "__main__":
    main()
