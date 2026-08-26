from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

LATITUDE = -1.633333333
LONGITUDE = 103.65

RUN_TIME = "2026-08-18T00:00"

URL = "https://single-runs-api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]

LEAD_HOURS = [6, 12, 18, 24]


def main():
    print("=" * 75)
    print("TEST A - ECMWF IFS SINGLE RUN - JAMBI")
    print("=" * 75)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": "ecmwf_ifs",
        "run": RUN_TIME,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": 30,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }

    print(f"Coordinate : {LATITUDE}, {LONGITUDE}")
    print("Model      : ecmwf_ifs")
    print(f"Run        : {RUN_TIME} UTC\n")

    response = requests.get(URL, params=params, timeout=90)

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print("\nAPI RESPONSE:")
        print(response.text)
        raise RuntimeError("Single Runs API request gagal.")

    data = response.json()

    print("\n=== API METADATA ===")
    print("Latitude :", data.get("latitude"))
    print("Longitude:", data.get("longitude"))
    print("Elevation:", data.get("elevation"))
    print("Timezone :", data.get("timezone"))

    print("\n=== UNITS ===")
    print(data.get("hourly_units"))

    hourly = data.get("hourly")

    if hourly is None:
        raise RuntimeError("Field hourly tidak ditemukan.")

    df = pd.DataFrame(hourly)

    df["time"] = pd.to_datetime(df["time"], utc=True)

    run_time = pd.Timestamp(RUN_TIME)

    if run_time.tzinfo is None:
        run_time = run_time.tz_localize("UTC")

    df["lead_hours"] = (
        (df["time"] - run_time).dt.total_seconds() / 3600
    )

    print("\n=== DATA INFO ===")
    print("Rows:", len(df))

    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    print("\n=== FIRST 10 ROWS ===")
    print(df.head(10).to_string(index=False))

    selected = df[df["lead_hours"].isin(LEAD_HOURS)].copy()

    show_columns = [
        "time",
        "lead_hours",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "cloud_cover",
        "precipitation",
    ]

    print("\n=== SELECTED LEAD TIMES ===")
    print(selected[show_columns].to_string(index=False))

    print("\n=== DATA QUALITY ===")
    print(
        "Missing wind speed:",
        df["wind_speed_10m"].isna().sum(),
    )
    print(
        "Duplicate timestamp:",
        df["time"].duplicated().sum(),
    )
    print(
        "Wind min:",
        df["wind_speed_10m"].min(),
        "m/s",
    )
    print(
        "Wind max:",
        df["wind_speed_10m"].max(),
        "m/s",
    )

    output_dir = Path("data/raw/single_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "single_run_20260818T00.csv"

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 75)
    print("TEST A PASSED")
    print("=" * 75)
    print("Saved:", output_file)


if __name__ == "__main__":
    main()