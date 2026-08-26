from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

LATITUDE = -1.633333333
LONGITUDE = 103.65

URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2026-08-18"
END_DATE = "2026-08-19"

REFERENCE_MODEL = "ecmwf_ifs_analysis_long_window"

REFERENCE_VARIABLES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]

EXPECTED_TIMES = [
    "2026-08-18T06:00:00Z",
    "2026-08-18T12:00:00Z",
    "2026-08-18T18:00:00Z",
    "2026-08-19T00:00:00Z",
]


def main():
    print("=" * 75)
    print("TEST B - ECMWF IFS ANALYSIS REFERENCE - JAMBI")
    print("=" * 75)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": REFERENCE_MODEL,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(REFERENCE_VARIABLES),
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }

    print(f"Coordinate      : {LATITUDE}, {LONGITUDE}")
    print(f"Reference model : {REFERENCE_MODEL}")
    print(f"Period          : {START_DATE} -> {END_DATE}")
    print("\nFetching Historical Weather API...")

    response = requests.get(URL, params=params, timeout=90)

    print(f"HTTP status     : {response.status_code}")

    if response.status_code != 200:
        print("\nAPI RESPONSE:")
        print(response.text)
        raise RuntimeError(
            "ECMWF Analysis reference request gagal."
        )

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
        raise RuntimeError(
            "Field hourly tidak ditemukan pada response."
        )

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    print("\n=== DATA INFO ===")
    print("Rows:", len(df))

    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    print("\n=== ALL REFERENCE ROWS ===")
    print(df.to_string(index=False))

    expected_times = pd.to_datetime(
        EXPECTED_TIMES,
        utc=True,
    )

    selected = df[df["time"].isin(expected_times)].copy()

    show_columns = [
        "time",
        "wind_speed_10m",
        "wind_direction_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "cloud_cover",
        "precipitation",
    ]

    print("\n=== MATCHING TEST A VALID TIMES ===")
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
        "Matching timestamps:",
        len(selected),
        "/",
        len(expected_times),
    )

    all_times_matched = len(selected) == len(expected_times)

    if not all_times_matched:
        print(
            "\n[WARNING] Tidak semua valid_time "
            "dari Test A tersedia pada reference."
        )

    output_dir = Path("data/raw/reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / "ecmwf_analysis_20260818_20260819.csv"
    )

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 75)

    if all_times_matched:
        print("TEST B PASSED")
    else:
        print("TEST B COMPLETED WITH WARNING")

    print("=" * 75)
    print("Saved:", output_file)


if __name__ == "__main__":
    main()