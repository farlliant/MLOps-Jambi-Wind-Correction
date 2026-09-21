from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_poc_dataset import (
    fetch_reference,
    fetch_single_run,
)


# ============================================================
# STORAGE
# ============================================================

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports")

CRITICAL_COLUMNS = [
    "forecast_wind_speed_10m",
    "reference_wind_speed_10m",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = ArgumentParser(
        description="LK03 dynamic data pipeline preview."
    )

    parser.add_argument(
        "--run-time",
        default="2026-08-18T00:00",
        help=(
            "ECMWF forecast run time in UTC. "
            "Example: 2026-08-18T00:00"
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_forecast(df):
    required_columns = {
        "run_time",
        "valid_time",
        "lead_hours",
        "forecast_wind_speed_10m",
        "forecast_wind_direction_10m",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Forecast missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_count = df.duplicated(
        subset=[
            "run_time",
            "valid_time",
            "lead_hours",
        ]
    ).sum()

    if duplicate_count > 0:
        raise RuntimeError(
            f"Forecast contains {duplicate_count} duplicate rows."
        )

    wind_speed = df[
        "forecast_wind_speed_10m"
    ].dropna()

    if (wind_speed < 0).any():
        raise RuntimeError(
            "Forecast wind speed contains negative values."
        )

    direction = df[
        "forecast_wind_direction_10m"
    ].dropna()

    invalid_direction = ~direction.between(0, 360)

    if invalid_direction.any():
        raise RuntimeError(
            "Forecast wind direction outside 0-360 degrees."
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    result = df.copy()

    result["valid_time"] = pd.to_datetime(
        result["valid_time"],
        utc=True,
    )

    result["run_time"] = pd.to_datetime(
        result["run_time"],
        utc=True,
    )

    # --------------------------------------------------------
    # Wind direction is circular.
    # --------------------------------------------------------

    direction_rad = np.deg2rad(
        result["forecast_wind_direction_10m"]
    )

    result["wind_direction_sin"] = np.sin(
        direction_rad
    )

    result["wind_direction_cos"] = np.cos(
        direction_rad
    )

    # --------------------------------------------------------
    # Temporal features use local Jambi time (WIB).
    # Storage remains UTC.
    # --------------------------------------------------------

    local_time = (
        result["valid_time"]
        .dt.tz_convert("Asia/Jakarta")
    )

    local_hour = local_time.dt.hour
    day_of_year = local_time.dt.dayofyear

    result["hour_sin"] = np.sin(
        2 * np.pi * local_hour / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * local_hour / 24
    )

    result["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    result["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    run_time = pd.Timestamp(args.run_time)

    if run_time.tzinfo is None:
        run_time = run_time.tz_localize("UTC")
    else:
        run_time = run_time.tz_convert("UTC")

    run_datetime = run_time.to_pydatetime()

    print("=" * 88)
    print("LK03 - DYNAMIC DATA PIPELINE PREVIEW")
    print("=" * 88)

    print(
        "Forecast source : "
        "ECMWF IFS via Open-Meteo Single Runs API"
    )

    print(
        "Reference       : "
        "ECMWF IFS Analysis"
    )

    print(
        "Forecast run    :",
        run_time.isoformat(),
    )

    print()

    # ========================================================
    # 1. EXTRACT FORECAST
    # ========================================================

    forecast = fetch_single_run(
        run_datetime
    )

    if forecast is None or forecast.empty:
        raise RuntimeError(
            "Forecast extraction returned no data."
        )

    validate_forecast(
        forecast
    )

    print("[1/5] FORECAST INGESTION")

    print(
        "      rows       :",
        len(forecast),
    )

    print(
        "      lead hours :",
        sorted(
            forecast[
                "lead_hours"
            ]
            .astype(int)
            .unique()
            .tolist()
        ),
    )

    # ========================================================
    # 2. EXTRACT REFERENCE
    # ========================================================

    reference_start = (
        forecast["valid_time"]
        .min()
        .date()
        .isoformat()
    )

    reference_end = (
        forecast["valid_time"]
        .max()
        .date()
        .isoformat()
    )

    reference = fetch_reference(
        reference_start,
        reference_end,
    )

    print("[2/5] REFERENCE INGESTION")

    print(
        "      rows       :",
        len(reference),
    )

    print(
        "      period     :",
        f"{reference_start} -> {reference_end}",
    )

    # ========================================================
    # 3. CLEANING + PAIRING
    # ========================================================

    paired = forecast.merge(
        reference,
        on="valid_time",
        how="left",
        validate="many_to_one",
    )

    invalid_mask = (
        paired[
            CRITICAL_COLUMNS
        ]
        .isna()
        .any(axis=1)
    )

    invalid_rows = int(
        invalid_mask.sum()
    )

    clean = (
        paired[
            ~invalid_mask
        ]
        .copy()
    )

    if clean.empty:
        raise RuntimeError(
            "No valid forecast-reference pairs remained."
        )

    clean["residual"] = (
        clean[
            "reference_wind_speed_10m"
        ]
        -
        clean[
            "forecast_wind_speed_10m"
        ]
    )

    clean["absolute_error"] = (
        clean["residual"].abs()
    )

    clean["squared_error"] = (
        clean["residual"] ** 2
    )

    print(
        "[3/5] CLEANING + VALID_TIME PAIRING"
    )

    print(
        "      paired rows :",
        len(paired),
    )

    print(
        "      valid rows  :",
        len(clean),
    )

    print(
        "      excluded    :",
        invalid_rows,
    )

    # ========================================================
    # 4. FEATURE ENGINEERING
    # ========================================================

    processed = engineer_features(
        clean
    )

    feature_columns = [
        "wind_direction_sin",
        "wind_direction_cos",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    print(
        "[4/5] FEATURE ENGINEERING"
    )

    for feature in feature_columns:
        print(
            "      +",
            feature,
        )

    # ========================================================
    # 5. STORAGE
    # ========================================================

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_label = (
        run_time
        .strftime("%Y%m%dT%H")
    )

    interim_file = (
        INTERIM_DIR
        / f"lk03_paired_{run_label}.csv"
    )

    processed_file = (
        PROCESSED_DIR
        / f"lk03_training_{run_label}.csv"
    )

    report_file = (
        REPORT_DIR
        / f"lk03_capture_{run_label}.csv"
    )

    clean.to_csv(
        interim_file,
        index=False,
    )

    processed.to_csv(
        processed_file,
        index=False,
    )

    duplicate_keys = int(
        forecast.duplicated(
            subset=[
                "run_time",
                "valid_time",
                "lead_hours",
            ]
        ).sum()
    )

    completeness = (
        len(clean)
        / len(paired)
        * 100
    )

    summary = pd.DataFrame(
        [
            {
                "forecast_run":
                    run_time.isoformat(),
                "forecast_rows":
                    len(forecast),
                "paired_rows":
                    len(paired),
                "valid_rows":
                    len(clean),
                "excluded_rows":
                    invalid_rows,
                "critical_completeness_percent":
                    completeness,
                "duplicate_forecast_keys":
                    duplicate_keys,
            }
        ]
    )

    summary.to_csv(
        report_file,
        index=False,
    )

    print(
        "[5/5] LAYERED STORAGE"
    )

    print(
        "      interim   :",
        interim_file,
    )

    print(
        "      processed :",
        processed_file,
    )

    print(
        "      report    :",
        report_file,
    )

    # ========================================================
    # OUTPUT PREVIEW
    # ========================================================

    display_columns = [
        "run_time",
        "valid_time",
        "lead_hours",
        "forecast_wind_speed_10m",
        "reference_wind_speed_10m",
        "residual",
        "wind_direction_sin",
        "wind_direction_cos",
        "hour_sin",
        "hour_cos",
    ]

    print(
        "\n=== PROCESSED PREVIEW ==="
    )

    print(
        processed[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== CAPTURE SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 88
    )

    print(
        "LK03 PIPELINE PREVIEW PASSED"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
