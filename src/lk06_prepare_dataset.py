
"""LK06: Prepare paired ECMWF forecast-reference regression data."""

from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_PATH = Path("data/versioned/forecast_history.csv")
REFERENCE_PATH = Path(
    "data/raw/reference/lk06_ecmwf_ifs_analysis.csv"
)
OUTPUT_PATH = Path("data/processed/lk06_training_dataset.csv")

TARGET_LEADS = [6, 12, 18, 24]

FORECAST_FEATURES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]


def prepare_dataset():
    print("=" * 65)
    print("LK06 - FORECAST REFERENCE PAIRING")
    print("=" * 65)

    # 1. Load forecast and reference.
    forecast = pd.read_csv(FORECAST_PATH)
    reference = pd.read_csv(REFERENCE_PATH)

    required_forecast = {
        "run_time",
        "valid_time",
        *FORECAST_FEATURES,
    }

    required_reference = {
        "valid_time",
        "reference_wind_speed_10m",
    }

    if not required_forecast.issubset(forecast.columns):
        raise ValueError("Forecast schema is incomplete.")

    if not required_reference.issubset(reference.columns):
        raise ValueError("Reference schema is incomplete.")

    # 2. Normalize timestamps.
    for column in ["run_time", "valid_time"]:
        forecast[column] = pd.to_datetime(
            forecast[column], utc=True, errors="raise"
        )

    reference["valid_time"] = pd.to_datetime(
        reference["valid_time"], utc=True, errors="raise"
    )

    # 3. Select LK01 prediction horizons.
    forecast["lead_hours"] = (
        forecast["valid_time"] - forecast["run_time"]
    ).dt.total_seconds() / 3600

    forecast = forecast[
        forecast["lead_hours"].isin(TARGET_LEADS)
    ].copy()

    if forecast.empty:
        raise ValueError("No matching forecast lead hours.")

    # 4. Validate keys to prevent accidental duplication.
    if forecast.duplicated(
        ["run_time", "valid_time"]
    ).any():
        raise ValueError("Duplicate forecast pairs.")

    if reference["valid_time"].duplicated().any():
        raise ValueError("Duplicate reference timestamps.")

    # 5. Pair forecast with reference.
    dataset = forecast.merge(
        reference,
        on="valid_time",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing_matches = (dataset["_merge"] != "both").sum()

    if missing_matches:
        raise ValueError(
            f"Missing reference matches: {missing_matches}"
        )

    dataset = dataset.drop(columns="_merge")

    # 6. Convert meteorological features to numeric.
    for column in FORECAST_FEATURES:
        dataset[column] = pd.to_numeric(
            dataset[column], errors="coerce"
        )

    dataset["reference_wind_speed_10m"] = pd.to_numeric(
        dataset["reference_wind_speed_10m"],
        errors="coerce",
    )

    critical = [
        "wind_speed_10m",
        "wind_direction_10m",
        "reference_wind_speed_10m",
    ]

    if dataset[critical].isna().any().any():
        raise ValueError("Missing critical wind-speed data.")

    if (
        (dataset["wind_speed_10m"] < 0).any()
        or (dataset["reference_wind_speed_10m"] < 0).any()
    ):
        raise ValueError("Negative wind speed detected.")

    if not dataset["wind_direction_10m"].between(
        0, 360
    ).all():
        raise ValueError("Invalid wind direction.")

    # 7. Rename feature columns to identify forecast origin.
    dataset = dataset.rename(
        columns={
            col: f"forecast_{col}"
            for col in FORECAST_FEATURES
        }
    )

    # 8. Cyclical wind-direction features.
    direction_rad = np.deg2rad(
        dataset["forecast_wind_direction_10m"] % 360
    )

    dataset["wind_direction_sin"] = np.sin(direction_rad)
    dataset["wind_direction_cos"] = np.cos(direction_rad)

    # 9. Cyclical time features in Jambi local time.
    local_time = dataset["valid_time"].dt.tz_convert(
        "Asia/Jakarta"
    )

    hour = local_time.dt.hour

    dataset["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    dataset["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    day = local_time.dt.dayofyear

    dataset["day_of_year_sin"] = np.sin(
        2 * np.pi * day / 365.25
    )
    dataset["day_of_year_cos"] = np.cos(
        2 * np.pi * day / 365.25
    )

    # 10. Calculate the regression target.
    dataset["residual"] = (
        dataset["reference_wind_speed_10m"]
        - dataset["forecast_wind_speed_10m"]
    )

    if dataset["residual"].isna().any():
        raise ValueError("Residual target contains missing values.")

    # 11. Keep relevant columns and save.
    output_columns = [
        "run_time",
        "valid_time",
        "lead_hours",
        *[f"forecast_{col}" for col in FORECAST_FEATURES],
        "wind_direction_sin",
        "wind_direction_cos",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "reference_wind_speed_10m",
        "residual",
    ]

    dataset = dataset[output_columns].sort_values(
        ["run_time", "valid_time"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    # 12. Summary and original forecast baseline.
    baseline_mae = dataset["residual"].abs().mean()
    baseline_rmse = np.sqrt(
        np.mean(dataset["residual"] ** 2)
    )

    print("Training rows:", len(dataset))
    print("Forecast runs:", dataset["run_time"].nunique())
    print("Duplicate pairs:", dataset.duplicated(
        ["run_time", "valid_time"]
    ).sum())
    print("Missing target:", dataset["residual"].isna().sum())

    print(f"Raw forecast MAE : {baseline_mae:.4f} m/s")
    print(f"Raw forecast RMSE: {baseline_rmse:.4f} m/s")

    print("\n=== RESIDUAL DATASET ===")
    print(
        dataset[
            [
                "run_time",
                "lead_hours",
                "forecast_wind_speed_10m",
                "reference_wind_speed_10m",
                "residual",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)
    print("DATASET PREPARATION PASSED")


if __name__ == "__main__":
    prepare_dataset()
