from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/single_runs")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports")

TARGET_LEADS = [6, 12, 18, 24]

REQUIRED_COLUMNS = [
    "run_time",
    "valid_time",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]

NUMERIC_COLUMNS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]

CRITICAL_COLUMNS = [
    "wind_speed_10m",
    "wind_direction_10m",
]


def parse_arguments() -> argparse.Namespace:
    """Parse preprocessing arguments."""
    parser = argparse.ArgumentParser(
        description="LK04 automated forecast preprocessing."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional raw CSV path.",
    )

    return parser.parse_args()


def latest_raw_file() -> Path:
    """Find the latest forecast snapshot."""
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "Tidak ada raw forecast pada data/raw/single_runs/."
        )

    return files[-1]


def validate_schema(dataframe: pd.DataFrame) -> None:
    """Ensure all required columns exist."""
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


def normalize_data_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize timestamps and numeric feature types."""
    dataframe = dataframe.copy()

    dataframe["run_time"] = pd.to_datetime(
        dataframe["run_time"],
        utc=True,
        errors="coerce",
    )

    dataframe["valid_time"] = pd.to_datetime(
        dataframe["valid_time"],
        utc=True,
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def add_lead_hours(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate forecast horizon from run and valid time."""
    dataframe = dataframe.copy()

    lead_hours = (
        dataframe["valid_time"] - dataframe["run_time"]
    ).dt.total_seconds() / 3600

    dataframe["lead_hours"] = (
        lead_hours.round().astype("Int64")
    )

    return dataframe


def invalid_physical_mask(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Identify meteorologically invalid rows."""
    invalid = pd.Series(
        False,
        index=dataframe.index,
    )

    invalid |= dataframe["wind_speed_10m"] < 0

    invalid |= ~dataframe["wind_direction_10m"].between(
        0,
        360,
    )

    invalid |= ~dataframe["relative_humidity_2m"].between(
        0,
        100,
    )

    invalid |= ~dataframe["cloud_cover"].between(
        0,
        100,
    )

    invalid |= dataframe["precipitation"] < 0

    return invalid


def add_cyclical_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create cyclical wind-direction and temporal features."""
    dataframe = dataframe.copy()

    radians = np.deg2rad(
        dataframe["wind_direction_10m"] % 360
    )

    dataframe["wind_direction_sin"] = np.sin(radians)
    dataframe["wind_direction_cos"] = np.cos(radians)

    local_time = dataframe["valid_time"].dt.tz_convert(
        "Asia/Jakarta"
    )

    hour = (
        local_time.dt.hour
        + local_time.dt.minute / 60
    )

    dataframe["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    dataframe["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    day_of_year = local_time.dt.dayofyear

    dataframe["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    dataframe["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return dataframe


def preprocess_file(input_path: Path) -> tuple[Path, Path]:
    """Preprocess one raw forecast snapshot."""
    print("=" * 70)
    print("LK04 - AUTOMATED PREPROCESSING")
    print("=" * 70)

    print(f"[INPUT] {input_path}")

    dataframe = pd.read_csv(input_path)
    raw_rows = len(dataframe)

    validate_schema(dataframe)
    print("[PASS] Schema validation")

    dataframe = normalize_data_types(dataframe)

    invalid_timestamp_mask = (
        dataframe["run_time"].isna()
        | dataframe["valid_time"].isna()
    )

    invalid_timestamp_rows = int(
        invalid_timestamp_mask.sum()
    )

    dataframe = dataframe.loc[
        ~invalid_timestamp_mask
    ].copy()

    dataframe = add_lead_hours(dataframe)

    dataframe = dataframe[
        dataframe["lead_hours"].isin(TARGET_LEADS)
    ].copy()

    selected_rows = len(dataframe)

    duplicate_mask = dataframe.duplicated(
        subset=[
            "run_time",
            "valid_time",
            "lead_hours",
        ]
    )

    duplicate_rows = int(duplicate_mask.sum())

    dataframe = dataframe.loc[
        ~duplicate_mask
    ].copy()

    missing_critical_mask = dataframe[
        CRITICAL_COLUMNS
    ].isna().any(axis=1)

    missing_critical_rows = int(
        missing_critical_mask.sum()
    )

    dataframe = dataframe.loc[
        ~missing_critical_mask
    ].copy()

    physical_mask = invalid_physical_mask(dataframe)
    invalid_physical_rows = int(physical_mask.sum())

    dataframe = dataframe.loc[
        ~physical_mask
    ].copy()

    dataframe = add_cyclical_features(dataframe)

    rename_map = {
        column: f"forecast_{column}"
        for column in NUMERIC_COLUMNS
    }

    dataframe = dataframe.rename(
        columns=rename_map
    )

    dataframe = dataframe.sort_values(
        ["run_time", "valid_time", "lead_hours"]
    ).reset_index(drop=True)

    processed_rows = len(dataframe)

    if processed_rows == 0:
        raise RuntimeError(
            "Preprocessing menghasilkan 0 valid rows."
        )

    completeness = (
        processed_rows / selected_rows * 100
        if selected_rows
        else 0.0
    )

    unique_runs = dataframe["run_time"].dropna().unique()

    if len(unique_runs) != 1:
        raise RuntimeError(
            "Raw snapshot harus berisi tepat satu run_time."
        )

    run_time = pd.Timestamp(unique_runs[0])

    run_id = run_time.strftime("%Y%m%dT%H")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_path = (
        PROCESSED_DIR
        / f"forecast_{run_id}_processed.csv"
    )

    report_path = (
        REPORT_DIR
        / f"forecast_{run_id}_quality.csv"
    )

    dataframe.to_csv(
        processed_path,
        index=False,
    )

    quality_report = pd.DataFrame(
        [
            {
                "run_time": run_time.isoformat(),
                "raw_rows": raw_rows,
                "selected_lead_rows": selected_rows,
                "invalid_timestamp_rows": (
                    invalid_timestamp_rows
                ),
                "duplicate_rows": duplicate_rows,
                "missing_critical_rows": (
                    missing_critical_rows
                ),
                "invalid_physical_rows": (
                    invalid_physical_rows
                ),
                "processed_rows": processed_rows,
                "completeness_percent": completeness,
            }
        ]
    )

    quality_report.to_csv(
        report_path,
        index=False,
    )

    print(f"Raw rows               : {raw_rows}")
    print(f"Target lead rows       : {selected_rows}")
    print(f"Duplicate rows         : {duplicate_rows}")
    print(
        "Missing critical rows : "
        f"{missing_critical_rows}"
    )
    print(
        "Invalid physical rows : "
        f"{invalid_physical_rows}"
    )
    print(f"Processed rows         : {processed_rows}")
    print(
        "Completeness          : "
        f"{completeness:.2f}%"
    )

    print("\n[FEATURES]")
    print("- wind_direction_sin")
    print("- wind_direction_cos")
    print("- hour_sin")
    print("- hour_cos")
    print("- day_of_year_sin")
    print("- day_of_year_cos")

    print(f"\n[SAVE] {processed_path}")
    print(f"[SAVE] {report_path}")

    print("\nPREPROCESSING PASSED")

    return processed_path, report_path


def main() -> None:
    """Run preprocessing."""
    args = parse_arguments()

    input_path = (
        args.input
        if args.input is not None
        else latest_raw_file()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input tidak ditemukan: {input_path}"
        )

    preprocess_file(input_path)


if __name__ == "__main__":
    main()