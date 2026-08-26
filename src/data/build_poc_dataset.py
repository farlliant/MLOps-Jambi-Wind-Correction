from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = -1.633333333
LONGITUDE = 103.65

SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
REFERENCE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_MODEL = "ecmwf_ifs"
REFERENCE_MODEL = "ecmwf_ifs_analysis_long_window"

START_DATE = "2026-05-15"
END_DATE = "2026-08-15"

RUN_HOURS = [0, 6, 12, 18]
LEAD_HOURS = [6, 12, 18, 24]

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

CRITICAL_COLUMNS = [
    "forecast_wind_speed_10m",
    "reference_wind_speed_10m",
]

MIN_COMPLETENESS_PERCENT = 98.0

MAX_REQUEST_ATTEMPTS = 3
REQUEST_DELAY_SECONDS = 0.25
RETRY_DELAY_SECONDS = 2

RAW_SINGLE_RUN_DIR = Path("data/raw/single_runs")
RAW_REFERENCE_DIR = Path("data/raw/reference")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports")


# ============================================================
# HELPERS
# ============================================================

def iter_dates(start_date, end_date):
    """Yield setiap tanggal dari start_date sampai end_date."""
    current = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    while current <= end:
        yield current
        current += timedelta(days=1)


def request_json(url, params, attempts=MAX_REQUEST_ATTEMPTS):
    """HTTP GET dengan retry untuk transient API/network failure."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=90)

            if response.status_code == 200:
                return response.json()

            last_error = f"HTTP {response.status_code}: {response.text}"

        except requests.RequestException as exc:
            last_error = repr(exc)

        print(f"[WARN] Request attempt {attempt}/{attempts} gagal.")

        if attempt < attempts:
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(last_error)


# ============================================================
# FORECAST INGESTION
# ============================================================

def fetch_single_run(run_time):
    """
    Mengambil satu ECMWF IFS historical run dan menyimpan hanya
    lead +6, +12, +18, dan +24 jam.
    """
    RAW_SINGLE_RUN_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = RAW_SINGLE_RUN_DIR / f"{run_time:%Y%m%dT%H}.csv"

    # Gunakan cache jika sebelumnya sudah pernah di-fetch.
    if cache_file.exists():
        print(f"  [CACHE] {cache_file.name}")

        df = pd.read_csv(cache_file)
        df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

        return df

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

    try:
        data = request_json(SINGLE_RUN_URL, params)
    except RuntimeError as exc:
        print(f"[WARN] Run {run_time} gagal:")
        print(exc)
        return None

    hourly = data.get("hourly")

    if hourly is None:
        print(f"[WARN] Run {run_time} tidak memiliki hourly data.")
        return None

    df = pd.DataFrame(hourly)

    if df.empty:
        print(f"[WARN] Run {run_time} menghasilkan DataFrame kosong.")
        return None

    # Normalize waktu.
    df["valid_time"] = pd.to_datetime(df["time"], utc=True)

    run_timestamp = pd.Timestamp(run_time)
    if run_timestamp.tzinfo is None:
        run_timestamp = run_timestamp.tz_localize("UTC")

    df["run_time"] = run_timestamp

    # Hitung model lead.
    df["lead_hours"] = (
        (df["valid_time"] - df["run_time"]).dt.total_seconds() / 3600
    )

    df = df[df["lead_hours"].isin(LEAD_HOURS)].copy()

    if df.empty:
        print(f"[WARN] Run {run_time} tidak memiliki lead {LEAD_HOURS}.")
        return None

    # Prefix semua feature agar jelas berasal dari forecast.
    rename_map = {
        variable: f"forecast_{variable}"
        for variable in FORECAST_VARIABLES
    }

    df = df.rename(columns=rename_map)

    keep_columns = [
        "run_time",
        "valid_time",
        "lead_hours",
        *rename_map.values(),
    ]

    df = df[keep_columns].copy()
    df.to_csv(cache_file, index=False)

    return df


# ============================================================
# REFERENCE INGESTION
# ============================================================

def fetch_reference(start_date, end_date):
    """
    Mengambil ECMWF IFS analysis sebagai reference awal PoC.

    Catatan:
    reference ini bukan sensor ground truth.
    """
    RAW_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = (
        RAW_REFERENCE_DIR
        / f"reference_{start_date}_{end_date}.csv"
    )

    if cache_file.exists():
        print(f"[CACHE] Reference: {cache_file.name}")

        df = pd.read_csv(cache_file)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

        return df

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": REFERENCE_MODEL,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "wind_speed_10m",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }

    data = request_json(REFERENCE_URL, params)
    hourly = data.get("hourly")

    if hourly is None:
        raise RuntimeError("Reference tidak memiliki field hourly.")

    df = pd.DataFrame(hourly)

    if df.empty:
        raise RuntimeError("Reference menghasilkan DataFrame kosong.")

    df["valid_time"] = pd.to_datetime(df["time"], utc=True)

    df = df.rename(
        columns={"wind_speed_10m": "reference_wind_speed_10m"}
    )

    df = df[
        ["valid_time", "reference_wind_speed_10m"]
    ].copy()

    duplicate_reference = df["valid_time"].duplicated().sum()

    if duplicate_reference > 0:
        raise RuntimeError(
            f"Reference memiliki {duplicate_reference} duplicate timestamp."
        )

    df.to_csv(cache_file, index=False)

    return df


# ============================================================
# BASELINE METRICS
# ============================================================

def calculate_baseline_report(clean_df):
    """Hitung MAE, RMSE, dan residual bias untuk setiap model lead."""
    report_rows = []

    for lead, group in clean_df.groupby("lead_hours"):
        mae = group["absolute_error"].mean()
        rmse = np.sqrt(group["squared_error"].mean())
        residual_bias = group["residual"].mean()

        report_rows.append(
            {
                "lead_hours": int(lead),
                "samples": len(group),
                "mae_ms": mae,
                "rmse_ms": rmse,
                "residual_bias_ms": residual_bias,
            }
        )

    return (
        pd.DataFrame(report_rows)
        .sort_values("lead_hours")
        .reset_index(drop=True)
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 90)
    print("TEST C/D - ECMWF WIND FORECAST CORRECTION DATASET - JAMBI")
    print("=" * 90)

    print(f"Period : {START_DATE} -> {END_DATE}")
    print(f"Leads  : {LEAD_HOURS}")
    print(
        f"Minimum critical completeness: "
        f"{MIN_COMPLETENESS_PERCENT:.2f}%"
    )
    print()

    # --------------------------------------------------------
    # 1. Fetch seluruh forecast runs
    # --------------------------------------------------------

    forecast_frames = []
    failed_runs = []

    total_runs = 0
    successful_runs = 0

    for date in iter_dates(START_DATE, END_DATE):
        for run_hour in RUN_HOURS:
            total_runs += 1

            run_time = datetime(
                date.year,
                date.month,
                date.day,
                run_hour,
                tzinfo=timezone.utc,
            )

            print("[FETCH]", run_time.isoformat())

            df = fetch_single_run(run_time)

            if df is not None and not df.empty:
                forecast_frames.append(df)
                successful_runs += 1
            else:
                failed_runs.append(run_time.isoformat())

            time.sleep(REQUEST_DELAY_SECONDS)

    if not forecast_frames:
        raise RuntimeError(
            "Tidak ada ECMWF forecast yang berhasil diambil."
        )

    # Suppress warning pandas concat untuk all-NA columns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        forecast = pd.concat(
            forecast_frames,
            ignore_index=True,
            sort=False,
        )

    ingestion_success_rate = successful_runs / total_runs * 100

    print("\n" + "=" * 90)
    print("FORECAST INGESTION")
    print("=" * 90)

    print(f"Runs success : {successful_runs}/{total_runs}")
    print(f"Success rate : {ingestion_success_rate:.2f}%")
    print(f"Forecast rows: {len(forecast)}")
    print(f"Failed runs  : {len(failed_runs)}")

    if failed_runs:
        print("\nFailed run list:")

        for failed_run in failed_runs:
            print("-", failed_run)

    # --------------------------------------------------------
    # 2. Fetch reference
    # --------------------------------------------------------

    # Buffer dua hari karena lead +24h dari akhir periode
    # dapat jatuh pada hari berikutnya.
    end_reference = (
        datetime.fromisoformat(END_DATE)
        + timedelta(days=2)
    )

    reference = fetch_reference(
        START_DATE,
        end_reference.date().isoformat(),
    )

    print(f"\nReference rows: {len(reference)}")

    # --------------------------------------------------------
    # 3. Pair forecast dan reference
    # --------------------------------------------------------

    merged = forecast.merge(
        reference,
        on="valid_time",
        how="inner",
        validate="many_to_one",
    )

    print("\n" + "=" * 90)
    print("TEST C - FORECAST / REFERENCE PAIRING")
    print("=" * 90)

    print(f"Forecast rows : {len(forecast)}")
    print(f"Reference rows: {len(reference)}")
    print(f"Merged rows   : {len(merged)}")

    if merged.empty:
        raise RuntimeError(
            "Forecast-reference pairing menghasilkan 0 row."
        )

    duplicate_pairs = merged.duplicated(
        subset=["run_time", "valid_time"]
    ).sum()

    print(f"\nDuplicate run-time pairs: {duplicate_pairs}")

    # --------------------------------------------------------
    # 4. Raw missing-value check
    # --------------------------------------------------------

    print("\n=== RAW MISSING VALUES ===")
    print(merged.isna().sum().to_string())

    # --------------------------------------------------------
    # 5. Critical validation
    # --------------------------------------------------------

    invalid_mask = merged[CRITICAL_COLUMNS].isna().any(axis=1)

    invalid_rows = merged[invalid_mask].copy()
    clean_merged = merged[~invalid_mask].copy()

    raw_row_count = len(merged)
    invalid_row_count = len(invalid_rows)
    valid_row_count = len(clean_merged)

    completeness = valid_row_count / raw_row_count * 100

    print("\n" + "=" * 90)
    print("CRITICAL DATA VALIDATION")
    print("=" * 90)

    print(f"Raw merged rows       : {raw_row_count}")
    print(f"Invalid critical rows : {invalid_row_count}")
    print(f"Valid critical rows   : {valid_row_count}")
    print(f"Critical completeness : {completeness:.2f}%")

    if invalid_row_count > 0:
        print("\n=== INVALID CRITICAL ROWS ===")

        invalid_preview_columns = [
            "run_time",
            "valid_time",
            "lead_hours",
            "forecast_wind_speed_10m",
            "reference_wind_speed_10m",
        ]

        print(
            invalid_rows[
                invalid_preview_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # 6. Residual target
    # --------------------------------------------------------

    clean_merged["residual"] = (
        clean_merged["reference_wind_speed_10m"]
        - clean_merged["forecast_wind_speed_10m"]
    )

    clean_merged["absolute_error"] = (
        clean_merged["residual"].abs()
    )

    clean_merged["squared_error"] = (
        clean_merged["residual"] ** 2
    )

    # --------------------------------------------------------
    # 7. Clean dataset quality
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CLEAN DATASET QUALITY")
    print("=" * 90)

    print(
        "Reference wind missing:",
        clean_merged["reference_wind_speed_10m"].isna().sum(),
    )

    print(
        "Forecast wind missing :",
        clean_merged["forecast_wind_speed_10m"].isna().sum(),
    )

    print(
        "Residual missing      :",
        clean_merged["residual"].isna().sum(),
    )

    residual_mean = clean_merged["residual"].mean()
    residual_std = clean_merged["residual"].std()

    print(f"\nResidual mean: {residual_mean:.4f} m/s")
    print(f"Residual std : {residual_std:.4f} m/s")

    # --------------------------------------------------------
    # 8. Preview
    # --------------------------------------------------------

    print("\n=== FIRST 15 CLEAN PAIRED ROWS ===")

    preview_columns = [
        "run_time",
        "valid_time",
        "lead_hours",
        "forecast_wind_speed_10m",
        "reference_wind_speed_10m",
        "residual",
    ]

    print(
        clean_merged[
            preview_columns
        ].head(15).to_string(index=False)
    )

    # --------------------------------------------------------
    # 9. Baseline by lead
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("TEST D - RAW ECMWF BASELINE BY MODEL LEAD")
    print("=" * 90)

    report = calculate_baseline_report(clean_merged)

    print(
        report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 10. Overall baseline
    # --------------------------------------------------------

    overall_mae = clean_merged["absolute_error"].mean()

    overall_rmse = np.sqrt(
        clean_merged["squared_error"].mean()
    )

    overall_bias = clean_merged["residual"].mean()

    print("\n=== OVERALL RAW BASELINE ===")
    print(f"MAE  : {overall_mae:.4f} m/s")
    print(f"RMSE : {overall_rmse:.4f} m/s")
    print(f"Bias : {overall_bias:.4f} m/s")

    # --------------------------------------------------------
    # 11. Data-quality per lead
    # --------------------------------------------------------

    print("\n=== VALID SAMPLE COUNT BY LEAD ===")

    raw_count_by_lead = (
        merged.groupby("lead_hours")
        .size()
        .rename("raw_samples")
    )

    valid_count_by_lead = (
        clean_merged.groupby("lead_hours")
        .size()
        .rename("valid_samples")
    )

    lead_quality = pd.concat(
        [raw_count_by_lead, valid_count_by_lead],
        axis=1,
    ).fillna(0)

    lead_quality["valid_percent"] = (
        lead_quality["valid_samples"]
        / lead_quality["raw_samples"]
        * 100
    )

    print(
        lead_quality
        .reset_index()
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # 12. Save outputs
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_merged_file = (
        PROCESSED_DIR
        / "wind_forecast_reference_raw.csv"
    )

    clean_dataset_file = (
        PROCESSED_DIR
        / "wind_forecast_reference_poc.csv"
    )

    invalid_file = (
        REPORT_DIR
        / "invalid_critical_rows.csv"
    )

    baseline_file = (
        REPORT_DIR
        / "baseline_by_lead.csv"
    )

    lead_quality_file = (
        REPORT_DIR
        / "data_quality_by_lead.csv"
    )

    failed_runs_file = (
        REPORT_DIR
        / "failed_runs.csv"
    )

    merged.to_csv(raw_merged_file, index=False)
    clean_merged.to_csv(clean_dataset_file, index=False)
    invalid_rows.to_csv(invalid_file, index=False)
    report.to_csv(baseline_file, index=False)

    lead_quality.reset_index().to_csv(
        lead_quality_file,
        index=False,
    )

    pd.DataFrame(
        {"failed_run": failed_runs}
    ).to_csv(
        failed_runs_file,
        index=False,
    )

    # --------------------------------------------------------
    # 13. PoC verdict
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("PoC CHECK")
    print("=" * 90)

    print(
        f"Ingestion success rate : "
        f"{ingestion_success_rate:.2f}%"
    )

    print(
        f"Critical completeness  : "
        f"{completeness:.2f}%"
    )

    print(
        f"Residual std           : "
        f"{residual_std:.4f} m/s"
    )

    print(
        f"Duplicate pairs        : "
        f"{duplicate_pairs}"
    )

    passed = (
        valid_row_count > 0
        and completeness >= MIN_COMPLETENESS_PERCENT
        and residual_std > 0
        and duplicate_pairs == 0
    )

    verdict = "PASSED" if passed else "FAILED"

    print(f"Test verdict           : {verdict}")

    print("\nSaved:")
    print("-", raw_merged_file)
    print("-", clean_dataset_file)
    print("-", invalid_file)
    print("-", baseline_file)
    print("-", lead_quality_file)
    print("-", failed_runs_file)

    print("\n" + "=" * 90)
    print(f"TEST C/D {verdict}")
    print("=" * 90)


if __name__ == "__main__":
    main()