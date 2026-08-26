from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("data/processed/wind_forecast_reference_poc.csv")
REPORT_DIR = Path("data/reports")

LOCAL_TIMEZONE = "Asia/Jakarta"

TEST_SIZE = 0.20
RANDOM_STATE = 42

LEAD_HOURS = [6, 12, 18, 24]

TARGET = "residual"

RAW_FORECAST = "forecast_wind_speed_10m"
REFERENCE = "reference_wind_speed_10m"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Membuat feature yang tersedia pada saat forecast dibuat.

    Timestamp disimpan dalam UTC, tetapi feature siklus harian
    menggunakan waktu lokal Asia/Jakarta.
    """
    df = df.copy()

    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)

    # --------------------------------------------------------
    # Wind direction adalah circular variable.
    #
    # 359 derajat dan 1 derajat sebenarnya sangat dekat,
    # sehingga tidak cocok dipakai sebagai angka linear biasa.
    # --------------------------------------------------------

    direction_rad = np.deg2rad(
        df["forecast_wind_direction_10m"]
    )

    df["wind_direction_sin"] = np.sin(direction_rad)
    df["wind_direction_cos"] = np.cos(direction_rad)

    # --------------------------------------------------------
    # Temporal feature berdasarkan waktu lokal Jambi (WIB).
    #
    # Storage tetap UTC; konversi hanya dilakukan di memory.
    # --------------------------------------------------------

    local_time = df["valid_time"].dt.tz_convert(LOCAL_TIMEZONE)

    local_hour = local_time.dt.hour
    day_of_year = local_time.dt.dayofyear

    df["hour_sin"] = np.sin(
        2 * np.pi * local_hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * local_hour / 24
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return df


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "forecast_wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "forecast_wind_gusts_10m",
    "forecast_temperature_2m",
    "forecast_relative_humidity_2m",
    "forecast_pressure_msl",
    "forecast_cloud_cover",
    "forecast_precipitation",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# ============================================================
# MODELS
# ============================================================

def get_models():
    """
    Tiga model sederhana untuk sanity test.

    Tujuannya bukan mencari model terbaik secara final,
    melainkan mengecek apakah residual memiliki learning signal.
    """

    linear_regression = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    hist_gradient_boosting = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "LinearRegression": linear_regression,
        "RandomForest": random_forest,
        "HistGradientBoosting": hist_gradient_boosting,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(reference, prediction):
    """
    Hitung MAE, RMSE, dan bias.

    Bias didefinisikan sebagai:
        reference - prediction

    Positif  -> prediction terlalu rendah.
    Negatif  -> prediction terlalu tinggi.
    """

    mae = mean_absolute_error(reference, prediction)

    rmse = np.sqrt(
        mean_squared_error(reference, prediction)
    )

    bias = np.mean(reference - prediction)

    return mae, rmse, bias


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(df):
    """
    Membagi data berdasarkan waktu.

    80% data terlama = training
    20% data terbaru = testing
    """

    df = df.sort_values("valid_time").reset_index(drop=True)

    split_index = int(
        len(df) * (1 - TEST_SIZE)
    )

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 95)
    print("TEST E - TEMPORAL SANITY RESIDUAL CORRECTION")
    print("=" * 95)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ada: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset : {DATA_FILE}")
    print(f"Rows    : {len(df)}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = engineer_features(df)

    print("\n=== FEATURE SET ===")

    for feature in FEATURE_COLUMNS:
        print("-", feature)

    print("\n=== FEATURE MISSING VALUES ===")

    print(
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nMissing feature akan ditangani "
        "menggunakan median imputation dari training data."
    )

    models = get_models()

    report_rows = []
    prediction_frames = []

    # ========================================================
    # PER-LEAD EXPERIMENT
    # ========================================================

    for lead in LEAD_HOURS:
        lead_df = df[
            df["lead_hours"] == lead
        ].copy()

        if lead_df.empty:
            print(
                f"\n[WARNING] Tidak ada data untuk lead +{lead}h."
            )
            continue

        train, test = temporal_split(lead_df)

        print("\n" + "=" * 95)
        print(f"MODEL LEAD +{lead} HOURS")
        print("=" * 95)

        print(f"Total samples : {len(lead_df)}")
        print(f"Train samples : {len(train)}")
        print(f"Test samples  : {len(test)}")

        print(
            "Train period  :",
            train["valid_time"].min(),
            "->",
            train["valid_time"].max(),
        )

        print(
            "Test period   :",
            test["valid_time"].min(),
            "->",
            test["valid_time"].max(),
        )

        X_train = train[FEATURE_COLUMNS]
        y_train = train[TARGET]

        X_test = test[FEATURE_COLUMNS]
        y_test = test[TARGET]

        reference_test = test[REFERENCE].to_numpy()
        raw_forecast_test = test[RAW_FORECAST].to_numpy()

        # ----------------------------------------------------
        # Raw ECMWF baseline pada test period
        # ----------------------------------------------------

        raw_mae, raw_rmse, raw_bias = calculate_metrics(
            reference_test,
            raw_forecast_test,
        )

        print("\n--- RAW ECMWF BASELINE ---")
        print(f"MAE  : {raw_mae:.4f} m/s")
        print(f"RMSE : {raw_rmse:.4f} m/s")
        print(f"Bias : {raw_bias:.4f} m/s")

        # ----------------------------------------------------
        # ML models
        # ----------------------------------------------------

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            predicted_residual = model.predict(X_test)

            corrected_forecast = (
                raw_forecast_test
                + predicted_residual
            )

            # Wind speed secara fisik tidak boleh negatif.
            corrected_forecast = np.clip(
                corrected_forecast,
                a_min=0,
                a_max=None,
            )

            corrected_mae, corrected_rmse, corrected_bias = (
                calculate_metrics(
                    reference_test,
                    corrected_forecast,
                )
            )

            mae_improvement = (
                (raw_mae - corrected_mae)
                / raw_mae
                * 100
            )

            rmse_improvement = (
                (raw_rmse - corrected_rmse)
                / raw_rmse
                * 100
            )

            print(f"\n--- {model_name} ---")
            print(
                f"Corrected MAE  : "
                f"{corrected_mae:.4f} m/s"
            )
            print(
                f"Corrected RMSE : "
                f"{corrected_rmse:.4f} m/s"
            )
            print(
                f"Corrected Bias : "
                f"{corrected_bias:.4f} m/s"
            )
            print(
                f"MAE improvement  : "
                f"{mae_improvement:+.2f}%"
            )
            print(
                f"RMSE improvement : "
                f"{rmse_improvement:+.2f}%"
            )

            report_rows.append(
                {
                    "lead_hours": lead,
                    "model": model_name,
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "raw_mae_ms": raw_mae,
                    "corrected_mae_ms": corrected_mae,
                    "mae_improvement_percent": mae_improvement,
                    "raw_rmse_ms": raw_rmse,
                    "corrected_rmse_ms": corrected_rmse,
                    "rmse_improvement_percent": rmse_improvement,
                    "raw_bias_ms": raw_bias,
                    "corrected_bias_ms": corrected_bias,
                }
            )

            prediction_df = pd.DataFrame(
                {
                    "run_time": test["run_time"].values,
                    "valid_time": test["valid_time"].values,
                    "lead_hours": lead,
                    "model": model_name,
                    "raw_forecast_wind_speed_10m":
                        raw_forecast_test,
                    "reference_wind_speed_10m":
                        reference_test,
                    "predicted_residual":
                        predicted_residual,
                    "corrected_wind_speed_10m":
                        corrected_forecast,
                }
            )

            prediction_frames.append(prediction_df)

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    report = pd.DataFrame(report_rows)

    if report.empty:
        raise RuntimeError(
            "Tidak ada hasil model yang dapat dievaluasi."
        )

    report = report.sort_values(
        ["lead_hours", "corrected_mae_ms"]
    ).reset_index(drop=True)

    print("\n" + "=" * 95)
    print("TEST E - MODEL COMPARISON")
    print("=" * 95)

    summary_columns = [
        "lead_hours",
        "model",
        "raw_mae_ms",
        "corrected_mae_ms",
        "mae_improvement_percent",
        "raw_rmse_ms",
        "corrected_rmse_ms",
        "corrected_bias_ms",
    ]

    print(
        report[summary_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # BEST MODEL PER LEAD
    # ========================================================

    best_by_lead = (
        report
        .sort_values("corrected_mae_ms")
        .groupby("lead_hours", as_index=False)
        .first()
        .sort_values("lead_hours")
    )

    print("\n=== BEST MODEL PER LEAD ===")

    best_columns = [
        "lead_hours",
        "model",
        "raw_mae_ms",
        "corrected_mae_ms",
        "mae_improvement_percent",
    ]

    print(
        best_by_lead[best_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SANITY VERDICT
    # ========================================================

    improved_leads = (
        best_by_lead[
            "mae_improvement_percent"
        ] > 0
    ).sum()

    total_leads = len(best_by_lead)

    mean_best_improvement = (
        best_by_lead[
            "mae_improvement_percent"
        ].mean()
    )

    print("\n" + "=" * 95)
    print("SANITY CHECK")
    print("=" * 95)

    print(
        f"Leads with MAE improvement : "
        f"{improved_leads}/{total_leads}"
    )

    print(
        f"Mean best MAE improvement  : "
        f"{mean_best_improvement:.2f}%"
    )

    # Sanity rule:
    # learning signal dianggap cukup meyakinkan
    # jika mayoritas lead menunjukkan improvement.
    if improved_leads >= 3 and mean_best_improvement > 0:
        verdict = "PASSED - LEARNING SIGNAL DETECTED"
    elif improved_leads > 0:
        verdict = "PARTIAL - WEAK/MIXED LEARNING SIGNAL"
    else:
        verdict = "FAILED - NO LEARNING SIGNAL DETECTED"

    print(f"Test verdict               : {verdict}")

    # ========================================================
    # SAVE
    # ========================================================

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_file = (
        REPORT_DIR
        / "test_e_model_comparison.csv"
    )

    best_model_file = (
        REPORT_DIR
        / "test_e_best_model_by_lead.csv"
    )

    prediction_file = (
        REPORT_DIR
        / "test_e_predictions.csv"
    )

    report.to_csv(
        comparison_file,
        index=False,
    )

    best_by_lead.to_csv(
        best_model_file,
        index=False,
    )

    if prediction_frames:
        predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        predictions.to_csv(
            prediction_file,
            index=False,
        )

    print("\nSaved:")
    print("-", comparison_file)
    print("-", best_model_file)
    print("-", prediction_file)

    print("\n" + "=" * 95)
    print("TEST E COMPLETED")
    print("=" * 95)


if __name__ == "__main__":
    main()