"""Unit tests for dataset integrity and temporal splitting."""

import pandas as pd
import pytest

import train


def sample_dataset():
    records = []
    start = pd.Timestamp("2026-10-01T00:00:00Z")

    for i in range(20):
        lead = (6, 12, 18, 24)[i % 4]
        valid_time = start + pd.Timedelta(hours=6 * i)
        run_time = valid_time - pd.Timedelta(hours=lead)

        row = {feature: 1.0 for feature in train.FEATURES}

        row.update({
            "run_time": run_time.isoformat(),
            "valid_time": valid_time.isoformat(),
            "lead_hours": lead,
            "forecast_wind_speed_10m": 2.0,
            "reference_wind_speed_10m": 2.5,
            "residual": 0.5,
        })

        records.append(row)

    return pd.DataFrame(records)


def load_temp_dataset(df, tmp_path, monkeypatch):
    path = tmp_path / "training.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(train, "DATA_PATH", path)
    return train.load_and_validate()


def test_valid_dataset_and_temporal_split(
    tmp_path, monkeypatch
):
    df, sha = load_temp_dataset(
        sample_dataset(), tmp_path, monkeypatch
    )

    training, validation, cutoff = (
        train.chronological_split(df)
    )

    assert len(sha) == 64
    assert len(df) == 20
    assert len(training) == 16
    assert len(validation) == 4

    assert set(training["valid_time"]).isdisjoint(
        set(validation["valid_time"])
    )

    assert training["valid_time"].max() < cutoff
    assert validation["valid_time"].min() >= cutoff


def test_reversed_residual_is_rejected(
    tmp_path, monkeypatch
):
    df = sample_dataset()
    df["residual"] = -0.5

    with pytest.raises(ValueError, match="Residual"):
        load_temp_dataset(df, tmp_path, monkeypatch)


def test_duplicate_forecast_key_is_rejected(
    tmp_path, monkeypatch
):
    df = sample_dataset()
    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplikasi"):
        load_temp_dataset(df, tmp_path, monkeypatch)


def test_missing_critical_value_is_rejected(
    tmp_path, monkeypatch
):
    df = sample_dataset()
    df.loc[0, "reference_wind_speed_10m"] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        load_temp_dataset(df, tmp_path, monkeypatch)
