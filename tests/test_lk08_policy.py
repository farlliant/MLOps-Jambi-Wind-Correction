import pytest

from src.lk08_policy import evaluate_metrics


BASE = {
    "mae": 0.6151984053126739,
    "rmse": 0.8084273468366523,
    "bias": -0.5551462058080778,
    "raw_mae": 0.6182608695652174,
    "reference_mae": 0.6151984053126739,
    "reference_rmse": 0.8084273468366523,
    "reference_abs_bias": 0.5551462058080778,
    "unique_times": 13,
    "lead_hours": [6, 12, 18, 24],
}


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"mae": 0.67}, False),
        ({"rmse": 0.90}, False),
        ({"bias": -0.80}, False),
        ({"unique_times": 12}, False),
        ({"lead_hours": [6, 12, 18]}, False),
        ({"mae": float("nan")}, False),
        ({"mae": float("inf")}, False),
        ({"raw_mae": 0.60}, False),
        ({"reference_mae": 0.61}, False),
    ],
)
def test_validation_gate(changes, expected):
    values = {**BASE, **changes}
    checks = evaluate_metrics(**values)

    assert all(checks.values()) is expected


def test_rejects_per_horizon_regression():
    checks = evaluate_metrics(
        **BASE,
        per_lead_mae={
            6: 0.5,
            12: 0.5,
            18: 0.7,
            24: 0.5,
        },
        reference_per_lead_mae={
            6: 0.6,
            12: 0.6,
            18: 0.6,
            24: 0.6,
        },
    )

    assert checks["mae_per_horizon_vs_reference"] is False


def test_accepts_per_horizon_nonregression():
    checks = evaluate_metrics(
        **BASE,
        per_lead_mae={
            6: 0.5,
            12: 0.5,
            18: 0.5,
            24: 0.5,
        },
        reference_per_lead_mae={
            6: 0.6,
            12: 0.6,
            18: 0.6,
            24: 0.6,
        },
    )

    assert checks["mae_per_horizon_vs_reference"] is True
