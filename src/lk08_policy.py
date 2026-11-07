"""Pure validation rules for LK08 model candidates."""

from math import isfinite


def evaluate_metrics(
    *,
    mae,
    rmse,
    bias,
    raw_mae,
    reference_mae,
    reference_rmse,
    reference_abs_bias,
    unique_times,
    lead_hours,
    min_unique_times=13,
    required_leads=(6, 12, 18, 24),
    epsilon=1e-6,
    per_lead_mae=None,
    reference_per_lead_mae=None,
):
    values = (
        mae,
        rmse,
        bias,
        raw_mae,
        reference_mae,
        reference_rmse,
        reference_abs_bias,
    )

    finite = all(
        isfinite(float(value))
        for value in values
    )

    checks = {
        "finite_metrics": finite,
        "enough_validation_times": unique_times >= min_unique_times,
        "all_horizons_present": set(required_leads).issubset(
            set(lead_hours)
        ),
        "mae_vs_raw_ecmwf": False,
        "mae_vs_reference": False,
        "rmse_vs_reference": False,
        "absolute_bias_vs_reference": False,
    }

    if finite:
        checks.update({
            "mae_vs_raw_ecmwf":
                mae <= raw_mae + epsilon,

            "mae_vs_reference":
                mae <= reference_mae + epsilon,

            "rmse_vs_reference":
                rmse <= reference_rmse + epsilon,

            "absolute_bias_vs_reference":
                abs(bias) <= reference_abs_bias + epsilon,
        })

    # Pemeriksaan per horizon hanya dapat dilakukan
    # ketika metrik pembanding tersedia.
    if reference_per_lead_mae is not None:
        checks["mae_per_horizon_vs_reference"] = (
            per_lead_mae is not None
            and all(
                lead in per_lead_mae
                and lead in reference_per_lead_mae
                and isfinite(float(per_lead_mae[lead]))
                and per_lead_mae[lead]
                    <= reference_per_lead_mae[lead] + epsilon
                for lead in required_leads
            )
        )

    return checks
