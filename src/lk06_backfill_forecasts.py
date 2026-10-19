
"""LK06 - Historical forecast backfill using the existing LK04/LK05 pipelines."""

import argparse
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

from ingest_data import ingest_run
from update_forecast_history import update_history


HISTORY_PATH = Path("data/versioned/forecast_history.csv")
VALID_RUN_HOURS = (0, 6, 12, 18)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill ECMWF IFS forecast runs for LK06."
    )

    parser.add_argument("--start-date", default="2026-10-01")
    parser.add_argument("--end-date", default="2026-10-14")
    parser.add_argument(
        "--run-hours",
        nargs="+",
        type=int,
        default=list(VALID_RUN_HOURS),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def generate_runs(start_date, end_date, run_hours):
    """Generate deterministic UTC forecast run timestamps."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("start-date must not exceed end-date.")

    if not run_hours or any(
        hour not in VALID_RUN_HOURS for hour in run_hours
    ):
        raise ValueError("Only run hours 0, 6, 12, 18 are allowed.")

    runs = []

    current = start

    while current <= end:
        for hour in sorted(set(run_hours)):
            run_time = datetime.combine(
                current,
                dt_time(hour=hour, tzinfo=timezone.utc),
            )
            runs.append(run_time)

        current += timedelta(days=1)

    return runs


def main():
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")

    runs = generate_runs(
        args.start_date,
        args.end_date,
        args.run_hours,
    )

    if args.limit is not None:
        runs = runs[:args.limit]

    print("=" * 65)
    print("LK06 - HISTORICAL FORECAST BACKFILL")
    print("=" * 65)

    print("Period:", args.start_date, "to", args.end_date)
    print("Run hours:", args.run_hours)
    print("Planned runs:", len(runs))
    print("Dry run:", args.dry_run)

    if args.dry_run:
        print("\n=== FORECAST RUN PLAN ===")

        for run_time in runs[:10]:
            print(run_time.isoformat())

        if len(runs) > 10:
            print(f"... and {len(runs) - 10} more runs")

        print("\nDRY RUN PASSED - No data modified.")
        return

    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f"DVC dataset not found: {HISTORY_PATH}"
        )

    completed = 0
    failed = []

    for index, run_time in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] {run_time.isoformat()}")

        try:
            # Reuse LK04 ingestion.
            snapshot_path = ingest_run(run_time)

            if snapshot_path is None:
                raise RuntimeError("Forecast unavailable.")

            # Reuse LK05 incremental dataset update.
            update_history(HISTORY_PATH, snapshot_path)

            completed += 1

        except Exception as exc:
            print(f"[ERROR] {run_time}: {exc}")
            failed.append(run_time.isoformat())

        if index < len(runs):
            time.sleep(args.delay)

    print("\n" + "=" * 65)
    print("BACKFILL SUMMARY")
    print("=" * 65)
    print("Planned runs:", len(runs))
    print("Completed:", completed)
    print("Failed:", len(failed))

    if failed:
        print("\nFailed forecast runs:")
        for item in failed:
            print("-", item)

        raise RuntimeError(
            "Backfill incomplete. Review failed forecast runs."
        )

    print("\nBACKFILL PASSED")


if __name__ == "__main__":
    main()
