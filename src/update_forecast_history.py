"""Incrementally update the DVC-tracked ECMWF forecast history."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

DEFAULT_HISTORY = Path("data/versioned/forecast_history.csv")
KEY_COLUMNS = ("run_time", "valid_time")


def read_dataset(path: Path):
    """Read a CSV dataset and return its columns and records."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)

    if not columns or not rows:
        raise ValueError(f"Dataset is empty: {path}")

    if not all(column in columns for column in KEY_COLUMNS):
        raise ValueError(f"Missing key columns in: {path}")

    return columns, rows


def build_index(rows, label):
    """Validate unique natural keys and index records."""
    index = {}

    for row in rows:
        key = tuple(row[column] for column in KEY_COLUMNS)

        if not all(key):
            raise ValueError(f"Empty forecast key in {label}")

        if key in index:
            raise ValueError(f"Duplicate forecast key in {label}: {key}")

        index[key] = row

    return index


def update_history(history_path: Path, input_path: Path):
    """Append unseen forecast pairs without duplicating existing data."""
    columns, history_rows = read_dataset(history_path)
    input_columns, input_rows = read_dataset(input_path)

    if columns != input_columns:
        raise ValueError("Input columns do not match history columns")

    history_index = build_index(history_rows, "history")
    input_index = build_index(input_rows, "input")

    input_runs = {row["run_time"] for row in input_rows}

    if len(input_runs) != 1:
        raise ValueError("Input snapshot must contain exactly one run")

    new_rows = []

    for key, row in input_index.items():
        if key not in history_index:
            new_rows.append(row)
        elif row != history_index[key]:
            raise ValueError(f"Conflicting data for existing key: {key}")

    print(f"[INPUT] History  : {history_path}")
    print(f"[INPUT] Snapshot : {input_path}")
    print(f"[INFO] Run time  : {next(iter(input_runs))}")
    print(f"[INFO] V1 rows   : {len(history_rows)}")
    print(f"[INFO] New rows  : {len(new_rows)}")

    if not new_rows:
        print("[SKIP] No new forecast pairs. History unchanged.")
        return

    # Write to a temporary file to avoid partially written datasets.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=history_path.parent,
            prefix=".forecast_history_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)

            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(history_rows + new_rows)

        os.replace(temp_path, history_path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"[PASS] V2 rows   : {len(history_rows) + len(new_rows)}")
    print(f"[SAVE] {history_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Append new ECMWF forecast runs to DVC history."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
    )

    args = parser.parse_args()
    update_history(args.history, args.input)


if __name__ == "__main__":
    main()
