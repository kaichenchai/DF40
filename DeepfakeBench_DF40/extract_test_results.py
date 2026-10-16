#!/usr/bin/env python3
"""Extract dataset metrics from a DeepfakeBench test log into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DATASET_RE = re.compile(r"^\S+:\s+dataset:\s+(?P<dataset>\S+)\s*$")
METRIC_RE = re.compile(
    r"^\S+:\s+(?P<metric>acc|auc|eer|ap|video_auc):\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
METRICS = ("acc", "auc", "eer", "ap", "video_auc")
CSV_FIELDS = ("dataset", "domain", *METRICS)


def extract_rows(log_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            dataset_match = DATASET_RE.match(line.rstrip("\n"))
            if dataset_match:
                if current is not None:
                    rows.append(finalize_row(current, log_path, line_number))
                current = {"dataset": dataset_match.group("dataset")}
                continue

            metric_match = METRIC_RE.match(line.rstrip("\n"))
            if metric_match and current is not None:
                current[metric_match.group("metric")] = float(
                    metric_match.group("value")
                )

    if current is not None:
        rows.append(finalize_row(current, log_path, line_number + 1))

    if not rows:
        raise ValueError(f"No dataset result blocks found in {log_path}")
    return rows


def finalize_row(
    row: dict[str, object], log_path: Path, line_number: int
) -> dict[str, object]:
    missing = [metric for metric in METRICS if metric not in row]
    if missing:
        raise ValueError(
            f"Incomplete result for {row['dataset']!r} near "
            f"{log_path}:{line_number}; missing: {', '.join(missing)}"
        )

    dataset = str(row["dataset"])
    domain = dataset.rsplit("_", 1)[-1].upper()
    if domain not in {"FF", "CDF"}:
        raise ValueError(
            f"Dataset {dataset!r} does not end in '_ff' or '_cdf'"
        )

    return {
        "dataset": dataset,
        "domain": domain,
        **{metric: row[metric] for metric in METRICS},
    }


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract scalar test metrics from a DeepfakeBench log."
    )
    parser.add_argument("log", type=Path, help="Path to the test log")
    parser.add_argument("output_csv", type=Path, help="Path for the output CSV")
    args = parser.parse_args()

    if not args.log.is_file():
        parser.error(f"log file does not exist: {args.log}")

    rows = extract_rows(args.log)
    write_csv(rows, args.output_csv)
    print(f"Extracted {len(rows)} result rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
