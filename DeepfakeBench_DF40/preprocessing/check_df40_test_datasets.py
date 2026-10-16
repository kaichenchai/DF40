"""Validate DF40 test JSON paths against the local dataset directory.

Run from the DeepfakeBench_DF40 directory:

    python preprocessing/check_df40_test_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METHODS = (
    "blendface",
    "e4s",
    "facedancer",
    "fsgan",
    "inswap",
    "mobileswap",
    "simswap",
    "uniface",
)
DOMAINS = ("ff", "cdf")


def frame_records(value: object) -> Iterable[tuple[str, list[str]]]:
    if isinstance(value, dict):
        frames = value.get("frames")
        if isinstance(frames, list):
            yield str(value.get("label", "<unknown>")), [
                frame for frame in frames if isinstance(frame, str)
            ]
            return
        for child in value.values():
            yield from frame_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from frame_records(child)


def load_test_records(json_path: Path, dataset_key: str) -> list[tuple[str, list[str]]]:
    with json_path.open(encoding="utf-8") as stream:
        content = json.load(stream)

    dataset = content.get(dataset_key)
    if not isinstance(dataset, dict):
        raise ValueError(f"{json_path} does not contain top-level key {dataset_key!r}")

    records: list[tuple[str, list[str]]] = []
    for label, data in dataset.items():
        if not isinstance(data, dict):
            raise ValueError(f"{json_path}: invalid data for label {label!r}")
        records.extend(
            (record_label, frames)
            for record_label, frames in frame_records(data.get("test", {}))
        )
    return records


def validate_dataset(repo_root: Path, method: str, domain: str) -> dict[str, object]:
    dataset_key = f"{method}_{domain}"
    json_path = repo_root / "preprocessing" / "dataset_json" / f"{dataset_key}.json"
    result: dict[str, object] = {
        "dataset": dataset_key,
        "json": json_path,
        "records": 0,
        "frames": 0,
        "missing": [],
        "outside_expected_root": [],
    }

    if not json_path.is_file():
        result["missing"] = [f"JSON file not found: {json_path}"]
        return result

    try:
        records = load_test_records(json_path, dataset_key)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result["missing"] = [str(error)]
        return result

    expected_fake_root = (
        f"deepfakes_detection_datasets/DF40/{method}/{domain}/frames/"
    )
    missing: list[str] = []
    outside_root: list[str] = []
    frame_count = 0
    for label, frames in records:
        result["records"] = int(result["records"]) + 1
        frame_count += len(frames)
        for frame in frames:
            frame_path = repo_root / frame
            if not frame_path.is_file():
                missing.append(frame)
            if "Fake" in label and not frame.startswith(expected_fake_root):
                outside_root.append(frame)

    result["frames"] = frame_count
    result["missing"] = missing
    result["outside_expected_root"] = outside_root
    return result


def print_result(result: dict[str, object], max_paths: int) -> bool:
    dataset = result["dataset"]
    missing = result["missing"]
    outside_root = result["outside_expected_root"]
    passed = not missing and not outside_root and result["records"] > 0
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] {dataset}: "
        f"{result['records']} videos, {result['frames']} test frames"
    )

    if missing:
        print(f"  missing paths: {len(missing)}")
        for path in missing[:max_paths]:
            print(f"    {path}")
    if outside_root:
        print(f"  fake paths outside expected root: {len(outside_root)}")
        for path in outside_root[:max_paths]:
            print(f"    {path}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="DeepfakeBench_DF40 directory (default: detected from this script)",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=5,
        help="Maximum example paths to print per failure (default: 5)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results = [
        validate_dataset(repo_root, method, domain)
        for method in METHODS
        for domain in DOMAINS
    ]

    passed = all(print_result(result, args.max_paths) for result in results)
    missing_roots = []
    for method in METHODS:
        for domain in DOMAINS:
            frame_root = (
                repo_root
                / "deepfakes_detection_datasets"
                / "DF40"
                / method
                / domain
                / "frames"
            )
            archive = frame_root.with_suffix(".zip")
            if not frame_root.is_dir() and archive.is_file():
                missing_roots.append(f"{method}_{domain}: {archive}")

    if missing_roots:
        passed = False
        print("\nArchive-only frame roots:")
        for root in missing_roots:
            print(f"  {root}")

    summary = defaultdict(int)
    for result in results:
        summary["passed" if not result["missing"] and not result["outside_expected_root"] else "failed"] += 1
    print(f"\nSummary: {summary['passed']}/16 dataset configs passed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
