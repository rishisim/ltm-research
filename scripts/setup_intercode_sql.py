#!/usr/bin/env python3
"""
Setup script for InterCode SQL experiments.

Creates train/dev/test splits from the InterCode Spider dataset.
The Spider dataset has ~1,034 questions across ~20 databases.

Split strategy:
  - train: Used for KB construction (first N tasks)
  - dev: Primary evaluation split
  - test: Held-out evaluation split

We split by database to avoid data leakage: tasks from the same database
should not appear in both train and eval splits.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

def load_intercode_data(data_path: str):
    """Load the InterCode SQL dataset (CSV or JSON format)."""
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        import pandas as pd
        df = pd.read_csv(data_path)
        return df.to_dict("records")


def split_by_database(data, train_ratio=0.4, dev_ratio=0.3, seed=42):
    """
    Split tasks by database to prevent data leakage.

    Args:
        data: List of InterCode SQL task records.
        train_ratio: Fraction of databases for training.
        dev_ratio: Fraction of databases for dev.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'dev', 'test' splits, each containing task indices.
    """
    # Group task indices by database
    db_to_indices = defaultdict(list)
    for idx, record in enumerate(data):
        db = record.get("db", f"unknown_{idx}")
        db_to_indices[db].append(idx)

    databases = sorted(db_to_indices.keys())
    random.seed(seed)
    random.shuffle(databases)

    n_dbs = len(databases)
    n_train = max(1, int(n_dbs * train_ratio))
    n_dev = max(1, int(n_dbs * dev_ratio))

    train_dbs = databases[:n_train]
    dev_dbs = databases[n_train : n_train + n_dev]
    test_dbs = databases[n_train + n_dev :]

    splits = {
        "train": {
            "databases": train_dbs,
            "task_indices": [],
        },
        "dev": {
            "databases": dev_dbs,
            "task_indices": [],
        },
        "test": {
            "databases": test_dbs,
            "task_indices": [],
        },
    }

    for split_name, split_info in splits.items():
        for db in split_info["databases"]:
            split_info["task_indices"].extend(db_to_indices[db])
        split_info["task_indices"].sort()

    return splits


def main():
    parser = argparse.ArgumentParser(description="Setup InterCode SQL data splits")
    parser.add_argument(
        "--intercode-path",
        type=str,
        default=None,
        help="Path to cloned InterCode repo (default: tries pip package location)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to InterCode SQL CSV data file. If not set, uses the pip package's built-in data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/intercode_sql",
        help="Output directory for split manifests",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.4,
        help="Fraction of databases for training (default: 0.4)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.3,
        help="Fraction of databases for dev (default: 0.3)",
    )
    parser.add_argument(
        "--max-dev",
        type=int,
        default=200,
        help="Maximum number of dev tasks (default: 200)",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=200,
        help="Maximum number of test tasks (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the InterCode data file
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        # Use the pip package's built-in data
        try:
            from intercode.assets import sql_test_data
            data_path = Path(sql_test_data)
        except ImportError:
            print("Error: InterCode not installed. Install with: pip install intercode-bench")
            sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    data = load_intercode_data(str(data_path))
    print(f"Total tasks: {len(data)}")

    # Count databases
    dbs = set(r.get("db", "") for r in data)
    print(f"Total databases: {len(dbs)}")

    # Create splits
    splits = split_by_database(data, args.train_ratio, args.dev_ratio, args.seed)

    # Cap dev/test sizes
    random.seed(args.seed)
    for split_name, max_size in [("dev", args.max_dev), ("test", args.max_test)]:
        indices = splits[split_name]["task_indices"]
        if len(indices) > max_size:
            random.shuffle(indices)
            splits[split_name]["task_indices"] = sorted(indices[:max_size])

    # Save manifests
    for split_name, split_info in splits.items():
        manifest = {
            "split": split_name,
            "num_tasks": len(split_info["task_indices"]),
            "num_databases": len(split_info["databases"]),
            "databases": split_info["databases"],
            "task_indices": split_info["task_indices"],
            "data_path": str(data_path),
        }
        manifest_path = output_dir / f"{split_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(
            f"  {split_name}: {manifest['num_tasks']} tasks across "
            f"{manifest['num_databases']} databases -> {manifest_path}"
        )

    # Save a combined metadata file
    metadata = {
        "source_data": str(data_path),
        "total_tasks": len(data),
        "total_databases": len(dbs),
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "splits": {
            name: {
                "num_tasks": len(info["task_indices"]),
                "num_databases": len(info["databases"]),
            }
            for name, info in splits.items()
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSetup complete. Manifests saved to: {output_dir}")


if __name__ == "__main__":
    main()
