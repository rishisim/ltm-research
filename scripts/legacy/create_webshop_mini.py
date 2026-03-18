#!/usr/bin/env python3
"""
Create WebShop Mini Dataset

Samples task IDs from WebShop's official instruction splits to create
a reproducible mini dataset for evaluation.

Output: webshop_mini/{train,dev,test}.json — each is a list of integer task IDs.

Usage:
    python scripts/create_webshop_mini.py [--num-tasks 200] [--seed 42]
"""

import argparse
import json
import os
import random
import sys


def load_webshop_goals(webshop_path: str):
    """
    Load WebShop goal/instruction data and return task indices per split.

    WebShop stores its goals in web_agent_site/data/ as JSON files.
    The official splits are:
      - train: indices 0..10586 (10,587 tasks)
      - dev:   indices 10587..11586 (1,000 tasks)
      - test:  indices 11587..12086 (500 tasks)

    These index ranges are based on the official WebShop code.
    """
    # Try to load the goals file to get the total count
    goals_file = os.path.join(webshop_path, "web_agent_site", "data", "goal_items.json")
    items_human_file = os.path.join(webshop_path, "web_agent_site", "data", "items_human_ins.json")

    total_goals = None

    for gf in [goals_file, items_human_file]:
        if os.path.exists(gf):
            try:
                with open(gf, "r") as f:
                    goals = json.load(f)
                total_goals = len(goals)
                print(f"Loaded {total_goals} goals from {gf}")
                break
            except Exception as e:
                print(f"Warning: Could not load {gf}: {e}")

    if total_goals is None:
        # Use default counts from the WebShop paper
        print("Warning: Could not load goals file. Using default split sizes from the paper.")
        total_goals = 12087

    # Official split boundaries
    train_end = 10587
    dev_end = 11587
    test_end = min(total_goals, 12087)

    splits = {
        "train": list(range(0, min(train_end, total_goals))),
        "dev": list(range(train_end, min(dev_end, total_goals))),
        "test": list(range(dev_end, min(test_end, total_goals))),
    }

    for split_name, indices in splits.items():
        print(f"  {split_name}: {len(indices)} tasks (indices {indices[0]}..{indices[-1]})")

    return splits


def create_mini_dataset(
    webshop_path: str,
    output_dir: str,
    num_tasks: int = 200,
    seed: int = 42,
):
    """Sample num_tasks from each split and save as JSON manifests."""

    print(f"=== Creating WebShop Mini Dataset ===")
    print(f"WebShop path: {webshop_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Tasks/split:  {num_tasks}")
    print(f"Random seed:  {seed}")
    print()

    splits = load_webshop_goals(webshop_path)

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(seed)

    summary = {}
    for split_name, all_indices in splits.items():
        sample_size = min(num_tasks, len(all_indices))
        sampled = sorted(rng.sample(all_indices, sample_size))

        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w") as f:
            json.dump(
                {
                    "split": split_name,
                    "num_tasks": len(sampled),
                    "seed": seed,
                    "task_ids": sampled,
                },
                f,
                indent=2,
            )

        summary[split_name] = {
            "count": len(sampled),
            "file": output_path,
            "id_range": f"{sampled[0]}..{sampled[-1]}" if sampled else "empty",
        }
        print(f"  {split_name}: sampled {len(sampled)} tasks → {output_path}")

    # Save a combined summary
    summary_path = os.path.join(output_dir, "manifest.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "description": "WebShop Mini dataset manifest",
                "num_tasks_per_split": num_tasks,
                "seed": seed,
                "splits": summary,
            },
            f,
            indent=2,
        )

    print(f"\nDone! WebShop Mini dataset created at: {output_dir}")
    print(f"Manifest: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create WebShop Mini dataset by sampling task IDs"
    )
    parser.add_argument(
        "--webshop-path",
        type=str,
        default=os.getenv("WEBSHOP_PATH", "./webshop"),
        help="Path to cloned WebShop repo (default: $WEBSHOP_PATH or ./webshop)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="webshop_mini",
        help="Output directory for mini dataset (default: webshop_mini)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=200,
        help="Number of tasks per split (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    create_mini_dataset(
        webshop_path=args.webshop_path,
        output_dir=args.output_dir,
        num_tasks=args.num_tasks,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
