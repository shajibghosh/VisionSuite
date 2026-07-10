#!/usr/bin/env python
"""Build a leaderboard + comparison plots across all runs of a task.

    python scripts/compare_runs.py --task detection
    python scripts/compare_runs.py --task classification --root runs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsuite.compare import compare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    choices=["detection", "classification", "segmentation"])
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="runs/_comparisons")
    args = ap.parse_args()
    print(json.dumps(compare(args.task, args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
