#!/usr/bin/env python3
"""Plot Figure 1(a) from the local SQLite experiment database."""

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALL_STRATEGIES = [
    "random", "confidence", "entropy", "margin",
    "coreset", "galaxy", "badge", "bait",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", nargs="+", choices=ALL_STRATEGIES,
        default=ALL_STRATEGIES, help="strategies to include",
    )
    parser.add_argument(
        "--min-trials", type=int, default=1,
        help="minimum completed trials required per strategy (default: 1)",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "results/figure1a.png"),
        help="output image path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    database_path = ROOT / "results/experiments.sqlite"
    if not database_path.exists():
        raise RuntimeError("no local experiment database")

    database = sqlite3.connect(str(database_path))
    completed = set(database.execute(
        "SELECT strategy, seed FROM runs WHERE status='complete'"
    ))
    figure, axis = plt.subplots(figsize=(7.2, 5))

    for strategy in args.strategies:
        seeds = sorted(seed for name, seed in completed if name == strategy)
        if len(seeds) < args.min_trials:
            database.close()
            raise RuntimeError(
                "{}: found {}/{} required complete trials".format(
                    strategy, len(seeds), args.min_trials
                )
            )

        trials = []
        labels = None
        for seed in seeds:
            rows = database.execute(
                "SELECT labels, test_accuracy FROM metrics "
                "WHERE strategy=? AND seed=? ORDER BY round",
                (strategy, seed),
            ).fetchall()
            labels = np.array([row[0] for row in rows])
            accuracies = np.array([row[1] for row in rows])
            trials.append(np.maximum.accumulate(accuracies))

        values = np.array(trials)
        mean = values.mean(axis=0)
        axis.plot(labels, mean, label=strategy.upper())
        if len(trials) > 1:
            standard_error = values.std(axis=0, ddof=1) / np.sqrt(len(trials))
            axis.fill_between(
                labels, mean - standard_error, mean + standard_error, alpha=0.22
            )

    database.close()
    axis.set(
        xlabel="Number of Labels", ylabel="Test Accuracy", ylim=(0.960, 0.982)
    )
    axis.grid(True, linestyle="--", alpha=0.6)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=220)
    plt.close(figure)
    print("saved {}".format(output_path))


if __name__ == "__main__":
    main()
