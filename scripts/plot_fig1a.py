#!/usr/bin/env python3
"""Plot Figure 1(a) mean and standard error from four completed seeds."""
import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fig1a import ALL, SEEDS, database_path


ROOT = Path(__file__).resolve().parents[1]
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
]
EXPECTED_LABELS = list(range(1000, 10001, 1000))


def read_trials(db, strategy):
    trials = []
    for seed in SEEDS:
        status = db.execute(
            "SELECT status FROM runs WHERE strategy=? AND seed=?",
            (strategy, seed),
        ).fetchone()
        if status != ("complete",):
            raise RuntimeError(f"{strategy}, seed {seed}: run is not complete")
        rows = db.execute(
            "SELECT labels,test_accuracy FROM metrics "
            "WHERE strategy=? AND seed=? ORDER BY round",
            (strategy, seed),
        ).fetchall()
        if [labels for labels, _ in rows] != EXPECTED_LABELS:
            raise RuntimeError(f"{strategy}, seed {seed}: expected 10 label budgets")
        accuracy = np.asarray([value for _, value in rows], dtype=float)
        if not np.all(np.isfinite(accuracy)) or np.any((accuracy < 0) | (accuracy > 1)):
            raise RuntimeError(f"{strategy}, seed {seed}: invalid test accuracy")
        trials.append(np.maximum.accumulate(accuracy))
    return np.asarray(trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", "--output-dir", dest="output", type=Path,
                        help="directory for the generated plot")
    parser.add_argument("--input", "-i", "--database", "--db", dest="input", type=Path,
                        help="input Figure 1(a) SQLite database")
    args = parser.parse_args()
    output_dir = (args.output or ROOT / "results").resolve()
    path = (args.input or database_path(output_dir)).resolve()
    if not path.is_file():
        parser.error(f"experiment database not found: {path}")

    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        summaries = []
        for strategy in ALL:
            values = read_trials(db, strategy)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
            summaries.append((strategy, mean, sem))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for (strategy, mean, sem), color in zip(summaries, COLORS):
        ax.plot(EXPECTED_LABELS, mean, label=strategy.upper(), color=color, linewidth=2)
        ax.fill_between(EXPECTED_LABELS, mean - sem, mean + sem, color=color, alpha=0.18)
    ax.set(xlabel="Number of Labels", ylabel="Test Accuracy", ylim=(.960, .982))
    ax.grid(True, linestyle="--", alpha=.6)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure1a.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output} from {len(SEEDS)} seeds per strategy")


if __name__ == "__main__":
    main()
