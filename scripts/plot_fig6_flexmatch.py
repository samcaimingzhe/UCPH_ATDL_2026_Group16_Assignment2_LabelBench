#!/usr/bin/env python3
"""Plot Figure 6(c) from the FlexMatch SQLite results."""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALL = ["random", "confidence", "entropy", "margin", "coreset", "galaxy", "badge", "bait"]
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive",
]
DEFAULT_SEEDS = np.linspace(1234, 9999999, num=3, dtype=int).tolist()
BATCH = 1000
ROUNDS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "fig6" / "flexmatch")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--strategies", nargs="+", choices=ALL, default=ALL)
    parser.add_argument("--smoothing", choices=["max", "none"], default="none",
                        help="Keep raw accuracies by default; max applies optional monotone post-processing.")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    path = (args.database or output / "figure6_flexmatch.sqlite").resolve()
    if not path.is_file():
        raise SystemExit(f"database not found: {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_expected = list(range(BATCH, BATCH * ROUNDS + 1, BATCH))
    summaries = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if metadata.get("panel") != "6c" or metadata.get("trainer") != "flexmatch":
            raise RuntimeError("database is not a Figure 6(c) FlexMatch database")
        if metadata.get("rounds") != str(ROUNDS):
            raise RuntimeError("plotting requires the full 10-round Figure 6 run; smoke-test databases are not final results")

        for strategy in args.strategies:
            trials = []
            for seed in args.seeds:
                status = db.execute(
                    "SELECT status FROM runs WHERE strategy=? AND seed=?",
                    (strategy, seed),
                ).fetchone()
                if status != ("complete",):
                    raise RuntimeError(f"{strategy}, seed {seed}: run is not complete")
                rows = db.execute(
                    "SELECT labels,test_accuracy FROM metrics WHERE strategy=? AND seed=? ORDER BY round",
                    (strategy, seed),
                ).fetchall()
                if [row[0] for row in rows] != labels_expected:
                    raise RuntimeError(f"{strategy}, seed {seed}: incomplete label-budget sequence")
                trials.append([row[1] for row in rows])

            values = np.asarray(trials, dtype=float)
            if args.smoothing == "max":
                values = np.maximum.accumulate(values, axis=1)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(args.seeds))
            summaries.append((strategy, np.asarray(labels_expected), mean, sem))

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    csv_rows = []
    for strategy, labels, mean, sem in summaries:
        color = COLORS[ALL.index(strategy)]
        ax.plot(labels, mean, label=strategy.upper(), color=color, linewidth=2)
        ax.fill_between(labels, mean - sem, mean + sem, color=color, alpha=.25)
        for n, m, e in zip(labels, mean, sem):
            csv_rows.append([strategy, int(n), float(m), float(e), len(args.seeds), args.smoothing])

    ax.set_xlabel("Number of Labels")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.960, 0.980)
    ax.grid(True, linestyle="--", alpha=.6)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    png = output / "figure6c_flexmatch.png"
    fig.savefig(png, dpi=250)
    plt.close(fig)

    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "strategy", "labels", "mean_test_accuracy", "standard_error", "trials", "smoothing"
        ])
        writer.writerows(csv_rows)

    print(f"Saved {png}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
