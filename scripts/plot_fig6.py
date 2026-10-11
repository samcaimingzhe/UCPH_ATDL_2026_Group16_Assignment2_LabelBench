#!/usr/bin/env python3
"""Plot Figure 6 results for a selected trainer."""
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
DEFAULT_SEEDS = [1234, 3334155, 6667077]
BATCH = 1000
ROUNDS = 10
TRAINERS = ("flexmatch", "softmatch", "freematch", "uda", "pseudolabel", "passive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=TRAINERS, default="flexmatch")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to results/fig6/<trainer>.")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--strategies", nargs="+", choices=ALL, default=ALL)
    parser.add_argument("--smoothing", choices=["max", "none"], default="none",
                        help="Keep raw accuracies by default; max applies optional monotone post-processing.")
    args = parser.parse_args()

    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    output = (args.output_dir or ROOT / "results" / "fig6" / args.trainer).resolve()
    default_database = output / f"figure6_{args.trainer}.sqlite"
    if not default_database.is_file():
        default_database = output / "figure6.sqlite"
    path = (args.database or default_database).resolve()
    if not path.is_file():
        raise SystemExit(f"database not found: {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_expected = list(range(BATCH, BATCH * ROUNDS + 1, BATCH))
    summaries = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if metadata.get("experiment") != "fig6" or metadata.get("trainer") != args.trainer:
            raise RuntimeError(f"database is not a Figure 6 {args.trainer} database")
        if metadata.get("rounds") != str(ROUNDS):
            raise RuntimeError("plotting requires the full 10-round Figure 6 run; smoke-test databases are not final results")

        for strategy in args.strategies:
            trials = []
            seeds = args.seeds if args.seeds is not None else [row[0] for row in db.execute(
                "SELECT seed FROM runs WHERE strategy=? AND status='complete' ORDER BY seed",
                (strategy,))]
            for seed in seeds:
                status = db.execute(
                    "SELECT status FROM runs WHERE strategy=? AND seed=?",
                    (strategy, seed),
                ).fetchone()
                if status != ("complete",):
                    print(f"Skipping {strategy} seed={seed}: not complete")
                    continue
                rows = db.execute(
                    "SELECT labels,test_accuracy FROM metrics WHERE strategy=? AND seed=? ORDER BY round",
                    (strategy, seed),
                ).fetchall()
                if [row[0] for row in rows] != labels_expected:
                    print(f"Skipping {strategy} seed={seed}: incomplete label-budget sequence")
                    continue
                accuracy = np.asarray([row[1] for row in rows], dtype=float)
                if not np.all(np.isfinite(accuracy)) or np.any((accuracy < 0) | (accuracy > 1)):
                    print(f"Skipping {strategy} seed={seed}: invalid accuracy")
                    continue
                trials.append(accuracy)

            if not trials:
                print(f"Skipping {strategy}: no complete valid trials")
                continue
            values = np.asarray(trials, dtype=float)
            if args.smoothing == "max":
                values = np.maximum.accumulate(values, axis=1)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(trials)) if len(trials) > 1 else None
            summaries.append((strategy, np.asarray(labels_expected), mean, sem, len(trials)))

    if not summaries:
        print("No complete valid runs; no plot saved.")
        return
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    csv_rows = []
    for strategy, labels, mean, sem, count in summaries:
        color = COLORS[ALL.index(strategy)]
        ax.plot(labels, mean, label=f"{strategy.upper()} (n={count})", color=color, linewidth=2)
        if sem is not None:
            ax.fill_between(labels, mean - sem, mean + sem, color=color, alpha=.25)
        errors = sem if sem is not None else [None] * len(labels)
        for n, m, e in zip(labels, mean, errors):
            csv_rows.append([strategy, int(n), float(m), float(e) if e is not None else None, count, args.smoothing])

    ax.set_xlabel("Number of Labels")
    ax.set_ylabel("Test Accuracy")
    if args.trainer == "flexmatch":
        ax.set_ylim(0.960, 0.980)
    ax.grid(True, linestyle="--", alpha=.6)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    png = output / f"figure6_{args.trainer}.png"
    fig.savefig(png, dpi=250)
    plt.close(fig)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
