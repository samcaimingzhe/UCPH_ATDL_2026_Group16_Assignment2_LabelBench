#!/usr/bin/env python3
"""Plot Figure 1(a) CIFAR-10 test accuracy."""
import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig1_common import ROOT, STRATEGIES, database_path, settings

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/plots")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    cfg = settings("cifar10")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    path = (args.input or database_path(cfg)).resolve()
    if not path.is_file():
        parser.error(f"database not found: {path}")
    labels = list(range(cfg["batch"], cfg["batch"] * cfg["rounds"] + 1, cfg["batch"]))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        for strategy in args.strategies:
            trials = []
            seeds = args.seeds if args.seeds is not None else [row[0] for row in db.execute(
                "SELECT seed FROM runs WHERE strategy=? AND status='complete' ORDER BY seed",
                (strategy,))]
            for seed in seeds:
                status = db.execute("SELECT status FROM runs WHERE strategy=? AND seed=?",
                                    (strategy, seed)).fetchone()
                if status != ("complete",):
                    print(f"Skipping {strategy} seed={seed}: not complete")
                    continue
                rows = db.execute("SELECT labels,test_accuracy FROM metrics WHERE strategy=? AND seed=? "
                                  "ORDER BY round", (strategy, seed)).fetchall()
                if [r[0] for r in rows] != labels:
                    print(f"Skipping {strategy} seed={seed}: missing or mismatched rounds")
                    continue
                values = np.asarray([r[1] for r in rows], dtype=float)
                if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                    print(f"Skipping {strategy} seed={seed}: invalid accuracy")
                    continue
                trials.append(np.maximum.accumulate(values))
            if not trials:
                print(f"Skipping {strategy}: no complete valid trials")
                continue
            trials = np.asarray(trials)
            mean = trials.mean(axis=0)
            sem = trials.std(axis=0, ddof=1) / np.sqrt(len(trials)) if len(trials) > 1 else None
            color = COLORS[STRATEGIES.index(strategy)]
            ax.plot(labels, mean, label=f"{strategy.upper()} (n={len(trials)})", color=color, linewidth=2)
            if sem is not None:
                ax.fill_between(labels, mean - sem, mean + sem, color=color, alpha=.18)
    if not ax.lines:
        plt.close(fig)
        print("No complete valid runs; no plot saved.")
        return
    ax.set(xlabel="Number of Labels", ylabel="Test Accuracy")
    ax.grid(True, linestyle="--", alpha=.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "figure1a.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
