#!/usr/bin/env python3
"""Read Figure 5 SQLite metrics; export PNG and CSV without training."""
import argparse
import csv
from pathlib import Path
import sqlite3

from fig5_common import COLORS, SEEDS, STRATEGIES, database_path, settings


def read_trials(db, cfg, strategy, seeds):
    expected = list(range(cfg["batch"], cfg["batch"] * cfg["rounds"] + 1, cfg["batch"]))
    trials = []
    for seed in seeds:
        status = db.execute("SELECT status FROM runs WHERE strategy=? AND seed=? AND phase='evaluate'",
                            (strategy, seed)).fetchone()
        if status != ("complete",):
            raise RuntimeError(f"{strategy}, seed {seed}: evaluation is not complete")
        rows = db.execute("SELECT labels,test_accuracy FROM metrics WHERE strategy=? AND seed=? "
                          "AND phase='evaluate' ORDER BY round", (strategy, seed)).fetchall()
        if [r[0] for r in rows] != expected:
            raise RuntimeError(f"{strategy}, seed {seed}: missing or mismatched label budgets")
        if any(not 0 <= r[1] <= 1 for r in rows):
            raise RuntimeError(f"{strategy}, seed {seed}: accuracy must be finite and in [0, 1]")
        trials.append([r[1] for r in rows])
    return expected, trials


def main(panel=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if panel is None:
        parser.add_argument("--panel", choices=list("abc"), required=True)
    parser.add_argument("--output", "-o", "--output-dir", dest="output", type=Path)
    parser.add_argument("--input", "-i", "--database", dest="input", type=Path)
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--smoothing", choices=["max", "none"], default="max",
                        help="official plotting uses cumulative max per trial; database stays raw")
    parser.add_argument("--ylim", nargs=2, type=float,
                        help="optional y-axis limits; default auto range includes all results")
    args = parser.parse_args()
    cfg = settings(panel or args.panel)
    strategies = args.strategies or cfg["strategies"]
    if not set(strategies) <= set(cfg["strategies"]):
        parser.error("unsupported strategy for this panel")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("standard error needs at least two distinct seeds")
    if len(set(strategies)) != len(strategies):
        parser.error("strategies must be unique")
    if args.ylim and args.ylim[0] >= args.ylim[1]:
        parser.error("ylim must be increasing")
    output = (args.output or cfg["output"]).resolve()
    path = (args.input or database_path(cfg["panel"], output)).resolve()
    if not path.is_file():
        parser.error(f"experiment database not found: {path}; run training first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summaries = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if metadata.get("panel") != cfg["panel"]:
            raise RuntimeError("database belongs to a different Figure 5 panel")
        for strategy in strategies:
            labels, trials = read_trials(db, cfg, strategy, args.seeds)
            values = np.asarray(trials, dtype=float)
            if args.smoothing == "max":
                values = np.maximum.accumulate(values, axis=1)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(args.seeds))
            summaries.append((strategy, labels, mean, sem))
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    csv_rows = []
    for strategy, labels, mean, sem in summaries:
        color = COLORS[STRATEGIES.index(strategy)]
        ax.plot(labels, mean, label=strategy.upper(), color=color, linewidth=2)
        ax.fill_between(labels, mean - sem, mean + sem, color=color, alpha=.25)
        csv_rows.extend((strategy, int(n), float(m), float(e), len(args.seeds), args.smoothing)
                        for n, m, e in zip(labels, mean, sem))
    ax.set(xlabel="Number of Labels", ylabel="Test Accuracy")
    if args.ylim:
        ax.set_ylim(*args.ylim)
    ax.grid(True, linestyle="--", alpha=.6)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    png = output / f"figure5{cfg['panel']}.png"
    fig.savefig(png, dpi=250)
    plt.close(fig)
    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["strategy", "labels", "mean_test_accuracy", "standard_error", "trials", "smoothing"])
        writer.writerows(csv_rows)
    print(f"Saved {png}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
