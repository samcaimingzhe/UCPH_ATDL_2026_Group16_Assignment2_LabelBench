#!/usr/bin/env python3
"""Plot every recorded metric and a contact sheet for each experiment.

By default, Figure 5 plots the final-model evaluate phase. Use --phase
selection to inspect proxy metrics for Figure 5(a) or 5(b).
"""
import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig5_common import COLORS as FIG5_COLORS, SEEDS as FIG5_SEEDS
from fig5_common import STRATEGIES, settings
from run_fig1a import ALL as FIG1_STRATEGIES, SEEDS as FIG1_SEEDS


ROOT = Path(__file__).resolve().parents[1]
PLOT_SIZE = (3.5, 2.8)
SKIP_KEYS = {"Epoch", "Num Labeled"}
FIG1_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]


def database_path(experiment):
    if experiment == "fig1a":
        return ROOT / "results/fig1a/figure1a.sqlite"
    panel = experiment[-1]
    flat = ROOT / f"results/fig5/figure5{panel}.sqlite"
    nested = ROOT / f"results/fig5/{panel}/figure5{panel}.sqlite"
    return flat if flat.is_file() else nested


def slug(metric):
    return re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")


def load_metrics(path, experiment, phase):
    is_fig1 = experiment == "fig1a"
    cfg = None if is_fig1 else settings(experiment[-1])
    strategies = FIG1_STRATEGIES if is_fig1 else cfg["strategies"]
    seeds = FIG1_SEEDS if is_fig1 else FIG5_SEEDS
    batch, rounds = (1000, 10) if is_fig1 else (cfg["batch"], cfg["rounds"])
    labels = np.arange(batch, batch * rounds + 1, batch)
    data = {}
    metric_names = None

    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        if not is_fig1:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            if metadata.get("panel") != cfg["panel"]:
                raise ValueError(f"{path} is not a Figure 5({cfg['panel']}) database")
        for strategy in strategies:
            data[strategy] = {}
            for seed in seeds:
                where = "strategy=? AND seed=?" + ("" if is_fig1 else " AND phase=?")
                params = (strategy, seed) if is_fig1 else (strategy, seed, phase)
                status = db.execute(f"SELECT status FROM runs WHERE {where}", params).fetchone()
                if status != ("complete",):
                    raise ValueError(f"{experiment} {phase}: {strategy} seed={seed} is not complete")
                rows = db.execute(
                    f"SELECT round,labels,payload FROM metrics WHERE {where} ORDER BY round", params
                ).fetchall()
                if [r[0] for r in rows] != list(range(1, rounds + 1)) or [r[1] for r in rows] != list(labels):
                    raise ValueError(f"{experiment} {phase}: {strategy} seed={seed} has missing rounds or labels")
                payloads = [json.loads(r[2]) for r in rows]
                names = [key for key, value in payloads[0].items()
                         if key not in SKIP_KEYS and isinstance(value, (int, float))
                         and not isinstance(value, bool)]
                if metric_names is None:
                    metric_names = names
                if set(names) != set(metric_names):
                    raise ValueError(f"{experiment} {phase}: inconsistent metrics for {strategy} seed={seed}")
                for name in metric_names:
                    values = np.asarray([payload.get(name) for payload in payloads], dtype=float)
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"{experiment} {phase}: invalid {name} for {strategy} seed={seed}")
                    data[strategy].setdefault(name, []).append(values)
    return labels, strategies, seeds, metric_names, data


def draw_metric(ax, labels, strategies, seeds, data, metric, experiment):
    for strategy in strategies:
        trials = np.asarray(data[strategy][metric], dtype=float)
        mean = trials.mean(axis=0)
        sem = trials.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        color = (FIG1_COLORS[FIG1_STRATEGIES.index(strategy)] if experiment == "fig1a"
                 else FIG5_COLORS[STRATEGIES.index(strategy)])
        ax.plot(labels, mean, label=strategy.upper(), color=color, linewidth=2)
        ax.fill_between(labels, mean - sem, mean + sem, color=color, alpha=.20)
    ax.set(xlabel="Number of Labels", ylabel=metric)
    ax.grid(True, linestyle="--", alpha=.6)
    ax.tick_params(labelsize=8)


def save_experiment(experiment, phase, path, output_root, columns):
    labels, strategies, seeds, metrics, data = load_metrics(path, experiment, phase)
    suffix = "" if experiment == "fig1a" else f"_{phase}"
    folder = output_root / f"{experiment}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=PLOT_SIZE)
        draw_metric(ax, labels, strategies, seeds, data, metric, experiment)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(folder / f"{slug(metric)}.png", dpi=250)
        plt.close(fig)

    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(PLOT_SIZE[0] * columns, PLOT_SIZE[1] * rows),
                             squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        draw_metric(ax, labels, strategies, seeds, data, metric, experiment)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=min(len(names), 8), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .975))
    combined = output_root / f"{experiment}{suffix}_all_metrics.png"
    fig.savefig(combined, dpi=220)
    plt.close(fig)
    print(f"{experiment}{suffix}: {len(metrics)} metric PNGs in {folder}; combined: {combined}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", "fig1a", "fig5a", "fig5b", "fig5c"],
                        default="all")
    parser.add_argument("--input", type=Path, help="database path (requires one experiment)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/plots/metrics")
    parser.add_argument("--phase", choices=["evaluate", "selection"], default="evaluate",
                        help="Figure 5(a/b) phase; Figure 5(c) only has evaluate")
    parser.add_argument("--columns", type=int, default=4, help="columns in each combined image")
    args = parser.parse_args()
    if args.columns < 1:
        parser.error("--columns must be positive")
    if args.input and args.experiment == "all":
        parser.error("--input requires a single --experiment")
    if args.phase == "selection" and args.experiment in ("fig5c", "all"):
        parser.error("selection is available only for fig5a or fig5b")
    experiments = (["fig1a", "fig5a", "fig5b", "fig5c"] if args.experiment == "all"
                   else [args.experiment])
    for experiment in experiments:
        path = (args.input or database_path(experiment)).resolve()
        if not path.is_file():
            if args.experiment == "all" and experiment == "fig5b":
                print(f"Skipping fig5b: database not found at {path}")
                continue
            parser.error(f"database not found: {path}")
        phase = "evaluate" if experiment == "fig5c" else args.phase
        save_experiment(experiment, phase, path, args.output_dir.resolve(), args.columns)


if __name__ == "__main__":
    main()
