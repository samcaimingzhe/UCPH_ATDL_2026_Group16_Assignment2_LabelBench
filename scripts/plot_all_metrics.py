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

from fig1_common import STRATEGIES as FIG1_STRATEGIES, settings as fig1_settings
from fig5_common import COLORS as FIG5_COLORS, STRATEGIES, settings as fig5_settings


ROOT = Path(__file__).resolve().parents[1]
PLOT_SIZE = (3.5, 3)
SKIP_KEYS = {"Epoch", "Num Labeled"}
EXCLUDED_PARTS = ("loss", "std", "min class", "max class", "number of positive",
                  "validation")
FIG1_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]


def database_path(experiment):
    if experiment.startswith("fig1"):
        panel = "a" if experiment == "fig1a" else "b"
        name = f"figure1{panel}.sqlite"
        candidates = [ROOT / f"results/fig1{panel}" / name,
                      ROOT / "results/fig1" / name]
    else:
        panel = experiment[-1]
        name = f"figure5{panel}.sqlite"
        candidates = [ROOT / "results/fig5" / name,
                      ROOT / "results/fig5" / panel / name]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted((ROOT / "results").rglob(name))
    return matches[0] if matches else candidates[0]


def slug(metric):
    if metric == "Test Accuracy":
        return "generalization_accuracy"
    return re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")


def include_metric(name):
    lowered = name.lower()
    return name not in SKIP_KEYS and not any(part in lowered for part in EXCLUDED_PARTS)


def load_metrics(path, experiment, phase):
    is_fig1 = experiment.startswith("fig1")
    cfg = (fig1_settings("cifar10" if experiment == "fig1a" else "imagenet")
           if is_fig1 else fig5_settings(experiment[-1]))
    strategies = FIG1_STRATEGIES if is_fig1 else cfg["strategies"]
    batch, rounds = cfg["batch"], cfg["rounds"]
    labels = np.arange(batch, batch * rounds + 1, batch)
    data = {}
    metric_names = None
    seed_counts = {}

    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        if not is_fig1:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            if metadata.get("panel") != cfg["panel"]:
                raise ValueError(f"{path} is not a Figure 5({cfg['panel']}) database")
        for strategy in strategies:
            completed = db.execute(
                "SELECT seed FROM runs WHERE strategy=? AND status='complete'"
                + ("" if is_fig1 else " AND phase=?") + " ORDER BY seed",
                (strategy,) if is_fig1 else (strategy, phase),
            ).fetchall()
            if not completed:
                continue
            strategy_data = {}
            valid_seeds = 0
            for (seed,) in completed:
                where = "strategy=? AND seed=?" + ("" if is_fig1 else " AND phase=?")
                params = (strategy, seed) if is_fig1 else (strategy, seed, phase)
                rows = db.execute(
                    f"SELECT round,labels,payload FROM metrics WHERE {where} ORDER BY round", params
                ).fetchall()
                if [r[0] for r in rows] != list(range(1, rounds + 1)) or [r[1] for r in rows] != list(labels):
                    print(f"Skipping {experiment} {strategy} seed={seed}: missing rounds or labels")
                    continue
                payloads = [json.loads(r[2]) for r in rows]
                names = [key for key, value in payloads[0].items()
                         if include_metric(key) and isinstance(value, (int, float))
                         and not isinstance(value, bool)]
                if metric_names is None:
                    metric_names = names
                if set(names) != set(metric_names):
                    print(f"Skipping {experiment} {strategy} seed={seed}: inconsistent metrics")
                    continue
                values_by_metric = {}
                valid = True
                for name in metric_names:
                    values = np.asarray([payload.get(name) for payload in payloads], dtype=float)
                    if not np.all(np.isfinite(values)):
                        print(f"Skipping {experiment} {strategy} seed={seed}: invalid {name}")
                        valid = False
                        break
                    values_by_metric[name] = values
                if not valid:
                    continue
                for name, values in values_by_metric.items():
                    strategy_data.setdefault(name, []).append(values)
                valid_seeds += 1
            if valid_seeds:
                data[strategy] = strategy_data
                seed_counts[strategy] = valid_seeds
    if not data:
        return None
    ordered = [strategy for strategy in strategies if strategy in data]
    priority = ["Test Accuracy", "Pool Accuracy"]
    metric_names = [m for m in priority if m in metric_names] + [
        m for m in metric_names if m not in priority
    ]
    return labels, ordered, seed_counts, metric_names, data


def draw_metric(ax, labels, strategies, seed_counts, data, metric, experiment):
    for strategy in strategies:
        trials = np.asarray(data[strategy][metric], dtype=float)
        mean = trials.mean(axis=0)
        sem = (trials.std(axis=0, ddof=1) / np.sqrt(len(trials))
               if len(trials) > 1 else None)
        color = (FIG1_COLORS[FIG1_STRATEGIES.index(strategy)] if experiment.startswith("fig1")
                 else FIG5_COLORS[STRATEGIES.index(strategy)])
        ax.plot(labels, mean, label=f"{strategy.upper()} (n={seed_counts[strategy]})",
                color=color, linewidth=2)
        if sem is not None:
            ax.fill_between(labels, mean - sem, mean + sem, color=color, alpha=.20)
    display_name = "Generalization Accuracy" if metric == "Test Accuracy" else metric
    ax.set(xlabel="Number of Labels", ylabel=display_name)
    ax.grid(True, linestyle="--", alpha=.6)
    ax.tick_params(labelsize=8)


def save_experiment(experiment, phase, path, output_root, columns):
    loaded = load_metrics(path, experiment, phase)
    if loaded is None:
        print(f"Skipping {experiment}: no complete runs")
        return
    labels, strategies, seed_counts, metrics, data = loaded
    suffix = "" if experiment.startswith("fig1") else f"_{phase}"
    folder = output_root / f"{experiment}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*.png"):
        if stale.stem == "test_accuracy" or not include_metric(stale.stem.replace("_", " ")):
            stale.unlink()

    for metric in metrics:
        fig, ax = plt.subplots(figsize=PLOT_SIZE)
        draw_metric(ax, labels, strategies, seed_counts, data, metric, experiment)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(folder / f"{slug(metric)}.png", dpi=250)
        plt.close(fig)

    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(PLOT_SIZE[0] * columns, PLOT_SIZE[1] * rows),
                             squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        draw_metric(ax, labels, strategies, seed_counts, data, metric, experiment)
        ax.set_title("Generalization Accuracy" if metric == "Test Accuracy" else metric,
                     fontsize=10)
        ax.set_ylabel("")
        ax.legend(fontsize=6, ncol=2, loc="best")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    combined = output_root / f"{experiment}{suffix}_all_metrics.png"
    fig.savefig(combined, dpi=220)
    plt.close(fig)
    print(f"{experiment}{suffix}: {len(metrics)} metric PNGs in {folder}; combined: {combined}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", "fig1a", "fig1b", "fig1c",
                                                 "fig5a", "fig5b", "fig5c"],
                        default="all")
    parser.add_argument("--input", type=Path, help="database path (requires one experiment)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/plots/metrics")
    parser.add_argument("--phase", choices=["evaluate", "selection"], default="evaluate",
                        help="Figure 5(a/b) phase; Figure 5(c) only has evaluate")
    parser.add_argument("--columns", type=int, default=2, help="columns in each combined image")
    args = parser.parse_args()
    if args.columns < 1:
        parser.error("--columns must be positive")
    if args.input and args.experiment == "all":
        parser.error("--input requires a single --experiment")
    if args.phase == "selection" and args.experiment in ("fig5c", "all"):
        parser.error("selection is available only for fig5a or fig5b")
    experiments = (["fig1a", "fig1b", "fig1c", "fig5a", "fig5b", "fig5c"]
                   if args.experiment == "all"
                   else [args.experiment])
    for experiment in experiments:
        path = (args.input or database_path(experiment)).resolve()
        if not path.is_file():
            if args.experiment == "all":
                print(f"Skipping {experiment}: database not found at {path}")
                continue
            parser.error(f"database not found: {path}")
        phase = "evaluate" if experiment == "fig5c" else args.phase
        save_experiment(experiment, phase, path, args.output_dir.resolve(), args.columns)


if __name__ == "__main__":
    main()
