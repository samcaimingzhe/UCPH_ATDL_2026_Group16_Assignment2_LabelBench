#!/usr/bin/env python3
"""Fetch matching W&B histories and render LabelBench Figure 1(a)."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["random", "confidence", "entropy", "margin", "coreset", "galaxy", "badge", "bait"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def cumulative_max(values):
    return np.maximum.accumulate(np.asarray(values, dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-entity", required=True)
    args = parser.parse_args()

    project = "Active Learning, cifar10, Batch Size=1000"
    runs = wandb.Api(timeout=120).runs(f"{args.wandb_entity}/{project}")
    histories = defaultdict(list)
    budgets = {}

    for run in runs:
        config = run.config
        if "none" not in config.get("embed_model_config", ""):
            continue
        if "clip_ViTB32_pretrained" not in config.get("classifier_model_config", ""):
            continue
        if "flexmatch" not in config.get("trainer_config", ""):
            continue
        if "noiseless" not in config.get("corrupter_config", "noiseless"):
            continue
        strategy = config.get("strategy_config", "").split("_")[0]
        if strategy not in STRATEGIES:
            continue
        x, y = [], []
        for row in run.scan_history(keys=["Num Labeled", "Test Accuracy"]):
            if row.get("Num Labeled") is not None and row.get("Test Accuracy") is not None:
                x.append(int(row["Num Labeled"]))
                y.append(float(row["Test Accuracy"]))
        if x:
            order = np.argsort(x)
            budgets[strategy] = np.asarray(x)[order]
            histories[strategy].append(cumulative_max(np.asarray(y)[order]))

    missing = [name for name in STRATEGIES if len(histories[name]) < 4]
    if missing:
        details = ", ".join(f"{name}={len(histories[name])}/4" for name in missing)
        raise RuntimeError(f"missing completed Figure 1(a) trials: {details}")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for strategy, color in zip(STRATEGIES, COLORS):
        values = np.vstack(histories[strategy][:4])
        mean = values.mean(axis=0)
        stderr = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        x = budgets[strategy]
        ax.plot(x, mean, linewidth=2, color=color, label=strategy.upper())
        ax.fill_between(x, mean - stderr, mean + stderr, color=color, alpha=0.25)

    ax.set_xlabel("Number of Labels")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.960, 0.982)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = ROOT / "results" / "figure1a.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

