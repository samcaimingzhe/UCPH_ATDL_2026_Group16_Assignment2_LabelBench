"""Re-create the single-seed Figure 6(c) plot from summary_single_seed.csv.

This plot intentionally has no standard-error band because only seed 1234 is present.
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "summary_single_seed.csv"
OUT_PATH = ROOT / "figure6c_flexmatch_single_seed.png"
STRATEGIES = ["random", "confidence", "entropy", "margin", "coreset", "galaxy", "badge", "bait"]

curves = defaultdict(list)
with CSV_PATH.open(newline="") as f:
    for row in csv.DictReader(f):
        curves[row["strategy"]].append((int(row["labels"]), float(row["test_accuracy"])))

expected = list(range(1000, 10001, 1000))
for strategy in STRATEGIES:
    points = sorted(curves[strategy])
    if [x for x, _ in points] != expected:
        raise ValueError(f"{strategy}: expected ten label budgets; found {points}")

all_y = [y for strategy in STRATEGIES for _, y in curves[strategy]]
fig, ax = plt.subplots(figsize=(8, 5.5))
for strategy in STRATEGIES:
    points = sorted(curves[strategy])
    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", markersize=3, linewidth=1.8, label=strategy.upper())
ax.set_xlabel("Number of Labels")
ax.set_ylabel("Test Accuracy")
ax.set_title("Figure 6(c): FlexMatch on CIFAR-10\nSingle trial, seed 1234")
ax.set_xticks(expected)
ax.set_ylim(max(0, min(all_y) - 0.002), min(1, max(all_y) + 0.002))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
