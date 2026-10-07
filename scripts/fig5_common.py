"""Paths and experiment settings shared by Figure 5 training and plotting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Same four seeds as the official np.linspace(1234, 9999999, 4, dtype=int).
SEEDS = [1234, 3334155, 6667077, 9999999]
STRATEGIES = ["random", "confidence", "entropy", "margin", "coreset",
              "galaxy", "badge", "bait"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:pink", "tab:olive"]


def settings(panel):
    if panel not in "abc" or len(panel) != 1:
        raise ValueError("panel must be a, b or c")
    batch = 200 if panel == "b" else 1000
    maximum = 4000 if panel == "b" else 10000
    return {
        "panel": panel,
        "batch": batch,
        "rounds": maximum // batch,
        "strategies": STRATEGIES if panel == "a" else STRATEGIES[:-1],
        "output": ROOT / "results" / "fig5" / panel,
        "ylim": (.960, .982) if panel != "c" else (.948, .978),
    }


def database_path(panel, output):
    return Path(output) / f"figure5{panel}.sqlite"
