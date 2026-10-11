"""Shared settings for Figure 1(a) CIFAR-10."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["random", "confidence", "entropy", "margin", "coreset",
              "galaxy", "badge", "bait"]
SEEDS = [1234, 3334155, 6667077, 9999999]


def settings(dataset="cifar10"):
    if dataset != "cifar10":
        raise ValueError("Figure 1 settings currently support only cifar10")
    return {"batch": 1000, "rounds": 10, "seeds": SEEDS,
            "trainer_config":
            ROOT / "configs/trainer/cifar10/flexmatch/clip_ViTB32_finetune.json",
            "output": ROOT / "results/fig1"}


def database_path(cfg, output_dir=None):
    output = Path(output_dir).resolve() if output_dir is not None else cfg["output"]
    return output / "figure1a.sqlite"
