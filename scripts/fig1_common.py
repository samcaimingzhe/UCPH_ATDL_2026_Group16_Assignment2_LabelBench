"""Shared settings for Figure 1(a) CIFAR-10 and Figure 1(b) ImageNet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["random", "confidence", "entropy", "margin", "coreset",
              "galaxy", "badge", "bait"]
SEEDS = [1234, 3334155, 6667077, 9999999]


def settings(dataset):
    if dataset == "cifar10":
        return {"dataset": dataset, "panel": "a", "batch": 1000, "rounds": 10,
                "seeds": SEEDS, "trainer_config":
                ROOT / "configs/trainer/cifar10/flexmatch/clip_ViTB32_finetune.json",
                "output": ROOT / "results/fig1a"}
    if dataset == "imagenet":
        return {"dataset": dataset, "panel": "b", "batch": 30000, "rounds": 20,
                "seeds": SEEDS[:2], "trainer_config":
                ROOT / "configs/trainer/imagenet/flexmatch/clip_ViTB32_finetune.json",
                "output": ROOT / "results/fig1b"}
    raise ValueError(f"unsupported Figure 1 dataset: {dataset}")


def database_path(cfg, output_dir=None):
    output = Path(output_dir).resolve() if output_dir is not None else cfg["output"]
    return output / f"figure1{cfg['panel']}.sqlite"
