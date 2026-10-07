#!/usr/bin/env python3
"""Launch the authors' complete CIFAR-10 experiment for LabelBench Figure 1(a)."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".labelbench"
COMMIT = "9e23393ba5dd48fc7c23293e441337a5e45a3edf"
REPOSITORY = "https://github.com/EfficientTraining/LabelBench.git"
STRATEGIES = [
    "random_sampling.json",
    "confidence_sampling.json",
    "entropy_sampling.json",
    "margin_sampling.json",
    "coreset_sampling.json",
    "galaxy_sampling.json",
    "badge_sampling.json",
    "bait_sampling.json",
]


def prepare_official_source() -> None:
    if not SOURCE.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(SOURCE)], check=True)
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    if current != COMMIT:
        subprocess.run(["git", "fetch", "origin", COMMIT], cwd=SOURCE, check=True)
        subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=SOURCE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--runs-per-gpu", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    prepare_official_source()
    command = [
        sys.executable,
        "mp_launcher.py",
        "--wandb_name", args.wandb_entity,
        "--dataset", "cifar10",
        "--data_dir", str(ROOT / "data"),
        "--metric", "multi_class",
        "--batch_size", "1000",
        "--num_batch", "10",
        "--embed_model_config", "none.json",
        "--classifier_model_config", "clip_ViTB32_pretrained.json",
        "--trainer_config", "cifar10/flexmatch/clip_ViTB32_finetune.json",
        "--corrupter_config", "noiseless.json",
        "--strategies", *STRATEGIES,
        "--num_runs", "4",
        "--device_per_run", "0",
        "--run_per_device", str(args.runs_per_gpu),
        "--gpu_masks", *(str(gpu) for gpu in args.gpus),
        "--skip", str(args.skip),
    ]
    subprocess.run(command, cwd=SOURCE, check=True)


if __name__ == "__main__":
    main()
