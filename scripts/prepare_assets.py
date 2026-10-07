#!/usr/bin/env python3
"""Download and verify every non-training asset needed by Figure 1(a)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".labelbench"
MODEL_DIR = ROOT / "model"
COMMIT = "9e23393ba5dd48fc7c23293e441337a5e45a3edf"
REPOSITORY = "https://github.com/EfficientTraining/LabelBench.git"


def prepare_source() -> None:
    if not SOURCE.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(SOURCE)], check=True)
    subprocess.run(["git", "fetch", "origin", COMMIT], cwd=SOURCE, check=True)
    subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=SOURCE, check=True)

    # Keep the official implementation, changing only where OpenAI CLIP caches its weight.
    path = SOURCE / "LabelBench/model/model_impl/clip.py"
    text = path.read_text()
    if "LABELBENCH_MODEL_DIR" not in text:
        text = text.replace("import clip\n", "import clip\nimport os\n", 1)
        text = text.replace(
            "model, preprocess = clip.load(model_name)",
            "model, preprocess = clip.load(\n"
            "            model_name, download_root=os.environ.get('LABELBENCH_MODEL_DIR'))",
            1,
        )
        path.write_text(text)


def verify_data() -> None:
    from torchvision.datasets import CIFAR10, CIFAR100

    cifar10_train = CIFAR10(ROOT / "data", train=True, download=True)
    cifar10_test = CIFAR10(ROOT / "data", train=False, download=True)
    cifar100_train = CIFAR100(ROOT / "data", train=True, download=True)
    cifar100_test = CIFAR100(ROOT / "data", train=False, download=True)
    assert (len(cifar10_train), len(cifar10_test)) == (50_000, 10_000)
    assert (len(cifar100_train), len(cifar100_test)) == (50_000, 10_000)


def prepare_clip() -> None:
    import clip

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, _ = clip.load("ViT-B/32", device="cpu", download_root=str(MODEL_DIR))
    del model


def main() -> None:
    prepare_source()
    verify_data()
    prepare_clip()
    print("assets ready")
    print(f"LabelBench: {SOURCE}")
    print(f"CLIP weights: {MODEL_DIR}")
    print(f"CIFAR data: {ROOT / 'data'}")


if __name__ == "__main__":
    main()

