#!/usr/bin/env python3
"""Download and verify every non-training asset needed by Figure 1(a)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


def verify_data() -> None:
    from torchvision.datasets import CIFAR10

    cifar10_train = CIFAR10(ROOT / "data", train=True, download=True)
    cifar10_test = CIFAR10(ROOT / "data", train=False, download=True)
    assert (len(cifar10_train), len(cifar10_test)) == (50_000, 10_000)


def prepare_clip() -> None:
    import clip

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, _ = clip.load("ViT-B/32", device="cpu", download_root=str(MODEL_DIR))
    del model


def main() -> None:
    verify_data()
    prepare_clip()
    print("assets ready")
    print(f"CLIP weights: {MODEL_DIR}")
    print(f"CIFAR-10 data: {ROOT / 'data'}")


if __name__ == "__main__":
    main()
