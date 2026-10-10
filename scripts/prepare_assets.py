#!/usr/bin/env python3
"""Prepare CIFAR-10 or verify a locally provided ImageNet-1K dataset."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


def detect_dataset(data_dir, dataset="auto"):
    if dataset != "auto":
        return dataset
    path = Path(data_dir).expanduser().resolve()
    if path.name.lower() in {"imagenet", "imagenet1k", "imagenet-1k", "ilsvrc2012"}:
        return "imagenet"
    if path.name.lower() in {"cifar10", "cifar-10"}:
        return "cifar10"
    if (path / "train").is_dir() and (path / "val").is_dir():
        return "imagenet"
    if path == (ROOT / "data").resolve() or (path / "cifar-10-batches-py").is_dir():
        return "cifar10"
    raise ValueError(f"Cannot identify dataset in {path}; specify --dataset cifar10 or imagenet")


def verify_data(data_dir=None, dataset="cifar10"):
    data_dir = Path(data_dir or ROOT / "data").expanduser().resolve()
    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10

        train = CIFAR10(data_dir, train=True, download=True)
        test = CIFAR10(data_dir, train=False, download=True)
        if (len(train), len(test)) != (50_000, 10_000):
            raise ValueError("CIFAR-10 has unexpected train/test sizes")
        print(f"CIFAR-10 ready: {data_dir}")
    elif dataset == "imagenet":
        from torchvision.datasets import ImageFolder

        train_dir, val_dir = data_dir / "train", data_dir / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {data_dir}")
        train, val = ImageFolder(train_dir), ImageFolder(val_dir)
        if len(train.classes) != 1000 or train.classes != val.classes:
            raise ValueError("ImageNet train/val need the same 1,000 synset folders")
        if (len(train), len(val)) != (1_281_167, 50_000):
            raise ValueError(f"Incomplete ImageNet-1K: found {len(train)} train and {len(val)} val images")
        print(f"ImageNet-1K verified: {data_dir}")
    else:
        raise ValueError(f"unsupported dataset: {dataset}")


def prepare_clip(model_dir=MODEL_DIR):
    import clip

    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    model, _ = clip.load("ViT-B/32", device="cpu", download_root=str(model_dir))
    del model
    print(f"CLIP weights ready: {model_dir}")


def prepare(data_dir, dataset="auto"):
    selected = detect_dataset(data_dir, dataset)
    verify_data(data_dir, selected)
    prepare_clip()
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset", choices=["auto", "cifar10", "imagenet"], default="auto")
    args = parser.parse_args()
    prepare(args.data_dir, args.dataset)


if __name__ == "__main__":
    main()
