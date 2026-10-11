"""NEW preflight; no dataset/model download. Use --gpu inside a Slurm GPU step."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

import clip
import numpy as np
import torch
import torchvision
from functorch import vmap
from LabelBench.model.model import get_model_fn
from LabelBench.strategy.strategies import get_strategy
from LabelBench.trainer.trainer import get_trainer

assert callable(clip.load), "The imported clip is not OpenAI CLIP."
assert np.__version__.split(".")[0] == "1", "Use the proposed NumPy 1.x environment."
print("Python:", sys.version)
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("NumPy:", np.__version__, "CUDA wheel runtime:", torch.version.cuda)
print("CLIP import:", clip.__file__)
print("CUDA_VISIBLE_DEVICES supplied by Slurm:", os.environ.get("CUDA_VISIBLE_DEVICES"))
device = "cpu"
if args.gpu:
    assert torch.cuda.is_available(), "CUDA unavailable inside allocated GPU job."
    assert torch.cuda.device_count() == 1, "This starter job requests exactly one GPU."
    print("GPU:", torch.cuda.get_device_name(0))
    device = "cuda"
    x = torch.randn(128, 128, device=device)
    assert torch.isfinite(x @ x).all()
for name, width in [("linear", None), ("shallow", 128), ("shallow", 512), ("shallow", 1024)]:
    config = {"input_dim": 512, "num_output": 10, "num_hidden": 2, "ret_emb": False}
    if width is not None:
        config["hidden_dim"] = width
    model = get_model_fn(name)(config).to(device)
    logits, embedding = model(torch.randn(4, 512, device=device), ret_features=True)
    assert tuple(logits.shape) == (4, 10)
    assert tuple(embedding.shape) == (4, 512 if width is None else width)
    logits.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    print(name, width, "forward/backward OK")
print("PREFLIGHT PASSED")
