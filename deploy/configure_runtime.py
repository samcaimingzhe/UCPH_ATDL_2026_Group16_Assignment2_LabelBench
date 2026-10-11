"""NEW deployment profile: use four DataLoader workers and feature batch 128.

Run once in a fresh checkout BEFORE configure_ablation.py and cache generation.
This profile may change augmentation RNG relative to historical results; all new
variants and the fresh baseline must use the same profile and cache.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "work/ablation_original_configs").exists():
    raise SystemExit("Configure resources before creating the ablation baseline backup.")
changes = {
    "configs/embed_model/clip_ViTB32.json": {"num_workers": 4, "inference_batch_size": 128},
    "configs/trainer/cifar10/flexmatch/clip_ViTB32_shallow.json": {"num_workers": 4},
    "configs/trainer/cifar10/flexmatch/clip_ViTB32_finetune.json": {"num_workers": 4},
}
for relative, fields in changes.items():
    path = ROOT / relative
    config = json.loads(path.read_text())
    config.update(fields)
    path.write_text(json.dumps(config, indent=2) + "\n")
for relative, old, new in [
    ("LabelBench/dataset/datasets.py", "num_workers=40", "num_workers=4"),
    ("LabelBench/trainer/trainer_impl/pytorch_passive_trainer.py", "num_workers=10", "num_workers=4"),
]:
    path = ROOT / relative
    content = path.read_text()
    if content.count(old) == 1:
        path.write_text(content.replace(old, new))
    elif old not in content and content.count(new) == 1:
        pass
    else:
        raise SystemExit(f"Unexpected source structure in {relative}; review manually.")
print("Deployment profile set: four workers; feature extraction batch 128.")
print("Proxy labeled batch/uratio/lr/epochs and downstream training batch remain unchanged.")
