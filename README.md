# LabelBench CIFAR-10 Reproduction

Group 16's reproduction of Figures 1(a) and 5(a) from *LabelBench: A Comprehensive Framework for Benchmarking Adaptive Label-Efficient Learning*. We use CIFAR-10, CLIP ViT-B/32, FlexMatch, and eight active-learning strategies. ImageNet experiments are outside our compute budget; Figure 5(a) selection alone took about nine hours on two NVIDIA A30 GPUs.

Our scripts build on the [authors' code](https://github.com/EfficientTraining/LabelBench). The original implementation logs runs to Weights & Biases; this repository stores run status, metrics, and Figure 5(a) selections locally in SQLite. Each panel uses four seeds (`1234`, `3334155`, `6667077`, `9999999`) and ten rounds of 1,000 new labels.

## Setup

An NVIDIA CUDA GPU is required for training. From the repository root:

```bash
conda create -n labelbench python=3.10.22
conda activate labelbench
python -m pip install -r requirements.txt
```

CIFAR-10 and the pretrained CLIP weights are loaded from `data/` and `model/`; the dataset/model loaders download missing assets when network access is available.

## Run

```bash
# Figure 1(a): full-model selection and evaluation
python scripts/run_fig1a.py --gpus 0 --output-dir results/fig1
python scripts/plot_fig1a.py

# Figure 5(a): proxy selection, then full-model evaluation
python scripts/run_fig5.py --panel a --phase selection --gpus 0 --output-dir results/fig5
python scripts/run_fig5.py --panel a --phase evaluate --gpus 0 --output-dir results/fig5
python scripts/plot_fig5.py --panel a
```
Completed method/seed runs are skipped when restarted with the same output directory. An interrupted run restarts its current method/seed from round 1. Plotting requires all four seeds to finish. Figure 5(a)'s plot uses evaluation metrics, not proxy-selection metrics.
```
# run_fig1a.py
usage: run_fig1a.py [-h]
                    --gpus GPUS [GPUS ...]
                    [--strategies {random,confidence,entropy,margin,coreset,galaxy,badge,bait} [{random,confidence,entropy,margin,coreset,galaxy,badge,bait} ...]]
                    [--skip SKIP]

options:
  -h, --help           show this help
                       message and exit
  --gpus GPUS [GPUS ...]
  --strategies {random,confidence,entropy,margin,coreset,galaxy,badge,bait} [{random,confidence,entropy,margin,coreset,galaxy,badge,bait} ...]
  --skip SKIP

# run_fig5.py
usage: run_fig5.py [-h] --panel {a,b,c} [--gpus GPUS [GPUS ...]]
                   [--strategies STRATEGIES [STRATEGIES ...]] [--seeds SEEDS [SEEDS ...]]
                   [--phase {all,selection,evaluate}] [--output-dir OUTPUT_DIR]
                   [--save-checkpoints] [--dry-run]

options:
  -h, --help            show this help message and exit
  --panel {a,b,c}
  --gpus GPUS [GPUS ...]
  --strategies STRATEGIES [STRATEGIES ...]
  --seeds SEEDS [SEEDS ...]
  --phase {all,selection,evaluate}
                        a/b: collect proxy labels, evaluate saved labels, or both
  --output-dir OUTPUT_DIR
  --save-checkpoints    save final-round weights (large for CLIP)
  --dry-run             print the plan without training or creating output files
```
