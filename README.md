# LabelBench CIFAR-10 Reproduction

Group 16's reproduction of Figures 1(a) and 5(a) from *LabelBench: A Comprehensive Framework for Benchmarking Adaptive Label-Efficient Learning*. We use CIFAR-10, CLIP ViT-B/32, FlexMatch, and eight active-learning strategies. ImageNet experiments are outside our compute budget; Figure 5(a) selection alone took about nine hours on two NVIDIA A30 GPUs.

Our scripts build on the [authors' code](https://github.com/EfficientTraining/LabelBench). The original implementation logs runs to Weights & Biases; this repository stores run status, metrics, and Figure 5(a) selections locally in SQLite. Each panel uses four seeds (`1234`, `3334155`, `6667077`, `9999999`) and ten rounds of 1,000 new labels.

## Setup

An NVIDIA CUDA GPU is required for training. From the repository root:

```bash
conda create -n labelbench python=3.10.22
conda activate labelbench
python -m pip install -r requirements.txt
pip install 'numpy<2'
```

CIFAR-10 and the pretrained CLIP weights are loaded from `data/` and `model/`; the dataset/model loaders download missing assets when network access is available.

## Run

```bash
# Figure 1(a): full-model selection and evaluation
python scripts/run_fig1a.py --gpus 0 1
python scripts/plot_fig1a.py

# Figure 5(a): proxy selection, then full-model evaluation
python scripts/run_fig5.py --panel a --phase selection --gpus 0 1
python scripts/run_fig5.py --panel a --phase evaluate --gpus 0 1
python scripts/plot_fig5.py --panel a
```

Defaults: Figure 1(a) writes to `results/` (`figure1a.sqlite`, `figure1a.png`; an existing `experiments.sqlite` is also reused). Figure 5(a) writes to `results/fig5/a/` (`figure5a.sqlite`, `figure5a.png`, `summary.csv`). To use another directory, pass the same `--output-dir /path/to/output` to that panel's training and plotting commands.

Completed method/seed runs are skipped when restarted with the same output directory. An interrupted run restarts its current method/seed from round 1. Plotting requires all four seeds to finish. Figure 5(a)'s plot uses evaluation metrics, not proxy-selection metrics.
