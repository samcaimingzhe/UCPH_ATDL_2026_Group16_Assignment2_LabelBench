# LabelBench Figure 1 reproduction

This is a compact launcher for reproducing Figure 1(a) of *LabelBench: A Framework for
Benchmarking Label-Efficient Learning*. It deliberately calls the authors' implementations
of FlexMatch and the active-learning strategies instead of maintaining local rewrites of
BAIT, BADGE, GALAXY and CORESET.

## Scope

Figure 1 contains three panels:

- Figure 1(a): CIFAR-10 generalization accuracy. This folder is ready for this panel.
- Figure 1(b): ImageNet generalization accuracy. ImageNet is not included.
- Figure 1(c): ImageNet pool accuracy. ImageNet is not included.

ImageNet requires separately licensed/access-controlled data, so the downloaded CIFAR data
cannot reproduce panels (b) and (c).

## Files

```text
LB/
├── data/                   # local CIFAR-10 and CIFAR-100
├── model/                  # reserved for model artifacts kept with this project
├── scripts/
│   ├── run_fig1a.py        # launches the official Figure 1(a) experiment
│   └── plot_fig1a.py       # reads Weights & Biases runs and plots Figure 1(a)
├── results/                # one PNG is written here after training
├── requirements.txt
└── README.md
```

No experiment JSON files are kept here. The paper settings are fixed in the launcher, and
the original LabelBench source is pinned to commit
`9e23393ba5dd48fc7c23293e441337a5e45a3edf`. On first use, the launcher downloads that
source into the hidden `.labelbench/` directory.

## Figure 1(a) settings

| Item | Value |
|---|---|
| Dataset | CIFAR-10 |
| Pool | 50,000 official training examples |
| Validation/test | official 10,000 test examples split deterministically into 5,000/5,000 with NumPy seed 42 |
| Model | pretrained OpenAI CLIP ViT-B/32, end-to-end fine-tuning |
| Semi-supervised trainer | FlexMatch |
| Active-learning methods | random, confidence, entropy, margin, CORESET, GALAXY, BADGE, BAIT |
| Initial labels | 1,000, selected uniformly at random |
| Labels added per round | 1,000 |
| Label budgets | 1,000 through 10,000 |
| Trials | 4 |
| Reported metric | Top-1 accuracy on the held-out 5,000-example test split |
| Confidence band | standard error over 4 trials |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `3e-5` |
| Schedule | cosine, 500 warm-up steps |
| Training limit | 20 epochs per active-learning round |
| Early stopping | patience 3, using the validation split |
| Labeled batch size | 64 |
| Unlabeled ratio | 3 unlabeled examples per labeled example |
| FlexMatch cutoff | 0.95 |
| Unlabeled loss weight | 1.0 |

The four seeds follow the authors' launcher: four integers evenly spaced from 1234 to
9,999,999. The plot applies the authors' cumulative-maximum smoothing before averaging.

## Data

The `data/` directory already contains torchvision's official Python-format downloads:

- CIFAR-10: 50,000 training and 10,000 test images;
- CIFAR-100: 50,000 training and 10,000 test images.

Figure 1(a) uses CIFAR-10 only. CIFAR-100 is retained for later LabelBench figures.

## Environment

The authors used Python 3.9 and did not publish an exact lock file. Create an isolated
environment and install:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CLIP weights are downloaded automatically when training starts and, following OpenAI CLIP's
default behavior, are stored in the user's normal `~/.cache/clip` cache. The local `model/`
directory is reserved for any model artifact that you later choose to copy into the project.

## Run Figure 1(a)

Weights & Biases is part of the authors' result pipeline. Log in before launching:

```bash
wandb login
python scripts/run_fig1a.py --wandb-entity YOUR_WANDB_ENTITY --gpus 0 1 2 3
```

Each of the eight strategies is run with four seeds, for 32 active-learning experiments in
total. The launcher delegates training to the pinned official source. It does not start unless
`--wandb-entity` and at least one CUDA GPU ID are supplied.

After every run finishes:

```bash
python scripts/plot_fig1a.py --wandb-entity YOUR_WANDB_ENTITY
```

This writes a single `results/figure1a.png`. Raw histories remain in Weights & Biases, so no
large collection of local CSV or JSON files is generated.

## Provenance and limitations

- Paper: Figure 1 and Sections 4.1-4.3 of the supplied LabelBench PDF.
- Code: <https://github.com/EfficientTraining/LabelBench>.
- Exact dependency versions and the historical pretrained-weight artifact were not locked by
  the authors. The requirement list follows their repository, while the launcher pins the
  LabelBench code commit.
- Figure 1(b,c) remains out of scope until ImageNet is supplied locally.
