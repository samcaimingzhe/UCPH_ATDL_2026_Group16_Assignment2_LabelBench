# Proxy architecture ablation

This extension studies proxy architecture for CIFAR-10
Figure 5(a)-style experiments using Margin sampling.

## Reported experiments

- baseline: one hidden layer with width 512
- linear: linear classifier without a hidden layer
- h128: one hidden layer with width 128
- h1024: one hidden layer with width 1024

All four configurations use seeds 1234, 3334155,
6667077, and 9999999, with label budgets from 1,000
to 10,000 in increments of 1,000.

The configuration helper also supports augmentation-count
and unlabeled-ratio variants. These are not part of the
reported architecture ablation.

## Environment

Use Python 3.10. Install the matching CUDA-enabled
PyTorch and torchvision packages first:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r deploy/requirements-cifar10.txt
python -m pip check
python deploy/check_runtime.py
```

In a fresh checkout, apply the Hendrix resource profile
before configuring any ablation or creating feature caches:

```bash
python deploy/configure_runtime.py
python scripts/prepare_assets.py --dataset cifar10
mkdir -p work logs
printf 'LB_PYTHON=%q\n' "$(command -v python)" > work/hendrix.env
```

Run these commands in the intended Python environment.
The resource profile uses four DataLoader workers and
a feature-extraction batch size of 128.

## Running on Slurm

Submit from the repository root. Configure baseline first:

```bash
python scripts/configure_ablation.py baseline \
  --output-dir results/ablation/baseline

sbatch --wait deploy/hendrix_job.sbatch check
sbatch --wait deploy/hendrix_job.sbatch features baseline
sbatch --wait deploy/hendrix_job.sbatch trial baseline 1234
```

Override GPU type, node, and time limit using the resources
available on your cluster. Complete all four seeds, then
repeat for linear, h128, and h1024, configuring each variant
before submission.

Run jobs sequentially. Shared JSON configurations must not
change while a job is pending or running.

Feature caches are reused from results/fig5/features/.
Each trial runs proxy selection followed by downstream
full CLIP evaluation.

Completed phases are skipped. An incomplete phase restarts
from its first labeling budget; epoch-level or round-level
checkpoint resumption is not implemented.

Manifests record the Git commit and source/configuration
hashes. Preserve the original checkout for resuming an
existing experiment. Use new output directories when the
recorded code or configuration changes.

## Analysis

Use scripts/plot_ablation.py with --phase evaluate for the
main downstream comparison and --phase selection for proxy
diagnostics. Run the script with --help for its arguments.

Plots show raw mean accuracy and SEM over paired seeds,
without cumulative-maximum smoothing. Normalized AULC
summarizes the learning curve over the label-budget range.

Recorded phase elapsed time includes training, validation,
prediction, and loop overhead. It excludes feature-cache
precomputation and is not a pure acquisition-time measure.
