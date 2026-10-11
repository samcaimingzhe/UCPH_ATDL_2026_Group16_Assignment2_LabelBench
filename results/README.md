# Figure 6(c) FlexMatch — Single-Seed Reproduction Artifacts

## Experiment
- Paper: Jifan Zhang et al., *LabelBench: A Comprehensive Framework for Benchmarking Adaptive Label-Efficient Learning* (2024), Figure 6(c).
- Dataset: CIFAR-10.
- Semi-supervised trainer: FlexMatch.
- Pretrained model: CLIP ViT-B/32.
- Acquisition strategies: Random, Confidence, Entropy, Margin, Coreset, Galaxy, BADGE, and BAIT.
- Label budgets: 1,000 to 10,000 labels, increasing by 1,000 per round.
- Metric: Test Accuracy.
- Seed: 1234 (one trial).
- Compute: Kaggle Notebook with two NVIDIA Tesla T4 GPUs.

## Main result
`figure6c_flexmatch_single_seed.png` is a single-trial line plot with no standard-error bands. `summary_single_seed.csv` stores the ten test/pool accuracy points for each strategy. `figure6_flexmatch.sqlite` stores the consolidated metrics used for plotting. The plotting code can recreate the figure from the CSV.

At 10,000 labels, the single-trial test accuracies are: Galaxy 98.18%; Margin 98.00%; BADGE 98.00%; BAIT 98.00%; Random 97.94%; Entropy 97.84%; Coreset 97.78%; Confidence 97.70%.

## Implementation adjustments
- Label-cache loading uses `torch.load(..., weights_only=False)` for compatibility with the Kaggle PyTorch version.
- DataLoader workers were reduced from 20 to 2 to fit the Kaggle runtime.
- The Figure 6 FlexMatch worker includes a device-compatibility adjustment for pseudo-label bookkeeping.

These adjustments are intended for runtime compatibility, but exact implementation equivalence is not claimed.

## Important data provenance / limitations
- The paper reports a mean over three trials with standard-error intervals; this package contains only one trial (seed 1234), so it does **not** reproduce the paper's three-trial estimate or error bands.
- The original SQLite database from the first four runs was lost after a Kaggle session reset. The Random, Confidence, Entropy, and Margin metrics were reconstructed from the saved interleaved console log. Their strategy-to-curve mapping was inferred from the recorded launch order and alternating round-level metric outputs; it could not be independently checked against the original database.
- Coreset, Galaxy, BADGE, and BAIT were recorded in the later run and stored in the SQLite database.
- Treat these as partial single-seed reproduction results, not an exact statistical replication.

## Replot
Requires Python and Matplotlib:

```bash
python plot_figure6c_from_csv.py
```

Reference implementation branch: `feat/figure6-flexmatch` in the group repository.
