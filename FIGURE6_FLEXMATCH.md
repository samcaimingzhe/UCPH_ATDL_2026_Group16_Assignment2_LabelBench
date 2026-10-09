# Figure 6(c): FlexMatch panel

This patch adds a dedicated pipeline for the FlexMatch panel of Figure 6 to the team's CIFAR-10 LabelBench reproduction repository.

## Scope

- Dataset: CIFAR-10
- Model: CLIP ViT-B/32
- Semi-supervised trainer: FlexMatch
- Active-learning strategies: Random, Confidence, Entropy, Margin, CORESET, GALAXY, BADGE, BAIT
- Label budgets: 1,000 to 10,000 (1,000 labels added at each round)
- Trials: 3
- Metric: test/generalization accuracy

The paper says Figure 6 averages three trials but does not give their exact seed values. Defaults follow the team's existing seed endpoints, yielding 1234, 5000616, and 9999999. Report those as the seeds used; do not claim they are the original authors' exact seeds.

## Smoke test (do this first)

Run only the Random strategy, one seed, and two rounds. This checks dataset/model setup, one training cycle, active selection for the second round, and metric/database writing. It is a diagnostic run, not a final reproduction result.

```bash
python scripts/run_fig6_flexmatch.py \
  --gpus 0 \
  --strategies random \
  --seeds 1234 \
  --rounds 2 \
  --output-dir results/fig6/flexmatch_smoke
```

## Full Figure 6(c) run

Only after the smoke test succeeds, run all eight active-learning strategies for all three trials and all ten label budgets:

```bash
python scripts/run_fig6_flexmatch.py \
  --gpus 0 \
  --output-dir results/fig6/flexmatch
```

A single Colab T4 will run jobs sequentially; do not pass `--gpus 0 1` because this runtime has only one GPU.

## Plot, after all jobs complete

```bash
python scripts/plot_fig6_flexmatch.py --output-dir results/fig6/flexmatch
```

The plot script uses raw mean accuracies and standard errors by default. The optional `--smoothing max` setting applies a monotone post-processing operation and should not be enabled for the primary reproduction plot unless explicitly justified and reported.

## Reproducibility record

Save the Git commit hash, package versions, GPU info, smoke/full commands, stdout/stderr logs, SQLite database, summary CSV, and PNG. Colab's `/content` storage is temporary, so copy final artifacts to Google Drive before the runtime ends.
