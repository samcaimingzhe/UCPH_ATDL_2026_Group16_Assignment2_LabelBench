#!/usr/bin/env python3
"""Run one Figure 6(c) FlexMatch strategy/seed and save raw metrics to SQLite."""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILES = {
    s: f"{s}_sampling.json"
    for s in ["random", "confidence", "entropy", "margin", "coreset", "galaxy", "badge", "bait"]
}
ROUNDS = 10
BATCH = 1000


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def connect(output: Path, rounds: int) -> sqlite3.Connection:
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "figure6_flexmatch.sqlite", timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
            strategy TEXT, seed INTEGER, status TEXT, started REAL, finished REAL,
            PRIMARY KEY(strategy, seed));
        CREATE TABLE IF NOT EXISTS metrics(
            strategy TEXT, seed INTEGER, round INTEGER, labels INTEGER,
            test_accuracy REAL, pool_accuracy REAL, payload TEXT,
            PRIMARY KEY(strategy, seed, round));
        """
    )
    for key, value in [("panel", "6c"), ("trainer", "flexmatch"),
                       ("dataset", "cifar10"), ("batch", str(BATCH)),
                       ("rounds", str(rounds))]:
        db.execute("INSERT OR IGNORE INTO metadata VALUES(?, ?)", (key, value))
        actual = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0]
        if actual != value:
            db.close()
            raise RuntimeError(f"output database contains a different {key}: {actual}")
    db.commit()
    return db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=FILES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "fig6" / "flexmatch")
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=10)
    args = parser.parse_args()

    db = connect(args.output_dir.resolve(), args.rounds)
    old = db.execute(
        "SELECT status FROM runs WHERE strategy=? AND seed=?",
        (args.strategy, args.seed),
    ).fetchone()
    if old and old[0] == "complete":
        print(f"already complete: {args.strategy}, seed {args.seed}")
        db.close()
        return

    db.execute(
        "INSERT OR REPLACE INTO runs VALUES(?,?,'running',?,NULL)",
        (args.strategy, args.seed, time.time()),
    )
    db.execute(
        "DELETE FROM metrics WHERE strategy=? AND seed=?",
        (args.strategy, args.seed),
    )
    db.commit()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed + 42)

    mc = load(ROOT / "configs/model/clip_ViTB32_pretrained.json")
    tc = load(ROOT / "configs/trainer/cifar10/flexmatch/clip_ViTB32_finetune.json")
    sc = load(ROOT / "configs/strategy" / FILES[args.strategy])
    cc = load(ROOT / "configs/corrupter/noiseless.json")

    from LabelBench.corrupter.corrupter import get_corrupter_fn
    from LabelBench.dataset.datasets import get_dataset
    from LabelBench.metric.metrics import get_metric
    from LabelBench.model.model import get_model_fn
    from LabelBench.strategy.strategies import get_strategy
    from LabelBench.trainer.trainer import get_fns, get_optimizer_fn, get_scheduler_fn, get_trainer
    from LabelBench.trainer.trainer_impl.flexmatch import FlexmatchTrainer

    # Match the CPU-side pseudo-label bookkeeping used by this team's Figure 5 worker.
    # The official trainer moves idx_u to CUDA but stores uhat/sigma on CPU, which can
    # cause a device-indexing error on some PyTorch builds.
    class Figure6Flexmatch(FlexmatchTrainer):
        trainer_name = "figure6_flexmatch"

        def train_step(self, model, img_l, target_l, class_weights, loss_fn,
                       idx_u, img_uw, img_us, iteration):
            n_l, n_u = len(img_l), len(img_uw)
            logits = model(torch.cat([img_l, img_uw, img_us]).cuda(), ret_features=False)
            logits = logits.squeeze(-1)
            logits_l = logits[:n_l]
            logits_uw = logits[n_l:n_l + n_u]
            logits_us = logits[-n_u:]
            supervised_loss = loss_fn(logits_l, target_l.cuda())
            with torch.no_grad():
                beta = self.sigma[:-1] / self.sigma.max()
                confidence, pseudo = F.softmax(logits_uw.detach().cpu().float(), dim=-1).max(dim=-1)
                mask = confidence.ge(self.trainer_config["p_cutoff"] *
                                     beta[pseudo] / (2. - beta[pseudo]))
                accepted = confidence.ge(self.trainer_config["p_cutoff"])
                self.uhat[idx_u.cpu()[accepted]] = pseudo[accepted]
                for c in range(self.dataset.get_num_classes()):
                    self.sigma[c] = self.uhat[self.unlabeled_idxs].eq(c).sum()
                self.sigma[-1] = self.n_ul - self.sigma[:-1].sum()
            unsupervised_loss = F.cross_entropy(logits_us, pseudo.cuda(), reduction="none")
            return supervised_loss + self.trainer_config["ulb_loss_ratio"] * (unsupervised_loss * mask.cuda()).mean()

    data = get_dataset("cifar10", str(ROOT / "data"))
    mc["num_output"] = data.get_num_classes()
    data.set_corrupter(get_corrupter_fn(cc["name"], cc))
    tc["trainer_name"] = "figure6_flexmatch"
    tc = get_scheduler_fn(get_optimizer_fn(get_fns(tc)))
    metric = get_metric("multi_class")
    trainer = get_trainer(
        tc["trainer_name"], tc, data, get_model_fn(mc["model_name"]), mc, metric, None
    )
    strategy = get_strategy(sc["strategy_name"], sc, data)

    try:
        for r in range(1, args.rounds + 1):
            if r == 1:
                selected = np.random.choice(np.arange(len(data)), BATCH, replace=False)
            else:
                selected = strategy.select(trainer, BATCH)
            data.update_labeled_idxs(selected)

            model = trainer.train()
            trainer.evaluate_on_train(model)
            trainer.evaluate_on_val(model)
            trainer.evaluate_on_test(model)
            values = trainer.compute_metric(r)
            payload = json.dumps(values, default=lambda x: x.item())
            db.execute(
                "INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?,?)",
                (
                    args.strategy,
                    args.seed,
                    r,
                    int(values["Num Labeled"]),
                    float(values["Test Accuracy"]),
                    float(values["Pool Accuracy"]),
                    payload,
                ),
            )
            db.commit()
            print(payload, flush=True)

        db.execute(
            "UPDATE runs SET status='complete', finished=? WHERE strategy=? AND seed=?",
            (time.time(), args.strategy, args.seed),
        )
        db.commit()
    except BaseException:
        db.execute(
            "UPDATE runs SET status='failed', finished=? WHERE strategy=? AND seed=?",
            (time.time(), args.strategy, args.seed),
        )
        db.commit()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
