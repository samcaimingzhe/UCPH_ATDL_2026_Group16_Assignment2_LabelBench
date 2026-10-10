#!/usr/bin/env python3
"""Run one Figure 1 strategy/seed and save metrics locally."""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LabelBench.corrupter.corrupter import get_corrupter_fn
from LabelBench.dataset.datasets import get_dataset
from LabelBench.metric.metrics import get_metric
from LabelBench.model.model import get_model_fn
from LabelBench.strategy.strategies import get_strategy
from LabelBench.trainer.trainer import (
    get_fns,
    get_optimizer_fn,
    get_scheduler_fn,
    get_trainer,
)
from fig1_common import STRATEGIES, database_path, settings

FILES = {strategy: f"{strategy}_sampling.json" for strategy in STRATEGIES}


def load(path):
    with open(path) as f:
        return json.load(f)


def connect(cfg, output_dir=None):
    p = database_path(cfg, output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(p, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS runs("
        "strategy TEXT, seed INTEGER, status TEXT, started REAL, finished REAL, "
        "PRIMARY KEY(strategy, seed))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS metrics("
        "strategy TEXT, seed INTEGER, round INTEGER, labels INTEGER, "
        "test_accuracy REAL, pool_accuracy REAL, payload TEXT, "
        "PRIMARY KEY(strategy, seed, round))"
    )
    db.commit()
    return db


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    p.add_argument("--strategy", choices=FILES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--data-dir", type=Path, required=True)
    a = p.parse_args()

    cfg = settings(a.dataset)
    output = (a.output_dir or cfg["output"]).resolve()
    run(a, cfg, output)


def run(a, cfg, output):
    db = connect(cfg, output)
    old = db.execute(
        "SELECT status FROM runs WHERE strategy=? AND seed=?",
        (a.strategy, a.seed),
    ).fetchone()

    if old and old[0] == "complete":
        print("already complete")
        db.close()
        return

    db.execute(
        "INSERT OR REPLACE INTO runs VALUES(?,?,'running',?,NULL)",
        (a.strategy, a.seed, time.time()),
    )
    db.execute(
        "DELETE FROM metrics WHERE strategy=? AND seed=?",
        (a.strategy, a.seed),
    )
    db.commit()

    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(a.seed + 42)

    mc = load(ROOT / "configs/model/clip_ViTB32_pretrained.json")
    tc = load(cfg["trainer_config"])
    sc = load(ROOT / "configs/strategy" / FILES[a.strategy])
    cc = load(ROOT / "configs/corrupter/noiseless.json")

    data = get_dataset(a.dataset, str(a.data_dir.resolve()))
    mc["num_output"] = data.get_num_classes()
    data.set_corrupter(get_corrupter_fn(cc["name"], cc))

    tc = get_scheduler_fn(get_optimizer_fn(get_fns(tc)))
    metric = get_metric("multi_class")
    trainer = get_trainer(
        tc["trainer_name"],
        tc,
        data,
        get_model_fn(mc["model_name"]),
        mc,
        metric,
        None,
    )
    strategy = get_strategy(sc["strategy_name"], sc, data)

    try:
        for r in range(1, cfg["rounds"] + 1):
            if r == 1:
                selected = np.random.choice(
                    np.arange(len(data)), cfg["batch"], replace=False
                )
            else:
                selected = strategy.select(trainer, cfg["batch"])

            data.update_labeled_idxs(selected)
            model = trainer.train()
            trainer.evaluate_on_train(model)
            trainer.evaluate_on_val(model)
            trainer.evaluate_on_test(model)

            v = trainer.compute_metric(r)
            payload = json.dumps(v, default=lambda x: x.item())

            db.execute(
                "INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?,?)",
                (
                    a.strategy,
                    a.seed,
                    r,
                    int(v["Num Labeled"]),
                    float(v["Test Accuracy"]),
                    float(v["Pool Accuracy"]),
                    payload,
                ),
            )
            db.commit()
            print(payload, flush=True)

        db.execute(
            "UPDATE runs SET status='complete', finished=? "
            "WHERE strategy=? AND seed=?",
            (time.time(), a.strategy, a.seed),
        )
        db.commit()

    except BaseException:
        db.execute(
            "UPDATE runs SET status='failed', finished=? "
            "WHERE strategy=? AND seed=?",
            (time.time(), a.strategy, a.seed),
        )
        db.commit()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
