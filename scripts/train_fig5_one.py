#!/usr/bin/env python3
"""Train one Figure 5 method/seed; record raw metrics and selected indices."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

from fig5_common import ROOT, database_path, settings


def load(path):
    return json.loads(Path(path).read_text())


def connect(output, cfg):
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database_path(cfg["panel"], output), timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
            strategy TEXT, seed INTEGER, phase TEXT, status TEXT, started REAL, finished REAL,
            PRIMARY KEY(strategy,seed,phase));
        CREATE TABLE IF NOT EXISTS metrics(
            strategy TEXT, seed INTEGER, phase TEXT, round INTEGER, labels INTEGER,
            test_accuracy REAL, pool_accuracy REAL, payload TEXT,
            PRIMARY KEY(strategy,seed,phase,round));
        CREATE TABLE IF NOT EXISTS selections(
            strategy TEXT, seed INTEGER, round INTEGER, indices TEXT,
            PRIMARY KEY(strategy,seed,round));
    """)
    for key, value in [("panel", cfg["panel"]), ("batch", str(cfg["batch"])),
                       ("rounds", str(cfg["rounds"]))]:
        db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, value))
        actual = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0]
        if actual != value:
            db.close()
            raise RuntimeError("output database contains a different experiment configuration")
    db.commit()
    return db


@contextmanager
def job_lock(output, strategy, seed):
    locks = output / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{strategy}_{seed}.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another worker is running {strategy}, seed {seed}") from None
        yield


def complete(db, strategy, seed, phase):
    return db.execute("SELECT status FROM runs WHERE strategy=? AND seed=? AND phase=?",
                      (strategy, seed, phase)).fetchone() == ("complete",)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed + 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_runtime():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("LABELBENCH_MODEL_DIR", str(ROOT / "model"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Figure 5 training uses CUDA in the existing LabelBench trainers. "
                           "Use an NVIDIA GPU host; --dry-run and plotting work without CUDA.")
    # Register the additional proxy model without modifying Figure 1 imports.
    import LabelBench.model.model_impl.shallow  # noqa: F401


def get_data():
    from LabelBench.corrupter.corrupter import get_corrupter_fn
    from LabelBench.dataset.datasets import get_dataset
    data = get_dataset("cifar10", str(ROOT / "data"))
    cc = load(ROOT / "configs/corrupter/noiseless.json")
    data.set_corrupter(get_corrupter_fn(cc["name"], cc))
    return data


def make_extractor():
    import numpy as np
    import torch
    from LabelBench.dataset.feature_extractor import FeatureExtractor
    from LabelBench.model.model import get_model_fn

    class TensorCacheExtractor(FeatureExtractor):
        """Cache tensors, so torch.load can use weights_only=True on modern PyTorch."""
        def get_feature(self, dataset, dataset_split, epoch, use_strong):
            seed = epoch % self.num_transform_seeds if dataset_split == "train" else 0
            strong = use_strong and dataset_split == "train"
            name = f"{self.file_name}_{seed}_{'strong' if strong else dataset_split}.pt"
            path = Path(name)
            # Even direct worker invocations safely share this cache.
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if path.exists():
                    tensors = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    np_state, py_state = np.random.get_state(), random.getstate()
                    try:
                        with torch.random.fork_rng():
                            torch.manual_seed(seed)
                            random.seed(seed)
                            features = self.get_feature_helper(dataset, strong, dataset_split, seed)
                        tensors = (tuple(torch.as_tensor(x) for x in features)
                                   if isinstance(features, tuple) else torch.as_tensor(features))
                        temporary = path.with_suffix(f".{os.getpid()}.tmp")
                        torch.save(tensors, temporary)
                        temporary.replace(path)
                    finally:
                        np.random.set_state(np_state)
                        random.setstate(py_state)
            if isinstance(tensors, tuple):
                return tuple(x.numpy() for x in tensors)
            return tensors.numpy()

    config = load(ROOT / "configs/embed_model/clip_ViTB32.json")
    config["num_output"] = 0
    cache = ROOT / "results/fig5/features"
    cache.mkdir(parents=True, exist_ok=True)
    return TensorCacheExtractor(get_model_fn(config["model_name"]),
                                str(cache / "cifar10_clip_vitb32"), config)


def precompute():
    import torch
    data, extractor = get_data(), make_extractor()
    for epoch in range(extractor.num_transform_seeds):
        extractor.get_feature(data.train_dataset, "train", epoch, False)
        extractor.get_feature(data.train_dataset, "train", epoch, True)
    for split in ["val", "test"]:
        extractor.get_feature(getattr(data, split + "_dataset"), split, 0, False)
    del data, extractor
    gc.collect()
    torch.cuda.empty_cache()


def make_trainer(data, panel, proxy=False):
    import torch
    import torch.nn.functional as F
    from LabelBench.metric.metrics import get_metric
    from LabelBench.model.model import get_model_fn
    from LabelBench.trainer.trainer import get_fns, get_optimizer_fn, get_scheduler_fn, get_trainer
    from LabelBench.trainer.trainer_impl.flexmatch import FlexmatchTrainer

    # Keep book-keeping indices on CPU, matching the official CPU pseudo-label
    # counters. Losses/logits remain on CUDA. Does not change Figure 1 trainers.
    class Fig5Flexmatch(FlexmatchTrainer):
        trainer_name = "figure5_flexmatch"

        def train_step(self, model, img_l, target_l, class_weights, loss_fn,
                       idx_u, img_uw, img_us, iteration):
            n_l, n_u = len(img_l), len(img_uw)
            logits = model(torch.cat([img_l, img_uw, img_us]).cuda(), ret_features=False)
            logits = logits.squeeze(-1)
            logits_l = logits[:n_l]
            logits_uw = logits[n_l:n_l + n_u]
            logits_us = logits[-n_u:]
            sup = loss_fn(logits_l, target_l.cuda())
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
            unsup = F.cross_entropy(logits_us, pseudo.cuda(), reduction="none")
            return sup + self.trainer_config["ulb_loss_ratio"] * (unsup * mask.cuda()).mean()

    if proxy:
        mc = load(ROOT / "configs/model/shallow.json")
        mc["input_dim"] = 512  # OpenAI CLIP ViT-B/32 output dimension.
        tc_path = ROOT / "configs/trainer/cifar10/flexmatch/clip_ViTB32_shallow.json"
        extractor = make_extractor()
    else:
        mc = load(ROOT / "configs/model/clip_ViTB32_pretrained.json")
        mode = "passive" if panel == "c" else "flexmatch"
        tc_path = ROOT / f"configs/trainer/cifar10/{mode}/clip_ViTB32_finetune.json"
        extractor = None
    mc["num_output"] = data.get_num_classes()
    tc = load(tc_path)
    if tc["trainer_name"] == "flexmatch":
        tc["trainer_name"] = "figure5_flexmatch"
    tc = get_scheduler_fn(get_optimizer_fn(get_fns(tc)))
    return get_trainer(tc["trainer_name"], tc, data, get_model_fn(mc["model_name"]), mc,
                       get_metric("multi_class"), extractor)


def evaluate(trainer, model, round_index):
    trainer.evaluate_on_train(model)
    trainer.evaluate_on_val(model)
    trainer.evaluate_on_test(model)
    return trainer.compute_metric(round_index)


def save_metric(db, args, phase, round_index, values):
    payload = json.dumps(values, default=lambda x: x.item())
    db.execute("INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?,?,?)",
               (args.strategy, args.seed, phase, round_index, int(values["Num Labeled"]),
                float(values["Test Accuracy"]), float(values["Pool Accuracy"]), payload))
    db.commit()
    print(f"{phase} {args.strategy} seed={args.seed}: {payload}", flush=True)


def save_indices(db, output, args, round_index, data, selected):
    import numpy as np
    db.execute("INSERT OR REPLACE INTO selections VALUES(?,?,?,?)",
               (args.strategy, args.seed, round_index, json.dumps([int(i) for i in selected])))
    folder = output / "selections"
    folder.mkdir(exist_ok=True)
    path = folder / f"{args.strategy}_seed{args.seed}.npy"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.save(file, data.labeled_idxs().astype(np.int64), allow_pickle=False)
    temporary.replace(path)
    db.commit()


def save_checkpoint(output, args, phase, model):
    import torch
    if args.save_checkpoints:
        folder = output / "checkpoints"
        folder.mkdir(exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.module.state_dict().items()},
                   folder / f"{args.strategy}_seed{args.seed}_{phase}_final.pt")


def run_phase(db, output, cfg, args, phase):
    import numpy as np
    import torch
    from LabelBench.strategy.strategies import get_strategy

    if complete(db, args.strategy, args.seed, phase):
        print(f"already complete: {phase} {args.strategy} seed={args.seed}", flush=True)
        return
    proxy = phase == "selection"
    saved = None
    if phase == "evaluate" and cfg["panel"] in "ab":
        if not complete(db, args.strategy, args.seed, "selection"):
            raise RuntimeError("run --phase selection before evaluating proxy-selected labels")
        rows = db.execute("SELECT round,indices FROM selections WHERE strategy=? AND seed=? ORDER BY round",
                          (args.strategy, args.seed)).fetchall()
        if [r[0] for r in rows] != list(range(1, cfg["rounds"] + 1)):
            raise RuntimeError("saved selection rounds are incomplete")
        saved = [json.loads(r[1]) for r in rows]
    set_seed(args.seed)
    data = get_data()
    # Label-cache construction may consume DataLoader RNG on the first run.
    # Reset before sampling/training so existing caches do not change a trial.
    set_seed(args.seed)
    trainer = make_trainer(data, cfg["panel"], proxy)
    sc = load(ROOT / f"configs/strategy/{args.strategy}_sampling.json")
    strategy = get_strategy(sc["strategy_name"], sc, data)
    db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,'running',?,NULL)",
               (args.strategy, args.seed, phase, time.time()))
    db.execute("DELETE FROM metrics WHERE strategy=? AND seed=? AND phase=?",
               (args.strategy, args.seed, phase))
    if saved is None:
        db.execute("DELETE FROM selections WHERE strategy=? AND seed=?", (args.strategy, args.seed))
    db.commit()
    try:
        for r in range(1, cfg["rounds"] + 1):
            if saved is not None:
                selected = np.asarray(saved[r - 1], dtype=int)
            elif r == 1:
                selected = np.random.choice(len(data), cfg["batch"], replace=False)
            else:
                selected = np.asarray(strategy.select(trainer, cfg["batch"]), dtype=int)
            if len(selected) != cfg["batch"] or len(np.unique(selected)) != len(selected):
                raise RuntimeError("selection returned an incorrect batch size or duplicate indices")
            if np.any(selected < 0) or np.any(selected >= len(data)):
                raise RuntimeError("selection returned an out-of-range index")
            if r > 1 and np.intersect1d(selected, data.labeled_idxs()).size:
                raise RuntimeError("selection includes previously labeled examples")
            data.update_labeled_idxs(selected)
            if saved is None:
                save_indices(db, output, args, r, data, selected)
            # Restart from pretrained CLIP / random proxy weights at each budget,
            # matching the official main.py and point_evaluation.py.
            model = trainer.train()
            values = evaluate(trainer, model, r)
            save_metric(db, args, phase, r, values)
            if r == cfg["rounds"]:
                save_checkpoint(output, args, phase, model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        db.execute("UPDATE runs SET status='complete',finished=? WHERE strategy=? AND seed=? AND phase=?",
                   (time.time(), args.strategy, args.seed, phase))
        db.commit()
    except BaseException:
        db.execute("UPDATE runs SET status='failed',finished=? WHERE strategy=? AND seed=? AND phase=?",
                   (time.time(), args.strategy, args.seed, phase))
        db.commit()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=list("abc"), required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=["all", "features", "selection", "evaluate"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()
    cfg = settings(args.panel)
    if args.strategy not in cfg["strategies"]:
        parser.error("unsupported strategy for this panel")
    if args.panel == "c" and args.phase not in ["all", "evaluate"]:
        parser.error("5(c) has no proxy selection or embedding phase")
    initialize_runtime()
    if args.phase == "features":
        set_seed(args.seed)
        precompute()
        return
    output = (args.output_dir or cfg["output"]).resolve()
    with job_lock(output, args.strategy, args.seed):
        db = connect(output, cfg)
        try:
            # Refuse to silently combine results produced with changed configs.
            paths = [ROOT / "configs/model/clip_ViTB32_pretrained.json",
                     ROOT / f"configs/strategy/{args.strategy}_sampling.json",
                     ROOT / "configs/corrupter/noiseless.json"]
            if args.panel in "ab":
                paths += [ROOT / "configs/model/shallow.json", ROOT / "configs/embed_model/clip_ViTB32.json",
                          ROOT / "configs/trainer/cifar10/flexmatch/clip_ViTB32_shallow.json",
                          ROOT / "configs/trainer/cifar10/flexmatch/clip_ViTB32_finetune.json"]
            else:
                paths += [ROOT / "configs/trainer/cifar10/passive/clip_ViTB32_finetune.json"]
            fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
            key = f"config_sha256_{args.strategy}"
            db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, fingerprint))
            if db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0] != fingerprint:
                raise RuntimeError("configs changed; use a new --output-dir to keep experiments separate")
            db.commit()
            phases = (["selection", "evaluate"] if args.phase == "all" and args.panel in "ab"
                      else ["evaluate"] if args.phase == "all" else [args.phase])
            for phase in phases:
                run_phase(db, output, cfg, args, phase)
        finally:
            db.close()


if __name__ == "__main__":
    main()
