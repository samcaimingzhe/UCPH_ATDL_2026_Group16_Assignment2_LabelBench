#!/usr/bin/env python3
"""Run Figure 1(a) CIFAR-10 or Figure 1(b/c) ImageNet experiments."""
import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from fig1_common import ROOT, STRATEGIES, database_path, settings
from prepare_assets import detect_dataset, prepare


def main(panel=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if panel is None:
        parser.add_argument("--panel", choices=list("abc"),
                            help="a=CIFAR-10; b/c=the same ImageNet training, different plot metrics")
    parser.add_argument("--dataset", choices=["auto", "cifar10", "imagenet"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chosen_panel = panel or args.panel
    try:
        selected = detect_dataset(args.data_dir, args.dataset)
    except ValueError as exc:
        if chosen_panel is None:
            parser.error(str(exc))
        selected = "cifar10" if chosen_panel == "a" else "imagenet"
    if chosen_panel is not None and (selected == "cifar10") != (chosen_panel == "a"):
        parser.error("panel a needs CIFAR-10; panels b/c need ImageNet")
    cfg = settings(selected)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    if len(set(args.strategies)) != len(args.strategies) or len(set(seeds)) != len(seeds):
        parser.error("strategies and seeds must be unique")
    if args.skip < 0:
        parser.error("--skip must be nonnegative")
    output = (args.output_dir or cfg["output"]).resolve()
    waiting = deque([(s, seed) for s in args.strategies for seed in seeds][args.skip:])
    figure = f"1({chosen_panel or cfg['panel']})"
    print(f"Figure {figure}: {selected}; {len(waiting)} jobs; "
          f"{cfg['rounds']} rounds x {cfg['batch']} labels", flush=True)
    print(f"Database: {database_path(cfg, output)}", flush=True)
    if args.dry_run:
        print("Jobs:", list(waiting))
        return

    # Prepare once before concurrent workers access the same dataset/model files.
    prepare(args.data_dir, selected)
    worker = ROOT / "scripts/train_fig1_one.py"
    while waiting:
        processes = []
        try:
            for gpu in args.gpus:
                if not waiting:
                    break
                strategy, seed = waiting.popleft()
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["LABELBENCH_MODEL_DIR"] = str(ROOT / "model")
                cmd = [sys.executable, str(worker), "--dataset", selected,
                       "--strategy", strategy, "--seed", str(seed),
                       "--data-dir", str(args.data_dir), "--output-dir", str(output)]
                print(f"start strategy={strategy} seed={seed} gpu={gpu}", flush=True)
                processes.append((subprocess.Popen(cmd, cwd=ROOT, env=env), strategy, seed, gpu))
            for process, strategy, seed, gpu in processes:
                code = process.wait()
                if code:
                    raise RuntimeError(f"failed strategy={strategy} seed={seed} gpu={gpu} exit={code}")
        finally:
            for process, *_ in processes:
                if process.poll() is None:
                    process.terminate()
            for process, *_ in processes:
                process.wait()


if __name__ == "__main__":
    main()
