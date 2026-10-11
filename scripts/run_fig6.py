#!/usr/bin/env python3
"""Run Figure 6 with a selectable implemented trainer (CIFAR-10, batch size 1000).

Results are isolated by trainer and use three trials.
"""
import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALL = ["random", "confidence", "entropy", "margin", "coreset", "galaxy", "badge", "bait"]
TRAINERS = ("flexmatch", "softmatch", "freematch", "uda", "pseudolabel", "passive")
# The paper states three trials but does not report the exact seeds.  The team
# repository already uses endpoints 1234 and 9999999 with np.linspace for its
# existing seed convention; for three trials this gives these seeds.
DEFAULT_SEEDS = [1234, 3334155, 6667077]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--trainer", choices=TRAINERS, default="flexmatch")
    parser.add_argument("--strategies", nargs="+", choices=ALL, default=ALL)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to results/fig6/<trainer>.")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip this many strategy/seed jobs from the start of the plan.")
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=10,
                        help="Number of label-budget rounds (use 2 for a smoke test; 10 for Figure 6).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategies = args.strategies
    seeds = args.seeds
    if len(set(strategies)) != len(strategies):
        parser.error("strategies must be unique")
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    if any(g < 0 for g in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique and nonnegative")

    output = (args.output_dir or ROOT / "results" / "fig6" / args.trainer).resolve()
    waiting = deque((strategy, seed) for strategy in strategies for seed in seeds)
    for _ in range(args.skip):
        if waiting:
            waiting.popleft()

    worker = Path(__file__).with_name("train_fig6_one.py")
    print(f"Figure 6 ({args.trainer}): {len(strategies)} strategies x {len(seeds)} seeds", flush=True)
    print(f"Strategies: {strategies}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Rounds: {args.rounds}; labels per round: 1000", flush=True)
    print(f"Output: {output}", flush=True)

    def command(strategy: str, seed: int):
        return [
            sys.executable,
            str(worker),
            "--trainer", args.trainer,
            "--strategy", strategy,
            "--seed", str(seed),
            "--output-dir", str(output),
            "--rounds", str(args.rounds),
        ]

    if args.dry_run:
        if waiting:
            print("First command:", " ".join(command(*waiting[0])))
        return

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
                cmd = command(strategy, seed)
                print(f"start strategy={strategy} seed={seed} gpu={gpu}", flush=True)
                processes.append((
                    subprocess.Popen(cmd, cwd=ROOT, env=env),
                    cmd,
                    strategy,
                    seed,
                    gpu,
                ))

            for process, cmd, strategy, seed, gpu in processes:
                code = process.wait()
                if code:
                    raise SystemExit(
                        f"failed: strategy={strategy} seed={seed} gpu={gpu} exit={code}"
                    )
        finally:
            for process, *_ in processes:
                if process.poll() is None:
                    process.terminate()
            for process, *_ in processes:
                process.wait()


if __name__ == "__main__":
    main()
