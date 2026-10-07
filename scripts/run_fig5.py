#!/usr/bin/env python3
"""Launch Figure 5 training only; plotting is a separate command."""
import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys

from fig5_common import ROOT, SEEDS, database_path, settings


def main(panel=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if panel is None:
        parser.add_argument("--panel", choices=list("abc"), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--phase", choices=["all", "selection", "evaluate"], default="all",
                        help="a/b: collect proxy labels, evaluate saved labels, or both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="save final-round weights (large for CLIP)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without training or creating output files")
    args = parser.parse_args()
    cfg = settings(panel or args.panel)
    strategies = args.strategies or cfg["strategies"]
    if not set(strategies) <= set(cfg["strategies"]):
        parser.error(f"supported strategies: {', '.join(cfg['strategies'])}")
    if len(set(strategies)) != len(strategies) or len(set(args.seeds)) != len(args.seeds):
        parser.error("strategies and seeds must be unique")
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    if cfg["panel"] == "c" and args.phase != "all":
        parser.error("5(c) uses one supervised selection/training loop; use --phase all")
    output = (args.output_dir or cfg["output"]).resolve()
    print(f"Figure 5({cfg['panel']}): {len(strategies)} methods x {len(args.seeds)} seeds; "
          f"{cfg['rounds']} rounds x {cfg['batch']} labels", flush=True)
    print(f"Database: {database_path(cfg['panel'], output)}", flush=True)
    worker = ROOT / "scripts" / "train_fig5_one.py"

    def command(strategy, seed, phase):
        cmd = [sys.executable, str(worker), "--panel", cfg["panel"],
               "--strategy", strategy, "--seed", str(seed), "--phase", phase,
               "--output-dir", str(output)]
        if args.save_checkpoints:
            cmd.append("--save-checkpoints")
        return cmd

    # Complete feature extraction on a single GPU before concurrent workers read
    # the shared cache; avoids writing the same embedding file from two jobs.
    phases = [args.phase]
    if cfg["panel"] in "ab" and args.phase == "all":
        phases = ["selection", "evaluate"]
    if args.dry_run:
        print(f"Seeds: {args.seeds}; strategies: {strategies}; phases: {phases}")
        print("Worker:", " ".join(command(strategies[0], args.seeds[0], phases[0])))
        return

    def environment(gpu):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["LABELBENCH_MODEL_DIR"] = str(ROOT / "model")
        return env

    if "selection" in phases:
        subprocess.run(command(strategies[0], args.seeds[0], "features"),
                       cwd=ROOT, env=environment(args.gpus[0]), check=True)
    for phase in phases:
        waiting = deque((s, n) for s in strategies for n in args.seeds)
        while waiting:
            processes = []
            try:
                for gpu in args.gpus:
                    if not waiting:
                        break
                    strategy, seed = waiting.popleft()
                    print(f"start {phase}: strategy={strategy} seed={seed} gpu={gpu}", flush=True)
                    cmd = command(strategy, seed, phase)
                    processes.append((subprocess.Popen(cmd, cwd=ROOT, env=environment(gpu)), cmd))
                for process, cmd in processes:
                    if process.wait():
                        raise RuntimeError("training failed: " + " ".join(cmd))
            finally:
                for process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                for process, _ in processes:
                    process.wait()


if __name__ == "__main__":
    main()
