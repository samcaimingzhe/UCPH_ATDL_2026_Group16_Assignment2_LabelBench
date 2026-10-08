#!/usr/bin/env python3
"""Run Figure 1(a): eight strategies, four seeds, local SQLite logging."""
import argparse, os, subprocess, sys
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALL=["random","confidence","entropy","margin","coreset","galaxy","badge","bait"]
SEEDS=[1234,3334155,6667077,9999999]
def database_path(output_dir=None):
 output=Path(output_dir).resolve() if output_dir is not None else ROOT/"results"
 preferred=output/"figure1a.sqlite"
 legacy=output/"experiments.sqlite"
 return preferred if preferred.is_file() or not legacy.is_file() else legacy
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--gpus",nargs="+",type=int,required=True)
 p.add_argument("--strategies",nargs="+",choices=ALL,default=ALL)
 p.add_argument("--skip",type=int,default=0)
 p.add_argument("--output-dir",type=Path,help="directory for Figure 1(a) database and plot")
 a=p.parse_args()
 output=(a.output_dir or ROOT/"results").resolve()
 print(f"Figure 1(a) database: {database_path(output)}",flush=True)
 waiting=deque([(s,n) for s in a.strategies for n in SEEDS][a.skip:])
 while waiting:
  batch=[]
  try:
   for gpu in a.gpus:
    if not waiting:break
    strategy,seed=waiting.popleft()
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["LABELBENCH_MODEL_DIR"]=str(ROOT/"model")
    cmd=[sys.executable,str(ROOT/"scripts/train_one.py"),"--strategy",strategy,"--seed",str(seed),"--output-dir",str(output)]
    print(
     "start strategy={} seed={} gpu={}".format(strategy, seed, gpu),
     flush=True
     )
    batch.append((subprocess.Popen(cmd,cwd=ROOT,env=env),gpu,strategy,seed))
   for process,gpu,strategy,seed in batch:
    code=process.wait()
    if code:
     raise SystemExit(
         "failed: strategy={} seed={} gpu={} exit={}".format(
             strategy, seed, gpu, code
         )
     )
  finally:
   for process,_,_,_ in batch:
    if process.poll() is None:
     process.terminate()
   for process,_,_,_ in batch:
    process.wait()
if __name__=="__main__":main()
