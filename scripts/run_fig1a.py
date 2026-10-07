#!/usr/bin/env python3
"""Run Figure 1(a): eight strategies, four seeds, local SQLite logging."""
import argparse, os, subprocess, sys
from collections import deque
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ALL=["random","confidence","entropy","margin","coreset","galaxy","badge","bait"]
SEEDS=np.linspace(1234,9999999,num=4,dtype=int).tolist()
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--gpus",nargs="+",type=int,required=True)
 p.add_argument("--strategies",nargs="+",choices=ALL,default=ALL)
 p.add_argument("--skip",type=int,default=0);a=p.parse_args()
 waiting=deque([(s,n) for s in a.strategies for n in SEEDS][a.skip:])
 while waiting:
  batch=[]
  for gpu in a.gpus:
   if not waiting:break
   strategy,seed=waiting.popleft();env=os.environ.copy()
   env["CUDA_VISIBLE_DEVICES"]=str(gpu)
   env["LABELBENCH_MODEL_DIR"]=str(ROOT/"model")
   cmd=[sys.executable,str(ROOT/"scripts/train_one.py"),"--strategy",strategy,"--seed",str(seed)]
   print("start strategy={} seed={} gpu={}".format(strategy, seed, gpu),flush=True)
   batch.append((subprocess.Popen(cmd,cwd=ROOT,env=env),gpu,strategy,seed))
  for process,gpu,strategy,seed in batch:
   code=process.wait()
   if code:raise SystemExit(
    "failed: strategy={} seed={} gpu={} exit={}".format(strategy, seed, gpu, code)
    )
if __name__=="__main__":main()
