#!/usr/bin/env python3
"""Plot Figure 1(a) from results/experiments.sqlite."""
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ALL=["random","confidence","entropy","margin","coreset","galaxy","badge","bait"]
def main():
 path=ROOT/"results/experiments.sqlite"
 if not path.exists():raise RuntimeError("no local experiment database")
 db=sqlite3.connect(path);complete=set(db.execute("SELECT strategy,seed FROM runs WHERE status='complete'"));fig,ax=plt.subplots(figsize=(7.2,5))
 for strategy in ALL:
  seeds=sorted(seed for name,seed in complete if name==strategy)
  if len(seeds)!=4:raise RuntimeError(f"{strategy}: found {len(seeds)}/4 complete trials")
  trials=[]
  for seed in seeds:
   rows=db.execute("SELECT labels,test_accuracy FROM metrics WHERE strategy=? AND seed=? ORDER BY round",(strategy,seed)).fetchall();labels=np.array([x for x,_ in rows]);trials.append(np.maximum.accumulate([y for _,y in rows]))
  values=np.array(trials);mean=values.mean(0);se=values.std(0,ddof=1)/2;ax.plot(labels,mean,label=strategy.upper());ax.fill_between(labels,mean-se,mean+se,alpha=.22)
 db.close();ax.set(xlabel="Number of Labels",ylabel="Test Accuracy",ylim=(.960,.982));ax.grid(True,linestyle="--",alpha=.6);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(ROOT/"results/figure1a.png",dpi=220);plt.close(fig)
if __name__=="__main__":main()
