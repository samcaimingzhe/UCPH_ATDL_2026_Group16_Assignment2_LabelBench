#!/usr/bin/env python3
"""New companion script, not part of the audited LabelBench repository.

Compare complete Figure 5(a/b) SQLite experiments using raw accuracies and
paired seeds. All database connections are read-only. SEM uses sample SD
(ddof=1) / sqrt(n); no cumulative-max smoothing is applied.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3

import numpy as np


DEFAULT_SEEDS = [1234, 3334155, 6667077, 9999999]


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(len(values))


def read_experiment(name, path, strategy, seeds, phase):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name}: database does not exist: {path}")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if metadata.get("panel") not in ("a", "b"):
            raise ValueError(f"{name}: only Figure 5(a/b) databases are supported")
        try:
            batch, rounds = int(metadata["batch"]), int(metadata["rounds"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{name}: invalid batch/rounds metadata") from exc
        if batch <= 0 or rounds < 2:
            raise ValueError(f"{name}: positive batch and at least two budgets are required")
        budgets = np.arange(1, rounds + 1, dtype=int) * batch
        expected_rounds = list(range(1, rounds + 1))
        accuracies, elapsed, initial = [], [], {}
        for seed in seeds:
            prefix = f"{name}, {strategy}, seed {seed}, {phase}"
            run = db.execute(
                "SELECT status,started,finished FROM runs "
                "WHERE strategy=? AND seed=? AND phase=?", (strategy, seed, phase)
            ).fetchone()
            if run is None or run[0] != "complete":
                raise ValueError(f"{prefix}: requested run is not complete")
            try:
                started, finished = float(run[1]), float(run[2])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{prefix}: invalid phase timestamps") from exc
            if not (math.isfinite(started) and math.isfinite(finished) and finished >= started):
                raise ValueError(f"{prefix}: invalid phase timestamps")
            elapsed.append((finished - started) / 60.0)
            rows = db.execute(
                "SELECT round,labels,test_accuracy FROM metrics "
                "WHERE strategy=? AND seed=? AND phase=? ORDER BY round",
                (strategy, seed, phase),
            ).fetchall()
            if [row[0] for row in rows] != expected_rounds:
                raise ValueError(f"{prefix}: incomplete or unexpected metric rounds")
            if [row[1] for row in rows] != budgets.tolist():
                raise ValueError(f"{prefix}: metric budgets disagree with metadata")
            values = np.asarray([row[2] for row in rows], dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{prefix}: accuracy must be finite and within [0, 1]")
            accuracies.append(values * 100.0)

            selection = db.execute(
                "SELECT status FROM runs WHERE strategy=? AND seed=? AND phase='selection'",
                (strategy, seed),
            ).fetchone()
            if selection != ("complete",):
                raise ValueError(f"{prefix}: selection run is not complete")
            selections = db.execute(
                "SELECT round,indices FROM selections WHERE strategy=? AND seed=? ORDER BY round",
                (strategy, seed),
            ).fetchall()
            if [row[0] for row in selections] != expected_rounds:
                raise ValueError(f"{prefix}: incomplete or unexpected selection rounds")
            seen = set()
            for round_index, encoded in selections:
                indices = json.loads(encoded)
                if not isinstance(indices, list) or len(indices) != batch:
                    raise ValueError(f"{prefix}: invalid selection count at round {round_index}")
                if any(type(index) is not int or index < 0 for index in indices):
                    raise ValueError(f"{prefix}: invalid selection index at round {round_index}")
                if len(set(indices)) != batch or seen.intersection(indices):
                    raise ValueError(f"{prefix}: repeated selected indices at round {round_index}")
                seen.update(indices)
                if round_index == 1:
                    initial[seed] = indices
    values = np.asarray(accuracies)
    # Integrate each seed separately, normalized to its observed budget span.
    aulc = ((values[:, :-1] + values[:, 1:]) * np.diff(budgets) / 2).sum(axis=1)
    aulc /= budgets[-1] - budgets[0]
    return {"name": name, "path": str(path), "panel": metadata["panel"],
            "budgets": budgets, "accuracy": values, "initial": initial,
            "aulc": aulc, "elapsed": np.asarray(elapsed)}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(experiments, baseline, seeds, strategy, phase, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    budgets = baseline["budgets"]
    summary_rows, curve_rows, seed_rows = [], [], []
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for experiment in experiments:
        name, values = experiment["name"], experiment["accuracy"]
        delta = values - baseline["accuracy"]
        mean, sem = mean_sem(values)
        delta_mean, delta_sem = mean_sem(delta)
        line, = axes[0].plot(budgets, mean, marker="o", ms=4, label=name)
        color = line.get_color()
        axes[0].fill_between(budgets, mean - sem, mean + sem, color=color, alpha=0.16)
        axes[1].plot(budgets, delta_mean, marker="o", ms=3, color=color, label=name)
        axes[1].fill_between(budgets, delta_mean - delta_sem, delta_mean + delta_sem,
                             color=color, alpha=0.16)
        row = {"variant": name, "database": experiment["path"], "strategy": strategy,
               "phase": phase, "n_seeds": len(seeds), "seeds": json.dumps(seeds),
               "budget_min": int(budgets[0]), "budget_max": int(budgets[-1]),
               "paired_baseline": baseline["name"]}
        for budget in (2000, 5000, 10000):
            positions = np.flatnonzero(budgets == budget)
            j = int(positions[0]) if len(positions) else None
            for suffix, data in (("mean_pct", mean), ("sem_pct", sem),
                                 ("paired_delta_mean_pp", delta_mean),
                                 ("paired_delta_sem_pp", delta_sem)):
                row[f"acc_{budget}_{suffix}"] = float(data[j]) if j is not None else ""
        for label, data in (("normalized_aulc_pct", experiment["aulc"]),
                            ("paired_aulc_delta_pp", experiment["aulc"] - baseline["aulc"]),
                            ("phase_elapsed_minutes_including_loop_overhead_and_prediction",
                             experiment["elapsed"])):
            row[label + "_mean"], row[label + "_sem"] = map(float, mean_sem(data))
        summary_rows.append(row)
        for j, budget in enumerate(budgets):
            curve_rows.append({"variant": name, "strategy": strategy, "phase": phase,
                               "labels": int(budget), "accuracy_mean_pct": float(mean[j]),
                               "accuracy_sem_pct": float(sem[j]),
                               "paired_delta_mean_pp": float(delta_mean[j]),
                               "paired_delta_sem_pp": float(delta_sem[j])})
        for i, seed in enumerate(seeds):
            seed_row = {"variant": name, "strategy": strategy, "phase": phase, "seed": seed,
                        "normalized_aulc_pct": float(experiment["aulc"][i]),
                        "paired_aulc_delta_pp": float(experiment["aulc"][i] - baseline["aulc"][i]),
                        "phase_elapsed_minutes_including_loop_overhead_and_prediction":
                        float(experiment["elapsed"][i])}
            for j, budget in enumerate(budgets):
                seed_row[f"acc_{budget}_pct"] = float(values[i, j])
                seed_row[f"paired_delta_{budget}_pp"] = float(delta[i, j])
            seed_rows.append(seed_row)
    axes[0].set_ylabel("Test accuracy (%)")
    phase_label = "Downstream evaluation" if phase == "evaluate" else "Proxy accuracy (selection phase)"
    axes[0].set_title(f"{strategy} | {phase_label}\nRaw mean ± SEM over {len(seeds)} paired seeds")
    axes[0].legend(loc="best", fontsize=9)
    axes[1].axhline(0, color="0.4", lw=0.8, ls="--")
    axes[1].set_ylabel(f"Paired difference vs\n{baseline['name']} (pp)")
    axes[1].set_xlabel("Labeled examples")
    for axis in axes:
        axis.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"ablation_{phase}.{suffix}", dpi=180)
    plt.close(fig)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "curves.csv", curve_rows)
    write_csv(output / "per_seed.csv", seed_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", action="append", required=True, metavar="NAME=PATH",
                        help="repeat for each variant, including the named baseline")
    parser.add_argument("--baseline", required=True, help="one of the names passed with --db")
    parser.add_argument("--strategy", default="margin")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--phase", choices=["evaluate", "selection"], default="evaluate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("SEM requires at least two distinct seeds; one-seed runs are not supported")
    sources = {}
    for item in args.db:
        name, separator, path = item.partition("=")
        if not separator or not name.strip() or not path.strip() or name in sources:
            parser.error("each --db must be a unique, nonempty NAME=PATH")
        sources[name] = Path(path)
    if args.baseline not in sources:
        parser.error("--baseline must match a --db name")
    try:
        experiments = [read_experiment(name, path, args.strategy, args.seeds, args.phase)
                       for name, path in sources.items()]
        baseline = next(item for item in experiments if item["name"] == args.baseline)
        for item in experiments:
            if item["panel"] != baseline["panel"] or not np.array_equal(item["budgets"], baseline["budgets"]):
                raise ValueError(f"{item['name']}: all databases must share the same panel and budgets")
            for seed in args.seeds:
                if item["initial"][seed] != baseline["initial"][seed]:
                    raise ValueError(f"{item['name']}, seed {seed}: initial selection or its order differs from baseline")
        report(experiments, baseline, args.seeds, args.strategy, args.phase,
               args.output_dir.expanduser().resolve())
    except (ValueError, sqlite3.Error, OSError) as exc:
        parser.error(str(exc))
    print(f"Validated {len(experiments)} variants × {len(args.seeds)} seeds; output: {args.output_dir.resolve()}")
    print("Accuracy/AULC use percent; paired differences use percentage points.")
    print("Phase elapsed time includes loop overhead, training, selection (when applicable), and prediction;")
    print("it excludes the separate cache-precomputation phase and is not pure training time.")


if __name__ == "__main__":
    main()
