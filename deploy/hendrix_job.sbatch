#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=labelbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# NEW starter workflow: one GPU, one trial at a time, preserving Slurm GPU IDs.
# Run sbatch from the repository root. Arguments: check|features|trial variant seed.
set -euo pipefail
MODE=${1:-check}
VARIANT=${2:-baseline}
SEED=${3:-1234}
case "$MODE" in check|features|trial) ;; *) echo "Use check, features, or trial" >&2; exit 2;; esac
case "$VARIANT" in baseline|linear|h128|h1024|k1|k3|u1|u3) ;; *) exit 2;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
SOURCE_ROOT=${SLURM_SUBMIT_DIR:?Submit from the repository root}
cd "$SOURCE_ROOT"
source work/hendrix.env
test -x "$LB_PYTHON"
mkdir -p work
exec 9>work/hendrix-single-job.lock
flock -n 9 || { echo "Another starter job is using this checkout. Run jobs sequentially." >&2; exit 2; }
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ "$MODE" != check ]]; then
  "$LB_PYTHON" - "$VARIANT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path.cwd()
manifest = json.loads((root / "results/ablation" / sys.argv[1] / "ablation_manifest.json").read_text())
assert manifest["variant"] == sys.argv[1]
for name, expected in manifest["sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Source/config changed: {name}. Configure the intended variant before submitting.")
PY
fi

# Run SQLite WAL on node-local storage, not the shared home filesystem.
RUN_ROOT=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/labelbench-${SLURM_JOB_ID}.XXXXXX")
echo "Node-local checkout: $RUN_ROOT"
rsync -a --exclude=/.git/ --exclude=/work/ --exclude=/logs/ --exclude=/results/ \
  --exclude=/data/ --exclude=/model/ --exclude=__pycache__ "$SOURCE_ROOT/" "$RUN_ROOT/"
mkdir -p "$RUN_ROOT/work" "$RUN_ROOT/results/fig5/features"
if [[ "$MODE" != check ]]; then
  test -d "$SOURCE_ROOT/data"
  test -d "$SOURCE_ROOT/model"
  rsync -a "$SOURCE_ROOT/data/" "$RUN_ROOT/data/"
  rsync -a "$SOURCE_ROOT/model/" "$RUN_ROOT/model/"
  if [[ -d "$SOURCE_ROOT/results/fig5/features" ]]; then
    rsync -a --include='*.pt' --exclude='*' "$SOURCE_ROOT/results/fig5/features/" "$RUN_ROOT/results/fig5/features/"
  fi
fi
OUTPUT_REL="results/ablation/$VARIANT"
mkdir -p "$RUN_ROOT/$OUTPUT_REL"
if [[ -d "$SOURCE_ROOT/$OUTPUT_REL" ]]; then
  # The persistent database is a closed DELETE-journal snapshot, never live WAL.
  test ! -e "$SOURCE_ROOT/$OUTPUT_REL/figure5a.sqlite-wal"
  test ! -e "$SOURCE_ROOT/$OUTPUT_REL/figure5a.sqlite-shm"
  rsync -a --exclude=locks "$SOURCE_ROOT/$OUTPUT_REL/" "$RUN_ROOT/$OUTPUT_REL/"
fi
cd "$RUN_ROOT"
export LABELBENCH_MODEL_DIR="$RUN_ROOT/model"
export MPLCONFIGDIR="$RUN_ROOT/work/matplotlib"
mkdir -p "$SOURCE_ROOT/$OUTPUT_REL/job_records/$SLURM_JOB_ID"
"$LB_PYTHON" -m pip freeze > "$SOURCE_ROOT/$OUTPUT_REL/job_records/$SLURM_JOB_ID/pip-freeze.txt"
srun --ntasks=1 nvidia-smi
srun --ntasks=1 "$LB_PYTHON" deploy/check_runtime.py --gpu

publish() {
  mkdir -p "$SOURCE_ROOT/results/fig5/features"
  rsync -a --include='*.pt' --exclude='*' "$RUN_ROOT/results/fig5/features/" "$SOURCE_ROOT/results/fig5/features/"
  if [[ -d "$RUN_ROOT/data" ]]; then
    rsync -a --include='labels_*.pt' --exclude='*' "$RUN_ROOT/data/" "$SOURCE_ROOT/data/"
  fi
  for folder in selections checkpoints; do
    if [[ -d "$RUN_ROOT/$OUTPUT_REL/$folder" ]]; then
      mkdir -p "$SOURCE_ROOT/$OUTPUT_REL/$folder"
      rsync -a "$RUN_ROOT/$OUTPUT_REL/$folder/" "$SOURCE_ROOT/$OUTPUT_REL/$folder/"
    fi
  done
  if [[ -f "$RUN_ROOT/$OUTPUT_REL/figure5a.sqlite" ]]; then
    "$LB_PYTHON" deploy/snapshot_db.py \
      "$RUN_ROOT/$OUTPUT_REL/figure5a.sqlite" "$SOURCE_ROOT/$OUTPUT_REL/figure5a.sqlite"
  fi
}

run_phase() {
  local phase=$1
  local code=0
  srun --ntasks=1 "$LB_PYTHON" scripts/train_fig5_one.py \
    --panel a --strategy margin --seed "$SEED" --phase "$phase" \
    --output-dir "$RUN_ROOT/$OUTPUT_REL" || code=$?
  # Called after the worker exits, on success or normal failure.
  publish
  if (( code != 0 )); then
    echo "Phase failed ($phase); snapshots published. Local directory: $RUN_ROOT" >&2
    exit "$code"
  fi
}

case "$MODE" in
  check) ;;
  features) run_phase features ;;
  trial)
    run_phase selection
    run_phase evaluate
    ;;
esac
echo "JOB COMPLETED: $MODE $VARIANT seed=$SEED"
# Only this newly created node-local directory is removed after success.
rm -rf -- "$RUN_ROOT"
