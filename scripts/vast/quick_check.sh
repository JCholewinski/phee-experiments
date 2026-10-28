#!/usr/bin/env bash
# One short run of every method with the real models (200 sentences, 1 epoch) to check
# that models download and run on the GPU, and to measure the time of one epoch.
# Results go to results_quick/ (not mixed with the real results).
#
#   bash scripts/vast/quick_check.sh
set -uo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/workspace/hf}"

configs=(
  seq/authors_baseline
  seq/own_crf
  eqa/authors_baseline
  eqa/own_current_predicted
  gqa/authors_baseline
  gqa/own_current_predicted
)
failed=()
for config in "${configs[@]}"; do
  echo "=== $config"
  if ! python scripts/run_experiment.py --config "configs/$config.yaml" --seed 1 --output_dir results_quick \
      --set data.limit=200 --set training.max_epochs=1 --force; then
    failed+=("$config")
  fi
done

python - <<'EOF'
import json
from pathlib import Path

# data.limit=200 limits every split; the full splits have 2898 train / 961 dev sentences
TRAIN_SCALE, DEV_SCALE = 2898 / 200, 961 / 200
print("\nOne epoch, extrapolated to the full data (train 2898, dev 961 sentences):")
print(f"  {'experiment':32s} {'stage':16s} {'train':>9s} {'dev eval':>9s} {'peak GPU':>9s}")
for path in sorted(Path("results_quick").glob("*/seed1/training.json")):
    for stage in json.loads(path.read_text()):
        history_path = path.parent / "history" / f"{stage['name']}.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        epoch = history[0] if history else {}
        print(f"  {path.parent.parent.name:32s} {stage['name']:16s}"
              f" {epoch.get('train_seconds', 0) * TRAIN_SCALE / 60:7.1f} m"
              f" {epoch.get('eval_seconds', 0) * DEV_SCALE / 60:7.1f} m"
              f" {stage.get('peak_gpu_memory_gb') or 0:6.1f} GB")
print("A run takes roughly (best epoch + 2) epochs per stage (early stopping, patience 2).")
EOF

if (( ${#failed[@]} )); then
  echo "FAILED: ${failed[*]} (see the output above)"
  exit 1
fi
echo "All methods ran."
