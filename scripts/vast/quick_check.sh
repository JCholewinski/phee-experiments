#!/usr/bin/env bash
# One short run of every method of the plan with the real models (200 sentences,
# 1 epoch; LLM: 20 sentences) to check that models download and run on the GPU, and to
# estimate the time of the real runs. Results go to results_quick/ (not mixed with the
# real results; their scores are meaningless).
#
#   bash scripts/vast/quick_check.sh
set -uo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/workspace/hf}"
if ! command -v python >/dev/null 2>&1 && [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate   # vast.ai PyTorch images keep Python in /venv/main
fi
LLM_PY="${LLM_PY:-/workspace/venv-llm/bin/python}"

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

if [[ -x "$LLM_PY" ]]; then
  for config in llm/qwen3_8b_authors_prompt llm/qwen3_8b; do
    echo "=== $config"
    if ! "$LLM_PY" scripts/run_experiment.py --config "configs/$config.yaml" --seed 1 --output_dir results_quick \
        --set data.limit=20 --set "shots=[0, 5]" --force; then
      failed+=("$config")
    fi
  done
else
  echo "!!! no vLLM environment at $LLM_PY (bash scripts/vast/setup.sh --llm): LLM not checked"
  failed+=("llm (no vLLM environment)")
fi

python - <<'EOF'
import json
from pathlib import Path

# data.limit=200 limits every split; the full splits have 2898 train / 961 dev / 968 test
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

print("\nLLM, extrapolated to dev + test (1929 sentences):")
for path in sorted(Path("results_quick").glob("llm__*/seed1/summary.json")):
    summary = json.loads(path.read_text())
    for variant, splits in summary["variants"].items():
        for split, entry in splits.items():
            inference = entry.get("inference", {})
            per_sentence = inference.get("seconds_per_sentence")
            if per_sentence is None:
                continue
            print(f"  {path.parent.parent.name:32s} {variant:8s} {split:5s} {per_sentence:5.2f} s/sentence"
                  f" -> {per_sentence * 1929 / 60:5.1f} min, wrong format {inference.get('invalid_json', 0)}"
                  f"/{inference.get('sentences', 0)}, prompt {inference.get('mean_prompt_tokens', 0):.0f} tokens")
EOF

if (( ${#failed[@]} )); then
  echo "FAILED: ${failed[*]} (see the output above)"
  exit 1
fi
echo "All methods ran."
