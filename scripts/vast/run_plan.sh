#!/usr/bin/env bash
# The whole experiment plan on a vast.ai instance, then the tables.
#
#   bash scripts/vast/run_plan.sh test     # test mode: seed 1 of every run
#   bash scripts/vast/run_plan.sh final    # final results: seeds 1-3 (finished seed-1 runs are reused)
#
# Start it in the background, so that it survives closing the terminal:
#   nohup bash scripts/vast/run_plan.sh test > plan_test.log 2>&1 &
#   tail -f plan_test.log
#
# Order: LLM part first (vLLM environment, about an hour, fails fast if vLLM does not
# work), then the trained methods (baselines, improved versions). Finished runs (DONE)
# are skipped, so after an interruption simply start the same command again. A failed
# run is logged (results/<run>/seed<N>/FAILED) and the next one starts.
# GPU_PRICE (USD per hour, shown by vast.ai for the instance) is used for the cost tables.
set -uo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/workspace/hf}"
if ! command -v python >/dev/null 2>&1 && [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate   # vast.ai PyTorch images keep Python in /venv/main
fi
LLM_PY="${LLM_PY:-/workspace/venv-llm/bin/python}"
GPU_PRICE="${GPU_PRICE:-0.25}"

case "${1:-}" in
  test)  seeds=(1) ;;
  final) seeds=(1 2 3) ;;   # DONE runs (seed 1 from the test mode) are skipped
  *) echo "usage: bash scripts/vast/run_plan.sh test|final" >&2; exit 2 ;;
esac

status=0
echo "##### $(date '+%F %T') plan '$1', seeds ${seeds[*]}, GPU price ${GPU_PRICE} USD/h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# 1. LLM: baseline prompt, improved prompt (one deterministic run each, done in test
#    mode), demonstrations from 50 sentences (one random subset per seed)
if [[ -x "$LLM_PY" ]]; then
  "$LLM_PY" scripts/run_matrix.py --groups llm llm_pool50 --seeds "${seeds[@]}" || status=1
else
  echo "!!! no vLLM environment at $LLM_PY: run 'bash scripts/vast/setup.sh --llm' first; LLM part skipped"
  status=1
fi

# 2. trained methods: the authors' baselines and the improved versions
python scripts/run_matrix.py --groups baselines improved --seeds "${seeds[@]}" || status=1

# 3. tables (all finished runs, every seed) and the list of runs
echo "##### $(date '+%F %T') tables"
python scripts/aggregate_results.py --gpu_price "$GPU_PRICE" || status=1
python scripts/run_matrix.py --groups llm llm_pool50 baselines improved --seeds 1 2 3 --list

if (( status )); then
  echo "##### finished WITH ERRORS (see FAILED runs above and results/tables/checks.md)"
else
  echo "##### finished: results/tables/comparison_test.md, main_test.md, checks.md"
fi
exit $status
