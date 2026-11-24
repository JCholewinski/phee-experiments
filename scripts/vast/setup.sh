#!/usr/bin/env bash
# Environment setup on a fresh vast.ai instance (PyTorch image with CUDA >= 12.8 for RTX 5090).
#
#   bash scripts/vast/setup.sh            # training environment
#   bash scripts/vast/setup.sh --llm      # + separate venv with vLLM for the LLM part
set -euo pipefail

cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"

# --- Python -------------------------------------------------------------------------
# vast.ai PyTorch images keep Python in a virtual environment (/venv/main) that is not
# always active in the Jupyter terminal; other images may only have `python3`.
if ! command -v python >/dev/null 2>&1; then
  if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
    grep -qs '/venv/main/bin/activate' ~/.bashrc || echo 'source /venv/main/bin/activate' >> ~/.bashrc
    echo "Activated /venv/main (added to ~/.bashrc)"
  elif command -v python3 >/dev/null 2>&1; then
    ln -sf "$(command -v python3)" /usr/local/bin/python
    echo "Linked python -> $(command -v python3)"
  else
    echo "No Python found on this machine" >&2
    exit 1
  fi
fi
PY="$(command -v python)"
echo "Python: $PY"

pip_install() {  # pip_install <python> <args...>   (PIP_VERBOSE=1: show the progress)
  local python="$1"; shift
  local quiet="-q"
  [[ "${PIP_VERBOSE:-0}" == "1" ]] && quiet="--progress-bar=off"
  if "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m pip install $quiet "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" "$@"
  else
    "$python" -m ensurepip --upgrade >/dev/null && "$python" -m pip install $quiet "$@"
  fi
}

"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
pip_install "$PY" -r requirements.txt pytest

"$PY" -m pytest tests -q

if [[ "${1:-}" == "--llm" ]]; then
  if ! "$PY" -m venv /workspace/venv-llm 2>/dev/null; then
    uv venv /workspace/venv-llm --python "$PY"
  fi
  # several GB (vLLM with its own torch): print every package, so it never looks frozen
  echo "Installing vLLM into /workspace/venv-llm (several GB, 5-20 min)..."
  PIP_VERBOSE=1 pip_install /workspace/venv-llm/bin/python -r requirements-llm.txt
  /workspace/venv-llm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
fi

cat <<'EOF'

Setup done. Next steps (README, "Uruchomienie na vast.ai"):
  0. In this terminal run:   source ~/.bashrc
  1. Quick check on real models (downloads models, ~1 epoch on 200 sentences per method,
     LLM on 20 sentences; needs the vLLM environment from 'setup.sh --llm'):
       nohup bash scripts/vast/quick_check.sh > quick.log 2>&1 &
  2. Test mode (seed 1 of every run, then tables in results/tables/):
       GPU_PRICE=<USD per hour of this instance> nohup bash scripts/vast/run_plan.sh test > plan_test.log 2>&1 &
  3. After checking the test results, the final seeds 2 and 3:
       GPU_PRICE=<USD per hour> nohup bash scripts/vast/run_plan.sh final > plan_final.log 2>&1 &
EOF
