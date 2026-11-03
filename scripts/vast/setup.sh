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

pip_install() {  # pip_install <python> <args...>
  local python="$1"; shift
  if "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m pip install -q "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" -q "$@"
  else
    "$python" -m ensurepip --upgrade >/dev/null && "$python" -m pip install -q "$@"
  fi
}

"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
pip_install "$PY" -r requirements.txt pytest

"$PY" -m pytest tests -q

if [[ "${1:-}" == "--llm" ]]; then
  if ! "$PY" -m venv /workspace/venv-llm 2>/dev/null; then
    uv venv /workspace/venv-llm --python "$PY"
  fi
  pip_install /workspace/venv-llm/bin/python -r requirements-llm.txt
  /workspace/venv-llm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
fi

cat <<'EOF'

Setup done. Next steps:
  0. In this terminal run:   source ~/.bashrc
  1. Accept the Gemma license for google/t5gemma-b-b-ul2 on huggingface.co, then:
       python -c "from huggingface_hub import login; login()"
  2. Quick check on real models (downloads models, ~1 epoch on 200 sentences per method):
       nohup bash scripts/vast/quick_check.sh > quick.log 2>&1 &
  3. Test mode (seed 1), then the remaining seeds (see README, section 5):
       nohup sh -c 'python scripts/run_matrix.py --groups baselines improved --seeds 1; /workspace/venv-llm/bin/python scripts/run_matrix.py --groups llm' > matrix.log 2>&1 &
EOF
