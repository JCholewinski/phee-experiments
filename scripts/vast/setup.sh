#!/usr/bin/env bash
# Environment setup on a fresh vast.ai instance (PyTorch image with CUDA >= 12.8 for RTX 5090).
#
#   bash scripts/vast/setup.sh            # training environment
#   bash scripts/vast/setup.sh --llm      # + separate venv with vLLM for the LLM part
set -euo pipefail

cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
pip install -q -r requirements.txt pytest

python -m pytest tests -q

if [[ "${1:-}" == "--llm" ]]; then
  python -m venv /workspace/venv-llm
  /workspace/venv-llm/bin/pip install -q -r requirements-llm.txt
  /workspace/venv-llm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
fi

cat <<'EOF'

Setup done. Next steps:
  1. Accept the Gemma license for google/t5gemma-b-b-ul2 (and Llama 3.1 for the LLM part)
     on huggingface.co, then:   hf auth login
  2. export HF_HOME=/workspace/hf PYTHONPATH=$PWD
  3. Quick check on real models (downloads models, ~1 epoch on 200 sentences per method):
       bash scripts/vast/quick_check.sh
  4. Full experiments:
       nohup python scripts/run_matrix.py --groups baselines seq eqa gqa --seeds 1 2 3 > matrix.log 2>&1 &
EOF
