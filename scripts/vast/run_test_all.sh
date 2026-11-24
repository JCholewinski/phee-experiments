#!/usr/bin/env bash
# Everything for the test mode on a FRESH instance, in one background job:
#   1. setup (training environment, tests, vLLM environment)
#   2. quick check on the real models (stops here if a method fails)
#   3. test mode: seed 1 of every run of the plan, tables
#   4. archive results_test.tar.gz (results with predictions + logs)
#
# Start it in the background, so a frozen browser terminal cannot stop or block it:
#   nohup bash scripts/vast/run_test_all.sh > all_test.log 2>&1 &
# Look at the progress any time (never needs Ctrl+C):
#   tail -n 30 all_test.log
# Closing the terminal or the browser does not stop it.
set -uo pipefail
cd "$(dirname "$0")/../.."

step() { echo; echo "##### [$(date '+%F %T')] $*"; }
fail() { echo; echo "##### [$(date '+%F %T')] FAILED: $*"; exit 1; }

# the Python environment of the vast.ai PyTorch image
if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

step "1/4 setup: training environment, tests, vLLM environment"
bash scripts/vast/setup.sh --llm || fail "setup (see the lines above)"
/workspace/venv-llm/bin/python -c "import vllm" || fail "vLLM is not importable"

step "2/4 quick check on the real models (~30 min, details in quick.log)"
bash scripts/vast/quick_check.sh > quick.log 2>&1
tail -n 45 quick.log
grep -q "All methods ran." quick.log || fail "quick check (see quick.log)"

step "3/4 test mode: seed 1 of every run (~8-9 h, details in plan_test.log)"
bash scripts/vast/run_plan.sh test > plan_test.log 2>&1
tail -n 25 plan_test.log

step "4/4 archive"
tar czf results_test.tar.gz results plan_test.log quick.log && ls -lh results_test.tar.gz

step "DONE. Download results_test.tar.gz: open Jupyter, folder phee-experiments, right click -> Download"
