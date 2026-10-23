#!/usr/bin/env bash
# Harness v2 test (2026-10-20): the single-run and RSI arms with --modular (modular parts, patch edits, per-part
# gate, run-time self-check guidance), 3 independent runs, under the matched study's budgets. The native agents
# and the v1 harness are not re-run: their frozen solvers are scored on the fresh splits (scripts/score_fresh_splits.py).
# Usage (on the login node): [SUITE=pdebench|inventory] [STAMP=YYYYMMDD] bash scripts/run_v2_study.sh <node> <wave: opus55|gpt55> <6 cores...>
set -euo pipefail
node=$1; wave=$2; shift 2; cores=("$@")
root=/depot/lin491/data/Binghang_project/PDE-RSI
suite=${SUITE:-pdebench}; stamp=${STAMP:-20261020}
tag=$([ "$suite" = inventory ] && echo "inv-" || echo "")
case $wave in
  opus55) config=claude_opus55_64k_code ;;
  gpt55) config=gpt55_code ;;
  *) echo "unknown wave $wave" >&2; exit 1 ;;
esac
common="--protocol $suite --modular --start scratch --probe-rounds 0 --attempts-per-round 3 --time-budget-seconds 7200 \
--concurrent-arms --history-rounds 6 --wall-seconds 10800 --cpu-seconds 10800 --max-validation-submissions 30 \
--spend-cap-usd 25"
launch() { timeout 20 ssh -n -f -o BatchMode=yes "$node" "cd $root && $1" >/dev/null 2>&1; }
for r in 1 2 3; do
  i=$(( (r-1)*2 ))
  name=v2-$tag$wave-harness-r$r-$stamp
  launch "setsid nohup bash scripts/run.sh code-agents $common --model-config configs/agent_models/$config.json \
--cores ${cores[$i]},${cores[$((i+1))]} --output runs/$name > runs/$name.log 2>&1 < /dev/null &"
  echo "$(date) $name on ${cores[$i]},${cores[$((i+1))]}"
done
