#!/usr/bin/env bash
# Matched-model study (2026-10-18): harness (single run + RSI arms) and the native coding agent,
# 3 independent runs each, for Opus 5.5 (Claude Code) and GPT-5.5 (Codex). Same 2 h budget, $25 spending cap,
# 30 validation submissions through the same gate, same solver sandbox; primary metric: the fresh sealed split.
# Usage (on the login node): [SUITE=pdebench|inventory] [STAMP=YYYYMMDD] bash scripts/run_matched_study.sh <node> <wave: opus55|gpt55> <14 cores...>
# Each wave uses 12 dedicated cores (3 harness comparisons x 2 arms, 3 native runs x agent+evaluation core)
# and the remaining 2 cores to score finished harness runs on both sealed splits.
set -euo pipefail
node=$1; wave=$2; shift 2; cores=("$@")
root=/depot/lin491/data/Binghang_project/PDE-RSI
suite=${SUITE:-pdebench}; stamp=${STAMP:-20261018}
tag=$([ "$suite" = inventory ] && echo "inv-" || echo "")
case $wave in
  opus55) model=us.anthropic.claude-opus-5-5; config=claude_opus55_64k_code; agent=claude-code ;;
  gpt55) model=us.openai.gpt-5.5; config=gpt55_code; agent=codex ;;
  *) echo "unknown wave $wave" >&2; exit 1 ;;
esac
common="--protocol $suite --start scratch --probe-rounds 0 --attempts-per-round 3 --time-budget-seconds 7200 \
--concurrent-arms --history-rounds 6 --wall-seconds 10800 --cpu-seconds 10800 --max-validation-submissions 30 \
--spend-cap-usd 25"
launch() { timeout 20 ssh -n -f -o BatchMode=yes "$node" "cd $root && $1" >/dev/null 2>&1; }
for r in 1 2 3; do
  i=$(( (r-1)*2 ))
  name=matched-$tag$wave-harness-r$r-$stamp
  launch "setsid nohup bash scripts/run.sh code-agents $common --model-config configs/agent_models/$config.json \
--cores ${cores[$i]},${cores[$((i+1))]} --output runs/$name > runs/$name.log 2>&1 < /dev/null &"
  echo "$(date) $name on ${cores[$i]},${cores[$((i+1))]}"
  j=$(( 6+(r-1)*2 ))
  name=matched-$tag$wave-$agent-r$r-$stamp
  launch "source scripts/env.sh && setsid nohup python scripts/run_native_agent.py --suite $suite --agent $agent --model $model \
--output runs/$name --budget-seconds 7200 --spend-cap-usd 25 --max-validation-submissions 30 \
--eval-core ${cores[$j]} --agent-core ${cores[$((j+1))]} > runs/$name.log 2>&1 < /dev/null &"
  echo "$(date) $name eval ${cores[$j]} agent ${cores[$((j+1))]}"
done
