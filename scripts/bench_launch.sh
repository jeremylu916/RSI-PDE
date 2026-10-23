#!/usr/bin/env bash
# Launch RSI-PDE Bench entrant runs on a GPU node, each detached, two OS cores per run.
# Usage (login node): [SMOKE=1] [STAMP=YYYYMMDD] bash scripts/bench_launch.sh <node> <track:entrant:model> ... -- <cores...>
# e.g. bash scripts/bench_launch.sh h001 tex:pde-rsi-v2:opus55 tex:claude-code:opus55 -- 14 15 16 17
set -euo pipefail
node=$1; shift
specs=(); while [[ $1 != -- ]]; do specs+=("$1"); shift; done; shift; cores=("$@")
root=/depot/lin491/data/Binghang_project/PDE-RSI
stamp=${STAMP:-$(date +%Y%m%d)}
prefix=$([ -n "${SMOKE:-}" ] && echo bench-smoke || echo bench-v1)
env=$([ -n "${SMOKE:-}" ] && echo "RSI_PDE_BENCH_SMOKE=1" || echo "")
(( ${#cores[@]} >= 2*${#specs[@]} )) || { echo "need two cores per run" >&2; exit 1; }
for i in "${!specs[@]}"; do
  IFS=: read -r track entrant model <<< "${specs[$i]}"
  name=$prefix-$track-$entrant-$model-$stamp
  pair=${cores[$((2*i))]},${cores[$((2*i+1))]}
  timeout 20 ssh -n -f -o BatchMode=yes "$node" "cd $root && source scripts/env.sh >/dev/null 2>&1 && \
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $env && setsid nohup python scripts/bench.py run \
--track $track --entrant $entrant --model $model --cores $pair --output runs/$name > runs/$name.log 2>&1 < /dev/null &" \
    >/dev/null 2>&1
  echo "$(date '+%F %T') $name on $node cores $pair"
done
