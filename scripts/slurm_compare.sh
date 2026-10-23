#!/usr/bin/env bash
#SBATCH --job-name=pde-rsi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:35:00
#SBATCH --output=/depot/lin491/data/Binghang_project/PDE-RSI/runs/%x-%j.out
set -euo pipefail
# Submit from the project root. Supply your allocation with sbatch --account=...
# if your account has no default allocation. No allocation is guessed here.
source /depot/lin491/data/Binghang_project/PDE-RSI/scripts/env.sh
test -x scripts/run.sh
scripts/run.sh compare --generations 4 --max-calls 240 --wall-seconds 600 --cpu-seconds 600 \
    --output "runs/slurm-${SLURM_JOB_ID}"
