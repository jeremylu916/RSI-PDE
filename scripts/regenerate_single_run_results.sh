#!/usr/bin/env bash
# Rebuild the results folders after the single arm became the one-shot single run (2026-10-20); see
# scripts/redo_single_one_shot.py. The earlier folders, whose "single" column was the iterated fixed-workflow
# arm, move to runs/archive-fixed-workflow-20261020/. Run after every listed run is scored, on a compute node
# (report_code_agents.py re-evaluates every saved solution), after sourcing scripts/env.sh:
#     bash scripts/regenerate_single_run_results.sh
set -euo pipefail
cd /depot/lin491/data/Binghang_project/PDE-RSI
archive=runs/archive-fixed-workflow-20261020
mkdir -p "$archive"
move() { if [ -e "runs/$1" ]; then mv "runs/$1" "$archive/"; fi; if [ -e "runs/$1.log" ]; then mv "runs/$1.log" "$archive/"; fi; }
models=(claude gpt-oss gpt55 gpt61 opus55)
pdebench_families="poisson heat convection_diffusion burgers wave2d fode caputo multi_term variable_order_t variable_order_xt"
student_families="heat convection_diffusion burgers caputo multi_term fpde_caputo fpde_variable_order fpde_variable_order_xt"
runs_for() { for m in "${models[@]}"; do printf 'runs/code-agents-%s-%s-%s ' "$1" "$m" "$2"; done; }
plot_families() {  # plot_score_vs_round.py, keeping one file set per family group
    python scripts/plot_score_vs_round.py "$1" --families "${@:3}" > /dev/null
    for stem in score_vs_round score_vs_round_overview; do
        for suffix in png svg; do mv "$1/$stem.$suffix" "$1/${stem}_$2.$suffix"; done
    done
}

# Manufactured code protocol (1-D, 2-D wave, group FPDE), fine evaluator.
R=runs/code-agent-results-fine-20261016; move "${R#runs/}"
python scripts/report_code_agents.py --runs runs/code-agents-fine-*-20261016 --output $R > $R.log
python scripts/per_equation_table.py $R > /dev/null
plot_families $R 1d_wave burgers poisson wave2d
plot_families $R group fode fpde_caputo fpde_variable_order fpde_variable_order_xt

# Student protocol (identify + solve): pilot and main.
R=runs/code-agent-results-student-pilot-20261016; move "${R#runs/}"
python scripts/report_code_agents.py --runs runs/code-agents-student-pilot-*-20261016 --output $R > $R.log
python scripts/per_equation_table.py $R > /dev/null
R=runs/code-agent-results-student-20261017; move "${R#runs/}"
python scripts/report_code_agents.py --runs $(runs_for student 20261016) --output $R > $R.log
python scripts/per_equation_table.py $R > /dev/null
python scripts/plot_score_vs_round.py $R --families $student_families > /dev/null
python scripts/export_student_cases.py --runs $(runs_for student 20261016) --output tasks/student > /dev/null

# pde_benchmarks.tex, 6 rounds.
R=runs/code-agent-results-pdebench-20261017; move "${R#runs/}"
runs=$(runs_for pdebench 20261017)
python scripts/report_code_agents.py --runs $runs --output $R > $R.log
python scripts/report_pdebench.py $R > /dev/null
python scripts/plot_score_vs_round.py $R --families $pdebench_families > /dev/null
python scripts/export_pdebench_table.py $R --runs $runs > /dev/null
python scripts/export_rsi_evolution.py $R --runs $runs > /dev/null
python scripts/plot_convergence.py $R > /dev/null
python scripts/per_problem_all_splits.py $R --prefix pdebench --stamp 20261017 > /dev/null
python scripts/export_pdebench_cases.py --runs $runs --output tasks/pdebench > /dev/null

# pde_benchmarks.tex, 2 h: standard, 100 % noise, hidden time dependence.
for variant in pdebench2h pdebench2h-noise100 pdebench2h-hiddentime; do
    R=runs/code-agent-results-$variant-20261017
    keep=$(mktemp -d -p .runtime/tmp)
    if [ -e $R/claude_code_comparison.md ]; then cp $R/claude_code_comparison.md "$keep/"; fi
    move "${R#runs/}"
    runs=$(runs_for $variant 20261017)
    python scripts/report_code_agents.py --runs $runs --output $R > $R.log
    python scripts/report_pdebench.py $R > /dev/null
    python scripts/export_pdebench_table.py $R --runs $runs > /dev/null
    python scripts/plot_convergence.py $R > /dev/null
    python scripts/plot_convergence.py $R --x minutes > /dev/null
    extra=""
    if [ $variant != pdebench2h-noise100 ]; then extra=$(ls -d runs/claudecode-pdebench2h-*-20261017 2>/dev/null || true); fi
    python scripts/api_cost.py --runs $runs $extra --output $R/cost > /dev/null 2>&1
    if [ -e "$keep/claude_code_comparison.md" ]; then cp "$keep/claude_code_comparison.md" $R/; fi
    rm -rf "$keep"
done
python scripts/report_per_pde.py runs/code-agent-results-pdebench2h-20261017 --runs $(runs_for pdebench2h 20261017) > /dev/null

# Matched-model studies.
cp -r runs/matched-study-results-20261018 "$archive/"
python scripts/report_matched_study.py --output runs/matched-study-results-20261018 > /dev/null
echo "regenerated"
