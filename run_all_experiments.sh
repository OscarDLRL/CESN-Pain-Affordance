#!/usr/bin/env bash
# CESN final experimental pipeline: frozen model, selected Method A + L-BFGS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-status}"
DEVICE="${CESN_DEVICE:-cuda}"
BATCH="${CESN_BATCH_CASES:-64}"
# Optional ManiSkill task-space FK, only if setup is available:
TASK_SPACE="${CESN_TASK_SPACE:-0}"
OUT="CESN/outputs/final_lbfgs"
MODEL="CESN/outputs/models/N_900/run_01/model.npz"
DATA="datasets/cesn_spatial_context_v1/multimodal_spatial_context_v1.h5"

run() { echo;printf '==> %s\n' "$1";shift;"$@"; }
require_env() { [[ "${CONDA_DEFAULT_ENV:-}" == 'maniskill-esn' ]] || { echo 'Activate: conda activate maniskill-esn' >&2;exit 1; }; }
require_data() { [[ -f "$MODEL" && -f "$DATA" ]] || { echo "Missing frozen model or dataset ($MODEL, $DATA)" >&2;exit 1; }; }
run_if_missing() {
  local check="$1";local label="$2";shift 2
  if [[ -f "$check" ]];then echo "REUSE $label: $check";else run "$label" "$@";fi
}
report() {
  run 'Regenerate figures/tables from saved results' \
    python3 -m CESN.analysis.final.reporting_spatial_final

  run 'Generate minimum context requirement table' \
    python3 -m CESN.analysis.final.context_requirement_table
}
case "$MODE" in
 status)
  echo "CESN repo: $ROOT";printf 'git: ';git rev-parse --short HEAD || true
  for f in "$DATA" "$MODEL" "${OUT}/known_task/known_task_detail.csv" \
    "${OUT}/unknown_task/unknown_task_detail.csv" "${OUT}/task_emergence/task_emergence_detail.csv" \
    "${OUT}/noise_robustness/noise_robustness_detail.csv" "${OUT}/ambiguity/ambiguity_detail.csv" \
    "${OUT}/movement_completion/movement_completion_detail.csv" "${OUT}/baselines/baselines_detail.csv" \
    ; do
      [[ -f "$f" ]] && echo "OK   $f" || echo "MISS $f"
  done
  ;;
 report)
  require_env;report
  ;;
 quick|final)
  require_env;require_data
  run 'Inversion engine parity (small, deterministic)' python3 -m CESN.analysis.final.check_inversion_parity --device "$DEVICE"
  if [[ "$MODE" == quick ]];then
    run 'Quick known-task' python3 -m CESN.analysis.final.known_task --quick --device "$DEVICE" --batch-cases 8 --out-dir "${OUT}/known_task_smoke"
    run 'Quick unknown-task' python3 -m CESN.analysis.final.unknown_task --device "$DEVICE" --batch-cases 8 --out-dir "${OUT}/unknown_task_smoke"
    run 'Quick exact-K task emergence' python3 -m CESN.analysis.final.task_emergence --device "$DEVICE" --batch-cases 8 --out-dir "${OUT}/task_emergence_smoke"
    run 'Quick noise' python3 -m CESN.analysis.final.noise_robustness --device "$DEVICE" --batch-cases 8 --out-dir "${OUT}/noise_robustness_smoke"
    run 'Quick ambiguity' python3 -m CESN.analysis.final.ambiguity --device "$DEVICE" --max-cases 2 --grid-size 25 --out-dir "${OUT}/ambiguity_smoke"
    run 'Quick completion' python3 -m CESN.analysis.final.movement_completion --device "$DEVICE" --out-dir "${OUT}/movement_completion_smoke"
    run 'Quick baselines' python3 -m CESN.analysis.final.baselines --device "$DEVICE" --max-cases 4 --grid-size 10 --out-dir "${OUT}/baselines_smoke"
    echo 'QUICK PASS: full report is NOT built from smoke outputs.'
    exit 0
  fi
  run_if_missing "${OUT}/known_task/known_task_detail.csv" 'Known-task XY' \
    python3 -m CESN.analysis.final.known_task --full --device "$DEVICE" --batch-cases "$BATCH"
  run_if_missing "${OUT}/unknown_task/unknown_task_detail.csv" 'Unknown-task Method A' \
    python3 -m CESN.analysis.final.unknown_task --full --device "$DEVICE" --batch-cases "$BATCH"
  run_if_missing "${OUT}/task_emergence/task_emergence_detail.csv" 'Task emergence exact-K' \
    python3 -m CESN.analysis.final.task_emergence --full --device "$DEVICE" --batch-cases "$BATCH"
  run_if_missing "${OUT}/noise_robustness/noise_robustness_detail.csv" 'Noise experiment (six sigma levels)' \
    python3 -m CESN.analysis.final.noise_robustness --full --device "$DEVICE" --batch-cases "$BATCH"
  run_if_missing "${OUT}/ambiguity/ambiguity_detail.csv" 'Known-task ambiguity/Hessian' \
    python3 -m CESN.analysis.final.ambiguity --device "$DEVICE"
  run_if_missing "${OUT}/movement_completion/movement_completion_detail.csv" 'Joint completion' \
    python3 -m CESN.analysis.final.movement_completion --device "$DEVICE"
  if [[ "$TASK_SPACE" == 1 ]];then
    run_if_missing "${OUT}/movement_completion_with_fk/movement_completion_detail.csv" 'Cartesian TCP completion (ManiSkill FK)' \
      python3 -m CESN.analysis.final.movement_completion --device "$DEVICE" --task-space \
      --out-dir "${OUT}/movement_completion_with_fk"
  fi
  run_if_missing "${OUT}/baselines/baselines_detail.csv" 'Grid / nearest-demo vs Method A' \
    python3 -m CESN.analysis.final.baselines --full --device "$DEVICE"
  if [[ ! -f "${OUT}/final_report/figures/11_loss_landscape_lbfgs_runs_K30.png" ]];then
    run 'Figure 11 loss landscape' python3 -m CESN.analysis.final.loss_landscape --device "$DEVICE" --verify
  else
    echo 'REUSE Figure 11 loss landscape'
  fi
  report
  ;;
 *)
  echo 'Usage: bash run_all_experiments.sh [status|quick|final|report]' >&2
  echo 'Optional environment variables: CESN_DEVICE=cuda CESN_BATCH_CASES=64 CESN_TASK_SPACE=1' >&2
  exit 2
  ;;
esac
