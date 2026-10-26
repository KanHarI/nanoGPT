#!/usr/bin/env bash
# Run on the dedicated training VM; the watchdog performs bounded recovery.
set -euo pipefail
run_dir="${MINMAX_OUT_DIR:-out/minmax-gpu-24layer-gamma12-20m20m}"
config="${MINMAX_CONFIG:-experiments/minmax/configs/gpu_calibrated.json}"
group="${MINMAX_WANDB_GROUP:-gpu-24layer-gamma12-20m20m}"
read -r -a conditions <<< "${MINMAX_CONDITIONS:-curriculum hard shuffled}"
mkdir -p "$run_dir"
finish() {
    status=$?
    printf '%s\n' "$status" > "$run_dir/exit-code.txt"
    if [[ "${AUTO_SHUTDOWN:-0}" == 1 ]]; then
        sudo shutdown -h +2
    fi
}
trap finish EXIT
.venv/bin/python -u -m experiments.minmax.sweep \
    --config "$config" \
    --out-dir "$run_dir" \
    --conditions "${conditions[@]}" --seeds 0 --resume \
    --wandb online --wandb-entity kanhar --wandb-project minmax-curriculum \
    --wandb-group "$group"
