#!/usr/bin/env bash
# Run from the repository root on the dedicated training VM.
# AUTO_SHUTDOWN=1 powers off that VM two minutes after success or failure.
set -euo pipefail
mkdir -p out/gpu-64layer-gamma12
finish() {
    status=$?
    printf '%s\n' "$status" > out/gpu-64layer-gamma12/exit-code.txt
    if [[ "${AUTO_SHUTDOWN:-0}" == 1 ]]; then
        sudo shutdown -h +2
    fi
}
trap finish EXIT
.venv/bin/python -u -m experiments.finite_field.sweep \
    --config experiments/finite_field/configs/gpu_calibrated.json \
    --out-dir out/gpu-64layer-gamma12 \
    --conditions curriculum hard mixture --seeds 0 --resume --use-run-plan --skip-complete \
    --wandb online --wandb-entity kanhar \
    --wandb-group gpu-64layer-gamma12
