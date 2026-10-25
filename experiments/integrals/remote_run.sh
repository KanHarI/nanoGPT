#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${INTEGRAL_CONFIG:?Set the calibrated config path}"
: "${INTEGRAL_DATA:?Set the complete training corpus directory}"
: "${INTEGRAL_OUTPUT:?Set the run output directory}"
: "${INTEGRAL_ARCHIVE:?Set a separately mounted persistent archive directory}"
integral_python="${INTEGRAL_PYTHON:-.venv/bin/python}"
integral_args=(--config "$INTEGRAL_CONFIG" --data-dir "$INTEGRAL_DATA"
  --out-dir "$INTEGRAL_OUTPUT" --archive-dir "$INTEGRAL_ARCHIVE" --resume
  --wandb "${INTEGRAL_WANDB:-online}" --wandb-project "${INTEGRAL_WANDB_PROJECT:-numerical-integrals-curriculum}")
if [[ -n "${INTEGRAL_WANDB_ENTITY:-}" ]]; then
  integral_args+=(--wandb-entity "$INTEGRAL_WANDB_ENTITY")
fi
if [[ "${INTEGRAL_SHUTDOWN_ON_SUCCESS:-0}" == 1 ]]; then
  integral_args+=(--shutdown-on-success)
fi
exec "$integral_python" -m experiments.integrals.sweep "${integral_args[@]}"
