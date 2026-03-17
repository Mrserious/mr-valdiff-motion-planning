#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Quick demo pipeline:
# 1) goal sampling
# 2) coarse/fine VI solving
# 3) dataset generation
# 4) training
# 5) sampling
#
# This script keeps the original parameter settings, including fine-level=6.

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

TASK_JSON="${TASK_JSON:-unicycle_value_cuda/tasks_standard24/standard10x10_0060_small.json}"
MAP_NAME="${MAP_NAME:-$(basename "${TASK_JSON%.json}")}"
OUT_ROOT="${OUT_ROOT:-data/unicycle_value_grids_demo}"
GOALS_PATH="${GOALS_PATH:-${OUT_ROOT}/${MAP_NAME}/goals.json}"
GOAL_INDEX="${GOAL_INDEX:-0}"

DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
SEED="${SEED:-4242}"
MARGIN="${MARGIN:-0.17}"

ZARR_PATH="${ZARR_PATH:-data/zarr/demo_${MAP_NAME}_goal${GOAL_INDEX}_valuewin23ch.zarr}"
RUN_DIR="${RUN_DIR:-outputs/demo_${MAP_NAME}_goal${GOAL_INDEX}_valuewin23ch}"
INFER_DIR="${INFER_DIR:-${OUT_ROOT}/${MAP_NAME}/goal_${GOAL_INDEX}/infer_demo}"

echo "[demo] map=${MAP_NAME} goal=${GOAL_INDEX}"
echo "[demo] goals=${GOALS_PATH}"
echo "[demo] zarr=${ZARR_PATH}"
echo "[demo] run_dir=${RUN_DIR}"
echo "[demo] infer_dir=${INFER_DIR}"

run_cmd "${PYTHON_BIN}" -m unicycle_value_guided.goal_sampling \
  --task "${TASK_JSON}" \
  --n 50 \
  --seed "${SEED}" \
  --seed-mode per_map \
  --margin "${MARGIN}" \
  --yaw-mode fixed \
  --yaw-deg 0 \
  --out-root "${OUT_ROOT}" \
  --overwrite

run_cmd "${PYTHON_BIN}" -m unicycle_value_guided.solve_value_grids \
  --goals "${GOALS_PATH}" \
  --levels coarse,fine \
  --goal-indices "${GOAL_INDEX}" \
  --grid-scheme multigrid \
  --coarse-level 2 \
  --fine-level 6 \
  --device "${DEVICE}" \
  --dtype float32 \
  --cell-size 0.0 \
  --cell-neighbor-radius 1 \
  --graph-chunk-nodes 2048 \
  --vi-chunk-nodes 8192 \
  --max-iters 500 \
  --tol 1e-6 \
  --keep-pkl \
  --overwrite

run_cmd "${PYTHON_BIN}" -m unicycle_value_guided.make_dataset \
  --goals "${GOALS_PATH}" \
  --out "${ZARR_PATH}" \
  --goal-indices "${GOAL_INDEX}" \
  --starts-per-goal 20 \
  --max-attempts-per-goal 2000 \
  --seed "${SEED}" \
  --seed-mode per_map \
  --clearance "${MARGIN}" \
  --min-goal-dist 1.0 \
  --max-steps 250 \
  --min-steps 8 \
  --crop-size 84 \
  --mpp 0.05 \
  --rotate-with-yaw \
  --crop-mode biased \
  --crop-bias-forward-m 0.9375 \
  --yaw-offsets-deg "-45,-36,-27,-18,-9,0,9,18,27,36,45,135,144,153,162,171,180,189,198,207,216,225" \
  --footprint-length-m 0.625 \
  --footprint-width-m 0.4375 \
  --strict-swept-collision \
  --collision-check-step 0.05 \
  --compressor none \
  --overwrite

run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${PYTHON_BIN}" train.py \
  --config-name=image_nav_diffusion_policy_cnn_valuewin23ch.yaml \
  task.dataset.zarr_path="${ZARR_PATH}" \
  training.device="${DEVICE}" \
  training.seed="${SEED}" \
  hydra.run.dir="${RUN_DIR}"

run_cmd "${PYTHON_BIN}" -m unicycle_value_guided.infer_diffusion \
  --goals "${GOALS_PATH}" \
  --goal-index "${GOAL_INDEX}" \
  --ckpt "${RUN_DIR}/checkpoints/latest.ckpt" \
  --device "${DEVICE}" \
  --use-ema \
  --episodes 10 \
  --max-steps 250 \
  --seed "${SEED}" \
  --clearance "${MARGIN}" \
  --min-goal-dist 3.0 \
  --crop-size 84 \
  --mpp 0.05 \
  --rotate-with-yaw \
  --crop-mode biased \
  --crop-bias-forward-m 0.9375 \
  --collision-check-step 0.05 \
  --collision-semantic swept \
  --projected-collision-stage pre \
  --opt-b-topk-children 5 \
  --plot \
  --out-dir "${INFER_DIR}"

echo "[demo] done"
echo "[demo] training checkpoint: ${RUN_DIR}/checkpoints/latest.ckpt"
echo "[demo] sampling summary:   ${INFER_DIR}/summary.json"
