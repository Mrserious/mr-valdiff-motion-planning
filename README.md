# Unicycle Quick Demo

This folder is a self-contained demo for the full pipeline:
- dataset generation
- training
- sampling (inference)

The quick demo keeps the original settings. In particular, `fine-level` is fixed to `6`.

## Prerequisites

- Python 3.9+ (3.10 recommended)
- A CUDA-capable GPU for training and inference

## Run

From inside `quick_demo`:

```bash
bash run_unicycle_quick_demo.sh
```

From any other working directory:

```bash
bash /path/to/quick_demo/run_unicycle_quick_demo.sh
```

The script always switches to the `quick_demo` directory before executing commands.

## Optional Environment Variables

You can override output paths or devices before running:

```bash
TASK_JSON=unicycle_value_cuda/tasks_standard24/standard10x10_0060_small.json
MAP_NAME=standard10x10_0060_small
OUT_ROOT=data/unicycle_value_grids_demo
GOALS_PATH=data/unicycle_value_grids_demo/standard10x10_0060_small/goals.json
GOAL_INDEX=0

DEVICE=cuda:0
CUDA_VISIBLE_DEVICES_VALUE=0
SEED=4242
MARGIN=0.17

ZARR_PATH=data/zarr/demo_standard10x10_0060_small_goal0_valuewin23ch.zarr
RUN_DIR=outputs/demo_standard10x10_0060_small_goal0_valuewin23ch
INFER_DIR=data/unicycle_value_grids_demo/standard10x10_0060_small/goal_0/infer_demo
PYTHON_BIN=python
```

Example:

```bash
DEVICE=cuda:1 CUDA_VISIBLE_DEVICES_VALUE=1 bash run_unicycle_quick_demo.sh
```

## Dry Run

Print all commands without executing:

```bash
DRY_RUN=1 bash run_unicycle_quick_demo.sh
```

## Outputs

- Training checkpoint:
  - `outputs/demo_<map>_goal<k>_valuewin23ch/checkpoints/latest.ckpt`
- Sampling summary:
  - `data/unicycle_value_grids_demo/<map>/goal_<k>/infer_demo/summary.json`

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```
