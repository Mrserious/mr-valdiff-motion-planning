from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.grid import available_levels

from unicycle_value_guided.task_io import dump_json, load_json, load_task
from unicycle_value_guided.value_grid import robot_to_value_grid_2d_min_theta, robot_to_value_grid_3d, save_array_and_meta, save_value_and_meta
from unicycle_value_guided.vi_io import load_vi_robot


def _parse_index_spec(spec: str, n: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            out.append(int(part))
    seen = set()
    dedup: list[int] = []
    for i in out:
        if i in seen:
            continue
        if i < 0 or i >= n:
            raise ValueError(f"goal index out of range: {i} (n={n})")
        seen.add(i)
        dedup.append(i)
    return dedup


def _write_task_with_goal_pose(base_task: dict[str, Any], goal_xyz: list[float], out_path: Path) -> None:
    task = json.loads(json.dumps(base_task))  # cheap deep copy
    robots = task.get("robots", [])
    if not robots:
        raise ValueError("Task has no robots[] entry, cannot set goal.")
    robots[0]["goal_pos"] = [float(goal_xyz[0]), float(goal_xyz[1])]
    # goal_state uses yaw in degrees in unicycle_value_cuda tasks
    robots[0]["goal_state"] = [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(task, out_path)


def _run_unicycle_cuda_vi(
    *,
    task_path: Path,
    grid_scheme: str,
    level: int,
    device: str,
    dtype: str,
    log_dir: Path,
    cell_size: float,
    cell_neighbor_radius: int,
    graph_chunk_nodes: int,
    vi_chunk_nodes: int,
    max_iters: int,
    tol: float,
    strict_zero: bool,
    zero_patience: int,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = log_dir / "vi_robot.pkl"
    summary_path = log_dir / "summary.json"
    if (not overwrite) and out_pkl.exists() and summary_path.exists():
        return out_pkl, json.loads(summary_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "-m",
        "unicycle_value_cuda.run_full_cuda_pipeline",
        "--task",
        str(task_path),
        "--grid-scheme",
        str(grid_scheme),
        "--level",
        str(int(level)),
        "--device",
        str(device),
        "--dtype",
        str(dtype),
        "--log-dir",
        str(log_dir),
        "--cell-size",
        str(float(cell_size)),
        "--cell-neighbor-radius",
        str(int(cell_neighbor_radius)),
        "--graph-chunk-nodes",
        str(int(graph_chunk_nodes)),
        "--vi-chunk-nodes",
        str(int(vi_chunk_nodes)),
        "--max-iters",
        str(int(max_iters)),
        "--tol",
        str(float(tol)),
        "--zero-patience",
        str(int(zero_patience)),
    ]
    if strict_zero:
        cmd.append("--strict-zero")

    subprocess.run(cmd, check=True)
    if not out_pkl.exists():
        raise FileNotFoundError(f"Missing expected VI output: {out_pkl}")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return out_pkl, summary


def solve_values_for_goals(
    *,
    goals_json_path: Path,
    levels: Iterable[str],
    grid_scheme: str,
    coarse_level: int,
    fine_level: int,
    device: str,
    dtype: str,
    cell_size: float,
    cell_neighbor_radius: int,
    graph_chunk_nodes: int,
    vi_chunk_nodes: int,
    max_iters: int,
    tol: float,
    goal_index_spec: str | None,
    overwrite: bool,
    keep_pkl: bool,
) -> None:
    goals_payload = load_json(goals_json_path)
    map_name = str(goals_payload.get("map_name", goals_json_path.parent.name))
    task_path = Path(goals_payload["task_path"])
    if not task_path.is_absolute():
        task_path = (Path.cwd() / task_path).resolve()

    goals: list[list[float]] = goals_payload["goals"]
    if not goals:
        raise ValueError("goals.json has empty goals list.")

    base_task = load_task(task_path)
    map_dir = goals_json_path.parent

    goal_indices = _parse_index_spec(goal_index_spec or "", n=len(goals))
    tmp_tasks_dir = map_dir / "tmp_tasks"
    tmp_tasks_dir.mkdir(parents=True, exist_ok=True)

    levels = [str(l).strip().lower() for l in levels]
    for level in levels:
        if level not in ("coarse", "fine"):
            raise ValueError(f"Unsupported level: {level}")

    for goal_idx in goal_indices:
        goal_xyz = goals[goal_idx]
        if len(goal_xyz) != 3:
            raise ValueError(f"Expected goal pose [x,y,yaw_deg], got: {goal_xyz}")

        goal_dir = map_dir / f"goal_{goal_idx}"
        goal_dir.mkdir(parents=True, exist_ok=True)

        tmp_task_path = tmp_tasks_dir / f"goal_{goal_idx}.json"
        _write_task_with_goal_pose(base_task, goal_xyz, tmp_task_path)

        for level in levels:
            grid_level = int(coarse_level if level == "coarse" else fine_level)
            avail = available_levels(scheme=str(grid_scheme))
            if grid_level not in avail:
                raise ValueError(f"grid_level={grid_level} not in {avail} (scheme={grid_scheme!r})")
            logs_dir = goal_dir / f"logs_{level}"
            vi_pkl, summary = _run_unicycle_cuda_vi(
                task_path=tmp_task_path,
                grid_scheme=str(grid_scheme),
                level=grid_level,
                device=device,
                dtype=dtype,
                log_dir=logs_dir,
                cell_size=float(cell_size),
                cell_neighbor_radius=int(cell_neighbor_radius),
                graph_chunk_nodes=int(graph_chunk_nodes),
                vi_chunk_nodes=int(vi_chunk_nodes),
                max_iters=int(max_iters),
                tol=float(tol),
                strict_zero=True,
                zero_patience=10,
                overwrite=bool(overwrite),
            )

            # Keep a stable filename at goal_dir root (optional but convenient).
            if keep_pkl:
                dst = goal_dir / f"vi_robot_{level}.pkl"
                try:
                    if dst.exists():
                        dst.unlink()
                    os.link(vi_pkl, dst)
                except Exception:
                    shutil.copy2(vi_pkl, dst)

            meta: dict[str, Any] = {
                "map_name": map_name,
                "task_path": str(task_path),
                "tmp_task_path": str(tmp_task_path),
                "goal_index": int(goal_idx),
                "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
                "level": str(level),
                "grid_level": int(grid_level),
                "grid_scheme": str(grid_scheme),
                "solver": {
                    "backend": "unicycle_value_cuda",
                    "device": str(device),
                    "dtype": str(dtype),
                    "cell_size": float(cell_size),
                    "cell_neighbor_radius": int(cell_neighbor_radius),
                    "graph_chunk_nodes": int(graph_chunk_nodes),
                    "vi_chunk_nodes": int(vi_chunk_nodes),
                    "max_iters": int(max_iters),
                    "tol": float(tol),
                },
                "output": {
                    "vi_robot_pkl": str(vi_pkl),
                },
                "summary": summary,
            }

            # Coarse: also export 2D value grid for observation.
            if level == "coarse":
                robot = load_vi_robot(vi_pkl)
                env = base_task["env"]
                # angle_scalor is not in summary; derive from env range (same convention as solver).
                x_limits = env["range"]["limits"][0]
                x0, x1 = float(min(x_limits)), float(max(x_limits))
                angle_scalor = (x1 - x0) / 2.0

                grid, grid_meta = robot_to_value_grid_2d_min_theta(
                    robot=robot,
                    env=env,
                    level=grid_level,
                    scheme=str(grid_scheme),
                    angle_scalor=float(angle_scalor),
                    fill_value=1.0,
                )
                meta.update(grid_meta)
                save_value_and_meta(goal_dir / "value_coarse.npy", goal_dir / "meta_coarse.json", grid, meta)

                # Latest unified pipeline: also export full V(x,y,theta) for yaw-window slices.
                grid3d, grid3d_meta = robot_to_value_grid_3d(
                    robot=robot,
                    env=env,
                    level=grid_level,
                    scheme=str(grid_scheme),
                    angle_scalor=float(angle_scalor),
                    fill_value=1.0,
                )
                meta3d = dict(meta)
                meta3d.update(grid3d_meta)
                save_array_and_meta(goal_dir / "value_coarse_3d.npy", goal_dir / "meta_coarse_3d.json", grid3d.V, meta3d)
            else:
                (goal_dir / "meta_fine.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt02: solve unicycle coarse/fine VI (CUDA) and export coarse value grid.")
    p.add_argument("--goals", type=str, required=True, help="Path to data/unicycle_value_grids/<map>/goals.json")
    p.add_argument("--levels", type=str, default="coarse,fine", help='Comma list: "coarse,fine" or "coarse".')
    p.add_argument("--goal-indices", type=str, default="", help='Subset of goals, e.g. "0-9,20". Empty=all.')
    p.add_argument("--grid-scheme", type=str, default="multigrid", choices=["legacy", "multigrid"], help="Grid scheme (default: multigrid).")
    p.add_argument("--coarse-level", type=int, default=2, help="Grid level for coarse VI (default: 2).")
    p.add_argument("--fine-level", type=int, default=6, help="Grid level for fine VI (default: 6).")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch CUDA device string.")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    p.add_argument("--cell-size", type=float, default=0.0, help="Bucket cell size (0 => use solver rho).")
    p.add_argument("--cell-neighbor-radius", type=int, default=1, help="Neighbor cell Chebyshev radius.")
    p.add_argument("--graph-chunk-nodes", type=int, default=2048, help="Nodes per CUDA graph build chunk.")
    p.add_argument("--vi-chunk-nodes", type=int, default=8192, help="Nodes per CUDA VI sub-chunk.")
    p.add_argument("--max-iters", type=int, default=500, help="Max VI sweeps.")
    p.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance (non-strict).")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if artifacts exist.")
    p.add_argument("--keep-pkl", action="store_true", help="Also hardlink/copy vi_robot_{level}.pkl to goal dir root.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    solve_values_for_goals(
        goals_json_path=Path(args.goals),
        levels=[s.strip() for s in str(args.levels).split(",") if s.strip()],
        grid_scheme=str(args.grid_scheme),
        coarse_level=int(args.coarse_level),
        fine_level=int(args.fine_level),
        device=str(args.device),
        dtype=str(args.dtype),
        cell_size=float(args.cell_size),
        cell_neighbor_radius=int(args.cell_neighbor_radius),
        graph_chunk_nodes=int(args.graph_chunk_nodes),
        vi_chunk_nodes=int(args.vi_chunk_nodes),
        max_iters=int(args.max_iters),
        tol=float(args.tol),
        goal_index_spec=str(args.goal_indices),
        overwrite=bool(args.overwrite),
        keep_pkl=bool(args.keep_pkl),
    )


if __name__ == "__main__":
    main()
