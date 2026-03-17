from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.grid import available_levels

from unicycle_value_guided.task_io import dump_json, load_task
from unicycle_value_guided.value_grid import (
    robot_to_value_grid_2d_min_theta,
    robot_to_value_grid_3d,
    save_array_and_meta,
    save_value_and_meta,
)
from unicycle_value_guided.value_grid3d import RegularValueGrid3D, load_regular_value_grid_3d
from unicycle_value_guided.vi_io import load_vi_robot


def _delta_to_dirname(delta: float) -> str:
    return f"delta_{float(delta):.3f}"


def _delta_bw_to_dirname(delta: float, boundary_wall_thickness_m: float) -> str:
    bw = float(boundary_wall_thickness_m)
    if not (np.isfinite(bw) and bw >= 0):
        raise ValueError(f"boundary_wall_thickness_m must be finite and >=0, got {boundary_wall_thickness_m}")
    if bw <= 0:
        return _delta_to_dirname(float(delta))
    return f"{_delta_to_dirname(float(delta))}_bw_{bw:.3f}"


def _add_boundary_walls_as_obstacles(task: dict[str, Any], *, thickness_m: float) -> dict[str, Any]:
    """
    Return a deep-copied task with 4 thin rectangular obstacles added along env.range's inner boundary.

    This is used to create a boundary "no-go" band in Opt-A planning semantics without changing env.range itself.
    """
    th = float(thickness_m)
    if not (np.isfinite(th) and th > 0):
        raise ValueError(f"thickness_m must be finite and >0, got {thickness_m}")

    out = deepcopy(task)
    env = dict(out.get("env", {}))
    rng = env.get("range", {})
    if rng.get("shape") != "rectangle":
        raise ValueError(f"Only rectangle range is supported for boundary walls, got: {rng.get('shape')!r}")
    limits = rng.get("limits", None)
    if limits is None or len(limits) != 2 or len(limits[0]) != 2 or len(limits[1]) != 2:
        raise ValueError(f"Invalid env.range.limits: {limits}")
    xmin, xmax = float(min(limits[0])), float(max(limits[0]))
    ymin, ymax = float(min(limits[1])), float(max(limits[1]))
    if th >= 0.5 * (xmax - xmin) or th >= 0.5 * (ymax - ymin):
        raise ValueError(f"boundary wall thickness too large for range: thickness={th} range=({xmin},{xmax},{ymin},{ymax})")

    obstacles = env.get("obstacles", [])
    if obstacles is None:
        obstacles = []
    if not isinstance(obstacles, list):
        raise ValueError(f"Invalid env.obstacles (not a list): {type(obstacles)}")
    obstacles = list(obstacles)

    obstacles.extend(
        [
            {
                "shape": "rectangle",
                "limits": [[float(xmin), float(xmin + th)], [float(ymin), float(ymax)]],
                "name": "boundary_wall_left",
            },
            {
                "shape": "rectangle",
                "limits": [[float(xmax - th), float(xmax)], [float(ymin), float(ymax)]],
                "name": "boundary_wall_right",
            },
            {
                "shape": "rectangle",
                "limits": [[float(xmin), float(xmax)], [float(ymin), float(ymin + th)]],
                "name": "boundary_wall_bottom",
            },
            {
                "shape": "rectangle",
                "limits": [[float(xmin), float(xmax)], [float(ymax - th), float(ymax)]],
                "name": "boundary_wall_top",
            },
        ]
    )

    env["obstacles"] = obstacles
    out["env"] = env
    return out


def _goal_xyz_tag(goal_xyz: Sequence[float]) -> str:
    arr = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:10]


def inflate_task_obstacles(task: dict[str, Any], *, delta: float) -> dict[str, Any]:
    """
    Return a deep-copied task where every obstacle is inflated outward by `delta`.

    Supported shapes:
      - circle: radius += delta
      - rectangle: limits expand by delta on all sides
    """
    delta = float(delta)
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    out = deepcopy(task)
    env = dict(out.get("env", {}))
    obstacles = env.get("obstacles", [])
    if obstacles is None:
        obstacles = []
    if not isinstance(obstacles, list):
        raise ValueError(f"Invalid env.obstacles (not a list): {type(obstacles)}")

    inflated: list[dict[str, Any]] = []
    for ob in obstacles:
        if not isinstance(ob, dict):
            raise ValueError(f"Invalid obstacle entry (not a dict): {type(ob)}")
        shape = ob.get("shape", None)
        ob2 = dict(ob)
        if shape == "circle":
            ob2["radius"] = float(ob2["radius"]) + delta
        elif shape == "rectangle":
            limits = ob2.get("limits", None)
            if limits is None or len(limits) != 2 or len(limits[0]) != 2 or len(limits[1]) != 2:
                raise ValueError(f"Invalid rectangle limits: {limits}")
            x1, x2 = float(limits[0][0]), float(limits[0][1])
            y1, y2 = float(limits[1][0]), float(limits[1][1])
            xmin, xmax = min(x1, x2) - delta, max(x1, x2) + delta
            ymin, ymax = min(y1, y2) - delta, max(y1, y2) + delta
            ob2["limits"] = [[float(xmin), float(xmax)], [float(ymin), float(ymax)]]
        else:
            raise ValueError(f"Unsupported obstacle shape for inflation: {shape}")
        inflated.append(ob2)

    env["obstacles"] = inflated
    out["env"] = env
    return out


def task_with_goal_pose(task: dict[str, Any], *, goal_xyz: Sequence[float]) -> dict[str, Any]:
    goal_xyz = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    out = deepcopy(task)
    robots = out.get("robots", [])
    if not isinstance(robots, list) or not robots:
        raise ValueError("Task has no robots[] entry, cannot set goal.")
    robots = [dict(r) for r in robots]
    robots[0] = dict(robots[0])
    robots[0]["goal_pos"] = [float(goal_xyz[0]), float(goal_xyz[1])]
    robots[0]["goal_state"] = [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])]
    out["robots"] = robots
    return out


try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as fh:
        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                print(f"[opt-a] waiting for cache lock: {lock_path}", flush=True)
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _safe_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _angle_scalor_from_task(task: dict[str, Any]) -> float:
    env = task.get("env", {})
    limits = env.get("range", {}).get("limits", None)
    if limits is None or len(limits) != 2 or len(limits[0]) != 2:
        raise ValueError(f"Invalid env.range.limits: {limits}")
    x0, x1 = float(min(limits[0])), float(max(limits[0]))
    return float((x1 - x0) / 2.0)


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
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = log_dir / "vi_robot.pkl"
    summary_path = log_dir / "summary.json"
    if (not overwrite) and _nonempty_file(out_pkl) and summary_path.exists():
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
        "--strict-zero",
        "--zero-patience",
        "10",
    ]
    subprocess.run(cmd, check=True)
    if not _nonempty_file(out_pkl):
        raise FileNotFoundError(f"Missing expected VI output: {out_pkl}")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return out_pkl, summary


@dataclass(frozen=True)
class InflatedGoalAssets:
    goal_dir: Path
    tmp_task_path: Path
    task_obs: dict[str, Any]
    coarse_grid3d: RegularValueGrid3D
    coarse_robot: Any
    fine_robot: Any | None


def _get_solver_cfg(meta_src: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(meta_src, dict):
        return {}
    solver = meta_src.get("solver", None)
    return solver if isinstance(solver, dict) else {}


def prepare_inflated_goal_assets(
    *,
    base_task: dict[str, Any],
    base_task_path: str | Path,
    map_name: str,
    goal_idx: int,
    goal_xyz: Sequence[float],
    cache_root: str | Path,
    delta: float,
    boundary_wall_thickness_m: float = 0.0,
    coarse_meta_src: dict[str, Any],
    fine_meta_src: dict[str, Any] | None,
    vi_device: str | None = None,
    vi_dtype: str | None = None,
    max_iters: int | None = None,
    tol: float | None = None,
    overwrite: bool = False,
    keep_pkl: bool = True,
    use_inflated_fine: bool = True,
) -> InflatedGoalAssets:
    """
    Opt-A (inflation) cache builder for unicycle VI assets.

    Returns assets to be used for inference-time planning semantics:
      - task_obs: inflated obstacles + goal pose (for occupancy)
      - coarse_grid3d: inflated coarse V(x,y,theta) (for yaw-window value slices)
      - coarse_robot: inflated coarse VI robot (for gpath)
      - fine_robot: inflated fine VI robot (optional; for snap-to-children semantics)
    """
    base_task_path = Path(base_task_path).resolve()
    map_name = str(map_name)
    goal_idx = int(goal_idx)
    goal_xyz = np.asarray(goal_xyz, dtype=np.float64).reshape(3)

    delta = float(delta)
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    cache_root = Path(cache_root)
    map_dir = cache_root / map_name / _delta_bw_to_dirname(delta, float(boundary_wall_thickness_m))
    map_dir.mkdir(parents=True, exist_ok=True)

    goal_tag = _goal_xyz_tag(goal_xyz)
    goal_dir = map_dir / f"goal_{goal_idx}_{goal_tag}"
    tmp_task_path = map_dir / "tmp_tasks" / f"goal_{goal_idx}_{goal_tag}.json"
    lock_path = map_dir / f".lock_goal_{goal_idx}_{goal_tag}.lock"

    coarse_solver = _get_solver_cfg(coarse_meta_src)
    grid_scheme = str(coarse_meta_src.get("grid_scheme", "multigrid"))
    coarse_level = int(coarse_meta_src.get("grid_level", coarse_meta_src.get("level", 2)))

    fine_solver = _get_solver_cfg(fine_meta_src) if fine_meta_src is not None else {}
    fine_level = int((fine_meta_src or {}).get("grid_level", (fine_meta_src or {}).get("level", 6))) if fine_meta_src is not None else 6

    vi_device_eff = str(vi_device or coarse_solver.get("device") or "cuda:0")
    vi_dtype_eff = str(vi_dtype or coarse_solver.get("dtype") or "float32")
    cell_size_eff = float(coarse_solver.get("cell_size", 0.0))
    cell_neighbor_radius_eff = int(coarse_solver.get("cell_neighbor_radius", 1))
    graph_chunk_nodes_eff = int(coarse_solver.get("graph_chunk_nodes", 2048))
    vi_chunk_nodes_eff = int(coarse_solver.get("vi_chunk_nodes", 8192))
    max_iters_eff = int(max_iters if max_iters is not None else coarse_solver.get("max_iters", 500))
    tol_eff = float(tol if tol is not None else coarse_solver.get("tol", 1e-6))

    def _coarse_paths() -> tuple[Path, Path, Path, Path]:
        return (
            goal_dir / "value_coarse.npy",
            goal_dir / "meta_coarse.json",
            goal_dir / "value_coarse_3d.npy",
            goal_dir / "meta_coarse_3d.json",
        )

    with _exclusive_lock(lock_path):
        # 1) build inflated task used for observation semantics
        if overwrite or (not tmp_task_path.exists()):
            bw = float(boundary_wall_thickness_m)
            if not (np.isfinite(bw) and bw >= 0):
                raise ValueError(f"boundary_wall_thickness_m must be finite and >=0, got {boundary_wall_thickness_m}")
            task0 = base_task
            if bw > 0:
                task0 = _add_boundary_walls_as_obstacles(task0, thickness_m=float(bw))
            task_obs = inflate_task_obstacles(task0, delta=delta)
            task_obs = task_with_goal_pose(task_obs, goal_xyz=goal_xyz)
            dump_json(task_obs, tmp_task_path)
        else:
            task_obs = load_task(tmp_task_path)

        goal_dir.mkdir(parents=True, exist_ok=True)

        # 2) coarse VI + export 2D/3D value grids
        avail = available_levels(scheme=str(grid_scheme))
        if coarse_level not in avail:
            raise ValueError(f"coarse_level={coarse_level} not in {avail} (scheme={grid_scheme!r})")
        logs_coarse = goal_dir / "logs_coarse"
        coarse_pkl, coarse_summary = _run_unicycle_cuda_vi(
            task_path=tmp_task_path,
            grid_scheme=str(grid_scheme),
            level=int(coarse_level),
            device=str(vi_device_eff),
            dtype=str(vi_dtype_eff),
            log_dir=logs_coarse,
            cell_size=float(cell_size_eff),
            cell_neighbor_radius=int(cell_neighbor_radius_eff),
            graph_chunk_nodes=int(graph_chunk_nodes_eff),
            vi_chunk_nodes=int(vi_chunk_nodes_eff),
            max_iters=int(max_iters_eff),
            tol=float(tol_eff),
            overwrite=bool(overwrite),
        )
        if keep_pkl:
            _safe_hardlink_or_copy(coarse_pkl, goal_dir / "vi_robot_coarse.pkl")

        value2d_path, meta2d_path, value3d_path, meta3d_path = _coarse_paths()
        if overwrite or (not _nonempty_file(value3d_path)) or (not meta3d_path.exists()):
            coarse_robot = load_vi_robot(coarse_pkl)
            angle_scalor = _angle_scalor_from_task(task_obs)
            grid2d, grid2d_meta = robot_to_value_grid_2d_min_theta(
                robot=coarse_robot,
                env=task_obs["env"],
                level=int(coarse_level),
                scheme=str(grid_scheme),
                angle_scalor=float(angle_scalor),
                fill_value=1.0,
            )
            grid3d, grid3d_meta = robot_to_value_grid_3d(
                robot=coarse_robot,
                env=task_obs["env"],
                level=int(coarse_level),
                scheme=str(grid_scheme),
                angle_scalor=float(angle_scalor),
                fill_value=1.0,
            )

            meta_common: dict[str, Any] = {
                "map_name": map_name,
                "task_path": str(base_task_path),
                "tmp_task_path": str(tmp_task_path),
                "goal_index": int(goal_idx),
                "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
                "level": "coarse",
                "grid_level": int(coarse_level),
                "grid_scheme": str(grid_scheme),
                "solver": {
                    "backend": "unicycle_value_cuda",
                    "device": str(vi_device_eff),
                    "dtype": str(vi_dtype_eff),
                    "cell_size": float(cell_size_eff),
                    "cell_neighbor_radius": int(cell_neighbor_radius_eff),
                    "graph_chunk_nodes": int(graph_chunk_nodes_eff),
                    "vi_chunk_nodes": int(vi_chunk_nodes_eff),
                    "max_iters": int(max_iters_eff),
                    "tol": float(tol_eff),
                },
                "output": {
                    "vi_robot_pkl": str(coarse_pkl),
                },
                "summary": coarse_summary,
                "opt_a": {
                    "enabled": True,
                    "delta": float(delta),
                },
            }

            meta2d = dict(meta_common)
            meta2d.update(grid2d_meta)
            save_value_and_meta(value2d_path, meta2d_path, grid2d, meta2d)

            meta3d = dict(meta_common)
            meta3d.update(grid3d_meta)
            save_array_and_meta(value3d_path, meta3d_path, grid3d.V, meta3d)

        coarse_grid3d = load_regular_value_grid_3d(value3d_path, meta3d_path)
        coarse_robot = load_vi_robot(logs_coarse / "vi_robot.pkl")

        # 3) fine VI (optional)
        fine_robot = None
        if bool(use_inflated_fine):
            if fine_level not in avail:
                raise ValueError(f"fine_level={fine_level} not in {avail} (scheme={grid_scheme!r})")

            vi_device_f = str(vi_device or fine_solver.get("device") or vi_device_eff)
            vi_dtype_f = str(vi_dtype or fine_solver.get("dtype") or vi_dtype_eff)
            cell_size_f = float(fine_solver.get("cell_size", cell_size_eff))
            cell_neighbor_radius_f = int(fine_solver.get("cell_neighbor_radius", cell_neighbor_radius_eff))
            graph_chunk_nodes_f = int(fine_solver.get("graph_chunk_nodes", graph_chunk_nodes_eff))
            vi_chunk_nodes_f = int(fine_solver.get("vi_chunk_nodes", vi_chunk_nodes_eff))
            max_iters_f = int(max_iters if max_iters is not None else fine_solver.get("max_iters", max_iters_eff))
            tol_f = float(tol if tol is not None else fine_solver.get("tol", tol_eff))

            logs_fine = goal_dir / "logs_fine"
            fine_pkl, fine_summary = _run_unicycle_cuda_vi(
                task_path=tmp_task_path,
                grid_scheme=str(grid_scheme),
                level=int(fine_level),
                device=str(vi_device_f),
                dtype=str(vi_dtype_f),
                log_dir=logs_fine,
                cell_size=float(cell_size_f),
                cell_neighbor_radius=int(cell_neighbor_radius_f),
                graph_chunk_nodes=int(graph_chunk_nodes_f),
                vi_chunk_nodes=int(vi_chunk_nodes_f),
                max_iters=int(max_iters_f),
                tol=float(tol_f),
                overwrite=bool(overwrite),
            )
            if keep_pkl:
                _safe_hardlink_or_copy(fine_pkl, goal_dir / "vi_robot_fine.pkl")

            meta_f: dict[str, Any] = {
                "map_name": map_name,
                "task_path": str(base_task_path),
                "tmp_task_path": str(tmp_task_path),
                "goal_index": int(goal_idx),
                "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
                "level": "fine",
                "grid_level": int(fine_level),
                "grid_scheme": str(grid_scheme),
                "solver": {
                    "backend": "unicycle_value_cuda",
                    "device": str(vi_device_f),
                    "dtype": str(vi_dtype_f),
                    "cell_size": float(cell_size_f),
                    "cell_neighbor_radius": int(cell_neighbor_radius_f),
                    "graph_chunk_nodes": int(graph_chunk_nodes_f),
                    "vi_chunk_nodes": int(vi_chunk_nodes_f),
                    "max_iters": int(max_iters_f),
                    "tol": float(tol_f),
                },
                "output": {
                    "vi_robot_pkl": str(fine_pkl),
                },
                "summary": fine_summary,
                "opt_a": {
                    "enabled": True,
                    "delta": float(delta),
                },
            }
            (goal_dir / "meta_fine.json").write_text(json.dumps(meta_f, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            fine_robot = load_vi_robot(fine_pkl)

        return InflatedGoalAssets(
            goal_dir=goal_dir,
            tmp_task_path=tmp_task_path,
            task_obs=task_obs,
            coarse_grid3d=coarse_grid3d,
            coarse_robot=coarse_robot,
            fine_robot=fine_robot,
        )
