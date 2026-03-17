from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .collision import obstacle_free_mask_unicycle
from .grid import available_levels, build_state_grid
from .task_io import get_range_limits, load_task
from .unicycle import Unicycle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unicycle CPU value solver (node-level, KDTree reference)")
    parser.add_argument("--task", type=str, required=True, help="Path to task json.")
    parser.add_argument("--grid-scheme", type=str, default="legacy", choices=["legacy", "multigrid"], help="Grid scheme.")
    parser.add_argument("--level", type=int, default=0, help="Grid level index (scheme-dependent).")
    parser.add_argument("--log-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--max-iters", type=int, default=500, help="Max sweeps.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance (used when strict-zero disabled).")
    parser.add_argument("--strict-zero", action="store_true", help="Stop only when max delta == 0 for patience steps.")
    parser.add_argument("--zero-patience", type=int, default=3, help="Strict-zero patience.")
    return parser.parse_args()


def _wrap_theta(theta_scaled: np.ndarray, angle_scalor: float) -> np.ndarray:
    a = float(angle_scalor)
    p = 2.0 * a
    return theta_scaled - np.floor((theta_scaled + a) / p) * p


def _build_children_kdtree(
    robot: Unicycle,
    states: np.ndarray,
    *,
    dt: float,
    rho: float,
    angle_scalor: float,
) -> List[List[int]]:
    controls = np.asarray(robot.controls, dtype=np.float32)
    u_lin = controls[:, 0].astype(np.float32)
    u_ang = controls[:, 1].astype(np.float32)

    children: List[List[int]] = []
    p = 2.0 * float(angle_scalor)
    k_pi = float(np.pi)

    for s in states.astype(np.float32, copy=False):
        theta_real = float(s[2]) * k_pi / float(angle_scalor)
        cos_t = float(np.cos(theta_real))
        sin_t = float(np.sin(theta_real))

        cand_x = float(s[0]) + float(dt) * cos_t * u_lin
        cand_y = float(s[1]) + float(dt) * sin_t * u_lin
        cand_theta = _wrap_theta(float(s[2]) + float(dt) * u_ang, angle_scalor=float(angle_scalor))
        cand = np.stack([cand_x, cand_y, cand_theta], axis=1).astype(np.float32, copy=False)

        phase = cand.copy()
        phase[:, 2] = np.where(phase[:, 2] > 0.0, phase[:, 2] - p, phase[:, 2] + p)
        queries = np.vstack([cand, phase])

        idx = robot.kdtree.query_radius(queries, float(rho), return_distance=False)
        neighbor_set: set[int] = set()
        for arr in idx:
            for j in arr.tolist():
                neighbor_set.add(int(j))
        children.append(sorted(neighbor_set))

    return children


def _value_iteration_cpu(
    children: List[List[int]],
    goal_mask: np.ndarray,
    initial_values: np.ndarray,
    *,
    delta: float,
    beta: float,
    max_iters: int,
    tol: float,
    strict_zero: bool,
    zero_patience: int,
) -> tuple[np.ndarray, int, float]:
    current = initial_values.astype(np.float32, copy=True)
    next_values = current.copy()
    delta_f = np.float32(delta)
    beta_f = np.float32(beta)
    one_f = np.float32(1.0)

    patience = int(max(1, zero_patience))
    zero_hits = 0
    iterations_run = 0
    last_delta = 0.0

    for _ in range(int(max(0, max_iters))):
        max_delta = 0.0
        for i, nbrs in enumerate(children):
            old_v = current[i]
            if bool(goal_mask[i]):
                new_v = np.float32(0.0)
            else:
                best = np.float32(np.inf)
                for j in nbrs:
                    cand = delta_f + beta_f * current[j]
                    if cand < best:
                        best = cand
                if np.isfinite(best):
                    new_v = best if best < one_f else one_f
                else:
                    new_v = old_v
            next_values[i] = new_v
            dv = float(np.abs(new_v - old_v))
            if dv > max_delta:
                max_delta = dv

        iterations_run += 1
        last_delta = float(max_delta)
        current, next_values = next_values, current

        if strict_zero:
            if max_delta == 0.0:
                zero_hits += 1
                if zero_hits >= patience:
                    break
            else:
                zero_hits = 0
        else:
            if max_delta <= float(tol):
                break

    return current, iterations_run, last_delta


def run_pipeline() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    levels = available_levels(scheme=str(args.grid_scheme))
    if int(args.level) not in levels:
        raise ValueError(f"Unsupported level={args.level} for scheme={args.grid_scheme!r}. Available: {levels}")
    task = load_task(args.task)
    env: Dict[str, Any] = task.env
    robots = task.robots
    if not robots:
        raise ValueError("Task has no robots.")
    robot_meta = robots[0]

    (x0, x1), _ = get_range_limits(env)
    angle_scalor = (x1 - x0) / 2.0
    robot = Unicycle(env, robot_meta, angle_scalor=angle_scalor, robot_id=0)

    # 1) Build grid nodes (half-open boundary convention).
    raw_states, grid_spec = build_state_grid(
        env,
        angle_scalor=angle_scalor,
        level=int(args.level),
        scheme=str(args.grid_scheme),
    )

    # 2) Filter collision states (not swept), aligned with notebook semantics.
    mask = obstacle_free_mask_unicycle(
        env=env,
        body_samples_xy=robot.body.samples,
        states=raw_states,
        angle_scalor=angle_scalor,
    )
    states = raw_states[mask]

    # 3) Ensure goal exists by explicitly inserting goal_state (if collision-free).
    goal_state = robot.goal_state.astype(np.float32)
    if obstacle_free_mask_unicycle(
        env=env,
        body_samples_xy=robot.body.samples,
        states=goal_state.reshape(1, 3),
        angle_scalor=angle_scalor,
    )[0]:
        already_present = bool(np.any(np.all(states == goal_state.reshape(1, 3), axis=1)))
        if not already_present:
            states = np.vstack([goal_state.reshape(1, 3), states])

    for s in states:
        robot.add_node_state(s)
    robot.init_nodes()
    robot.update_kdtree()

    d_val = robot.get_spatial_res()
    dt = robot.get_temporal_res()
    rho = robot.get_perturbation_radius()

    # 4) Build graph via KDTree (phase copy query).
    build_start = time.time()
    children = _build_children_kdtree(robot, states, dt=dt, rho=rho, angle_scalor=angle_scalor)
    build_sec = time.time() - build_start

    # 5) Value iteration.
    goal_mask = np.asarray([bool(robot.within_goal(n.state)) for n in robot.nodes], dtype=bool)
    initial_values = np.asarray([float(n.value) for n in robot.nodes], dtype=np.float32)
    delta = 1.0 - math.exp(-(dt - d_val))
    beta = 1.0 - delta

    vi_start = time.time()
    values, iterations, last_delta = _value_iteration_cpu(
        children,
        goal_mask,
        initial_values,
        delta=float(delta),
        beta=float(beta),
        max_iters=int(args.max_iters),
        tol=float(args.tol),
        strict_zero=bool(args.strict_zero),
        zero_patience=int(args.zero_patience),
    )
    vi_sec = time.time() - vi_start

    # 6) Write back values/children and save robot.
    for idx, node in enumerate(robot.nodes):
        node.value = float(values[idx])
        node.children.indices = list(children[idx])
        node.children.update_iteration = -1

    out_pkl = log_dir / "vi_robot.pkl"
    tmp_pkl = out_pkl.with_name(f".{out_pkl.name}.tmp.{os.getpid()}")
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    try:
        with tmp_pkl.open("wb") as fh:
            import pickle  # noqa: WPS433

            pickle.dump({"robot": robot}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
        os.replace(tmp_pkl, out_pkl)
    finally:
        if gc_enabled:
            gc.enable()
        try:
            if tmp_pkl.exists():
                tmp_pkl.unlink()
        except Exception:
            pass

    summary = {
        "task": str(Path(args.task).resolve()),
        "level": int(args.level),
        "grid_total_raw": int(grid_spec.total),
        "nodes_after_filter": int(states.shape[0]),
        "dt": float(dt),
        "rho": float(rho),
        "d": float(d_val),
        "delta": float(delta),
        "beta": float(beta),
        "controls": int(len(robot.controls)),
        "unique_edges_total": int(sum(len(c) for c in children)),
        "vi_iterations": int(iterations),
        "vi_last_delta": float(last_delta),
        "timing_sec": {
            "total": float(time.time() - t0),
            "graph_build": float(build_sec),
            "value_iteration": float(vi_sec),
        },
        "output": {
            "vi_robot_pkl": str(out_pkl),
        },
    }

    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_pipeline()
