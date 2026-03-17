from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from .unicycle import Unicycle


@dataclass(frozen=True)
class TrajectoryResult:
    path_indices: List[int]
    path_states: np.ndarray  # (T,3)
    controls: np.ndarray  # (T-1,2)


def _load_robot(pkl_path: Path) -> Unicycle:
    with pkl_path.open("rb") as fh:
        obj = pickle.load(fh)
    robot = obj.get("robot", obj)
    if not isinstance(robot, Unicycle):
        raise TypeError(f"Expected Unicycle in {pkl_path}, got {type(robot)}")
    return robot


def _preferred_run_dir(outputs_root: Path, *, level: int, prefer: Sequence[str]) -> Path:
    candidates = [f"{prefix}_level{level}" for prefix in prefer]
    candidates.extend([f"gpu2_level{level}", f"gpu1_level{level}", f"cpu_level{level}"])
    for name in candidates:
        run_dir = outputs_root / name
        if (run_dir / "vi_robot.pkl").exists():
            return run_dir
    raise FileNotFoundError(f"No vi_robot.pkl found for level={level} under {outputs_root}")


def _draw_env(ax: plt.Axes, env: Dict[str, Any]) -> None:
    rng = env.get("range", {})
    if rng.get("shape") == "rectangle":
        lims = rng["limits"]
        x0, x1 = float(min(lims[0])), float(max(lims[0]))
        y0, y1 = float(min(lims[1])), float(max(lims[1]))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    for obs in env.get("obstacles", []):
        if obs.get("shape") == "circle":
            center = obs["center"]
            r = float(obs["radius"])
            ax.add_patch(Circle((float(center[0]), float(center[1])), r, facecolor="black", edgecolor="none", alpha=0.35))
        elif obs.get("shape") == "rectangle":
            lims = obs["limits"]
            x0, x1 = float(min(lims[0])), float(max(lims[0]))
            y0, y1 = float(min(lims[1])), float(max(lims[1]))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="black", edgecolor="none", alpha=0.35))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.25)


def _draw_robot_body(
    ax: plt.Axes,
    robot: Unicycle,
    state: np.ndarray,
    *,
    edgecolor: str,
    facecolor: str,
    alpha: float,
    linewidth: float = 1.25,
) -> None:
    state = np.asarray(state, dtype=np.float32)
    pos = state[:2]
    shape = robot.body.shape

    if shape == "point":
        ax.plot([float(pos[0])], [float(pos[1])], marker="o", markersize=3.5, color=edgecolor, alpha=alpha)
        return

    if shape == "circle":
        r = float(robot.body.radius or 0.0)
        ax.add_patch(
            Circle(
                (float(pos[0]), float(pos[1])),
                r,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        )
        return

    if shape == "rectangle":
        if robot.body.size is None:
            raise ValueError("Rectangle body missing size.")
        length, width = float(robot.body.size[0]), float(robot.body.size[1])
        hl, hw = length / 2.0, width / 2.0
        corners_local = np.array(
            [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]],
            dtype=np.float32,
        )
        angle = float(robot.get_real_angle(float(state[2]), unit="radian"))
        c, s = float(math.cos(angle)), float(math.sin(angle))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners = corners_local @ rot.T + pos[None, :]
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                joinstyle="round",
            )
        )
        front_local = np.array([hl, 0.0], dtype=np.float32)
        front = front_local @ rot.T + pos
        ax.plot([float(pos[0]), float(front[0])], [float(pos[1]), float(front[1])], color=edgecolor, linewidth=linewidth, alpha=alpha)
        return

    raise ValueError(f"Unsupported body shape: {shape!r}")


def _rollout_greedy(robot: Unicycle, *, max_steps: int) -> TrajectoryResult:
    if robot.kdtree is None:
        robot.update_kdtree()
    if robot.kdtree is None:
        raise RuntimeError("Robot KDTree is not built.")

    dt = float(robot.get_temporal_res())
    rho = float(robot.get_perturbation_radius())
    values = np.asarray([float(n.value) for n in robot.nodes], dtype=np.float32)

    init_xy = np.asarray(robot.init_pos, dtype=np.float32).reshape(-1)
    if init_xy.size != 2:
        raise ValueError(f"Expected init_pos to have 2 elements, got {init_xy!r}")
    start_state = np.array([float(init_xy[0]), float(init_xy[1]), 0.0], dtype=np.float32)
    current_idx = int(robot.query_kdtree(start_state)[0])

    controls = np.asarray(robot.controls, dtype=np.float32)
    u_lin = controls[:, 0]
    u_ang = controls[:, 1]

    goal_state = np.asarray(robot.goal_state, dtype=np.float32)
    period = 2.0 * float(robot.angle_scalor)
    k_pi = float(np.pi)

    path: List[int] = [current_idx]
    chosen_controls: List[np.ndarray] = []
    visited: set[int] = {current_idx}

    for _ in range(int(max(1, max_steps))):
        state = np.asarray(robot.nodes[current_idx].state, dtype=np.float32)
        if bool(robot.within_goal(state)):
            break

        theta_real = float(state[2]) * k_pi / float(robot.angle_scalor)
        cos_t = float(math.cos(theta_real))
        sin_t = float(math.sin(theta_real))

        cand = np.empty((controls.shape[0], 3), dtype=np.float32)
        cand[:, 0] = float(state[0]) + dt * cos_t * u_lin
        cand[:, 1] = float(state[1]) + dt * sin_t * u_lin
        cand[:, 2] = state[2] + dt * u_ang
        cand[:, 2] = cand[:, 2] - np.floor((cand[:, 2] + float(robot.angle_scalor)) / period) * period

        phase = cand.copy()
        phase[:, 2] = np.where(phase[:, 2] > 0.0, phase[:, 2] - period, phase[:, 2] + period)
        queries = np.vstack([cand, phase])
        idx_lists = robot.kdtree.query_radius(queries, rho, return_distance=False)

        best_next: Optional[int] = None
        best_val = float("inf")
        best_dist_goal = float("inf")
        best_u: Optional[np.ndarray] = None

        a_count = cand.shape[0]
        for a in range(a_count):
            nbr0 = idx_lists[a]
            nbr1 = idx_lists[a + a_count]
            if nbr0.size == 0 and nbr1.size == 0:
                continue
            nbrs = nbr0 if nbr1.size == 0 else (nbr1 if nbr0.size == 0 else np.concatenate([nbr0, nbr1]))
            nbr_vals = values[nbrs]
            local_argmin = int(np.argmin(nbr_vals))
            next_idx = int(nbrs[local_argmin])
            next_val = float(nbr_vals[local_argmin])
            next_state = np.asarray(robot.nodes[next_idx].state, dtype=np.float32)
            dist_goal = float(np.sum((next_state[:2] - goal_state[:2]) ** 2))

            if next_val < best_val - 1e-12 or (abs(next_val - best_val) <= 1e-12 and dist_goal < best_dist_goal):
                best_val = next_val
                best_dist_goal = dist_goal
                best_next = next_idx
                best_u = controls[a]

        if best_next is None or best_u is None:
            break
        if best_next in visited:
            break

        chosen_controls.append(best_u.copy())
        path.append(best_next)
        visited.add(best_next)
        current_idx = best_next

    path_states = np.asarray([robot.nodes[i].state for i in path], dtype=np.float32)
    controls_arr = np.asarray(chosen_controls, dtype=np.float32) if chosen_controls else np.empty((0, 2), dtype=np.float32)
    return TrajectoryResult(path_indices=path, path_states=path_states, controls=controls_arr)


def _plot_trajectory(ax: plt.Axes, robot: Unicycle, result: TrajectoryResult, *, title: str) -> None:
    _draw_env(ax, robot.env)
    states = result.path_states
    ax.plot(states[:, 0], states[:, 1], "-o", linewidth=2.0, markersize=3.5, color="#1f77b4")

    init_xy = np.asarray(robot.init_pos, dtype=np.float32).reshape(-1)
    ax.plot([float(init_xy[0])], [float(init_xy[1])], marker="s", markersize=7, color="#2ca02c", label="start")

    goal_xy = np.asarray(robot.goal_state[:2], dtype=np.float32)
    ax.plot([float(goal_xy[0])], [float(goal_xy[1])], marker="*", markersize=10, color="#d62728", label="goal")

    r = float(robot.goal_region_threshold)
    ax.add_patch(Circle((float(goal_xy[0]), float(goal_xy[1])), r, fill=False, edgecolor="#d62728", linewidth=1.5, alpha=0.8))

    if states.shape[0] >= 1:
        _draw_robot_body(ax, robot, states[0], edgecolor="#2ca02c", facecolor="none", alpha=0.85)
        if states.shape[0] >= 2:
            for i in range(1, states.shape[0] - 1):
                _draw_robot_body(ax, robot, states[i], edgecolor="#1f77b4", facecolor="none", alpha=0.25, linewidth=1.0)
        _draw_robot_body(ax, robot, states[-1], edgecolor="#d62728", facecolor="none", alpha=0.85)

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser("Plot greedy rollouts from vi_robot.pkl outputs.")
    parser.add_argument("--outputs-root", type=str, default="unicycle_value_cuda/outputs", help="Root outputs directory.")
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2], help="Levels to plot (0/1/2).")
    parser.add_argument("--prefer", type=str, default="gpu2,gpu1,cpu", help="Preferred run dir prefixes, comma-separated.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max rollout steps.")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for figures (default: outputs-root).")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)

    prefer = [p.strip() for p in str(args.prefer).split(",") if p.strip()]
    levels = list(dict.fromkeys(int(l) for l in args.levels))

    results: List[Tuple[int, Path, Unicycle, TrajectoryResult]] = []
    for level in levels:
        run_dir = _preferred_run_dir(outputs_root, level=level, prefer=prefer)
        robot = _load_robot(run_dir / "vi_robot.pkl")
        rollout = _rollout_greedy(robot, max_steps=int(args.max_steps))
        results.append((level, run_dir, robot, rollout))

        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        _plot_trajectory(ax, robot, rollout, title=f"Unicycle rollout (level={level})")
        fig.tight_layout()
        fig_path = out_dir / f"trajectory_level{level}.png"
        fig.savefig(fig_path, dpi=int(args.dpi))
        plt.close(fig)

    if len(results) > 1:
        fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 6.0), squeeze=False)
        for col, (level, _, robot, rollout) in enumerate(results):
            ax = axes[0, col]
            _plot_trajectory(ax, robot, rollout, title=f"level={level}")
        fig.tight_layout()
        fig_path = out_dir / "trajectory_levels.png"
        fig.savefig(fig_path, dpi=int(args.dpi))
        plt.close(fig)


if __name__ == "__main__":
    main()
