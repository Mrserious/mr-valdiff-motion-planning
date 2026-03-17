from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from value_guided.observe_coarse import load_regular_value_grid

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.se2 import angle_scalor_from_range, theta_scaled_from_yaw
from unicycle_value_guided.task_io import get_range, load_task


def _plot_episode(
    *,
    task: dict[str, Any],
    coarse_grid: Any,
    goal_xy: Sequence[float],
    states: np.ndarray,
    out_path: Path,
    title: str,
    footprint_length_m: float,
    footprint_width_m: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    def footprint_corners_xy(*, x: float, y: float, yaw: float, length_m: float, width_m: float) -> np.ndarray:
        L = float(length_m)
        W = float(width_m)
        if not (np.isfinite(L) and np.isfinite(W) and L > 0 and W > 0):
            raise ValueError(f"Invalid footprint size: length_m={L} width_m={W}")
        c = float(math.cos(float(yaw)))
        s = float(math.sin(float(yaw)))
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        hx = 0.5 * L
        hy = 0.5 * W
        corners_local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
        corners_world = corners_local @ R.T + np.array([float(x), float(y)], dtype=np.float32)[None, :]
        return corners_world

    grid = coarse_grid.as_ascending()
    V = grid.V
    xs = grid.x_samples
    ys = grid.y_samples
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.imshow(V, origin="lower", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis", alpha=0.95)

    for ob in task.get("env", {}).get("obstacles", []):
        shape = ob.get("shape", None)
        if shape == "circle":
            cx, cy = ob["center"]
            r = ob["radius"]
            ax.add_patch(Circle((cx, cy), r, edgecolor="k", facecolor="none", linewidth=1.5))
        elif shape == "rectangle":
            (x1, x2), (y1, y2) = ob["limits"]
            rx0 = min(x1, x2)
            ry0 = min(y1, y2)
            rw = max(x1, x2) - rx0
            rh = max(y1, y2) - ry0
            ax.add_patch(Rectangle((rx0, ry0), rw, rh, edgecolor="k", facecolor="none", linewidth=1.5))

    goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=80, c="red", marker="x", linewidths=2, label="goal")

    states = np.asarray(states, dtype=np.float32).reshape(-1, 3)
    if states.size > 0:
        states_xy = states[:, 0:2]
        ax.scatter([float(states_xy[0, 0])], [float(states_xy[0, 1])], s=40, c="white", edgecolors="k", label="start")
        ax.plot(states_xy[:, 0], states_xy[:, 1], "-o", color="white", linewidth=2, markersize=3, label="traj")

        T = max(int(states.shape[0] - 1), 0)
        stride = max(1, T // 30) if T > 0 else 1
        idxs = list(range(0, states.shape[0], stride))
        if idxs and idxs[-1] != states.shape[0] - 1:
            idxs.append(states.shape[0] - 1)
        for i in idxs:
            x, y, yaw = float(states[i, 0]), float(states[i, 1]), float(states[i, 2])
            corners = footprint_corners_xy(x=x, y=y, yaw=yaw, length_m=float(footprint_length_m), width_m=float(footprint_width_m))
            ax.add_patch(Polygon(corners, closed=True, edgecolor="cyan", facecolor="none", linewidth=1.0, alpha=0.35))

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _goal_xyz_tag(goal_xyz: Sequence[float]) -> str:
    arr = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:10]


def _delta_to_dirname(delta: float) -> str:
    return f"delta_{float(delta):.3f}"


def _find_plot_goal_dir(*, base_goal_dir: Path, summary: dict[str, Any]) -> Path:
    opt_a = summary.get("opt_a", {}) if isinstance(summary.get("opt_a", {}), dict) else {}
    if not bool(opt_a.get("enabled", False)):
        return base_goal_dir
    cache_dir = opt_a.get("cache_dir", None)
    delta = opt_a.get("delta", None)
    if cache_dir is None or delta is None:
        return base_goal_dir

    map_name = str(summary.get("map_name", "")).strip()
    goal_idx = int(summary.get("goal_index", 0))
    goal_xyz = summary.get("goal_pose", None)
    if not map_name or goal_xyz is None:
        return base_goal_dir

    try:
        goal_tag = _goal_xyz_tag(goal_xyz)
    except Exception:
        return base_goal_dir

    inflated_goal_dir = Path(cache_dir) / map_name / _delta_to_dirname(float(delta)) / f"goal_{goal_idx}_{goal_tag}"
    if inflated_goal_dir.exists():
        return inflated_goal_dir
    return base_goal_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Re-render infer_diffusion episode_*.png with car footprint overlays.")
    p.add_argument("--infer-dir", type=str, required=True, help="Path to .../goal_k/infer_diffusion directory.")
    p.add_argument("--suffix", type=str, default="_car", help="Output suffix before .png (default: _car). Use '' to overwrite.")
    args = p.parse_args(argv)

    infer_dir = Path(args.infer_dir)
    if not infer_dir.exists():
        raise FileNotFoundError(f"--infer-dir does not exist: {infer_dir}")
    if not infer_dir.is_dir():
        raise NotADirectoryError(f"--infer-dir is not a directory: {infer_dir}")

    summary_path = infer_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    base_goal_dir = infer_dir.parent
    plot_goal_dir = _find_plot_goal_dir(base_goal_dir=base_goal_dir, summary=summary)

    # Plot background matches infer_diffusion (--opt-a uses inflated coarse goal_dir).
    coarse_grid2d = load_regular_value_grid(plot_goal_dir / "value_coarse.npy", plot_goal_dir / "meta_coarse.json")

    # Obstacles match the original (base) task, not the inflated task_obs.
    meta_coarse = json.loads((base_goal_dir / "meta_coarse.json").read_text(encoding="utf-8"))
    # IMPORTANT: for within_goal() semantics we must use the *goal-specific* tmp_task_path,
    # because the original map task JSON often contains a placeholder goal_state.
    task_path_raw = meta_coarse.get("tmp_task_path", None) or meta_coarse.get("task_path", None)
    if not task_path_raw:
        raise KeyError(f"meta_coarse.json missing task_path/tmp_task_path: {base_goal_dir / 'meta_coarse.json'}")
    task_path = Path(str(task_path_raw))
    if not task_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        task_path = (project_root / task_path).resolve()
    task = load_task(task_path)

    # Footprint size from task.
    robot0 = task.get("robots", [{}])[0]
    size = (robot0.get("configuration", {}) or {}).get("size", None)
    if size is None or len(size) != 2:
        raise ValueError(f"Task robot configuration.size must be [length,width], got: {size}")
    footprint_length_m = float(size[0])
    footprint_width_m = float(size[1])

    # Optional: annotate reached_goal vs not_reached using within_goal().
    xmin, xmax, _, _ = get_range(task)
    angle_scalor = angle_scalor_from_range(xmin, xmax)
    goal_check_robot = Unicycle(task["env"], robot0, angle_scalor=float(angle_scalor), robot_id=0)

    goal_pose = np.asarray(summary.get("goal_pose", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    goal_xy = goal_pose[:2]
    goal_idx = int(summary.get("goal_index", 0))

    state_paths = sorted(infer_dir.glob("episode_*_states.npy"))
    if not state_paths:
        raise FileNotFoundError(f"No episode_*_states.npy files found in: {infer_dir}")

    suffix = str(args.suffix)
    for sp in state_paths:
        ep_name = sp.name.replace("_states.npy", "")
        states = np.load(sp).astype(np.float32, copy=False)
        actions_path = infer_dir / f"{ep_name}_actions.npy"
        T = int(np.load(actions_path).shape[0]) if actions_path.exists() else max(int(states.shape[0] - 1), 0)

        last = states[-1]
        last_scaled = np.array([float(last[0]), float(last[1]), float(theta_scaled_from_yaw(float(last[2]), float(angle_scalor)))], dtype=np.float32)
        reached = bool(goal_check_robot.within_goal(last_scaled))
        reason = "reached_goal" if reached else "not_reached"

        out_path = infer_dir / f"{ep_name}{suffix}.png"
        _plot_episode(
            task=task,
            coarse_grid=coarse_grid2d,
            goal_xy=goal_xy,
            states=states,
            out_path=out_path,
            title=f"goal_{goal_idx} | {reason} | T={T}",
            footprint_length_m=float(footprint_length_m),
            footprint_width_m=float(footprint_width_m),
        )
        print(f"[replot] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
