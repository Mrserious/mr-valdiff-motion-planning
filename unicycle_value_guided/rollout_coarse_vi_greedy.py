from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle
from value_guided.observe_coarse import load_regular_value_grid

from unicycle_value_guided.rollout_fine import reconstruct_action_from_transition
from unicycle_value_guided.se2 import angle_scalor_from_range, theta_scaled_from_yaw, wrap_theta_scaled, yaw_from_theta_scaled
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.task_io import get_range, load_task
from unicycle_value_guided.inflation import prepare_inflated_goal_assets
from unicycle_value_guided.vi_io import load_vi_robot


def _clip_action_v_omega(
    action_v_omega: np.ndarray,
    *,
    control_limits_scaled: Sequence[Sequence[float]],
    angle_scalor: float,
) -> np.ndarray:
    action_v_omega = np.asarray(action_v_omega, dtype=np.float32).reshape(2)
    lims = np.asarray(control_limits_scaled, dtype=np.float32).reshape(2, 2)
    vmin, vmax = float(min(lims[0])), float(max(lims[0]))
    omin_s, omax_s = float(min(lims[1])), float(max(lims[1]))
    omin = omin_s / float(angle_scalor) * float(np.pi)
    omax = omax_s / float(angle_scalor) * float(np.pi)
    v = float(np.clip(action_v_omega[0], vmin, vmax))
    w = float(np.clip(action_v_omega[1], omin, omax))
    return np.array([v, w], dtype=np.float32)


def _integrate_unicycle_scaled(
    *,
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
) -> np.ndarray:
    s = np.asarray(state_scaled, dtype=np.float64).reshape(3)
    a = np.asarray(action_v_omega, dtype=np.float64).reshape(2)
    v = float(a[0])
    omega_rad = float(a[1])
    yaw = float(yaw_from_theta_scaled(float(s[2]), float(angle_scalor)))
    x = float(s[0] + math.cos(yaw) * v * float(dt))
    y = float(s[1] + math.sin(yaw) * v * float(dt))
    omega_scaled = float(omega_rad / math.pi * float(angle_scalor))
    th = wrap_theta_scaled(float(s[2] + omega_scaled * float(dt)), float(angle_scalor))
    return np.array([x, y, th], dtype=np.float32)


def _discrete_step_collision_free(
    *,
    robot: Any,
    task: dict[str, Any],
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
) -> bool:
    """
    Discrete (end-state only) collision check for one constant-control step.

    This matches the "lenient discrete-step" semantic used in simple-car-parking.ipynb's simulator:
      - integrate one dt step
      - check boundary
      - check obstacle_free(next_state) only at the end-state
    """
    xmin, xmax, ymin, ymax = get_range(task)
    nxt = _integrate_unicycle_scaled(state_scaled=state_scaled, action_v_omega=action_v_omega, dt=float(dt), angle_scalor=float(angle_scalor))
    x = float(nxt[0])
    y = float(nxt[1])
    if x < xmin or x > xmax or y < ymin or y > ymax:
        return False
    return bool(getattr(robot, "obstacle_free")(nxt))


def _plot_episode(
    *,
    task: dict[str, Any],
    coarse_grid: Any,
    goal_xy: Sequence[float],
    states: np.ndarray,
    out_path: Path,
    title: str,
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
        c = float(math.cos(float(yaw)))
        s = float(math.sin(float(yaw)))
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        hx = 0.5 * L
        hy = 0.5 * W
        corners_local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
        corners_world = corners_local @ R.T + np.array([float(x), float(y)], dtype=np.float32)[None, :]
        return corners_world

    robot0 = task.get("robots", [{}])[0]
    size = (robot0.get("configuration", {}) or {}).get("size", None)
    if size is None or len(size) != 2:
        raise ValueError(f"Task robot configuration.size must be [length,width], got: {size}")
    footprint_length_m = float(size[0])
    footprint_width_m = float(size[1])

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
            corners = footprint_corners_xy(x=x, y=y, yaw=yaw, length_m=footprint_length_m, width_m=footprint_width_m)
            ax.add_patch(Polygon(corners, closed=True, edgecolor="cyan", facecolor="none", linewidth=1.0, alpha=0.35))

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _infer_tag(infer_dir: Path) -> str:
    name = infer_dir.name
    if name.startswith("infer_diffusion_"):
        return name[len("infer_diffusion_") :]
    return name


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Re-run a VI-greedy rollout using coarse value (like simple-car-parking.ipynb Simulator.run), "
            "reusing the same start states from an existing infer_diffusion directory. "
            "Collision checking uses strict swept collision."
        )
    )
    p.add_argument("--infer-dir", type=str, required=True, help="Path to an existing infer_diffusion output dir (e.g., infer_diffusion1).")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory. Default: <goal_dir>/coarse_vi_greedy_swept_<tag>.")
    p.add_argument("--max-steps", type=int, default=250, help="Max rollout steps per episode.")
    p.add_argument(
        "--collision",
        type=str,
        default="swept",
        choices=["swept", "discrete"],
        help="Collision semantic: swept (strict, default) or discrete (end-state only).",
    )
    p.add_argument("--collision-check-step", type=float, default=0.05, help="Swept collision linear step (meters). Used only when --collision swept.")
    p.add_argument(
        "--interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "nearby"],
        help="Control interpolation mode (matches ipynb Simulator.run). Default: nearest.",
    )
    p.add_argument("--nearby-radius", type=float, default=0.5, help="Radius for interpolation=nearby (meters).")
    p.add_argument(
        "--unreachable-eps",
        type=float,
        default=1e-4,
        help="Skip reference nodes whose value is within eps of 1.0 (treated as unreachable).",
    )
    p.add_argument("--plot", action="store_true", help="Also write episode_*.png visualizations.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")

    # Opt-A: obstacle inflation planning semantics (cached).
    p.add_argument("--opt-a", action="store_true", help="Plan using inflated VI assets (Opt-A). Collision checking remains on real obstacles.")
    p.add_argument("--opt-a-delta", type=float, default=0.05, help="Opt-A obstacle inflation radius in meters (default: 0.05).")
    p.add_argument(
        "--opt-a-cache-dir",
        type=str,
        default="data/unicycle_value_grids_inflated_standard24",
        help="Cache directory for inflated VI assets (default: data/unicycle_value_grids_inflated_standard24).",
    )
    p.add_argument("--opt-a-overwrite-cache", action="store_true", help="Recompute Opt-A cached assets even if they exist.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    infer_dir = Path(args.infer_dir).expanduser()
    if not infer_dir.exists():
        raise FileNotFoundError(f"--infer-dir does not exist: {infer_dir}")
    if not infer_dir.is_dir():
        raise NotADirectoryError(f"--infer-dir is not a directory: {infer_dir}")

    src_summary_path = infer_dir / "summary.json"
    if not src_summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in infer dir: {src_summary_path}")
    src_summary = json.loads(src_summary_path.read_text(encoding="utf-8"))

    goal_dir = infer_dir.parent
    meta_coarse_path = goal_dir / "meta_coarse.json"
    if not meta_coarse_path.exists():
        raise FileNotFoundError(f"Missing meta_coarse.json next to infer dir: {meta_coarse_path}")
    meta_coarse = json.loads(meta_coarse_path.read_text(encoding="utf-8"))
    task = load_task(meta_coarse.get("task_path", meta_coarse.get("tmp_task_path")))
    xmin, xmax, _, _ = get_range(task)
    angle_scalor = float(angle_scalor_from_range(xmin, xmax))

    opt_a = None
    if bool(getattr(args, "opt_a", False)):
        map_name = str(src_summary.get("map_name", meta_coarse.get("map_name", goal_dir.parent.name)))
        goal_index = int(src_summary.get("goal_index", meta_coarse.get("goal_index", 0)))
        goal_pose = src_summary.get("goal_pose", meta_coarse.get("goal_pose", [0.0, 0.0, 0.0]))
        base_task_path = meta_coarse.get("task_path", None) or meta_coarse.get("tmp_task_path", None)
        if base_task_path is None:
            raise RuntimeError("Opt-A requested but meta_coarse has no task_path/tmp_task_path.")

        opt_a = prepare_inflated_goal_assets(
            base_task=task,
            base_task_path=Path(str(base_task_path)),
            map_name=str(map_name),
            goal_idx=int(goal_index),
            goal_xyz=goal_pose,
            cache_root=Path(str(args.opt_a_cache_dir)),
            delta=float(args.opt_a_delta),
            coarse_meta_src=meta_coarse,
            fine_meta_src=None,
            overwrite=bool(args.opt_a_overwrite_cache),
            keep_pkl=True,
            use_inflated_fine=False,
        )

    # VI robot (coarse) provides KDTree, node values and goal semantics.
    if opt_a is not None:
        coarse_robot = opt_a.coarse_robot
    else:
        vi_robot_path = goal_dir / "vi_robot_coarse.pkl"
        if not vi_robot_path.exists():
            vi_robot_path = goal_dir / "logs_coarse" / "vi_robot.pkl"
        coarse_robot = load_vi_robot(vi_robot_path)

    dt = float(getattr(coarse_robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from coarse_robot.get_temporal_res(): {dt}")

    control_limits_scaled = np.asarray(getattr(coarse_robot, "control_limits", [[-1, 1], [-1, 1]]), dtype=np.float32).reshape(2, 2)

    # Collision robot uses the same (real) obstacles and robot footprint as inference.
    collision_robot = Unicycle(task["env"], task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)

    # Read starts from source infer_dir episodes (x,y,yaw_rad) at t=0.
    state_paths = sorted(infer_dir.glob("episode_*_states.npy"))
    if not state_paths:
        raise FileNotFoundError(f"No episode_*_states.npy found in: {infer_dir}")
    starts_yaw: list[np.ndarray] = []
    for sp in state_paths:
        arr = np.load(sp).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
            raise ValueError(f"Invalid states array in {sp}: shape={arr.shape}")
        starts_yaw.append(arr[0].copy())

    tag = _infer_tag(infer_dir)
    collision_mode = str(args.collision).strip().lower()
    if collision_mode not in ("swept", "discrete"):
        raise ValueError(f"Invalid --collision: {args.collision!r}")

    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir is not None
        else (goal_dir / f"coarse_vi_greedy_{collision_mode}_{tag}")
    )
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    if out_dir.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Refusing to overwrite existing out dir without --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot background: coarse 2D value grid from the goal_dir.
    coarse_grid2d = None
    if bool(args.plot):
        grid_goal_dir = (opt_a.goal_dir if opt_a is not None else goal_dir)
        coarse_grid2d = load_regular_value_grid(grid_goal_dir / "value_coarse.npy", grid_goal_dir / "meta_coarse.json")

    goal_pose = np.asarray(src_summary.get("goal_pose", meta_coarse.get("goal_pose", [0.0, 0.0, 0.0])), dtype=np.float64).reshape(3)
    goal_xy = goal_pose[:2]

    # Cache node->optimal control (as in ipynb Simulator.reconstruct_all_controls()).
    ctrl_cache: dict[int, np.ndarray] = {}

    def _node_optimal_control(idx: int) -> np.ndarray:
        idx = int(idx)
        if idx in ctrl_cache:
            return ctrl_cache[idx]
        node = coarse_robot.nodes[idx]
        children = list(getattr(node.children, "indices", []))
        if not children:
            ctrl_cache[idx] = np.zeros(2, dtype=np.float32)
            return ctrl_cache[idx]
        child_vals = [float(getattr(coarse_robot.nodes[int(c)], "value", 1.0)) for c in children]
        min_val = min(child_vals) if child_vals else 1.0
        # Match ipynb behavior: take all ties for min value (exact), then average controls.
        candidates = [int(children[i]) for i, v in enumerate(child_vals) if abs(float(v) - float(min_val)) <= 1e-12]
        if not candidates:
            candidates = [int(children[int(np.argmin(np.asarray(child_vals, dtype=np.float64)))])]
        cur_state = np.asarray(node.state, dtype=np.float32).reshape(3)
        controls: list[np.ndarray] = []
        for c in candidates:
            nxt_state = np.asarray(coarse_robot.nodes[int(c)].state, dtype=np.float32).reshape(3)
            a = reconstruct_action_from_transition(
                cur_state_scaled=cur_state,
                nxt_state_scaled=nxt_state,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
            )
            a = _clip_action_v_omega(a, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))
            controls.append(a)
        ctrl = np.mean(np.stack(controls, axis=0), axis=0).astype(np.float32, copy=False)
        ctrl_cache[idx] = ctrl
        return ctrl

    def _get_control(state_scaled: np.ndarray) -> tuple[np.ndarray | None, list[int]]:
        state_scaled = np.asarray(state_scaled, dtype=np.float32).reshape(3)
        if str(args.interpolation) == "nearest":
            ref_node_indices = list(getattr(coarse_robot, "query_kdtree")(state_scaled))
            ref_node_indices = ref_node_indices[:1]
        else:
            ref_node_indices = list(getattr(coarse_robot, "query_kdtree")(state_scaled, radius=float(args.nearby_radius)))

        ref_nodes: list[int] = []
        for i in ref_node_indices:
            if abs(float(getattr(coarse_robot.nodes[int(i)], "value", 1.0)) - 1.0) <= float(args.unreachable_eps):
                continue
            ref_nodes.append(int(i))
        if not ref_nodes:
            return None, []

        u = np.zeros(2, dtype=np.float32)
        for i in ref_nodes:
            u += _node_optimal_control(int(i))
        u = (u / float(len(ref_nodes))).astype(np.float32, copy=False)
        u = _clip_action_v_omega(u, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))
        return u, ref_nodes

    per_episode: list[dict[str, Any]] = []
    successes = 0
    Ts: list[int] = []
    reason_counts: dict[str, int] = {}

    for ep, start_yaw in enumerate(starts_yaw):
        start_yaw = np.asarray(start_yaw, dtype=np.float32).reshape(3)
        start_scaled = np.array(
            [float(start_yaw[0]), float(start_yaw[1]), float(theta_scaled_from_yaw(float(start_yaw[2]), float(angle_scalor)))],
            dtype=np.float32,
        )
        start_scaled[2] = wrap_theta_scaled(float(start_scaled[2]), float(angle_scalor))

        states_yaw: list[np.ndarray] = [start_yaw.copy()]
        actions: list[np.ndarray] = []
        reason = "timeout"
        success = False

        s = start_scaled.copy()
        # Early success check (rare).
        if bool(getattr(coarse_robot, "within_goal")(s)):
            reason = "reached_goal"
            success = True
        else:
            for _ in range(int(args.max_steps)):
                u, refs = _get_control(s)
                if u is None:
                    reason = "unreachable"
                    break
                ok_collision = False
                if collision_mode == "swept":
                    ok_collision = trajectory_collision_free(
                        robot=collision_robot,
                        task=task,
                        state_scaled=s,
                        action_v_omega=u,
                        dt=float(dt),
                        angle_scalor=float(angle_scalor),
                        step_size=float(args.collision_check_step),
                    )
                else:
                    ok_collision = _discrete_step_collision_free(
                        robot=collision_robot,
                        task=task,
                        state_scaled=s,
                        action_v_omega=u,
                        dt=float(dt),
                        angle_scalor=float(angle_scalor),
                    )
                if not ok_collision:
                    reason = "collision"
                    break
                actions.append(u.astype(np.float32, copy=False))

                s = _integrate_unicycle_scaled(state_scaled=s, action_v_omega=u, dt=float(dt), angle_scalor=float(angle_scalor))
                yaw = float(yaw_from_theta_scaled(float(s[2]), float(angle_scalor)))
                states_yaw.append(np.array([float(s[0]), float(s[1]), float(yaw)], dtype=np.float32))

                if bool(getattr(coarse_robot, "within_goal")(s)):
                    reason = "reached_goal"
                    success = True
                    break

        T = int(len(actions))
        if success:
            successes += 1
            Ts.append(T)
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

        states_arr = np.asarray(states_yaw, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32)
        ep_name = f"episode_{ep:04d}"
        np.save(out_dir / f"{ep_name}_states.npy", states_arr)
        np.save(out_dir / f"{ep_name}_actions.npy", actions_arr)
        # keep naming consistency with infer_diffusion (optional artifacts)
        np.save(out_dir / f"{ep_name}_projected_actions.npy", actions_arr)
        np.save(out_dir / f"{ep_name}_resample_counts.npy", np.zeros((actions_arr.shape[0],), dtype=np.int64))

        if bool(args.plot):
            if coarse_grid2d is None:
                raise RuntimeError("Internal error: coarse_grid2d is None while --plot is set.")
            _plot_episode(
                task=task,
                coarse_grid=coarse_grid2d,
                goal_xy=goal_xy,
                states=states_arr,
                out_path=out_dir / f"{ep_name}.png",
                title=f"coarse_vi_greedy | {reason} | T={T}",
            )

        per_episode.append({"episode": ep, "success": bool(success), "reason": reason, "T": T})
        print(f"[coarse_vi_greedy] ep={ep} success={success} reason={reason} T={T}", flush=True)

    out_summary = {
        "map_name": str(src_summary.get("map_name", meta_coarse.get("map_name", ""))),
        "goal_index": int(src_summary.get("goal_index", meta_coarse.get("goal_index", 0))),
        "goal_pose": goal_pose.tolist(),
        "opt_a": (
            None
            if opt_a is None
            else {
                "enabled": True,
                "delta": float(args.opt_a_delta),
                "cache_dir": str(args.opt_a_cache_dir),
                "goal_dir": str(opt_a.goal_dir),
                "collision_check": "real",
            }
        ),
        "episodes": int(len(starts_yaw)),
        "successes": int(successes),
        "success_rate": float(successes) / float(len(starts_yaw)) if starts_yaw else 0.0,
        "avg_T": (float(np.mean(Ts)) if Ts else None),
        "median_T": (float(np.median(Ts)) if Ts else None),
        "source_infer_dir": str(infer_dir),
        "planner": {
            "type": "coarse_vi_greedy",
            "interpolation": str(args.interpolation),
            "nearby_radius": (None if str(args.interpolation) == "nearest" else float(args.nearby_radius)),
            "dt": float(dt),
            "unreachable_eps": float(args.unreachable_eps),
            "collision": {
                "semantic": str(collision_mode),
                "check_step": (None if collision_mode != "swept" else float(args.collision_check_step)),
            },
        },
        "reason_counts": reason_counts,
        "per_episode": per_episode,
    }
    (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(f"[coarse_vi_greedy] wrote summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
