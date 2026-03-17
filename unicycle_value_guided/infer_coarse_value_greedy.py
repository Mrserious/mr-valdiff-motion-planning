from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.inflation import prepare_inflated_goal_assets
from unicycle_value_guided.rollout_fine import reconstruct_action_from_transition
from unicycle_value_guided.se2 import angle_scalor_from_range, theta_scaled_from_yaw, wrap_theta_scaled, yaw_from_theta_scaled
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.task_io import get_range, load_task
from unicycle_value_guided.value_grid3d import RegularValueGrid3D, load_regular_value_grid_3d
from unicycle_value_guided.vi_io import load_vi_robot


@dataclass(frozen=True)
class AntiRepeatConfig:
    enabled: bool
    xy_q: float
    xy_recent_n: int
    child_recent_n: int
    edge_recent_n: int
    avoid_uturn: bool


@dataclass
class _AntiRepeatState:
    xy_recent: deque[tuple[int, int]]
    xy_counts: Counter[tuple[int, int]]
    child_recent: deque[int]
    child_counts: Counter[int]
    edge_recent: deque[tuple[int, int]]
    edge_counts: Counter[tuple[int, int]]
    prev_edge: tuple[int, int] | None


def _xy_key(x: float, y: float, *, q: float) -> tuple[int, int]:
    qq = float(q)
    if not (np.isfinite(qq) and qq > 0):
        raise ValueError(f"Invalid anti-repeat quantization q: {q}")
    return (int(round(float(x) / qq)), int(round(float(y) / qq)))


def _anti_push_xy(state: _AntiRepeatState, key: tuple[int, int], *, max_n: int) -> None:
    n = int(max_n)
    if n <= 0:
        return
    state.xy_recent.append(key)
    state.xy_counts[key] += 1
    while len(state.xy_recent) > n:
        k0 = state.xy_recent.popleft()
        state.xy_counts[k0] -= 1
        if state.xy_counts[k0] <= 0:
            del state.xy_counts[k0]


def _anti_push_child(state: _AntiRepeatState, idx_next: int, *, max_n: int) -> None:
    n = int(max_n)
    if n <= 0:
        return
    idx_next = int(idx_next)
    state.child_recent.append(idx_next)
    state.child_counts[idx_next] += 1
    while len(state.child_recent) > n:
        i0 = int(state.child_recent.popleft())
        state.child_counts[i0] -= 1
        if state.child_counts[i0] <= 0:
            del state.child_counts[i0]


def _anti_push_edge(state: _AntiRepeatState, edge: tuple[int, int], *, max_n: int) -> None:
    n = int(max_n)
    if n <= 0:
        return
    e = (int(edge[0]), int(edge[1]))
    state.edge_recent.append(e)
    state.edge_counts[e] += 1
    while len(state.edge_recent) > n:
        e0 = state.edge_recent.popleft()
        state.edge_counts[e0] -= 1
        if state.edge_counts[e0] <= 0:
            del state.edge_counts[e0]


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


def _integrate_unicycle_scaled_swept(
    *,
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
    step_size: float,
) -> np.ndarray:
    """
    Integrate one constant-control step using the same sub-stepping scheme as swept collision checking.

    This does *not* perform collision checks; it only computes the end state.
    """
    s = np.asarray(state_scaled, dtype=np.float64).reshape(3)
    a = np.asarray(action_v_omega, dtype=np.float64).reshape(2)
    v = float(a[0])
    omega = float(a[1])  # rad/s
    yaw = float(yaw_from_theta_scaled(float(s[2]), float(angle_scalor)))

    total_lin = abs(v) * float(dt)
    total_ang = abs(omega) * float(dt)
    n_lin = int(math.ceil(total_lin / max(float(step_size), 1e-6))) if total_lin > 0 else 1
    n_ang = int(math.ceil(total_ang / (math.pi / 36.0))) if total_ang > 0 else 1  # ~5deg per step
    n = max(2, n_lin, n_ang)
    dt_sub = float(dt) / float(n)

    x = float(s[0])
    y = float(s[1])
    th = float(s[2])
    for _ in range(n):
        x = float(x + math.cos(yaw) * v * dt_sub)
        y = float(y + math.sin(yaw) * v * dt_sub)
        yaw = float(yaw + omega * dt_sub)
        th = float(theta_scaled_from_yaw(yaw, float(angle_scalor)))
        th = wrap_theta_scaled(th, float(angle_scalor))
    return np.array([x, y, th], dtype=np.float32)


def _pick_action_with_antirepeat(
    *,
    ranked_children: Sequence[int],
    idx_cur: int,
    state_scaled: np.ndarray,
    fine_node_states: np.ndarray,
    dt: float,
    angle_scalor: float,
    control_limits_scaled: Sequence[Sequence[float]],
    collision_check_step: float,
    anti_repeat: AntiRepeatConfig | None,
    anti_state: _AntiRepeatState | None,
) -> tuple[int, np.ndarray] | None:
    """
    Select a child from ranked_children and reconstruct the projected edge action.

    Anti-repeat semantics intentionally match infer_diffusion:
      key = (recent_xy, back_edge, recent_edge, recent_child, rank_i)
    """
    if not ranked_children:
        return None

    if anti_repeat is None or (not bool(getattr(anti_repeat, "enabled", False))):
        cand_idx = int(ranked_children[0])
        cur_node = fine_node_states[int(idx_cur)].astype(np.float32, copy=False)
        cand_state = fine_node_states[cand_idx].astype(np.float32, copy=False)
        a = reconstruct_action_from_transition(
            cur_state_scaled=np.asarray(cur_node, dtype=np.float32).reshape(3),
            nxt_state_scaled=np.asarray(cand_state, dtype=np.float32).reshape(3),
            dt=float(dt),
            angle_scalor=float(angle_scalor),
        )
        a = _clip_action_v_omega(a, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))
        return (cand_idx, np.asarray(a, dtype=np.float32).reshape(2))

    if anti_state is None:
        raise RuntimeError("anti_repeat enabled but anti_state is None")

    cur_node = fine_node_states[int(idx_cur)].astype(np.float32, copy=False)
    best_key: tuple[bool, bool, bool, bool, int] | None = None
    best: tuple[int, np.ndarray] | None = None

    for rank_i, cand_idx in enumerate(ranked_children):
        cand_idx = int(cand_idx)
        cand_state = fine_node_states[cand_idx].astype(np.float32, copy=False)
        a = reconstruct_action_from_transition(
            cur_state_scaled=np.asarray(cur_node, dtype=np.float32).reshape(3),
            nxt_state_scaled=np.asarray(cand_state, dtype=np.float32).reshape(3),
            dt=float(dt),
            angle_scalor=float(angle_scalor),
        )
        a = _clip_action_v_omega(a, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))

        s_pred: np.ndarray | None = None

        is_recent_xy = False
        if int(anti_repeat.xy_recent_n) > 0:
            s_pred = _integrate_unicycle_scaled_swept(
                state_scaled=state_scaled,
                action_v_omega=a,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                step_size=float(collision_check_step),
            )
            k = _xy_key(float(s_pred[0]), float(s_pred[1]), q=float(anti_repeat.xy_q))
            is_recent_xy = bool(anti_state.xy_counts.get(k, 0) > 0)

        is_back_edge = False
        if bool(anti_repeat.avoid_uturn) and anti_state.prev_edge is not None:
            prev_cur, prev_next = anti_state.prev_edge
            if int(idx_cur) == int(prev_next) and int(cand_idx) == int(prev_cur):
                is_back_edge = True

        is_recent_edge = False
        if int(anti_repeat.edge_recent_n) > 0:
            is_recent_edge = bool(anti_state.edge_counts.get((int(idx_cur), int(cand_idx)), 0) > 0)

        is_recent_child = False
        if int(anti_repeat.child_recent_n) > 0:
            is_recent_child = bool(anti_state.child_counts.get(int(cand_idx), 0) > 0)

        key = (bool(is_recent_xy), bool(is_back_edge), bool(is_recent_edge), bool(is_recent_child), int(rank_i))
        if best_key is None or key < best_key:
            best_key = key
            best = (int(cand_idx), np.asarray(a, dtype=np.float32).reshape(2))
            if key[:4] == (False, False, False, False):
                break

    return best


def _postcheck_action(
    *,
    collision_robot: Any,
    collision_task: dict[str, Any],
    cur_state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
    collision_check_step: float,
    collision_semantic: str,
) -> tuple[bool, str | None, np.ndarray]:
    """
    Returns: (ok, fail_detail, s_end)
      - ok=True => fail_detail=None
      - ok=False => fail_detail in {"collision","out_of_bounds"} (best-effort)
    """
    semantic = str(collision_semantic).strip().lower()
    if semantic not in ("swept", "discrete"):
        raise ValueError(f"Invalid --collision-semantic: {collision_semantic!r} (expected: 'swept' or 'discrete')")

    s_end = _integrate_unicycle_scaled_swept(
        state_scaled=cur_state_scaled,
        action_v_omega=action_v_omega,
        dt=float(dt),
        angle_scalor=float(angle_scalor),
        step_size=float(collision_check_step),
    )

    xmin, xmax, ymin, ymax = get_range(collision_task)
    x = float(s_end[0])
    y = float(s_end[1])
    if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
        return (False, "out_of_bounds", s_end)

    if semantic == "swept":
        ok = trajectory_collision_free(
            robot=collision_robot,
            task=collision_task,
            state_scaled=cur_state_scaled,
            action_v_omega=action_v_omega,
            dt=float(dt),
            angle_scalor=float(angle_scalor),
            step_size=float(collision_check_step),
        )
        if not bool(ok):
            return (False, "collision", s_end)
        return (True, None, s_end)

    # discrete: end-state only
    if not bool(getattr(collision_robot, "obstacle_free")(np.asarray(s_end, dtype=np.float32).reshape(3))):
        return (False, "collision", s_end)
    return (True, None, s_end)


def _load_goals(goals_path: Path) -> dict[str, Any]:
    goals_path = Path(goals_path).expanduser()
    if not goals_path.exists():
        raise FileNotFoundError(f"--goals does not exist: {goals_path}")
    payload = json.loads(goals_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "goals" not in payload:
        raise ValueError(f"Invalid goals JSON: expected object with key 'goals', got: {type(payload)}")
    return payload


def _load_starts(starts_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Supports:
      - list of [x,y,yaw_rad]
      - dict with key 'starts' containing that list
    Returns (starts_yaw, meta)
    """
    starts_path = Path(starts_path).expanduser()
    if not starts_path.exists():
        raise FileNotFoundError(f"--starts-file does not exist: {starts_path}")
    payload = json.loads(starts_path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    if isinstance(payload, dict):
        meta = payload
        starts_list = payload.get("starts", None)
    elif isinstance(payload, list):
        starts_list = payload
    else:
        raise ValueError(f"Invalid --starts-file JSON: expected object or list, got {type(payload)}")
    if starts_list is None:
        raise ValueError("--starts-file JSON missing key: 'starts'")
    starts = np.asarray(starts_list, dtype=np.float32)
    if starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError(f"--starts-file starts must be shape (N,3) [x,y,yaw_rad], got {starts.shape}")
    return starts, meta


def _coarse_values_at_states(
    *,
    grid: RegularValueGrid3D,
    states_scaled: np.ndarray,  # (N,3) theta_scaled
) -> np.ndarray:
    """
    Vectorized trilinear sampling of RegularValueGrid3D at (x,y,theta_scaled).
    Returns float32 values shape (N,).
    """
    s = np.asarray(states_scaled, dtype=np.float64).reshape(-1, 3)
    N = int(s.shape[0])
    if N == 0:
        return np.zeros((0,), dtype=np.float32)

    g = grid.as_ascending()
    V = g.V.astype(np.float32, copy=False)
    xs = g.x_samples.astype(np.float64, copy=False)
    ys = g.y_samples.astype(np.float64, copy=False)
    a = float(g.angle_scalor)
    nt = int(g.nt)
    if nt <= 1:
        raise ValueError(f"coarse value grid nt must be > 1, got {nt}")

    xq = s[:, 0].astype(np.float64, copy=False)
    yq = s[:, 1].astype(np.float64, copy=False)

    # clamp to grid bounds
    xq = np.clip(xq, float(xs[0]), float(xs[-1]))
    yq = np.clip(yq, float(ys[0]), float(ys[-1]))

    ix = np.searchsorted(xs, xq, side="right") - 1
    iy = np.searchsorted(ys, yq, side="right") - 1
    ix = np.clip(ix, 0, int(xs.shape[0]) - 2).astype(np.int64, copy=False)
    iy = np.clip(iy, 0, int(ys.shape[0]) - 2).astype(np.int64, copy=False)

    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]

    tx = ((xq - x0) / np.maximum(x1 - x0, 1e-12)).astype(np.float32, copy=False)
    ty = ((yq - y0) / np.maximum(y1 - y0, 1e-12)).astype(np.float32, copy=False)

    th = s[:, 2].astype(np.float64, copy=False)
    th = np.array([wrap_theta_scaled(float(t), a) for t in th], dtype=np.float64)
    dth = 2.0 * a / float(nt)
    u = (th + a) / dth
    it0 = np.floor(u).astype(np.int64) % nt
    it1 = (it0 + 1) % nt
    tt = (u - np.floor(u)).astype(np.float32, copy=False)

    # bilinear per theta slice (it0/it1)
    v00 = V[it0, iy, ix]
    v01 = V[it0, iy, ix + 1]
    v10 = V[it0, iy + 1, ix]
    v11 = V[it0, iy + 1, ix + 1]
    v0 = (1.0 - tx) * v00 + tx * v01
    v1 = (1.0 - tx) * v10 + tx * v11
    b0 = (1.0 - ty) * v0 + ty * v1

    w00 = V[it1, iy, ix]
    w01 = V[it1, iy, ix + 1]
    w10 = V[it1, iy + 1, ix]
    w11 = V[it1, iy + 1, ix + 1]
    w0 = (1.0 - tx) * w00 + tx * w01
    w1 = (1.0 - tx) * w10 + tx * w11
    b1 = (1.0 - ty) * w0 + ty * w1

    out = (1.0 - tt) * b0 + tt * b1
    return out.astype(np.float32, copy=False).reshape(N)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Baseline inference: pick fine-graph children greedily by coarse 3D value grid (no diffusion), "
            "execute the reconstructed projected action, and stop the episode immediately on post-check collision."
        )
    )
    p.add_argument("--goals", type=str, required=True, help="Path to goals.json (e.g. .../standard10x10_0060_small/goals.json)")
    p.add_argument("--goal-index", type=int, required=True, help="Which goal_<k> to run.")
    p.add_argument("--starts-file", type=str, required=True, help="Starts JSON (list or {'starts': ...}) with [x,y,yaw_rad].")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory. Default: <goal_dir>/infer_coarse_value_greedy.")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to run (must be <= starts in --starts-file).")
    p.add_argument("--max-steps", type=int, default=250, help="Max rollout steps per episode.")
    p.add_argument(
        "--collision-semantic",
        type=str,
        default="discrete",
        choices=["discrete", "swept"],
        help="Collision check semantic for the executed (projected) action.",
    )
    p.add_argument("--collision-check-step", type=float, default=0.05, help="Swept collision linear step (meters).")
    p.add_argument("--allow-self-candidate", action="store_true", help="Allow staying at current node as a candidate child.")
    p.add_argument("--topk-children", type=int, default=0, help="Only consider the K best (lowest coarse value) children (0=all).")
    p.add_argument("--unreachable-eps", type=float, default=1e-4, help="Treat value >= 1-eps as unreachable (default: 1e-4).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")

    p.add_argument(
        "--anti-repeat",
        action="store_true",
        help="Enable best-effort anti-repeat (tabu) when selecting the fine child/action.",
    )
    p.add_argument("--anti-repeat-xy-q", type=float, default=0.005, help="XY quantization step (meters) for anti-repeat tabu (default: 0.005).")
    p.add_argument("--anti-repeat-xy-recent-n", type=int, default=80, help="How many recent XY keys to tabu (0=disable).")
    p.add_argument("--anti-repeat-child-recent-n", type=int, default=20, help="How many recent child indices to tabu (0=disable).")
    p.add_argument("--anti-repeat-edge-recent-n", type=int, default=20, help="How many recent edges (idx_cur,idx_next) to tabu (0=disable).")
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--anti-repeat-avoid-uturn",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Prefer not to immediately reverse the previous edge (A->B then B->A). Default: enabled.",
        )
    else:
        g = p.add_mutually_exclusive_group()
        g.add_argument(
            "--anti-repeat-avoid-uturn",
            dest="anti_repeat_avoid_uturn",
            action="store_true",
            default=True,
            help="Prefer not to immediately reverse the previous edge (A->B then B->A). Default: enabled.",
        )
        g.add_argument(
            "--no-anti-repeat-avoid-uturn",
            dest="anti_repeat_avoid_uturn",
            action="store_false",
            help="Allow immediate edge reversal (A->B then B->A).",
        )

    # Opt-A: obstacle inflation planning semantics (cached).
    p.add_argument("--opt-a", action="store_true", help="Enable Opt-A (inflated obstacles for coarse value and fine graph).")
    p.add_argument("--opt-a-delta", type=float, default=0.05, help="Opt-A obstacle inflation radius in meters (default: 0.05).")
    p.add_argument(
        "--opt-a-cache-dir",
        type=str,
        default="data/unicycle_value_grids_inflated_standard24",
        help="Cache directory for inflated VI assets.",
    )
    p.add_argument("--opt-a-overwrite-cache", action="store_true", help="Recompute Opt-A cached assets even if they exist.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    goals_path = Path(args.goals).expanduser()
    if not goals_path.is_absolute():
        goals_path = (Path.cwd() / goals_path).resolve()
    goals_payload = _load_goals(goals_path)

    goals = goals_payload.get("goals", [])
    goal_index = int(args.goal_index)
    if goal_index < 0 or goal_index >= int(len(goals)):
        raise ValueError(f"--goal-index out of range: {goal_index} (n_goals={len(goals)})")
    goal_xyz = np.asarray(goals[goal_index], dtype=np.float64).reshape(3)

    goal_dir = goals_path.parent / f"goal_{goal_index}"
    task_path = Path(str(goals_payload.get("task_path", ""))).expanduser()
    if not str(task_path).strip():
        raise ValueError("goals.json missing key: task_path")
    if not task_path.is_absolute():
        task_path = (Path.cwd() / task_path).resolve()
    task = load_task(task_path)

    xmin, xmax, _, _ = get_range(task)
    angle_scalor = float(angle_scalor_from_range(xmin, xmax))

    starts_path = Path(args.starts_file).expanduser()
    if not starts_path.is_absolute():
        starts_path = (Path.cwd() / starts_path).resolve()
    starts_yaw, starts_meta = _load_starts(starts_path)
    if isinstance(starts_meta, dict) and "goal_index" in starts_meta:
        if int(starts_meta["goal_index"]) != int(goal_index):
            raise ValueError(f"--starts-file goal_index={starts_meta['goal_index']} does not match --goal-index={goal_index}")

    episodes = int(args.episodes)
    if episodes <= 0:
        raise ValueError("--episodes must be >= 1")
    if episodes > int(starts_yaw.shape[0]):
        raise ValueError(f"--episodes={episodes} exceeds starts in --starts-file (N={int(starts_yaw.shape[0])})")

    # Load assets.
    coarse_meta_path = goal_dir / "meta_coarse.json"
    fine_meta_path = goal_dir / "meta_fine.json"
    if not coarse_meta_path.exists():
        raise FileNotFoundError(f"Missing meta_coarse.json next to goal dir: {coarse_meta_path}")
    if not fine_meta_path.exists():
        raise FileNotFoundError(f"Missing meta_fine.json next to goal dir: {fine_meta_path}")
    coarse_meta = json.loads(coarse_meta_path.read_text(encoding="utf-8"))
    fine_meta = json.loads(fine_meta_path.read_text(encoding="utf-8"))

    opt_a = None
    if bool(getattr(args, "opt_a", False)):
        opt_a = prepare_inflated_goal_assets(
            base_task=task,
            base_task_path=task_path,
            map_name=str(goals_payload.get("map_name", goals_path.parent.name)),
            goal_idx=int(goal_index),
            goal_xyz=goal_xyz,
            cache_root=Path(str(args.opt_a_cache_dir)),
            delta=float(args.opt_a_delta),
            coarse_meta_src=coarse_meta,
            fine_meta_src=fine_meta,
            overwrite=bool(args.opt_a_overwrite_cache),
            keep_pkl=True,
            use_inflated_fine=True,
        )
        if opt_a.fine_robot is None:
            raise RuntimeError("Opt-A enabled but inflated fine robot is missing (fine_robot is None).")
        coarse_grid3d = opt_a.coarse_grid3d
        fine_robot = opt_a.fine_robot
    else:
        coarse_grid3d = load_regular_value_grid_3d(goal_dir / "value_coarse_3d.npy", goal_dir / "meta_coarse_3d.json")
        vi_robot_path = goal_dir / "vi_robot_fine.pkl"
        if not vi_robot_path.exists():
            vi_robot_path = goal_dir / "logs_fine" / "vi_robot.pkl"
        fine_robot = load_vi_robot(vi_robot_path)

    dt = float(getattr(fine_robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from fine_robot.get_temporal_res(): {dt}")
    control_limits_scaled = np.asarray(getattr(fine_robot, "control_limits", [[-1, 1], [-1, 1]]), dtype=np.float32).reshape(2, 2)

    fine_node_states = np.stack([np.asarray(n.state, dtype=np.float32).reshape(3) for n in fine_robot.nodes], axis=0)

    # Collision robot always uses the real task obstacles/footprint.
    if not task.get("robots"):
        raise RuntimeError(f"Task has no robots[] entry: {task_path}")
    collision_robot: Any = Unicycle(task["env"], task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)

    # Output directory.
    out_dir = goal_dir / "infer_coarse_value_greedy" if args.out_dir is None else Path(str(args.out_dir)).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    if out_dir.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Refusing to overwrite existing out dir without --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    Ts_success: list[int] = []
    reason_counts: Counter[str] = Counter()
    postcheck_fail_detail_counts: Counter[str] = Counter()
    per_episode: list[dict[str, Any]] = []

    unreachable_eps = float(args.unreachable_eps)
    topk_children = int(args.topk_children)
    if topk_children < 0:
        raise ValueError("--topk-children must be >= 0")

    anti_repeat = AntiRepeatConfig(
        enabled=bool(getattr(args, "anti_repeat", False)),
        xy_q=float(getattr(args, "anti_repeat_xy_q", 0.005)),
        xy_recent_n=int(getattr(args, "anti_repeat_xy_recent_n", 0)),
        child_recent_n=int(getattr(args, "anti_repeat_child_recent_n", 0)),
        edge_recent_n=int(getattr(args, "anti_repeat_edge_recent_n", 0)),
        avoid_uturn=bool(getattr(args, "anti_repeat_avoid_uturn", True)),
    )
    if not bool(anti_repeat.enabled):
        anti_repeat = None
    else:
        if int(anti_repeat.xy_recent_n) < 0 or int(anti_repeat.child_recent_n) < 0 or int(anti_repeat.edge_recent_n) < 0:
            raise ValueError("Anti-repeat recent window sizes must be >= 0.")
        if not (np.isfinite(float(anti_repeat.xy_q)) and float(anti_repeat.xy_q) > 0):
            raise ValueError(f"Invalid --anti-repeat-xy-q: {anti_repeat.xy_q}")
        print(
            "[infer_coarse_value_greedy] anti_repeat enabled:"
            f" xy_q={float(anti_repeat.xy_q):.4f}"
            f" xy_recent_n={int(anti_repeat.xy_recent_n)}"
            f" child_recent_n={int(anti_repeat.child_recent_n)}"
            f" edge_recent_n={int(anti_repeat.edge_recent_n)}"
            f" avoid_uturn={bool(anti_repeat.avoid_uturn)}",
            flush=True,
        )

    for ep in range(int(episodes)):
        start_yaw = starts_yaw[int(ep)].astype(np.float32, copy=False).reshape(3)
        start_scaled = np.array(
            [
                float(start_yaw[0]),
                float(start_yaw[1]),
                float(theta_scaled_from_yaw(float(start_yaw[2]), float(angle_scalor))),
            ],
            dtype=np.float32,
        )
        start_scaled[2] = wrap_theta_scaled(float(start_scaled[2]), float(angle_scalor))

        states_yaw: list[np.ndarray] = [start_yaw.copy()]
        actions: list[np.ndarray] = []
        projected_actions: list[np.ndarray] = []
        resample_counts: list[int] = []

        reason = "timeout"
        success = False
        state_scaled = start_scaled.copy()

        anti_state: _AntiRepeatState | None = None
        if anti_repeat is not None and bool(anti_repeat.enabled):
            anti_state = _AntiRepeatState(
                xy_recent=deque(),
                xy_counts=Counter(),
                child_recent=deque(),
                child_counts=Counter(),
                edge_recent=deque(),
                edge_counts=Counter(),
                prev_edge=None,
            )
            k0 = _xy_key(float(state_scaled[0]), float(state_scaled[1]), q=float(anti_repeat.xy_q))
            _anti_push_xy(anti_state, k0, max_n=int(anti_repeat.xy_recent_n))

        for _ in range(int(args.max_steps)):
            if bool(getattr(fine_robot, "within_goal")(np.asarray(state_scaled, dtype=np.float32).reshape(3))):
                reason = "reached_goal"
                success = True
                break

            idx_list = getattr(fine_robot, "query_kdtree")(state_scaled)
            if not idx_list:
                reason = "snap_failed"
                break
            idx_cur = int(idx_list[0])

            children = list(getattr(fine_robot.nodes[int(idx_cur)].children, "indices", []))
            if bool(args.allow_self_candidate):
                children = [int(idx_cur)] + children
            if not children:
                reason = "no_children"
                break

            child_states = fine_node_states[np.asarray(children, dtype=np.int64)]
            vals = _coarse_values_at_states(grid=coarse_grid3d, states_scaled=child_states)
            reachable = vals < float(1.0 - unreachable_eps)
            reach_idx = np.flatnonzero(reachable)
            if int(reach_idx.shape[0]) == 0:
                reason = "unreachable"
                break
            children_reach = [int(children[int(i)]) for i in reach_idx]
            vals_reach = vals[reach_idx]

            order = np.argsort(vals_reach, kind="stable")
            ranked_children = [int(children_reach[int(i)]) for i in order]
            if topk_children > 0:
                ranked_children = ranked_children[: int(min(topk_children, len(ranked_children)))]

            pick = _pick_action_with_antirepeat(
                ranked_children=ranked_children,
                idx_cur=int(idx_cur),
                state_scaled=state_scaled,
                fine_node_states=fine_node_states,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                control_limits_scaled=control_limits_scaled,
                collision_check_step=float(args.collision_check_step),
                anti_repeat=anti_repeat,
                anti_state=anti_state,
            )
            if pick is None:
                reason = "no_children"
                break
            idx_next, proj = pick

            idx_anchor = int(idx_cur)
            ok, fail_detail, s_end = _postcheck_action(
                collision_robot=collision_robot,
                collision_task=task,
                cur_state_scaled=state_scaled,
                action_v_omega=proj,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                collision_check_step=float(args.collision_check_step),
                collision_semantic=str(args.collision_semantic),
            )
            if not bool(ok):
                reason = "resample_exhausted"
                if fail_detail is not None:
                    postcheck_fail_detail_counts[str(fail_detail)] += 1
                break

            actions.append(proj.astype(np.float32, copy=False))
            projected_actions.append(proj.astype(np.float32, copy=False))
            resample_counts.append(0)

            state_scaled = np.asarray(s_end, dtype=np.float32).reshape(3)
            yaw = float(yaw_from_theta_scaled(float(state_scaled[2]), float(angle_scalor)))
            states_yaw.append(np.array([float(state_scaled[0]), float(state_scaled[1]), float(yaw)], dtype=np.float32))

            if anti_repeat is not None and bool(anti_repeat.enabled) and anti_state is not None:
                k1 = _xy_key(float(state_scaled[0]), float(state_scaled[1]), q=float(anti_repeat.xy_q))
                _anti_push_xy(anti_state, k1, max_n=int(anti_repeat.xy_recent_n))
                _anti_push_child(anti_state, int(idx_next), max_n=int(anti_repeat.child_recent_n))
                _anti_push_edge(anti_state, (int(idx_anchor), int(idx_next)), max_n=int(anti_repeat.edge_recent_n))
                anti_state.prev_edge = (int(idx_anchor), int(idx_next))

        T = int(len(actions))
        if bool(success):
            successes += 1
            Ts_success.append(T)
        reason_counts[str(reason)] += 1

        ep_name = f"episode_{ep:04d}"
        states_arr = np.asarray(states_yaw, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32).reshape(-1, 2)
        projected_arr = np.asarray(projected_actions, dtype=np.float32).reshape(-1, 2)
        resample_arr = np.asarray(resample_counts, dtype=np.int64).reshape(-1)

        np.save(out_dir / f"{ep_name}_states.npy", states_arr)
        np.save(out_dir / f"{ep_name}_actions.npy", actions_arr)
        np.save(out_dir / f"{ep_name}_projected_actions.npy", projected_arr)
        np.save(out_dir / f"{ep_name}_resample_counts.npy", resample_arr)

        per_episode.append({"episode": int(ep), "success": bool(success), "reason": str(reason), "T": int(T)})
        print(f"[infer_coarse_value_greedy] ep={ep} success={success} reason={reason} T={T}", flush=True)

    out_summary: dict[str, Any] = {
        "map_name": str(goals_payload.get("map_name", goal_dir.parent.name)),
        "goal_index": int(goal_index),
        "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
        "execute_mode": "coarse_value_greedy",
        "episodes": int(episodes),
        "successes": int(successes),
        "success_rate": float(successes) / float(max(int(episodes), 1)),
        "avg_T": (float(np.mean(Ts_success)) if Ts_success else None),
        "median_T": (float(np.median(Ts_success)) if Ts_success else None),
        "reason_counts": dict(reason_counts),
        "postcheck_fail_detail_counts": dict(postcheck_fail_detail_counts),
        "avg_resample_per_step": 0.0,
        "start_filter": {
            "source": "file",
            "starts_file": str(starts_path),
            "starts_file_n": int(starts_yaw.shape[0]),
            "starts_meta": {k: starts_meta.get(k) for k in ("version", "source_infer_dir", "n_starts", "map_name", "goal_index") if k in starts_meta},
        },
        "sampling": {
            "collision_semantic": str(args.collision_semantic),
            "projected_collision_stage": "post",
            "value_source": "coarse_grid3d",
            "topk_children": int(topk_children),
            "allow_self_candidate": bool(args.allow_self_candidate),
            "anti_repeat": (
                None
                if anti_repeat is None
                else {
                    "enabled": bool(anti_repeat.enabled),
                    "xy_q": float(anti_repeat.xy_q),
                    "xy_recent_n": int(anti_repeat.xy_recent_n),
                    "child_recent_n": int(anti_repeat.child_recent_n),
                    "edge_recent_n": int(anti_repeat.edge_recent_n),
                    "avoid_uturn": bool(anti_repeat.avoid_uturn),
                }
            ),
        },
        "planner": {
            "type": "coarse_value_greedy_on_fine_children",
            "dt": float(dt),
            "unreachable_eps": float(unreachable_eps),
            "collision_check_step": float(args.collision_check_step),
        },
        "opt_a": (
            {
                "enabled": True,
                "delta": float(args.opt_a_delta),
                "cache_dir": str(args.opt_a_cache_dir),
                "goal_dir": (None if opt_a is None else str(opt_a.goal_dir)),
                "use_inflated_fine": True,
                "collision_check": "real",
            }
            if opt_a is not None
            else {"enabled": False, "delta": None}
        ),
        "per_episode": per_episode,
    }
    (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[infer_coarse_value_greedy] wrote summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
