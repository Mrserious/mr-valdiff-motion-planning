from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from value_guided.occupancy import rotate_points

from unicycle_value_guided.se2 import theta_scaled_from_yaw, wrap_yaw, yaw_from_theta_scaled


@dataclass(frozen=True)
class GlobalRefConfig:
    path_len: int = 100
    # 3 arclength-quantile waypoints + final goal pose -> gpath(4,3).
    # Fractions are w.r.t. the total arclength of the rolled-out coarse node path.
    quantile_fractions: tuple[float, float, float] = (0.2, 0.4, 0.6)
    min_value_drop: float = 1e-6
    max_stuck_steps: int = 10


def _pick_child_lowest_value(robot: Any, idx: int) -> int | None:
    node = robot.nodes[int(idx)]
    children = list(getattr(node.children, "indices", []))
    if not children:
        return None
    return int(min(children, key=lambda i: float(getattr(robot.nodes[int(i)], "value", 1.0))))


def _to_robot_frame(dx: np.ndarray, dy: np.ndarray, yaw: float) -> np.ndarray:
    rx, ry = rotate_points(dx, dy, -float(yaw))
    return np.stack([rx, ry], axis=-1).astype(np.float32, copy=False)

def rollout_coarse_vi_path_states_scaled(
    *,
    robot: Any,
    start_state: Sequence[float],
    angle_scalor: float,
    cfg: GlobalRefConfig = GlobalRefConfig(),
    stop_on_cycle: bool = True,
) -> np.ndarray:
    """
    Greedy roll-out on the coarse VI graph (in 3D), returning node states (L,3) in scaled theta.
    """
    if len(start_state) < 3:
        raise ValueError(f"start_state must be (x,y,yaw), got {start_state}")
    x, y, yaw = float(start_state[0]), float(start_state[1]), float(start_state[2])
    start_scaled = np.array([x, y, theta_scaled_from_yaw(yaw, float(angle_scalor))], dtype=np.float32)

    idx_list = getattr(robot, "query_kdtree")(start_scaled)
    if not idx_list:
        return np.asarray([[x, y, start_scaled[2]]], dtype=np.float32)
    cur_idx = int(idx_list[0])

    # Start from the snapped node state so that every waypoint is a real graph node.
    states = [np.asarray(robot.nodes[cur_idx].state, dtype=np.float32).reshape(3)]
    visited = {cur_idx}
    for _ in range(int(cfg.path_len) - 1):
        nxt = _pick_child_lowest_value(robot, cur_idx)
        if nxt is None:
            break
        if stop_on_cycle and int(nxt) in visited:
            break
        s = np.asarray(robot.nodes[int(nxt)].state, dtype=np.float32).reshape(3)
        states.append(s.astype(np.float32, copy=False))
        visited.add(int(nxt))
        cur_idx = int(nxt)
    return np.stack(states, axis=0).astype(np.float32, copy=False)


def _pick_node_indices_by_arclength(
    pts_xy: np.ndarray,
    fractions: Sequence[float],
    *,
    eps: float = 1e-12,
) -> list[int]:
    """
    Node-based arclength quantiles (no continuous interpolation).

    Returns indices k such that cum[k] >= frac * total.
    """
    pts_xy = np.asarray(pts_xy, dtype=np.float64)
    if pts_xy.ndim != 2 or pts_xy.shape[1] != 2 or pts_xy.shape[0] < 1:
        raise ValueError(f"pts_xy must be (L,2) with L>=1, got {pts_xy.shape}")
    fractions = [float(f) for f in fractions]
    if len(fractions) == 0:
        return []

    if pts_xy.shape[0] == 1:
        return [0 for _ in fractions]

    diffs = np.diff(pts_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg_lens, dtype=np.float64)], axis=0)  # (L,)
    total = float(cum[-1])
    if (not np.isfinite(total)) or total <= eps:
        return [0 for _ in fractions]

    out: list[int] = []
    for frac in fractions:
        frac = float(np.clip(frac, 0.0, 1.0))
        s = frac * total
        k = int(np.searchsorted(cum, s, side="left"))
        k = max(0, min(k, int(cum.shape[0]) - 1))
        out.append(k)
    return out


def compute_gpath_from_vi(
    *,
    coarse_robot: Any,
    state: Sequence[float],
    goal_xy: Sequence[float],
    goal_yaw: float | None,
    angle_scalor: float,
    cfg: GlobalRefConfig = GlobalRefConfig(),
) -> np.ndarray:
    """
    Build gpath (4,3) in robot frame:
      gpath[i] = [dx_i, dy_i, dyaw_i]

    Waypoints:
      - i=0..2: coarse VI node path arclength quantiles (20/40/60 by default)
      - i=3: goal pose (x_goal,y_goal,yaw_goal)

    Notes:
      - Waypoints are selected as nodes (no continuous interpolation) so that (x,y,yaw)
        come from the same coarse VI node.
      - dyaw is normalized by pi to [-1,1].
      - If goal_yaw is None, dyaw for the goal row is set to 0.
    """
    if len(state) < 3:
        raise ValueError(f"state must be (x,y,yaw), got {state}")
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(2)

    states_scaled = rollout_coarse_vi_path_states_scaled(
        robot=coarse_robot,
        start_state=[x, y, yaw],
        angle_scalor=float(angle_scalor),
        cfg=cfg,
    )
    # Node selection by arclength (on XY only).
    pts_xy = states_scaled[:, 0:2].astype(np.float32, copy=False)
    ks = _pick_node_indices_by_arclength(pts_xy, cfg.quantile_fractions)
    # Ensure exactly 3 indices.
    if len(ks) != 3:
        raise ValueError(f"cfg.quantile_fractions must have length 3, got {cfg.quantile_fractions}")

    wp_scaled = states_scaled[np.asarray(ks, dtype=np.int64)]  # (3,3) scaled theta
    wp_xy = wp_scaled[:, 0:2].astype(np.float32, copy=False)
    wp_yaw = np.array([yaw_from_theta_scaled(float(th), float(angle_scalor)) for th in wp_scaled[:, 2]], dtype=np.float32)  # (3,)

    # dx,dy in robot frame
    dxw = wp_xy[:, 0] - np.float32(x)
    dyw = wp_xy[:, 1] - np.float32(y)
    dxy_robot = _to_robot_frame(dxw, dyw, yaw)  # (3,2)

    dyaw = np.zeros((3,), dtype=np.float32)
    for i in range(3):
        dyaw[i] = float(wrap_yaw(float(wp_yaw[i]) - float(yaw)) / float(np.pi))

    gpath = np.zeros((4, 3), dtype=np.float32)
    gpath[0:3, 0:2] = dxy_robot
    gpath[0:3, 2] = dyaw

    # goal row
    dxg = float(goal_xy[0]) - float(x)
    dyg = float(goal_xy[1]) - float(y)
    dxyg = _to_robot_frame(np.asarray([dxg], dtype=np.float32), np.asarray([dyg], dtype=np.float32), yaw).reshape(2)
    gpath[3, 0:2] = dxyg.astype(np.float32, copy=False)
    if goal_yaw is not None and np.isfinite(float(goal_yaw)):
        gpath[3, 2] = float(wrap_yaw(float(goal_yaw) - float(yaw)) / float(np.pi))
    else:
        gpath[3, 2] = 0.0

    return gpath
