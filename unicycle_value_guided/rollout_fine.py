from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from unicycle_value_guided.se2 import signed_delta_theta_scaled, yaw_from_theta_scaled


def _pick_child_lowest_value(robot: Any, idx: int) -> int | None:
    node = robot.nodes[int(idx)]
    children = list(getattr(node.children, "indices", []))
    if not children:
        return None
    return int(min(children, key=lambda i: float(getattr(robot.nodes[int(i)], "value", 1.0))))


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


def reconstruct_action_from_transition(
    *,
    cur_state_scaled: np.ndarray,
    nxt_state_scaled: np.ndarray,
    dt: float,
    angle_scalor: float,
) -> np.ndarray:
    """
    Approximate (v, omega_rad) that takes cur -> nxt in one dt step.

    Notes:
      - omega is derived from the minimal signed delta in theta_scaled, then converted to rad/s.
      - v is derived by projecting xy displacement onto current heading direction.
    """
    cur = np.asarray(cur_state_scaled, dtype=np.float64).reshape(3)
    nxt = np.asarray(nxt_state_scaled, dtype=np.float64).reshape(3)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    yaw = float(yaw_from_theta_scaled(float(cur[2]), float(angle_scalor)))
    dx = float(nxt[0] - cur[0])
    dy = float(nxt[1] - cur[1])

    v = (dx * float(np.cos(yaw)) + dy * float(np.sin(yaw))) / float(dt)

    dtheta_scaled = float(signed_delta_theta_scaled(float(cur[2]), float(nxt[2]), float(angle_scalor)))
    omega_scaled = dtheta_scaled / float(dt)
    omega_rad = omega_scaled / float(angle_scalor) * float(np.pi)

    return np.array([v, omega_rad], dtype=np.float32)


@dataclass(frozen=True)
class RolloutResult:
    states_scaled: np.ndarray  # (T,3) pre-action, theta_scaled
    actions_v_omega: np.ndarray  # (T,2) (v, omega_rad/s)
    success: bool
    reason: str


def rollout_greedy(
    *,
    robot: Any,
    start_state_scaled: Sequence[float],
    max_steps: int,
    angle_scalor: float,
    stop_on_cycle: bool = True,
) -> RolloutResult:
    """
    Greedy descent on the VI graph:
      - snap start to nearest node (in 3D, periodic theta handled by robot.query_kdtree)
      - repeatedly pick child with lowest value
      - reconstruct (v,omega) from consecutive node states
    """
    start_state_scaled = np.asarray(start_state_scaled, dtype=np.float32).reshape(3)

    dt = float(getattr(robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from robot.get_temporal_res(): {dt}")

    # nearest node (unicycle robot uses 3D KDTree + periodic theta logic)
    idx_list = getattr(robot, "query_kdtree")(start_state_scaled)
    if not idx_list:
        raise RuntimeError("robot.query_kdtree returned empty indices.")
    cur_idx = int(idx_list[0])

    visited = {cur_idx}
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for _ in range(int(max_steps)):
        cur_state = np.asarray(robot.nodes[cur_idx].state, dtype=np.float32).reshape(3)
        if bool(getattr(robot, "within_goal")(cur_state)):
            return RolloutResult(
                states_scaled=np.asarray(states, dtype=np.float32),
                actions_v_omega=np.asarray(actions, dtype=np.float32),
                success=True,
                reason="reached_goal",
            )

        nxt_idx = _pick_child_lowest_value(robot, cur_idx)
        if nxt_idx is None:
            return RolloutResult(
                states_scaled=np.asarray(states, dtype=np.float32),
                actions_v_omega=np.asarray(actions, dtype=np.float32),
                success=False,
                reason="no_children",
            )
        if stop_on_cycle and int(nxt_idx) in visited:
            return RolloutResult(
                states_scaled=np.asarray(states, dtype=np.float32),
                actions_v_omega=np.asarray(actions, dtype=np.float32),
                success=False,
                reason="cycle",
            )

        nxt_state = np.asarray(robot.nodes[int(nxt_idx)].state, dtype=np.float32).reshape(3)
        a = reconstruct_action_from_transition(
            cur_state_scaled=cur_state,
            nxt_state_scaled=nxt_state,
            dt=dt,
            angle_scalor=float(angle_scalor),
        )
        a = _clip_action_v_omega(a, control_limits_scaled=getattr(robot, "control_limits", [[-1, 1], [-1, 1]]), angle_scalor=float(angle_scalor))

        states.append(cur_state)
        actions.append(a)
        visited.add(int(nxt_idx))
        cur_idx = int(nxt_idx)

    return RolloutResult(
        states_scaled=np.asarray(states, dtype=np.float32),
        actions_v_omega=np.asarray(actions, dtype=np.float32),
        success=False,
        reason="max_steps",
    )

