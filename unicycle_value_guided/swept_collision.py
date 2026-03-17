from __future__ import annotations

import math
from typing import Any

import numpy as np

from unicycle_value_guided.se2 import theta_scaled_from_yaw, wrap_theta_scaled, yaw_from_theta_scaled
from unicycle_value_guided.task_io import get_range


def trajectory_collision_free(
    *,
    robot: Any,
    task: dict,
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
    step_size: float,
) -> bool:
    """
    Swept/continuous collision check for one constant-control step.

    Integrates the unicycle dynamics with constant (v, omega) for duration dt,
    checking:
      - boundary (task.env.range)
      - robot.obstacle_free() at multiple sub-steps along the segment

    Notes:
      - `state_scaled` uses theta_scaled (not yaw).
      - `action_v_omega` uses omega in rad/s.
      - This is the same semantic check used in inference to filter candidate children.
    """
    xmin, xmax, ymin, ymax = get_range(task)
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
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
        if not bool(getattr(robot, "obstacle_free")(np.array([x, y, th], dtype=np.float32))):
            return False
    return True

