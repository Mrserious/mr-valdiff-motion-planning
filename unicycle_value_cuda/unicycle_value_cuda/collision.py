from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .geometry import rectangle_limits


def _rotate_points(points_xy: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    points_xy: (S,2)
    cos_t/sin_t: (N,)
    Returns rotated coordinates in world frame (without translation):
      rx: (N,S), ry: (N,S)
    """
    px = points_xy[:, 0][None, :]  # (1,S)
    py = points_xy[:, 1][None, :]  # (1,S)
    cos_v = cos_t[:, None]
    sin_v = sin_t[:, None]
    rx = cos_v * px - sin_v * py
    ry = sin_v * px + cos_v * py
    return rx, ry


def obstacle_free_mask_unicycle(
    *,
    env: Dict[str, Any],
    body_samples_xy: np.ndarray,
    states: np.ndarray,
    angle_scalor: float,
    chunk: int = 50_000,
) -> np.ndarray:
    """
    Vectorized version of the notebook-style collision check:
    - Treat each state as collision-free iff none of the body sample points lies inside any obstacle.
    - Does NOT perform swept/continuous collision checks along motion.

    Args:
        states: (N,3) float32, theta is theta_scaled in [-angle_scalor, angle_scalor)
    Returns:
        mask: (N,) bool, True means obstacle_free
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must be (N,3) array")
    obstacles = env.get("obstacles", [])
    if not obstacles:
        return np.ones((states.shape[0],), dtype=bool)

    a = float(angle_scalor)
    # theta_real = theta_scaled / angle_scalor * pi
    theta_real = states[:, 2].astype(np.float64) / a * np.pi
    cos_t = np.cos(theta_real).astype(np.float32)
    sin_t = np.sin(theta_real).astype(np.float32)
    pos = states[:, :2].astype(np.float32)

    out = np.ones((states.shape[0],), dtype=bool)
    n = states.shape[0]
    for start in range(0, n, int(chunk)):
        end = min(start + int(chunk), n)
        cos_c = cos_t[start:end]
        sin_c = sin_t[start:end]
        pos_c = pos[start:end]

        rx, ry = _rotate_points(body_samples_xy.astype(np.float32), cos_c, sin_c)
        wx = rx + pos_c[:, 0:1]
        wy = ry + pos_c[:, 1:2]

        collision = np.zeros((end - start,), dtype=bool)
        for obs in obstacles:
            shape = obs.get("shape")
            if shape == "rectangle":
                xmin, xmax, ymin, ymax = rectangle_limits(obs)
                inside = (wx >= xmin) & (wx <= xmax) & (wy >= ymin) & (wy <= ymax)
                collision |= inside.any(axis=1)
            elif shape == "circle":
                center = np.asarray(obs["center"], dtype=np.float32)
                r2 = float(obs["radius"]) ** 2
                dx = wx - center[0]
                dy = wy - center[1]
                inside = (dx * dx + dy * dy) <= r2
                collision |= inside.any(axis=1)
            else:
                raise ValueError(f"Unsupported obstacle shape: {shape!r}")
            if collision.all():
                break
        out[start:end] = ~collision
    return out

