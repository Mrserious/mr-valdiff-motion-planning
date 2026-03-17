from __future__ import annotations

from typing import Any

import numpy as np

from value_guided.geometry import in_circle, in_rectangle
from value_guided.task_io import get_obstacles, get_range


def outside_range_mask(task: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax = get_range(task)
    return (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)


def obstacle_mask(task: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(np.asarray(x, dtype=bool), dtype=bool)
    for ob in get_obstacles(task):
        shape = ob.get("shape", None)
        if shape == "circle":
            mask |= in_circle(x, y, center=ob["center"], radius=float(ob["radius"]))
        elif shape == "rectangle":
            mask |= in_rectangle(x, y, limits=ob["limits"])
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")
    return mask


def occupancy_mask(task: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask indicating occupied points:
      - outside env.range
      - inside any obstacle region
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return outside_range_mask(task, x, y) | obstacle_mask(task, x, y)


def rotate_points(dx: np.ndarray, dy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate local offsets (dx,dy) by yaw (rad) into world frame.
    """
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    rx = c * dx - s * dy
    ry = s * dx + c * dy
    return rx, ry

