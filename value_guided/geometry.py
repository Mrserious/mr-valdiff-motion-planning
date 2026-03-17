from __future__ import annotations

from typing import Sequence

import numpy as np


def _minmax_2(x: Sequence[float]) -> tuple[float, float]:
    if len(x) != 2:
        raise ValueError(f"Expected length-2 sequence, got: {x}")
    a = float(x[0])
    b = float(x[1])
    return (a, b) if a <= b else (b, a)


def in_rectangle(
    x: float | np.ndarray,
    y: float | np.ndarray,
    limits: Sequence[Sequence[float]],
) -> bool | np.ndarray:
    """
    Axis-aligned rectangle membership.
    limits: [[x1,x2],[y1,y2]] (order may be swapped).
    """
    xmin, xmax = _minmax_2(limits[0])
    ymin, ymax = _minmax_2(limits[1])
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return (xmin <= x_arr) & (x_arr <= xmax) & (ymin <= y_arr) & (y_arr <= ymax)


def in_circle(
    x: float | np.ndarray,
    y: float | np.ndarray,
    center: Sequence[float],
    radius: float,
) -> bool | np.ndarray:
    cx = float(center[0])
    cy = float(center[1])
    r = float(radius)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return (x_arr - cx) ** 2 + (y_arr - cy) ** 2 <= r**2


def distance_to_rectangle(
    x: float | np.ndarray,
    y: float | np.ndarray,
    limits: Sequence[Sequence[float]],
) -> float | np.ndarray:
    """
    Euclidean distance to an axis-aligned rectangle *region*.
    Returns 0 if the point is inside or on the rectangle.
    """
    xmin, xmax = _minmax_2(limits[0])
    ymin, ymax = _minmax_2(limits[1])
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    dx = np.maximum(np.maximum(xmin - x_arr, 0.0), x_arr - xmax)
    dy = np.maximum(np.maximum(ymin - y_arr, 0.0), y_arr - ymax)
    return np.sqrt(dx * dx + dy * dy)


def distance_to_circle(
    x: float | np.ndarray,
    y: float | np.ndarray,
    center: Sequence[float],
    radius: float,
) -> float | np.ndarray:
    """
    Euclidean distance to a circle *region* (disk).
    Returns 0 if the point is inside or on the circle.
    """
    cx = float(center[0])
    cy = float(center[1])
    r = float(radius)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    dist_center = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)
    return np.maximum(dist_center - r, 0.0)


