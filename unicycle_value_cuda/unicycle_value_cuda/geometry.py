from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def rectangle_limits(region: Dict[str, Any]) -> Tuple[float, float, float, float]:
    limits = region["limits"]
    xmin, xmax = float(min(limits[0])), float(max(limits[0]))
    ymin, ymax = float(min(limits[1])), float(max(limits[1]))
    return xmin, xmax, ymin, ymax


def within_region(region: Dict[str, Any], pos_xy: np.ndarray) -> bool:
    shape = region.get("shape")
    if shape == "circle":
        center = np.asarray(region["center"], dtype=np.float32)
        radius = float(region["radius"])
        return float(np.linalg.norm(center - pos_xy)) <= radius
    if shape == "rectangle":
        xmin, xmax, ymin, ymax = rectangle_limits(region)
        x, y = float(pos_xy[0]), float(pos_xy[1])
        return xmin <= x <= xmax and ymin <= y <= ymax
    raise ValueError(f"Unsupported region shape: {shape!r}")

