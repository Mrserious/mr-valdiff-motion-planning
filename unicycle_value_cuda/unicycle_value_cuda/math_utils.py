from __future__ import annotations

import numpy as np


def kruzhkov(x: float) -> float:
    return 1.0 - float(np.exp(-x))


def inv_kruzhkov(x: float, max_val: float) -> float:
    if x >= 1:
        return float(max_val)
    return -float(np.log(1 - x))


def rotation_matrix(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float32)

