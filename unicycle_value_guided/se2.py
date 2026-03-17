from __future__ import annotations

import math

import numpy as np


def angle_scalor_from_range(xmin: float, xmax: float) -> float:
    return float((float(xmax) - float(xmin)) / 2.0)


def yaw_from_theta_scaled(theta_scaled: float, angle_scalor: float) -> float:
    angle_scalor = float(angle_scalor)
    if angle_scalor <= 0:
        raise ValueError(f"angle_scalor must be positive, got {angle_scalor}")
    return float(theta_scaled / angle_scalor * math.pi)


def theta_scaled_from_yaw(yaw: float, angle_scalor: float) -> float:
    angle_scalor = float(angle_scalor)
    if angle_scalor <= 0:
        raise ValueError(f"angle_scalor must be positive, got {angle_scalor}")
    return float(yaw / math.pi * angle_scalor)


def wrap_yaw(yaw: float) -> float:
    """
    Wrap to [-pi, pi).
    """
    return float((yaw + math.pi) % (2.0 * math.pi) - math.pi)


def wrap_theta_scaled(theta_scaled: float, angle_scalor: float) -> float:
    """
    Wrap scaled angle to [-angle_scalor, angle_scalor).
    """
    a = float(angle_scalor)
    p = 2.0 * a
    return float(theta_scaled - math.floor((theta_scaled + a) / p) * p)


def signed_delta_theta_scaled(theta_from: float, theta_to: float, angle_scalor: float) -> float:
    """
    Minimal signed difference (theta_to - theta_from) in scaled space, wrapped to [-P/2, P/2),
    where P = 2*angle_scalor.
    """
    p = 2.0 * float(angle_scalor)
    d = float(theta_to - theta_from)
    d = (d + p / 2.0) % p - p / 2.0
    return float(d)


def distance_state_scaled(a: np.ndarray, b: np.ndarray, angle_scalor: float) -> float:
    """
    L2 distance in (x,y,theta_scaled) with periodic theta.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dth = float(abs(a[2] - b[2]))
    p = 2.0 * float(angle_scalor)
    dth = min(dth, p - dth)
    return float(math.sqrt(dx * dx + dy * dy + dth * dth))

