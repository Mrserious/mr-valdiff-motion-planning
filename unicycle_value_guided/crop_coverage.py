from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def crop_forward_backward_m(*, crop_size: int, meters_per_pixel: float, crop_bias_forward_m: float) -> tuple[float, float]:
    """
    Approximate forward/backward coverage (in meters) of the local crop in robot frame.

    For an image of width W=crop_size and resolution mpp:
      half = ((W-1)/2) * mpp
      forward  ~ half + bias
      backward ~ half - bias
    """
    W = int(crop_size)
    mpp = float(meters_per_pixel)
    bias = float(crop_bias_forward_m)
    if W <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if mpp <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}")
    half = ((float(W) - 1.0) / 2.0) * mpp
    return float(half + bias), float(half - bias)


def required_child_radius_m(*, fine_robot: Any) -> float:
    """
    Conservative estimate of how far a one-step child node can be from the current node in XY.

    Graph construction is:
      - integrate one dt step using a discretized control
      - connect to any neighbor within radius rho of that candidate

    So a conservative bound is: dt * max_speed + rho
    """
    dt = float(getattr(fine_robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from fine_robot.get_temporal_res(): {dt}")
    rho = float(getattr(fine_robot, "get_perturbation_radius")())
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"Invalid rho from fine_robot.get_perturbation_radius(): {rho}")

    lims = np.asarray(getattr(fine_robot, "control_limits", [[-1, 1], [-1, 1]]), dtype=np.float64).reshape(2, 2)
    max_speed = float(max(abs(float(lims[0, 0])), abs(float(lims[0, 1]))))
    return float(dt * max_speed + rho)


def validate_crop_covers_children(
    *,
    fine_robot: Any,
    crop_size: int,
    meters_per_pixel: float,
    crop_bias_forward_m: float,
    strict: bool,
    extra_margin_m: float = 0.0,
    context: str = "",
) -> None:
    """
    Validate that both forward/backward crop coverage can contain one-step children (conservative bound).
    """
    req = required_child_radius_m(fine_robot=fine_robot) + float(extra_margin_m)
    fwd, back = crop_forward_backward_m(crop_size=int(crop_size), meters_per_pixel=float(meters_per_pixel), crop_bias_forward_m=float(crop_bias_forward_m))
    ok = min(fwd, back) >= req
    if ok:
        return

    msg = (
        f"Crop may not cover one-step children (conservative). "
        f"need_radius≈{req:.3f}m but forward≈{fwd:.3f}m backward≈{back:.3f}m. "
        f"(crop_size={int(crop_size)} mpp={float(meters_per_pixel)} bias={float(crop_bias_forward_m)})"
    )
    if context:
        msg = f"{context}: {msg}"
    if strict:
        raise ValueError(msg)
    print(f"[warn] {msg}", flush=True)

