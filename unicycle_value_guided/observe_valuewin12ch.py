from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from value_guided.occupancy import occupancy_mask, rotate_points

from unicycle_value_guided.se2 import wrap_yaw
from unicycle_value_guided.value_grid3d import RegularValueGrid3D


@lru_cache(maxsize=64)
def _cached_axis_offsets(crop_size: int, meters_per_pixel: float, bias_forward_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cache 1D axis offsets (float64) for local crop grids.

    Returns:
      - x_offsets: (W,) where W=crop_size, robot-frame +x is forward.
      - y_offsets: (H,) where H=crop_size, robot-frame +y is left/up.
    """
    H = int(crop_size)
    W = int(crop_size)
    mpp = float(meters_per_pixel)
    bias = float(bias_forward_m)
    if H <= 0 or W <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if mpp <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}")

    x_offsets = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) * mpp + bias
    y_offsets = ((H - 1) / 2.0 - np.arange(H, dtype=np.float64)) * mpp  # image y-down -> world y-up
    return x_offsets, y_offsets


@lru_cache(maxsize=32)
def _cached_footprint_kernels(
    *,
    length_m: float,
    width_m: float,
    meters_per_pixel: float,
    yaw_offsets_deg: tuple[int, ...],
    num_samples_per_side: int = 11,
) -> tuple[torch.Tensor, int]:
    """
    Build footprint kernels for each yaw offset (relative to the local image axes).

    Returns:
      - kernels: (N,1,Kh,Kw) float32 on CPU, suitable for conv2d
      - zero_idx: index in yaw_offsets_deg that corresponds to 0 deg
    """
    L = float(length_m)
    W = float(width_m)
    mpp = float(meters_per_pixel)
    n = int(num_samples_per_side)
    if L <= 0 or W <= 0:
        raise ValueError(f"Invalid footprint size: length={L} width={W}")
    if mpp <= 0:
        raise ValueError(f"Invalid meters_per_pixel: {mpp}")
    if n <= 0:
        raise ValueError(f"Invalid num_samples_per_side: {n}")

    if 0 not in yaw_offsets_deg:
        raise ValueError("yaw_offsets_deg must include 0 for occupancy(mask_cur).")
    zero_idx = int(list(yaw_offsets_deg).index(0))

    # Local robot footprint samples (robot frame).
    sx = np.linspace(-L / 2.0, L / 2.0, n, dtype=np.float64)
    sy = np.linspace(-W / 2.0, W / 2.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(sx, sy, indexing="xy")
    samples = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # (Ns,2)

    # First pass: find max |dr|,|dc| across all offsets to fix a common kernel size.
    max_abs = 0
    offsets_rad = [float(np.deg2rad(int(d))) for d in yaw_offsets_deg]
    for d in offsets_rad:
        c = float(np.cos(d))
        s = float(np.sin(d))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts = samples @ rot.T  # (Ns,2) in local image axes
        dc = np.rint(pts[:, 0] / mpp).astype(np.int64)
        dr = np.rint(-pts[:, 1] / mpp).astype(np.int64)  # +y(up) -> -row
        max_abs = int(max(max_abs, int(np.max(np.abs(dc))), int(np.max(np.abs(dr)))))

    k = int(2 * max_abs + 1)
    center = int(max_abs)
    kernels = torch.zeros((len(offsets_rad), 1, k, k), dtype=torch.float32, device="cpu")

    # Second pass: populate kernels
    for i, d in enumerate(offsets_rad):
        c = float(np.cos(d))
        s = float(np.sin(d))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts = samples @ rot.T
        dc = np.rint(pts[:, 0] / mpp).astype(np.int64)
        dr = np.rint(-pts[:, 1] / mpp).astype(np.int64)
        rr = center + dr
        cc = center + dc
        rr = np.clip(rr, 0, k - 1)
        cc = np.clip(cc, 0, k - 1)
        kernels[i, 0, torch.from_numpy(rr), torch.from_numpy(cc)] = 1.0

    return kernels, zero_idx


def make_local_valuewin12ch(
    *,
    task: dict[str, Any],
    coarse_grid3d: RegularValueGrid3D,
    state: Sequence[float],
    crop_size: int = 84,
    meters_per_pixel: float = 0.05,
    rotate_with_yaw: bool = True,
    crop_bias_forward_m: float = 0.9375,
    yaw_offsets_deg: tuple[int, ...] = (
        -45,
        -36,
        -27,
        -18,
        -9,
        0,
        9,
        18,
        27,
        36,
        45,
        135,
        144,
        153,
        162,
        171,
        180,
        189,
        198,
        207,
        216,
        225,
    ),
    footprint_length_m: float = 0.625,
    footprint_width_m: float = 0.4375,
    value_clip: tuple[float, float] = (0.0, 1.0),
    scale_255: bool = True,
) -> np.ndarray:
    """
    Build unicycle local observation with:
      - channel 0: occupancy = footprint-aware mask at yaw_cur
      - channels 1..Nθ: V(x,y,yaw_i) slices masked by footprint-aware mask at yaw_i

    Output: (crop_size, crop_size, 1+Nθ) float32, scaled to [0,255] if scale_255=True.
    """
    if len(state) < 3:
        raise ValueError(f"state must be (x,y,yaw), got {state}")
    robot_x = float(state[0])
    robot_y = float(state[1])
    yaw_cur = float(state[2])
    yaw_cur = wrap_yaw(yaw_cur)

    H = int(crop_size)
    W = int(crop_size)

    x_offsets, y_offsets = _cached_axis_offsets(H, float(meters_per_pixel), float(crop_bias_forward_m))
    dx = x_offsets[None, :]  # (1,W)
    dy = y_offsets[:, None]  # (H,1)
    if rotate_with_yaw and abs(yaw_cur) > 1e-12:
        rdx, rdy = rotate_points(dx, dy, yaw_cur)
    else:
        rdx, rdy = dx, dy
    world_x = robot_x + rdx
    world_y = robot_y + rdy
    world_x, world_y = np.broadcast_arrays(world_x, world_y)  # (H,W)

    # Base point-occupancy (pixel centers) in world frame.
    point_occ = occupancy_mask(task, world_x, world_y).astype(np.float32)  # (H,W) in {0,1}

    # Footprint masks via conv2d on point_occ. Orientation is relative to local image axes.
    kernels, zero_idx = _cached_footprint_kernels(
        length_m=float(footprint_length_m),
        width_m=float(footprint_width_m),
        meters_per_pixel=float(meters_per_pixel),
        yaw_offsets_deg=tuple(int(d) for d in yaw_offsets_deg),
        num_samples_per_side=11,
    )
    k = int(kernels.shape[-1])
    pad = int(k // 2)
    occ_t = torch.from_numpy(point_occ[None, None, :, :])  # (1,1,H,W)
    masks_t = F.conv2d(occ_t, kernels, padding=pad) > 0.0  # (1,N,H,W) bool
    masks = masks_t[0].cpu().numpy().astype(bool, copy=False)  # (N,H,W)

    # occupancy channel uses mask_cur (yaw_cur only)
    occ = masks[int(zero_idx)].astype(np.float32)

    # Prepare bilinear indices once for all slices (xy grid is shared).
    ix, iy, tx, ty = coarse_grid3d.prepare_bilinear_indices(world_x, world_y)

    vmin, vmax = float(value_clip[0]), float(value_clip[1])
    slices: list[np.ndarray] = []
    for i, d in enumerate(yaw_offsets_deg):
        yaw_i = wrap_yaw(yaw_cur + float(np.deg2rad(int(d))))
        val = coarse_grid3d.sample_trilinear_prepared(yaw_rad=yaw_i, ix=ix, iy=iy, tx=tx, ty=ty)
        val = np.clip(val, vmin, vmax).astype(np.float32, copy=False)
        # footprint mask at yaw_i
        val = np.where(masks[i], np.float32(vmax), val).astype(np.float32, copy=False)
        slices.append(val)

    img = np.stack([occ] + slices, axis=-1).astype(np.float32, copy=False)  # (H,W,1+N)
    if scale_255:
        img = (img * 255.0).astype(np.float32, copy=False)
    return img
