from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from value_guided.occupancy import occupancy_mask, rotate_points


@dataclass(frozen=True)
class RegularValueGrid:
    """
    Regular grid value field V[row, col] defined over (x_samples[col], y_samples[row]).
    Value range is expected to be [0,1] (goal≈0, far/unreachable≈1).
    """

    V: np.ndarray  # (H, W), float32
    x_samples: np.ndarray  # (W,), float64/float32
    y_samples: np.ndarray  # (H,), float64/float32

    @property
    def H(self) -> int:
        return int(self.V.shape[0])

    @property
    def W(self) -> int:
        return int(self.V.shape[1])

    def as_ascending(self) -> "RegularValueGrid":
        V = self.V
        xs = self.x_samples
        ys = self.y_samples
        if xs.shape[0] >= 2 and xs[0] > xs[-1]:
            xs = xs[::-1].copy()
            V = V[:, ::-1].copy()
        if ys.shape[0] >= 2 and ys[0] > ys[-1]:
            ys = ys[::-1].copy()
            V = V[::-1, :].copy()
        return RegularValueGrid(V=V, x_samples=xs, y_samples=ys)

    def sample_bilinear(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Bilinear sampling on a regular axis-aligned grid.
        x,y can be arbitrary shapes; returns same shape float32.
        """
        grid = self.as_ascending()
        V = grid.V.astype(np.float32, copy=False)
        xs = grid.x_samples.astype(np.float64, copy=False)
        ys = grid.y_samples.astype(np.float64, copy=False)

        xq = np.asarray(x, dtype=np.float64)
        yq = np.asarray(y, dtype=np.float64)

        # clamp to grid bounds
        xq = np.clip(xq, xs[0], xs[-1])
        yq = np.clip(yq, ys[0], ys[-1])

        # find left indices
        ix = np.searchsorted(xs, xq, side="right") - 1
        iy = np.searchsorted(ys, yq, side="right") - 1
        ix = np.clip(ix, 0, xs.shape[0] - 2)
        iy = np.clip(iy, 0, ys.shape[0] - 2)

        x0 = xs[ix]
        x1 = xs[ix + 1]
        y0 = ys[iy]
        y1 = ys[iy + 1]

        tx = (xq - x0) / np.maximum(x1 - x0, 1e-12)
        ty = (yq - y0) / np.maximum(y1 - y0, 1e-12)

        v00 = V[iy, ix]
        v01 = V[iy, ix + 1]
        v10 = V[iy + 1, ix]
        v11 = V[iy + 1, ix + 1]

        v0 = (1.0 - tx) * v00 + tx * v01
        v1 = (1.0 - tx) * v10 + tx * v11
        out = (1.0 - ty) * v0 + ty * v1
        return out.astype(np.float32, copy=False)


def load_regular_value_grid(value_path: str | Path, meta_path: str | Path) -> RegularValueGrid:
    value_path = Path(value_path)
    meta_path = Path(meta_path)
    V = np.load(value_path).astype(np.float32, copy=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    x_samples = np.asarray(meta["x_samples"], dtype=np.float64)
    y_samples = np.asarray(meta["y_samples"], dtype=np.float64)
    if V.shape != (y_samples.shape[0], x_samples.shape[0]):
        raise ValueError(f"Value grid shape mismatch: V{V.shape} vs axes ({y_samples.shape[0]},{x_samples.shape[0]})")
    return RegularValueGrid(V=V, x_samples=x_samples, y_samples=y_samples)


@lru_cache(maxsize=64)
def _cached_axis_offsets(crop_size: int, meters_per_pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cache 1D axis offsets (float64) for local crop grids.

    Returns:
      - x_offsets: (W,) where W=crop_size
      - y_offsets: (H,) where H=crop_size
    """
    H = int(crop_size)
    W = int(crop_size)
    mpp = float(meters_per_pixel)
    if H <= 0 or W <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if mpp <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}")

    x_offsets = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) * mpp
    y_offsets = ((H - 1) / 2.0 - np.arange(H, dtype=np.float64)) * mpp  # image y-down -> world y-up
    return x_offsets, y_offsets


def make_local_image(
    *,
    task: dict[str, Any],
    coarse_grid: RegularValueGrid,
    state: Sequence[float],
    crop_size: int = 84,
    meters_per_pixel: float = 0.05,
    rotate_with_yaw: bool = False,
    value_clip: tuple[float, float] = (0.0, 1.0),
    scale_255: bool = True,
) -> np.ndarray:
    """
    Build local observation image centered at robot state.

    Output shape: (crop_size, crop_size, 2) with channel-last:
      - channel 0: occupancy (0/1)
      - channel 1: coarse value (0-1), masked by occupancy

    If scale_255=True: both channels are multiplied by 255 and kept in float32.
    This matches NavImageDataset's `/255` preprocessing.
    """
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if meters_per_pixel <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}")
    if len(state) < 2:
        raise ValueError(f"state must have at least (x,y), got {state}")
    robot_x = float(state[0])
    robot_y = float(state[1])
    yaw = float(state[2]) if len(state) >= 3 else 0.0

    H = int(crop_size)
    W = int(crop_size)

    x_offsets, y_offsets = _cached_axis_offsets(H, float(meters_per_pixel))
    dx = x_offsets[None, :]  # (1,W)
    dy = y_offsets[:, None]  # (H,1)
    if rotate_with_yaw and abs(yaw) > 1e-12:
        rdx, rdy = rotate_points(dx, dy, yaw)
    else:
        rdx, rdy = dx, dy
    world_x = robot_x + rdx
    world_y = robot_y + rdy
    world_x, world_y = np.broadcast_arrays(world_x, world_y)

    occ = occupancy_mask(task, world_x, world_y).astype(np.float32)

    val = coarse_grid.sample_bilinear(world_x, world_y)
    vmin, vmax = float(value_clip[0]), float(value_clip[1])
    val = np.clip(val, vmin, vmax).astype(np.float32)
    # mask obstacles / outside range
    val = np.where(occ > 0.5, np.float32(vmax), val).astype(np.float32, copy=False)

    img = np.stack([occ, val], axis=-1).astype(np.float32, copy=False)
    if scale_255:
        img = (img * 255.0).astype(np.float32, copy=False)
    return img


def make_local_images_batch(
    *,
    task: dict[str, Any],
    coarse_grid: RegularValueGrid,
    states: np.ndarray,
    crop_size: int = 84,
    meters_per_pixel: float = 0.05,
    rotate_with_yaw: bool = False,
    value_clip: tuple[float, float] = (0.0, 1.0),
    scale_255: bool = True,
) -> np.ndarray:
    """
    Batched version of make_local_image.

    Args:
        states: (T,2) or (T,3) array of (x,y[,yaw]).

    Returns:
        (T, crop_size, crop_size, 2) float32 (scaled by 255 if scale_255=True).
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError(f"states must be (T,2+) array, got {states.shape}")
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if meters_per_pixel <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}")

    robot_x = states[:, 0].astype(np.float64, copy=False)
    robot_y = states[:, 1].astype(np.float64, copy=False)
    if states.shape[1] >= 3:
        yaw = states[:, 2].astype(np.float64, copy=False)
    else:
        yaw = np.zeros((states.shape[0],), dtype=np.float64)

    H = int(crop_size)
    W = int(crop_size)
    x_offsets, y_offsets = _cached_axis_offsets(H, float(meters_per_pixel))

    if rotate_with_yaw:
        dx = x_offsets[None, :]  # (1,W)
        dy = y_offsets[:, None]  # (H,1)
        dx_grid, dy_grid = np.broadcast_arrays(dx, dy)  # (H,W)
        c = np.cos(yaw)[:, None, None]
        s = np.sin(yaw)[:, None, None]
        rdx = c * dx_grid[None, :, :] - s * dy_grid[None, :, :]
        rdy = s * dx_grid[None, :, :] + c * dy_grid[None, :, :]
        world_x = robot_x[:, None, None] + rdx
        world_y = robot_y[:, None, None] + rdy
    else:
        world_x = robot_x[:, None, None] + x_offsets[None, None, :]  # (T,1,W)
        world_y = robot_y[:, None, None] + y_offsets[None, :, None]  # (T,H,1)
        world_x, world_y = np.broadcast_arrays(world_x, world_y)  # (T,H,W)

    occ = occupancy_mask(task, world_x, world_y).astype(np.float32)

    val = coarse_grid.sample_bilinear(world_x, world_y)
    vmin, vmax = float(value_clip[0]), float(value_clip[1])
    val = np.clip(val, vmin, vmax).astype(np.float32)
    val = np.where(occ > 0.5, np.float32(vmax), val).astype(np.float32, copy=False)

    img = np.stack([occ, val], axis=-1).astype(np.float32, copy=False)
    if scale_255:
        img = (img * 255.0).astype(np.float32, copy=False)
    return img
