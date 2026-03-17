from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from unicycle_value_guided.se2 import theta_scaled_from_yaw, wrap_theta_scaled


@dataclass(frozen=True)
class RegularValueGrid3D:
    """
    Regular grid value field V[it, row, col] defined over:
      - x_samples[col]
      - y_samples[row]
      - theta_samples_scaled[it]  (theta_scaled in [-angle_scalor, angle_scalor))

    Expected value range: [0,1] (goal≈0, far/unreachable≈1).
    axis_order is fixed to "theta,y,x".
    """

    V: np.ndarray  # (nt, ny, nx), float32
    x_samples: np.ndarray  # (nx,), float64/float32
    y_samples: np.ndarray  # (ny,), float64/float32
    theta_samples_scaled: np.ndarray  # (nt,), float64/float32
    angle_scalor: float

    @property
    def nt(self) -> int:
        return int(self.V.shape[0])

    @property
    def ny(self) -> int:
        return int(self.V.shape[1])

    @property
    def nx(self) -> int:
        return int(self.V.shape[2])

    def as_ascending(self) -> "RegularValueGrid3D":
        V = self.V
        xs = self.x_samples
        ys = self.y_samples
        if xs.shape[0] >= 2 and xs[0] > xs[-1]:
            xs = xs[::-1].copy()
            V = V[:, :, ::-1].copy()
        if ys.shape[0] >= 2 and ys[0] > ys[-1]:
            ys = ys[::-1].copy()
            V = V[:, ::-1, :].copy()
        return RegularValueGrid3D(
            V=V.astype(np.float32, copy=False),
            x_samples=xs,
            y_samples=ys,
            theta_samples_scaled=self.theta_samples_scaled,
            angle_scalor=float(self.angle_scalor),
        )

    def prepare_bilinear_indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute bilinear indices/weights for a fixed (x,y) query grid.

        Returns:
          ix, iy: int64 arrays
          tx, ty: float32 arrays
        """
        grid = self.as_ascending()
        xs = grid.x_samples.astype(np.float64, copy=False)
        ys = grid.y_samples.astype(np.float64, copy=False)

        xq = np.asarray(x, dtype=np.float64)
        yq = np.asarray(y, dtype=np.float64)

        # clamp to grid bounds
        xq = np.clip(xq, xs[0], xs[-1])
        yq = np.clip(yq, ys[0], ys[-1])

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

        return ix.astype(np.int64, copy=False), iy.astype(np.int64, copy=False), tx.astype(np.float32, copy=False), ty.astype(np.float32, copy=False)

    def sample_bilinear_prepared(
        self,
        it: int,
        *,
        ix: np.ndarray,
        iy: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
    ) -> np.ndarray:
        """
        Bilinear sampling on slice V[it,:,:] with prepared (ix,iy,tx,ty).
        Returns float32 with same shape as ix/iy.
        """
        grid = self.as_ascending()
        V = grid.V.astype(np.float32, copy=False)
        it = int(it) % int(grid.nt)

        v00 = V[it, iy, ix]
        v01 = V[it, iy, ix + 1]
        v10 = V[it, iy + 1, ix]
        v11 = V[it, iy + 1, ix + 1]

        v0 = (1.0 - tx) * v00 + tx * v01
        v1 = (1.0 - tx) * v10 + tx * v11
        out = (1.0 - ty) * v0 + ty * v1
        return out.astype(np.float32, copy=False)

    def sample_trilinear_prepared(
        self,
        *,
        yaw_rad: float,
        ix: np.ndarray,
        iy: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
    ) -> np.ndarray:
        """
        Trilinear sampling at (x,y,yaw_rad) where (x,y) part is fixed by prepared indices.
        yaw is converted to theta_scaled and interpolated with periodic wrap.
        """
        a = float(self.angle_scalor)
        if a <= 0:
            raise ValueError(f"angle_scalor must be positive, got {a}")
        nt = int(self.nt)
        if nt <= 1:
            raise ValueError(f"nt must be >1 for trilinear, got {nt}")

        theta_scaled = theta_scaled_from_yaw(float(yaw_rad), a)
        theta_scaled = wrap_theta_scaled(theta_scaled, a)
        dtheta = 2.0 * a / float(nt)
        u = (theta_scaled + a) / dtheta
        it0 = int(np.floor(u)) % nt
        it1 = (it0 + 1) % nt
        tt = float(u - np.floor(u))

        v0 = self.sample_bilinear_prepared(it0, ix=ix, iy=iy, tx=tx, ty=ty)
        v1 = self.sample_bilinear_prepared(it1, ix=ix, iy=iy, tx=tx, ty=ty)
        out = (1.0 - tt) * v0 + tt * v1
        return out.astype(np.float32, copy=False)


def load_regular_value_grid_3d(value_path: str | Path, meta_path: str | Path) -> RegularValueGrid3D:
    value_path = Path(value_path)
    meta_path = Path(meta_path)
    V = np.load(value_path).astype(np.float32, copy=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    axis_order = str(meta.get("axis_order", "theta,y,x"))
    if axis_order != "theta,y,x":
        raise ValueError(f"Unsupported axis_order: {axis_order!r} (expected 'theta,y,x')")

    x_samples = np.asarray(meta["x_samples"], dtype=np.float64)
    y_samples = np.asarray(meta["y_samples"], dtype=np.float64)
    theta_samples_scaled = np.asarray(meta["theta_samples_scaled"], dtype=np.float64)
    angle_scalor = float(meta.get("angle_scalor"))

    expected = (theta_samples_scaled.shape[0], y_samples.shape[0], x_samples.shape[0])
    if V.shape != expected:
        raise ValueError(f"Value grid shape mismatch: V{V.shape} vs axes {expected} (nt,ny,nx)")

    return RegularValueGrid3D(
        V=V,
        x_samples=x_samples,
        y_samples=y_samples,
        theta_samples_scaled=theta_samples_scaled,
        angle_scalor=float(angle_scalor),
    )
