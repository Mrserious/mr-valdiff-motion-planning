from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.grid import GRID_SCHEMES, GridSpec

from unicycle_value_guided.se2 import wrap_theta_scaled


@dataclass(frozen=True)
class ValueGrid2D:
    """
    2D value grid for observation (y-major like value_guided.RegularValueGrid):
      - V: (H, W) where H=len(y_samples), W=len(x_samples)
    """

    V: np.ndarray
    x_samples: np.ndarray
    y_samples: np.ndarray


@dataclass(frozen=True)
class ValueGrid3D:
    """
    3D value grid (axis_order=theta,y,x):
      - V: (nt, ny, nx)
    """

    V: np.ndarray
    x_samples: np.ndarray
    y_samples: np.ndarray
    theta_samples_scaled: np.ndarray


def _axes_from_env(*, env: dict[str, Any], spec: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    limits = env["range"]["limits"]
    x0, x1 = float(min(limits[0])), float(max(limits[0]))
    y0, y1 = float(min(limits[1])), float(max(limits[1]))
    x_samples = np.linspace(x0, x1, int(spec.nx), endpoint=False, dtype=np.float64)
    y_samples = np.linspace(y0, y1, int(spec.ny), endpoint=False, dtype=np.float64)
    return x_samples, y_samples


def robot_to_value_grid_2d_min_theta(
    *,
    robot: Any,
    env: dict[str, Any],
    level: int,
    scheme: str = "multigrid",
    angle_scalor: float,
    fill_value: float = 1.0,
) -> Tuple[ValueGrid2D, dict[str, Any]]:
    """
    Project a 3D unicycle VI result (x,y,theta_scaled) to a 2D grid by:
      V2D(x,y) = min_theta V3D(x,y,theta)
    Missing (x,y,theta) samples are filled with `fill_value` before min.
    """
    if scheme not in GRID_SCHEMES:
        raise ValueError(f"Unknown grid scheme: {scheme!r}. Available: {sorted(GRID_SCHEMES)}")
    level_specs = GRID_SCHEMES[str(scheme)]
    if level not in level_specs:
        raise ValueError(f"Unknown grid level: {level}. Available: {sorted(level_specs)} (scheme={scheme!r})")
    spec = level_specs[int(level)]

    x_samples, y_samples = _axes_from_env(env=env, spec=spec)
    limits = env["range"]["limits"]
    x0, x1 = float(min(limits[0])), float(max(limits[0]))
    y0, y1 = float(min(limits[1])), float(max(limits[1]))

    dx = (x1 - x0) / float(spec.nx)
    dy = (y1 - y0) / float(spec.ny)
    dth = (2.0 * float(angle_scalor)) / float(spec.nt)
    if dx <= 0 or dy <= 0 or dth <= 0:
        raise ValueError(f"Invalid grid spacing: dx={dx} dy={dy} dth={dth}")

    V3 = np.full((int(spec.nx), int(spec.ny), int(spec.nt)), float(fill_value), dtype=np.float32)

    for node in getattr(robot, "nodes", []):
        s = np.asarray(getattr(node, "state", None), dtype=np.float64).reshape(3)
        x, y, th = float(s[0]), float(s[1]), float(s[2])
        th = wrap_theta_scaled(th, float(angle_scalor))

        ix = int(np.floor((x - x0) / dx))
        iy = int(np.floor((y - y0) / dy))
        it = int(np.floor((th + float(angle_scalor)) / dth))

        ix = max(0, min(ix, int(spec.nx) - 1))
        iy = max(0, min(iy, int(spec.ny) - 1))
        it = int(it % int(spec.nt))

        v = float(getattr(node, "value", float(fill_value)))
        if not np.isfinite(v):
            continue
        if v < float(V3[ix, iy, it]):
            V3[ix, iy, it] = float(v)

    V2_xy = np.min(V3, axis=2)  # (nx,ny)
    V2 = V2_xy.T.astype(np.float32, copy=False)  # (ny,nx)

    grid = ValueGrid2D(V=V2, x_samples=x_samples, y_samples=y_samples)
    meta = {
        "grid_kind": "unicycle_min_theta",
        "grid_scheme": str(scheme),
        "level": int(level),
        "nx": int(spec.nx),
        "ny": int(spec.ny),
        "nt": int(spec.nt),
        "angle_scalor": float(angle_scalor),
        "fill_value": float(fill_value),
        "x_samples": x_samples.astype(float).tolist(),
        "y_samples": y_samples.astype(float).tolist(),
    }
    return grid, meta


def robot_to_value_grid_3d(
    *,
    robot: Any,
    env: dict[str, Any],
    level: int,
    scheme: str = "multigrid",
    angle_scalor: float,
    fill_value: float = 1.0,
) -> tuple[ValueGrid3D, dict[str, Any]]:
    """
    Re-bin a 3D unicycle VI result (x,y,theta_scaled) to a regular 3D grid:
      - axis_order = theta,y,x
      - V[it, iy, ix] = min(value over nodes mapped to that bin)

    Missing bins are filled with `fill_value`.
    """
    if scheme not in GRID_SCHEMES:
        raise ValueError(f"Unknown grid scheme: {scheme!r}. Available: {sorted(GRID_SCHEMES)}")
    level_specs = GRID_SCHEMES[str(scheme)]
    if level not in level_specs:
        raise ValueError(f"Unknown grid level: {level}. Available: {sorted(level_specs)} (scheme={scheme!r})")
    spec = level_specs[int(level)]

    x_samples, y_samples = _axes_from_env(env=env, spec=spec)
    limits = env["range"]["limits"]
    x0, x1 = float(min(limits[0])), float(max(limits[0]))
    y0, y1 = float(min(limits[1])), float(max(limits[1]))

    dx = (x1 - x0) / float(spec.nx)
    dy = (y1 - y0) / float(spec.ny)
    dth = (2.0 * float(angle_scalor)) / float(spec.nt)
    if dx <= 0 or dy <= 0 or dth <= 0:
        raise ValueError(f"Invalid grid spacing: dx={dx} dy={dy} dth={dth}")

    theta_samples_scaled = np.linspace(-float(angle_scalor), float(angle_scalor), int(spec.nt), endpoint=False, dtype=np.float64)
    V3 = np.full((int(spec.nt), int(spec.ny), int(spec.nx)), float(fill_value), dtype=np.float32)

    for node in getattr(robot, "nodes", []):
        s = np.asarray(getattr(node, "state", None), dtype=np.float64).reshape(3)
        x, y, th = float(s[0]), float(s[1]), float(s[2])
        th = wrap_theta_scaled(th, float(angle_scalor))

        ix = int(np.floor((x - x0) / dx))
        iy = int(np.floor((y - y0) / dy))
        it = int(np.floor((th + float(angle_scalor)) / dth))

        ix = max(0, min(ix, int(spec.nx) - 1))
        iy = max(0, min(iy, int(spec.ny) - 1))
        it = int(it % int(spec.nt))

        v = float(getattr(node, "value", float(fill_value)))
        if not np.isfinite(v):
            continue
        if v < float(V3[it, iy, ix]):
            V3[it, iy, ix] = float(v)

    grid = ValueGrid3D(
        V=V3,
        x_samples=x_samples,
        y_samples=y_samples,
        theta_samples_scaled=theta_samples_scaled,
    )
    meta = {
        "grid_kind": "unicycle_v3d",
        "grid_scheme": str(scheme),
        "axis_order": "theta,y,x",
        "level": int(level),
        "grid_level": int(level),
        "nx": int(spec.nx),
        "ny": int(spec.ny),
        "nt": int(spec.nt),
        "angle_scalor": float(angle_scalor),
        "fill_value": float(fill_value),
        "x_samples": x_samples.astype(float).tolist(),
        "y_samples": y_samples.astype(float).tolist(),
        "theta_samples_scaled": theta_samples_scaled.astype(float).tolist(),
    }
    return grid, meta


def save_array_and_meta(value_path: str | Path, meta_path: str | Path, arr: np.ndarray, meta: dict[str, Any]) -> None:
    value_path = Path(value_path)
    meta_path = Path(meta_path)
    value_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(value_path, np.asarray(arr, dtype=np.float32))

    tmp_path = meta_path.with_name(f".{meta_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def save_value_and_meta(value_path: str | Path, meta_path: str | Path, grid: ValueGrid2D, meta: dict[str, Any]) -> None:
    value_path = Path(value_path)
    meta_path = Path(meta_path)
    value_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    save_array_and_meta(value_path, meta_path, grid.V.astype(np.float32, copy=False), meta)
