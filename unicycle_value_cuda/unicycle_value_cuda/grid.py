from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .task_io import get_range_limits


@dataclass(frozen=True)
class GridSpec:
    level: int
    nx: int
    ny: int
    nt: int

    @property
    def total(self) -> int:
        return int(self.nx * self.ny * self.nt)


LEGACY_LEVEL_SPECS: Mapping[int, GridSpec] = {
    0: GridSpec(level=0, nx=19, ny=19, nt=19),
    1: GridSpec(level=1, nx=37, ny=37, nt=37),
    2: GridSpec(level=2, nx=73, ny=73, nt=73),
}

# Multigrid-friendly nested refinement (half-open interval with endpoint=False).
# Each level doubles one or more axes so that coarse grid points are a subset of fine grid points.
MULTIGRID_LEVEL_SPECS: Mapping[int, GridSpec] = {
    0: GridSpec(level=0, nx=19, ny=19, nt=19),
    1: GridSpec(level=1, nx=38, ny=38, nt=19),
    2: GridSpec(level=2, nx=76, ny=76, nt=19),
    3: GridSpec(level=3, nx=152, ny=152, nt=19),
    4: GridSpec(level=4, nx=152, ny=152, nt=38),
    5: GridSpec(level=5, nx=152, ny=152, nt=76),
    6: GridSpec(level=6, nx=152, ny=152, nt=152),
}

GRID_SCHEMES: Mapping[str, Mapping[int, GridSpec]] = {
    "legacy": LEGACY_LEVEL_SPECS,
    "multigrid": MULTIGRID_LEVEL_SPECS,
}


def available_levels(*, scheme: str) -> Tuple[int, ...]:
    if scheme not in GRID_SCHEMES:
        raise ValueError(f"Unsupported grid scheme: {scheme!r}. Available: {sorted(GRID_SCHEMES)}")
    return tuple(sorted(GRID_SCHEMES[scheme].keys()))


def build_state_grid(
    env: Dict[str, Any],
    *,
    angle_scalor: float,
    level: int,
    scheme: str = "legacy",
) -> Tuple[np.ndarray, GridSpec]:
    if scheme not in GRID_SCHEMES:
        raise ValueError(f"Unsupported grid scheme: {scheme!r}. Available: {sorted(GRID_SCHEMES)}")
    level_specs = GRID_SCHEMES[scheme]
    if level not in level_specs:
        raise ValueError(f"Unsupported level: {level}. Available: {sorted(level_specs)} (scheme={scheme!r})")
    spec = level_specs[level]
    (x0, x1), (y0, y1) = get_range_limits(env)

    # Boundary convention aligned with notebook's random sampling semantics:
    # x,y in [min,max) and theta_scaled in [-angle_scalor, angle_scalor)
    x = np.linspace(x0, x1, spec.nx, endpoint=False, dtype=np.float32)
    y = np.linspace(y0, y1, spec.ny, endpoint=False, dtype=np.float32)
    theta = np.linspace(-float(angle_scalor), float(angle_scalor), spec.nt, endpoint=False, dtype=np.float32)

    xx, yy, tt = np.meshgrid(x, y, theta, indexing="ij")
    states = np.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], axis=1).astype(np.float32)
    return states, spec
