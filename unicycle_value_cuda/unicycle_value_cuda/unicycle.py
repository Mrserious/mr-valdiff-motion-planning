from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import within_region
from .math_utils import rotation_matrix
from .robot_base import RobotBase


def _range_volume(env: Dict[str, Any]) -> float:
    rng = env.get("range", {})
    if rng.get("shape") != "rectangle":
        return 0.0
    limits = rng["limits"]
    x0, x1 = float(min(limits[0])), float(max(limits[0]))
    y0, y1 = float(min(limits[1])), float(max(limits[1]))
    return abs((x1 - x0) * (y1 - y0))


class Unicycle(RobotBase):
    def __init__(self, env: Dict[str, Any], robot_meta: Dict[str, Any], *, angle_scalor: float, robot_id: int = 0) -> None:
        self.angle_scalor = float(angle_scalor)
        super().__init__(env, robot_meta, robot_id=robot_id)

        self.goal_state = np.asarray(robot_meta["goal_state"], dtype=np.float32).copy()
        # goal_state angle is in degree in task json; convert to scaled angle
        self.goal_state[-1] = self.goal_state[-1] / 180.0 * self.angle_scalor

        # Keep optional meta for within_goal.
        if "goal_region_norm_weight" in robot_meta:
            self.goal_region_norm_weight = robot_meta["goal_region_norm_weight"]

        # max velocity norm (linear/angular) for Lipschitz bound
        lims = np.asarray(robot_meta["control_limits"], dtype=np.float32)
        max_vel = np.max([np.max(lims, axis=1), -np.min(lims, axis=1)], axis=0)
        self.M = float(np.linalg.norm(max_vel))
        self.Lipschitz = 1.0 * self.M
        self.dim = 3

        # free-space volume in (x,y,theta_scaled)
        vol_unit_ball = float(np.pi * 4.0 / 3.0)
        vol_total = _range_volume(env) * self.angle_scalor * 2.0
        obs_vol = 0.0
        for obs in env.get("obstacles", []):
            if obs["shape"] == "circle":
                obs_vol += float(obs["radius"]) ** 2 * float(np.pi) * self.angle_scalor * 2.0
            elif obs["shape"] == "rectangle":
                limits = obs["limits"]
                obs_vol += abs(
                    (float(limits[0][1]) - float(limits[0][0]))
                    * (float(limits[1][1]) - float(limits[1][0]))
                ) * self.angle_scalor * 2.0
        vol_free = max(vol_total - obs_vol, 1e-6)
        self.spatial_res_scalor = float((vol_free / vol_unit_ball) ** (1.0 / self.dim))

        self._controls = self.discretize_control_space(robot_meta["control_limits"], num=(11, 11))
        self.is_dubin = bool(robot_meta["control_limits"][0][0] + robot_meta["control_limits"][0][1] == 0)

        self.preset_spatial_res: Optional[float] = None
        self.preset_temporal_res: Optional[float] = None

    @property
    def controls(self) -> Sequence[np.ndarray]:
        return self._controls

    def discretize_control_space(
        self,
        limits: List[List[float]],
        *,
        num: Tuple[int, int],
    ) -> List[np.ndarray]:
        lims = np.asarray(limits, dtype=np.float32)
        u1 = np.linspace(float(lims[0, 0]), float(lims[0, 1]), int(num[0]), dtype=np.float32)
        u2 = np.linspace(float(lims[1, 0]), float(lims[1, 1]), int(num[1]), dtype=np.float32)
        controls: List[np.ndarray] = []
        for a in u1:
            for b in u2:
                controls.append(np.array([a, b], dtype=np.float32))
        return controls

    def proj_zero_intval(self, theta_scaled: float) -> float:
        # Map to [-angle_scalor, angle_scalor)
        a = float(self.angle_scalor)
        p = 2.0 * a
        return float(theta_scaled - np.floor((theta_scaled + a) / p) * p)

    def add_phase(self, state: np.ndarray) -> np.ndarray:
        new_state = np.asarray(state, dtype=np.float32).copy()
        if new_state[-1] > 0:
            new_state[-1] -= 2.0 * self.angle_scalor
        else:
            new_state[-1] += 2.0 * self.angle_scalor
        return new_state

    def flip_angle(self, state: np.ndarray) -> np.ndarray:
        flipped = np.asarray(state, dtype=np.float32).copy()
        if flipped[-1] > 0:
            flipped[-1] -= self.angle_scalor
        else:
            flipped[-1] += self.angle_scalor
        return flipped

    def get_real_angle(self, theta_scaled: float, *, unit: str = "radian") -> float:
        theta_scaled = self.proj_zero_intval(theta_scaled)
        if unit == "radian":
            return float(theta_scaled / self.angle_scalor * np.pi)
        if unit == "degree":
            return float(theta_scaled / self.angle_scalor * 180.0)
        raise ValueError(f"Unsupported unit: {unit!r}")

    def within_goal(self, state: np.ndarray) -> bool:
        def user_defined_norm(vec: np.ndarray) -> float:
            if hasattr(self, "goal_region_norm_weight"):
                prod = np.matmul(np.asarray(self.goal_region_norm_weight), vec)
                return float(np.matmul(vec, prod))
            return float(np.linalg.norm(vec))

        s = np.asarray(state, dtype=np.float32)
        phased = self.add_phase(s)
        dist = min(float(np.linalg.norm(s - self.goal_state)), float(np.linalg.norm(phased - self.goal_state)))
        if self.is_dubin:
            flipped = self.flip_angle(s)
            flipped_phased = self.add_phase(phased)
            dist = min(
                dist,
                user_defined_norm(flipped - self.goal_state),
                user_defined_norm(flipped_phased - self.goal_state),
            )
        return dist <= float(self.goal_region_threshold)

    def get_spatial_res(self) -> float:
        if self.preset_spatial_res is not None:
            return float(self.preset_spatial_res)
        total_nodes = max(len(self.nodes), 1)
        return float(self.spatial_res_scalor * (np.log(total_nodes) / total_nodes) ** (1.0 / self.dim) / 2.0)

    def get_temporal_res(self) -> float:
        if self.preset_temporal_res is not None:
            return float(self.preset_temporal_res)
        return float((self.get_spatial_res() * 5.0) ** (2.0 / 3.0))

    def get_perturbation_radius(self) -> float:
        d = self.get_spatial_res()
        return float(2.0 * d)

    def obstacle_free(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float32)
        pos = state[:2]
        angle = self.get_real_angle(float(state[-1]), unit="radian")
        obstacles = self.env.get("obstacles", [])

        if self.body.shape == "point":
            for obs in obstacles:
                if within_region(obs, pos):
                    return False
            return True

        if self.body.shape == "circle":
            # sample points are in local coordinates
            for obs in obstacles:
                for pt in self.body.samples:
                    if within_region(obs, pos + pt):
                        return False
            return True

        if self.body.shape == "rectangle":
            rot = rotation_matrix(angle)
            body_samples = (rot @ self.body.samples.T).T + pos[None, :]
            for obs in obstacles:
                for pt in body_samples:
                    if within_region(obs, pt):
                        return False
            return True

        raise ValueError(f"Unsupported body shape: {self.body.shape!r}")

    def query_kdtree(self, state: np.ndarray, *, radius: Optional[float] = None) -> List[int]:
        if self.kdtree is None:
            raise RuntimeError("KDTree is not built. Call update_kdtree() first.")
        state = np.asarray(state, dtype=np.float32)
        n_state = state.copy()
        if n_state[-1] > 0:
            n_state[-1] -= 2.0 * self.angle_scalor
        else:
            n_state[-1] += 2.0 * self.angle_scalor
        if radius is not None:
            idx, _ = self.kdtree.query_radius([state, n_state], float(radius), return_distance=True)
            indices = list(idx[0]) + list(idx[1])
            return sorted(set(int(i) for i in indices))
        dist, idx = self.kdtree.query([state, n_state], return_distance=True)
        pairs = list(zip(list(dist[0]) + list(dist[1]), list(idx[0]) + list(idx[1])))
        pairs.sort(key=lambda x: x[0])
        return [int(pairs[0][1])]

    def get_vel(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        theta = self.get_real_angle(float(state[-1]), unit="radian")
        u_lin = float(control[0])
        u_ang = float(control[1])
        return np.array([np.cos(theta) * u_lin, np.sin(theta) * u_lin, u_ang], dtype=np.float32)

