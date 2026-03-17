from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import KDTree

from .math_utils import rotation_matrix
from .structures import Node


@dataclass(frozen=True)
class RobotBody:
    shape: str
    samples: np.ndarray
    size: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "RobotBody":
        shape = config["shape"]
        if shape == "point":
            samples = np.array([[0.0, 0.0]], dtype=np.float32)
            return RobotBody(shape=shape, samples=samples)
        if shape == "rectangle":
            size = (float(config["size"][0]), float(config["size"][1]))
            x = np.linspace(-size[0] / 2.0, size[0] / 2.0, 11, dtype=np.float32)
            y = np.linspace(-size[1] / 2.0, size[1] / 2.0, 11, dtype=np.float32)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            samples = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            return RobotBody(shape=shape, samples=samples, size=size)
        if shape == "circle":
            radius = float(config["radius"])
            pts: List[Tuple[float, float]] = [(0.0, 0.0)]
            for r in np.linspace(0.0, radius, 6, dtype=np.float32)[1:]:
                for t in np.linspace(0.0, np.pi * 2.0, 21, dtype=np.float32)[:-1]:
                    pts.append((float(np.cos(t) * r), float(np.sin(t) * r)))
            samples = np.array(pts, dtype=np.float32)
            return RobotBody(shape=shape, samples=samples, radius=radius)
        raise ValueError(f"Unsupported body shape: {shape!r}")


class RobotBase(ABC):
    def __init__(self, env: Dict[str, Any], robot_meta: Dict[str, Any], *, robot_id: int = 0) -> None:
        self.env = env
        self.meta = robot_meta
        self.ID = robot_id

        self.dyn_type = robot_meta["dyn_type"]
        self.init_pos = robot_meta["init_pos"]
        self.goal_pos = robot_meta["goal_pos"]
        self.goal_region_threshold = float(robot_meta["goal_region_threshold"])
        self.max_val = float(env["MAX_VAL"])

        self.control_limits = robot_meta["control_limits"]
        self.nodes: List[Node] = []
        self.states: List[np.ndarray] = []
        self.kdtree: Optional[KDTree] = None

        self.staleness_thre = 50
        self.body = RobotBody.from_config(robot_meta["configuration"])

    def update_kdtree(self) -> None:
        if not self.states:
            self.kdtree = None
            return
        self.kdtree = KDTree(np.asarray(self.states, dtype=np.float32), metric="l2")

    def init_nodes(self, node_indices: Optional[Iterable[int]] = None, init_val: float = 1.0) -> None:
        if node_indices is None:
            node_indices = range(len(self.nodes))
        for i in node_indices:
            node = self.nodes[i]
            node.staleness = int(self.max_val)
            if self.within_goal(node.state):
                node.value = 0.0
            else:
                node.value = float(init_val)

    def add_node_state(self, state: np.ndarray) -> None:
        node = Node(state=np.asarray(state, dtype=np.float32))
        if self.within_goal(node.state):
            node.value = 0.0
        self.nodes.append(node)
        self.states.append(node.state)

    @abstractmethod
    def obstacle_free(self, state: np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def within_goal(self, state: np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_spatial_res(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_temporal_res(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_perturbation_radius(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def query_kdtree(self, state: np.ndarray, *, radius: Optional[float] = None) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def get_vel(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_vels(self, state: np.ndarray) -> np.ndarray:
        return np.asarray([self.get_vel(state, u) for u in self.controls], dtype=np.float32)

    @property
    @abstractmethod
    def controls(self) -> Sequence[np.ndarray]:
        raise NotImplementedError

