from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ChildNodes:
    update_iteration: int = -1
    indices: List[int] = field(default_factory=list)


@dataclass
class Node:
    state: np.ndarray
    value: float = 1.0
    staleness: int = 0
    children: ChildNodes = field(default_factory=ChildNodes)
    id: Optional[int] = None

    @property
    def pos(self) -> np.ndarray:
        return self.state[:2]

