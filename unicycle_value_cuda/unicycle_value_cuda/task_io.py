from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TaskSpec:
    env: Dict[str, Any]
    robots: list[Dict[str, Any]]


def load_task(task_path: str | Path) -> TaskSpec:
    path = Path(task_path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "env" not in payload or "robots" not in payload:
        raise ValueError(f"Invalid task json (missing env/robots): {path}")
    return TaskSpec(env=payload["env"], robots=payload["robots"])


def get_range_limits(env: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if env.get("range", {}).get("shape") != "rectangle":
        raise ValueError("Only rectangle range is supported in MVP.")
    limits = env["range"]["limits"]
    x0, x1 = float(min(limits[0])), float(max(limits[0]))
    y0, y1 = float(min(limits[1])), float(max(limits[1]))
    return (x0, x1), (y0, y1)

