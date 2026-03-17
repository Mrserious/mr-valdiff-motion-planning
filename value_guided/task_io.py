from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_task(task_path: str | Path) -> dict[str, Any]:
    task_path = Path(task_path)
    with task_path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    if not isinstance(task, dict):
        raise ValueError(f"Invalid task json (not an object): {task_path}")
    return task


def get_range(task: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Returns (xmin, xmax, ymin, ymax) from task['env']['range']['limits'].
    """
    env = task.get("env", {})
    range_spec = env.get("range", {})
    if range_spec.get("shape") != "rectangle":
        raise ValueError(f"Unsupported env.range.shape: {range_spec.get('shape')}")
    limits = range_spec.get("limits", None)
    if limits is None or len(limits) != 2:
        raise ValueError(f"Invalid env.range.limits: {limits}")

    x_pair = limits[0]
    y_pair = limits[1]
    if x_pair is None or y_pair is None or len(x_pair) != 2 or len(y_pair) != 2:
        raise ValueError(f"Invalid env.range.limits: {limits}")

    xmin = float(min(x_pair[0], x_pair[1]))
    xmax = float(max(x_pair[0], x_pair[1]))
    ymin = float(min(y_pair[0], y_pair[1]))
    ymax = float(max(y_pair[0], y_pair[1]))
    return xmin, xmax, ymin, ymax


def get_obstacles(task: dict[str, Any]) -> list[dict[str, Any]]:
    env = task.get("env", {})
    obstacles = env.get("obstacles", [])
    if obstacles is None:
        return []
    if not isinstance(obstacles, list):
        raise ValueError(f"Invalid env.obstacles (not a list): {type(obstacles)}")
    return obstacles


def get_map_name(task_path: str | Path) -> str:
    task_path = Path(task_path)
    return task_path.stem


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

