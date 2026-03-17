from __future__ import annotations

from pathlib import Path

import numpy as np

from .collision import obstacle_free_mask_unicycle
from .grid import build_state_grid
from .task_io import get_range_limits, load_task
from .unicycle import Unicycle


def main() -> None:
    task_path = Path(__file__).resolve().parents[1] / "tasks" / "simple-car-parking.json"
    task = load_task(task_path)
    (x0, x1), _ = get_range_limits(task.env)
    angle_scalor = (x1 - x0) / 2.0
    robot = Unicycle(task.env, task.robots[0], angle_scalor=angle_scalor)

    states, spec = build_state_grid(task.env, angle_scalor=angle_scalor, level=0)
    mask = obstacle_free_mask_unicycle(
        env=task.env,
        body_samples_xy=robot.body.samples,
        states=states,
        angle_scalor=angle_scalor,
        chunk=10_000,
    )
    kept = int(mask.sum())
    print(f"level0 raw={spec.total}, obstacle_free={kept}")

    goal_state = robot.goal_state.astype(np.float32)
    goal_ok = obstacle_free_mask_unicycle(
        env=task.env,
        body_samples_xy=robot.body.samples,
        states=goal_state.reshape(1, 3),
        angle_scalor=angle_scalor,
    )[0]
    print(f"goal_state={goal_state.tolist()}, obstacle_free={bool(goal_ok)}")


if __name__ == "__main__":
    main()
