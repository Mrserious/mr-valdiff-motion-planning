from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from value_guided.geometry import distance_to_circle, distance_to_rectangle
from value_guided.task_io import dump_json, get_map_name, get_obstacles, get_range, load_task


def _check_supported_obstacle(ob: dict[str, Any]) -> None:
    shape = ob.get("shape", None)
    if shape not in ("circle", "rectangle"):
        raise ValueError(f"Unsupported obstacle shape: {shape}. Obstacle: {ob}")
    if shape == "circle":
        if "center" not in ob or "radius" not in ob:
            raise ValueError(f"Invalid circle obstacle: {ob}")
    if shape == "rectangle":
        if "limits" not in ob:
            raise ValueError(f"Invalid rectangle obstacle: {ob}")


def point_clear_of_obstacles(
    x: float,
    y: float,
    obstacles: list[dict[str, Any]],
    margin: float,
) -> bool:
    """
    Returns True if point (x,y) is at least `margin` meters away from every obstacle region.
    """
    for ob in obstacles:
        shape = ob.get("shape", None)
        if shape == "circle":
            d = float(distance_to_circle(x, y, ob["center"], ob["radius"]))
        elif shape == "rectangle":
            d = float(distance_to_rectangle(x, y, ob["limits"]))
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")
        if d < margin:
            return False
    return True


def batch_clear_of_obstacles(
    xy: np.ndarray,
    obstacles: list[dict[str, Any]],
    margin: float,
) -> np.ndarray:
    """
    Vectorized obstacle clearance check.
    Args:
        xy: (N,2)
    Returns:
        mask: (N,) bool
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N,2), got shape {xy.shape}")
    x = xy[:, 0]
    y = xy[:, 1]
    ok = np.ones(x.shape[0], dtype=bool)
    for ob in obstacles:
        shape = ob.get("shape", None)
        if shape == "circle":
            d = distance_to_circle(x, y, ob["center"], ob["radius"])
        elif shape == "rectangle":
            d = distance_to_rectangle(x, y, ob["limits"])
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")
        ok &= d >= margin
        if not ok.any():
            return ok
    return ok


def sample_goals(
    task: dict[str, Any],
    n_goals: int = 100,
    seed: int = 42,
    margin: float = 0.2,
    max_attempts: int | None = None,
) -> np.ndarray:
    """
    Rejection sampling of goals in free space with boundary/obstacle clearance.
    Returns:
        goals: (n_goals,2) float64 array in world meters.
    """
    if n_goals <= 0:
        raise ValueError(f"n_goals must be positive, got {n_goals}")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    xmin, xmax, ymin, ymax = get_range(task)
    x_low = xmin + margin
    x_high = xmax - margin
    y_low = ymin + margin
    y_high = ymax - margin
    if x_low >= x_high or y_low >= y_high:
        raise ValueError(
            "No feasible area after applying boundary margin. "
            f"range=({xmin},{xmax},{ymin},{ymax}), margin={margin}"
        )

    obstacles = get_obstacles(task)
    for ob in obstacles:
        _check_supported_obstacle(ob)

    rng = np.random.default_rng(seed)
    goals: list[np.ndarray] = []

    # Count how many random candidate points have been tested.
    attempts = 0
    if max_attempts is None:
        # High default to avoid false failures on cluttered maps.
        max_attempts = max(1_000_000, n_goals * 100_000)

    while sum(g.shape[0] for g in goals) < n_goals:
        remaining = n_goals - sum(g.shape[0] for g in goals)
        batch = int(min(max(remaining * 50, 1024), 200_000))

        xs = rng.uniform(x_low, x_high, size=batch)
        ys = rng.uniform(y_low, y_high, size=batch)
        cand = np.stack([xs, ys], axis=-1)

        ok = batch_clear_of_obstacles(cand, obstacles, margin=margin)
        accepted = cand[ok]
        if accepted.shape[0] > 0:
            goals.append(accepted[:remaining])

        attempts += batch
        if attempts >= max_attempts:
            collected = sum(g.shape[0] for g in goals)
            raise RuntimeError(
                "Failed to sample enough goals within max_attempts. "
                f"collected={collected}/{n_goals}, attempts={attempts}, margin={margin}. "
                "Try reducing margin or increasing max_attempts."
            )

    return np.concatenate(goals, axis=0)[:n_goals]


def validate_goals(
    task: dict[str, Any],
    goals_xy: np.ndarray,
    margin: float,
    atol: float = 1e-9,
) -> None:
    """
    Raises if any goal violates boundary/obstacle margin constraints.
    """
    if goals_xy.ndim != 2 or goals_xy.shape[1] != 2:
        raise ValueError(f"goals_xy must be (N,2), got {goals_xy.shape}")

    xmin, xmax, ymin, ymax = get_range(task)
    x = goals_xy[:, 0]
    y = goals_xy[:, 1]
    if np.any(x < xmin + margin - atol) or np.any(x > xmax - margin + atol):
        raise AssertionError("Some goals violate boundary margin in x.")
    if np.any(y < ymin + margin - atol) or np.any(y > ymax - margin + atol):
        raise AssertionError("Some goals violate boundary margin in y.")

    obstacles = get_obstacles(task)
    ok = batch_clear_of_obstacles(goals_xy, obstacles, margin=margin - atol)
    if not bool(np.all(ok)):
        bad = np.where(~ok)[0][:10]
        raise AssertionError(f"Some goals are too close to obstacles. bad_indices={bad.tolist()}")


def default_goals_out_path(task_path: str | Path, out_root: str | Path = "data/value_grids") -> Path:
    map_name = get_map_name(task_path)
    return Path(out_root) / map_name / "goals.json"


def save_goals_json(
    out_path: str | Path,
    map_name: str,
    task_path: str,
    seed: int,
    margin: float,
    goals_xy: np.ndarray,
) -> None:
    obj = {
        "map_name": map_name,
        "task_path": task_path,
        "seed": int(seed),
        "margin": float(margin),
        "goals": goals_xy.astype(float).tolist(),
    }
    dump_json(obj, out_path)


def _plot_goals(task: dict[str, Any], goals_xy: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for --plot, but it is not available."
        ) from e

    xmin, xmax, ymin, ymax = get_range(task)
    obstacles = get_obstacles(task)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # Draw obstacles
    for ob in obstacles:
        shape = ob.get("shape", None)
        if shape == "circle":
            cx, cy = ob["center"]
            r = ob["radius"]
            patch = Circle((cx, cy), r, edgecolor="k", facecolor="none", linewidth=1.5)
            ax.add_patch(patch)
        elif shape == "rectangle":
            (x1, x2), (y1, y2) = ob["limits"]
            xmin_r = min(x1, x2)
            xmax_r = max(x1, x2)
            ymin_r = min(y1, y2)
            ymax_r = max(y1, y2)
            patch = Rectangle(
                (xmin_r, ymin_r),
                width=(xmax_r - xmin_r),
                height=(ymax_r - ymin_r),
                edgecolor="k",
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_patch(patch)
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")

    ax.scatter(goals_xy[:, 0], goals_xy[:, 1], s=10, c="tab:blue", alpha=0.9)
    ax.set_title("Goal sampling (free space)")
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample 2D goals on a continuous map task JSON.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=str, help="Path to a single task json.")
    g.add_argument(
        "--tasks-dir",
        type=str,
        help="Directory containing multiple task json files (will output one goals.json per map).",
    )
    p.add_argument("--n", type=int, default=100, help="Number of goals to sample per map.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    p.add_argument("--margin", type=float, default=0.2, help="Clearance to obstacles/boundary (meters).")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output goals.json path (only valid with --task). Default uses data/value_grids/{map_name}/goals.json",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="data/value_grids",
        help="Root directory for fixed output structure when using --tasks-dir (or as default for --task).",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max number of candidate points to try (per map).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a PNG visualization next to goals.json (or under out_root/map_name/).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate constraints after sampling (raises on violation).",
    )
    return p


def _run_single(task_path: str, out: str | None, out_root: str, n: int, seed: int, margin: float, max_attempts: int | None, plot: bool, validate: bool) -> Path:
    task = load_task(task_path)
    map_name = get_map_name(task_path)
    goals = sample_goals(task, n_goals=n, seed=seed, margin=margin, max_attempts=max_attempts)
    if validate:
        validate_goals(task, goals, margin=margin)

    out_path = Path(out) if out is not None else default_goals_out_path(task_path, out_root=out_root)
    save_goals_json(out_path, map_name=map_name, task_path=task_path, seed=seed, margin=margin, goals_xy=goals)
    if plot:
        _plot_goals(task, goals, out_path.with_suffix(".png"))
    return out_path


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    if args.task is not None:
        out_path = _run_single(
            task_path=args.task,
            out=args.out,
            out_root=args.out_root,
            n=args.n,
            seed=args.seed,
            margin=args.margin,
            max_attempts=args.max_attempts,
            plot=args.plot,
            validate=args.validate,
        )
        print(f"[goal_sampling] wrote: {out_path}")
        return

    tasks_dir = Path(args.tasks_dir)
    tasks = sorted(tasks_dir.glob("*.json"))
    if not tasks:
        raise FileNotFoundError(f"No *.json tasks found under: {tasks_dir}")

    for task_path in tasks:
        out_path = _run_single(
            task_path=str(task_path),
            out=None,
            out_root=args.out_root,
            n=args.n,
            seed=args.seed,
            margin=args.margin,
            max_attempts=args.max_attempts,
            plot=args.plot,
            validate=args.validate,
        )
        print(f"[goal_sampling] wrote: {out_path}")


if __name__ == "__main__":
    main()

