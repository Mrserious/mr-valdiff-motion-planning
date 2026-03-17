from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.task_io import get_range_limits as get_range_limits_unicycle
from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from value_guided.goal_sampling import batch_clear_of_obstacles
from value_guided.task_io import dump_json, get_map_name, get_obstacles, get_range, load_task


def _parse_yaw_set(spec: str) -> list[float]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _pick_yaws_deg(rng: np.random.Generator, n: int, *, yaw_mode: str, yaw_deg: float, yaw_set_deg: list[float]) -> np.ndarray:
    yaw_mode = str(yaw_mode).strip().lower()
    if yaw_mode == "fixed":
        return np.full((n,), float(yaw_deg), dtype=np.float32)
    if yaw_mode == "uniform":
        return rng.uniform(-180.0, 180.0, size=n).astype(np.float32)
    if yaw_mode == "discrete":
        if not yaw_set_deg:
            raise ValueError("--yaw-mode=discrete requires --yaw-set-deg")
        idx = rng.integers(0, len(yaw_set_deg), size=n)
        return np.asarray([yaw_set_deg[int(i)] for i in idx], dtype=np.float32)
    raise ValueError(f"Unknown yaw_mode: {yaw_mode}. Expected fixed|uniform|discrete.")


def _stable_hash32(text: str) -> int:
    """
    Stable (cross-run) 32-bit hash for seed mixing.

    Notes:
    - Do NOT use Python's built-in hash(), it is salted per process.
    - This returns an unsigned 32-bit int.
    """
    b = hashlib.sha1(str(text).encode("utf-8")).digest()
    return int.from_bytes(b[:4], byteorder="little", signed=False)


def _mix_seed_per_map(seed_base: int, map_name: str) -> int:
    return (int(seed_base) + _stable_hash32(map_name)) & 0xFFFFFFFF


def sample_goal_poses(
    *,
    task: dict[str, Any],
    n_goals: int,
    seed: int,
    margin: float,
    yaw_mode: str,
    yaw_deg: float,
    yaw_set_deg: list[float],
    max_attempts: int | None,
) -> np.ndarray:
    """
    Sample collision-free goal poses for a unicycle task.

    Returns:
        (n_goals, 3) array: [x, y, yaw_deg].
    """
    if n_goals <= 0:
        raise ValueError(f"n_goals must be positive, got {n_goals}")
    margin = float(margin)
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
    rng = np.random.default_rng(int(seed))

    # Instantiate unicycle robot for obstacle-free check (supports rectangle footprint + yaw).
    # We rely on the task's own robot meta.
    env = task["env"]
    robots = task["robots"]
    if not robots:
        raise ValueError("Task has no robots[] entry.")
    robot_meta = robots[0]
    (x0, x1), _ = get_range_limits_unicycle(env)
    angle_scalor = (x1 - x0) / 2.0
    robot = Unicycle(env, robot_meta, angle_scalor=float(angle_scalor), robot_id=0)

    goals: list[list[float]] = []
    attempts = 0
    if max_attempts is None:
        max_attempts = max(1_000_000, int(n_goals) * 100_000)

    while len(goals) < int(n_goals):
        remaining = int(n_goals) - len(goals)
        batch = int(min(max(remaining * 50, 1024), 200_000))

        xs = rng.uniform(x_low, x_high, size=batch)
        ys = rng.uniform(y_low, y_high, size=batch)
        cand_xy = np.stack([xs, ys], axis=-1)
        ok_xy = batch_clear_of_obstacles(cand_xy, obstacles, margin=margin)
        if not bool(np.any(ok_xy)):
            attempts += batch
            if attempts >= int(max_attempts):
                break
            continue

        yaw_batch = _pick_yaws_deg(rng, batch, yaw_mode=yaw_mode, yaw_deg=float(yaw_deg), yaw_set_deg=yaw_set_deg)

        cand = np.concatenate([cand_xy, yaw_batch[:, None]], axis=1).astype(np.float32, copy=False)
        accepted: list[list[float]] = []
        for row in cand[ok_xy]:
            x, y, yaw_d = float(row[0]), float(row[1]), float(row[2])
            theta_scaled = yaw_d / 180.0 * float(angle_scalor)
            state = np.array([x, y, theta_scaled], dtype=np.float32)
            if not robot.obstacle_free(state):
                continue
            accepted.append([x, y, yaw_d])
            if len(accepted) >= remaining:
                break
        goals.extend(accepted)

        attempts += batch
        if attempts >= int(max_attempts) and len(goals) < int(n_goals):
            break

    if len(goals) < int(n_goals):
        raise RuntimeError(
            "Failed to sample enough goal poses within max_attempts. "
            f"collected={len(goals)}/{n_goals}, attempts={attempts}, margin={margin}. "
            "Try reducing margin or increasing max_attempts."
        )

    return np.asarray(goals[: int(n_goals)], dtype=np.float32)


def default_goals_out_path(task_path: str | Path, out_root: str | Path = "data/unicycle_value_grids") -> Path:
    map_name = get_map_name(task_path)
    return Path(out_root) / map_name / "goals.json"


def save_goals_json(
    out_path: str | Path,
    *,
    map_name: str,
    task_path: str,
    seed: int,
    seed_base: int | None = None,
    seed_mode: str | None = None,
    margin: float,
    yaw_mode: str,
    yaw_deg: float,
    yaw_set_deg: list[float],
    goals_xyz: np.ndarray,
) -> None:
    obj = {
        "map_name": map_name,
        "task_path": task_path,
        "seed": int(seed),
        "seed_base": int(seed_base) if seed_base is not None else None,
        "seed_mode": str(seed_mode) if seed_mode is not None else None,
        "margin": float(margin),
        "yaw_mode": str(yaw_mode),
        "yaw_deg": float(yaw_deg),
        "yaw_set_deg": [float(v) for v in yaw_set_deg],
        "goals": goals_xyz.astype(float).tolist(),
    }
    if obj["seed_base"] is None:
        obj.pop("seed_base", None)
    if obj["seed_mode"] is None:
        obj.pop("seed_mode", None)
    dump_json(obj, out_path)


def _plot_goals(task: dict[str, Any], goals_xyz: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot") from e

    xmin, xmax, ymin, ymax = get_range(task)
    obstacles = get_obstacles(task)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    for ob in obstacles:
        shape = ob.get("shape", None)
        if shape == "circle":
            cx, cy = ob["center"]
            r = ob["radius"]
            ax.add_patch(Circle((cx, cy), r, edgecolor="k", facecolor="none", linewidth=1.5))
        elif shape == "rectangle":
            (x1, x2), (y1, y2) = ob["limits"]
            rx0 = min(x1, x2)
            ry0 = min(y1, y2)
            rw = max(x1, x2) - rx0
            rh = max(y1, y2) - ry0
            ax.add_patch(Rectangle((rx0, ry0), rw, rh, edgecolor="k", facecolor="none", linewidth=1.5))
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")

    goals_xy = goals_xyz[:, :2]
    ax.scatter(goals_xy[:, 0], goals_xy[:, 1], s=12, c="tab:blue", alpha=0.9)
    ax.set_title("Unicycle goal pose sampling (x,y; yaw in goals.json)")
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _iter_task_paths(task: str | None, tasks_dir: str | None) -> Iterable[Path]:
    if task:
        yield Path(task)
        return
    if not tasks_dir:
        raise ValueError("Either --task or --tasks-dir is required.")
    tasks_dir_p = Path(tasks_dir)
    for p in sorted(tasks_dir_p.glob("*.json")):
        yield p


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt01: sample unicycle goal poses (x,y,yaw_deg) on a task json.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=str, help="Path to a single task json.")
    g.add_argument("--tasks-dir", type=str, help="Directory containing multiple task json files.")
    p.add_argument("--n", type=int, default=50, help="Number of goals per map.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    p.add_argument(
        "--seed-mode",
        type=str,
        default="per_map",
        choices=["global", "per_map"],
        help=(
            "How to apply --seed across multiple maps. "
            "global: same seed for every map (can cause similar goals across maps); "
            "per_map: mix seed with map_name to decorrelate maps."
        ),
    )
    p.add_argument("--margin", type=float, default=0.2, help="Clearance to obstacles/boundary (meters).")
    p.add_argument("--yaw-mode", type=str, default="fixed", choices=["fixed", "uniform", "discrete"])
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Only for --yaw-mode=fixed.")
    p.add_argument("--yaw-set-deg", type=str, default="", help='Only for --yaw-mode=discrete, e.g. "0,90,180,-90".')
    p.add_argument("--out", type=str, default="", help="Output goals.json path (single --task only).")
    p.add_argument("--out-root", type=str, default="data/unicycle_value_grids", help="Root folder for per-map outputs.")
    p.add_argument("--plot", action="store_true", help="Also save a goals.png next to goals.json.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite goals.json if exists.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    yaw_set_deg = _parse_yaw_set(args.yaw_set_deg)

    seed_mode = str(args.seed_mode).strip().lower()
    if seed_mode not in ("global", "per_map"):
        raise ValueError(f"Invalid --seed-mode: {seed_mode}")

    for task_path in _iter_task_paths(args.task, args.tasks_dir):
        task = load_task(task_path)
        map_name = get_map_name(task_path)
        seed_eff = int(args.seed) & 0xFFFFFFFF
        if seed_mode == "per_map":
            seed_eff = _mix_seed_per_map(seed_eff, map_name)

        out_path = Path(args.out) if args.out else default_goals_out_path(task_path, out_root=args.out_root)
        if out_path.exists() and not bool(args.overwrite):
            print(f"[goal_sampling] exists, skip: {out_path}", flush=True)
            continue

        goals_xyz = sample_goal_poses(
            task=task,
            n_goals=int(args.n),
            seed=int(seed_eff),
            margin=float(args.margin),
            yaw_mode=str(args.yaw_mode),
            yaw_deg=float(args.yaw_deg),
            yaw_set_deg=yaw_set_deg,
            max_attempts=None,
        )

        save_goals_json(
            out_path,
            map_name=map_name,
            task_path=str(Path(task_path).resolve()),
            seed=int(seed_eff),
            seed_base=int(args.seed),
            seed_mode=str(seed_mode),
            margin=float(args.margin),
            yaw_mode=str(args.yaw_mode),
            yaw_deg=float(args.yaw_deg),
            yaw_set_deg=yaw_set_deg,
            goals_xyz=goals_xyz,
        )
        if bool(args.plot):
            _plot_goals(task, goals_xyz, out_path.with_suffix(".png"))
        print(f"[goal_sampling] wrote: {out_path} (n={goals_xyz.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
