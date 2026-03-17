from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from value_guided.geometry import distance_to_circle, distance_to_rectangle


@dataclass(frozen=True)
class Standard24Config:
    # env range (meters)
    xmin: float = 0.0
    xmax: float = 10.0
    ymin: float = 0.0
    ymax: float = 10.0

    # obstacles
    n_interior_obstacles: int = 24
    n_circles: int = 12
    n_rectangles: int = 12
    circle_radii: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
    rect_wh_set: tuple[tuple[float, float], ...] = (
        (0.3, 0.3),
        (0.5, 0.3),
        (0.5, 0.7),
        (0.5, 0.5),
        (0.7, 0.5),
    )

    # placement range for centers (meters) to "leave boundary walls"
    center_min: float = 0.5
    center_max: float = 9.5

    # minimum clearances (meters)
    min_wall_dist: float = 0.2
    min_obstacle_gap: float = 0.2

    # start/goal feasibility check (for downstream sampling)
    robot_radius: float = 0.17
    start_goal_min_dist: float = 5.0
    feasibility_points: int = 20_000
    feasibility_pair_trials: int = 2_000

    # generation attempts
    max_place_attempts_per_obstacle: int = 5000
    max_maps_attempts: int = 5000

    # output formatting
    float_ndigits: int = 3


def _round_floats(x: Any, *, ndigits: int) -> Any:
    if isinstance(x, float):
        return float(round(x, int(ndigits)))
    if isinstance(x, list):
        return [_round_floats(v, ndigits=ndigits) for v in x]
    if isinstance(x, dict):
        return {k: _round_floats(v, ndigits=ndigits) for k, v in x.items()}
    return x


def _rect_limits_from_center(*, cx: float, cy: float, w: float, h: float) -> list[list[float]]:
    hw = float(w) / 2.0
    hh = float(h) / 2.0
    return [[float(cx - hw), float(cx + hw)], [float(cy - hh), float(cy + hh)]]


def _rect_rect_distance(a_limits: Sequence[Sequence[float]], b_limits: Sequence[Sequence[float]]) -> float:
    ax0, ax1 = float(min(a_limits[0])), float(max(a_limits[0]))
    ay0, ay1 = float(min(a_limits[1])), float(max(a_limits[1]))
    bx0, bx1 = float(min(b_limits[0])), float(max(b_limits[0]))
    by0, by1 = float(min(b_limits[1])), float(max(b_limits[1]))
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(math.hypot(dx, dy))


def _obstacle_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    sa = str(a.get("shape"))
    sb = str(b.get("shape"))
    if sa == "circle" and sb == "circle":
        ca = a["center"]
        cb = b["center"]
        ra = float(a["radius"])
        rb = float(b["radius"])
        d = float(math.hypot(float(ca[0]) - float(cb[0]), float(ca[1]) - float(cb[1]))) - (ra + rb)
        return float(max(d, 0.0))
    if sa == "rectangle" and sb == "rectangle":
        return _rect_rect_distance(a["limits"], b["limits"])
    if sa == "circle" and sb == "rectangle":
        ca = a["center"]
        ra = float(a["radius"])
        d = float(distance_to_rectangle(float(ca[0]), float(ca[1]), b["limits"]))
        return float(max(d - ra, 0.0))
    if sa == "rectangle" and sb == "circle":
        return _obstacle_distance(b, a)
    raise ValueError(f"Unsupported shapes: {sa!r}, {sb!r}")


def _fits_wall_clearance(*, ob: dict[str, Any], cfg: Standard24Config) -> bool:
    xmin, xmax, ymin, ymax = float(cfg.xmin), float(cfg.xmax), float(cfg.ymin), float(cfg.ymax)
    d_wall = float(cfg.min_wall_dist)
    shape = str(ob.get("shape"))
    if shape == "circle":
        cx, cy = float(ob["center"][0]), float(ob["center"][1])
        r = float(ob["radius"])
        return (
            (cx - xmin) - r >= d_wall
            and (xmax - cx) - r >= d_wall
            and (cy - ymin) - r >= d_wall
            and (ymax - cy) - r >= d_wall
        )
    if shape == "rectangle":
        (x0, x1), (y0, y1) = ob["limits"]
        rx0, rx1 = float(min(x0, x1)), float(max(x0, x1))
        ry0, ry1 = float(min(y0, y1)), float(max(y0, y1))
        return (rx0 - xmin) >= d_wall and (xmax - rx1) >= d_wall and (ry0 - ymin) >= d_wall and (ymax - ry1) >= d_wall
    raise ValueError(f"Unsupported obstacle shape: {shape!r}")


def _fits_obstacle_gap(*, ob: dict[str, Any], obstacles: list[dict[str, Any]], cfg: Standard24Config) -> bool:
    gap = float(cfg.min_obstacle_gap)
    for other in obstacles:
        if _obstacle_distance(ob, other) < gap:
            return False
    return True


def _point_clear_of_obstacles(x: float, y: float, obstacles: list[dict[str, Any]], margin: float) -> bool:
    for ob in obstacles:
        shape = str(ob.get("shape"))
        if shape == "circle":
            d = float(distance_to_circle(x, y, ob["center"], float(ob["radius"])))
        elif shape == "rectangle":
            d = float(distance_to_rectangle(x, y, ob["limits"]))
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape!r}")
        if d < float(margin):
            return False
    return True


def _feasible_for_sampling(obstacles: list[dict[str, Any]], cfg: Standard24Config, rng: np.random.Generator) -> bool:
    """
    Quick feasibility check for downstream start/goal sampling:
      - sample many free points with obstacle clearance >= robot_radius
      - ensure there exists at least one pair with distance >= start_goal_min_dist
    """
    r = float(cfg.robot_radius)
    xmin, xmax, ymin, ymax = float(cfg.xmin), float(cfg.xmax), float(cfg.ymin), float(cfg.ymax)
    x_low = xmin + r
    x_high = xmax - r
    y_low = ymin + r
    y_high = ymax - r
    if x_low >= x_high or y_low >= y_high:
        return False

    n = int(cfg.feasibility_points)
    xs = rng.uniform(x_low, x_high, size=n).astype(np.float64)
    ys = rng.uniform(y_low, y_high, size=n).astype(np.float64)
    ok = np.ones((n,), dtype=bool)
    for ob in obstacles:
        shape = str(ob.get("shape"))
        if shape == "circle":
            d = distance_to_circle(xs, ys, ob["center"], float(ob["radius"]))
        elif shape == "rectangle":
            d = distance_to_rectangle(xs, ys, ob["limits"])
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape!r}")
        ok &= d >= r
        if not bool(np.any(ok)):
            return False
    pts = np.stack([xs[ok], ys[ok]], axis=1)
    if pts.shape[0] < 500:
        return False

    # Try random pairs; existence of a far-enough pair is enough.
    dmin = float(cfg.start_goal_min_dist)
    trials = int(cfg.feasibility_pair_trials)
    idx0 = rng.integers(0, pts.shape[0], size=trials)
    idx1 = rng.integers(0, pts.shape[0], size=trials)
    dx = pts[idx0, 0] - pts[idx1, 0]
    dy = pts[idx0, 1] - pts[idx1, 1]
    far = np.sqrt(dx * dx + dy * dy) >= dmin
    return bool(np.any(far))


def _make_task(
    *,
    obstacles: list[dict[str, Any]],
    cfg: Standard24Config,
    robot_size: tuple[float, float],
    control_limits: tuple[tuple[float, float], tuple[float, float]],
    goal_region_threshold: float,
) -> dict[str, Any]:
    env = {
        "obstacles": obstacles,
        "MAX_VAL": 100000000,
        "INFINI": 0.000001,
        "range": {
            "shape": "rectangle",
            "limits": [[float(cfg.xmin), float(cfg.xmax)], [float(cfg.ymin), float(cfg.ymax)]],
        },
    }
    robots = [
        {
            "dyn_type": "Unicycle",
            "configuration": {"shape": "rectangle", "size": [float(robot_size[0]), float(robot_size[1])]},
            "goal_pos": [5.0, 5.0],
            "goal_state": [5.0, 5.0, 0.0],
            "goal_region_threshold": float(goal_region_threshold),
            "init_pos": [1.0, 1.0],
            "control_limits": [[float(control_limits[0][0]), float(control_limits[0][1])], [float(control_limits[1][0]), float(control_limits[1][1])]],
        }
    ]
    return {"env": env, "robots": robots}


def _generate_one(cfg: Standard24Config, rng: np.random.Generator) -> list[dict[str, Any]]:
    if cfg.n_circles + cfg.n_rectangles != cfg.n_interior_obstacles:
        raise ValueError("n_circles + n_rectangles must equal n_interior_obstacles.")
    if cfg.n_interior_obstacles <= 0:
        raise ValueError("n_interior_obstacles must be positive.")

    obstacles: list[dict[str, Any]] = []
    cmin = float(cfg.center_min)
    cmax = float(cfg.center_max)

    # circles
    for _ in range(int(cfg.n_circles)):
        for _attempt in range(int(cfg.max_place_attempts_per_obstacle)):
            r = float(rng.choice(np.asarray(cfg.circle_radii, dtype=np.float64)))
            cx = float(rng.uniform(cmin, cmax))
            cy = float(rng.uniform(cmin, cmax))
            ob = {"shape": "circle", "center": [cx, cy], "radius": r}
            if not _fits_wall_clearance(ob=ob, cfg=cfg):
                continue
            if not _fits_obstacle_gap(ob=ob, obstacles=obstacles, cfg=cfg):
                continue
            obstacles.append(ob)
            break
        else:
            raise RuntimeError("Failed to place a circle obstacle (increase attempts or relax gaps).")

    # rectangles
    for _ in range(int(cfg.n_rectangles)):
        for _attempt in range(int(cfg.max_place_attempts_per_obstacle)):
            w, h = rng.choice(np.asarray(cfg.rect_wh_set, dtype=np.float64))
            cx = float(rng.uniform(cmin, cmax))
            cy = float(rng.uniform(cmin, cmax))
            limits = _rect_limits_from_center(cx=cx, cy=cy, w=float(w), h=float(h))
            ob = {"shape": "rectangle", "limits": limits}
            if not _fits_wall_clearance(ob=ob, cfg=cfg):
                continue
            if not _fits_obstacle_gap(ob=ob, obstacles=obstacles, cfg=cfg):
                continue
            obstacles.append(ob)
            break
        else:
            raise RuntimeError("Failed to place a rectangle obstacle (increase attempts or relax gaps).")

    return obstacles


def _plot_task(task: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --write-preview") from e

    env = task.get("env", {})
    rng = env.get("range", {})
    limits = rng.get("limits", [[0, 1], [0, 1]])
    xmin, xmax = float(min(limits[0])), float(max(limits[0]))
    ymin, ymax = float(min(limits[1])), float(max(limits[1]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    for ob in env.get("obstacles", []) or []:
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

    ax.set_title(out_path.stem)
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prompt07 (unicycle): generate standardized 10x10 maps with exactly 24 interior obstacles "
            "(12 circles + 12 rectangles) and fixed size sets, plus clearance constraints."
        )
    )
    p.add_argument("--out-dir", type=str, required=True, help="Output dir for generated task JSONs.")
    p.add_argument("--variants", type=int, default=100, help="How many maps to generate.")
    p.add_argument("--name-prefix", type=str, default="standard10x10", help="Filename prefix.")
    p.add_argument("--start-index", type=int, default=0, help="Start index for prefix_index naming.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing task files if present.")
    p.add_argument("--write-preview", action="store_true", help="Also save <name>.preview.png per map.")
    p.add_argument("--print-metrics", action="store_true", help="Print basic per-map stats.")

    # Expose a few key knobs (defaults match the user's standardized spec)
    p.add_argument("--min-wall-dist", type=float, default=0.2, help="Min distance to boundary walls (meters).")
    p.add_argument("--min-obstacle-gap", type=float, default=0.2, help="Min edge gap between obstacles (meters).")
    p.add_argument("--robot-radius", type=float, default=0.17, help="Robot radius used for feasibility check (meters).")
    p.add_argument("--start-goal-min-dist", type=float, default=5.0, help="Min start-goal distance used for feasibility check (meters).")

    # Robot params in the generated task (unicycle_value_cuda compatible)
    p.add_argument("--robot-length", type=float, default=0.625, help="Unicycle rectangle length (meters).")
    p.add_argument("--robot-width", type=float, default=0.4375, help="Unicycle rectangle width (meters).")
    p.add_argument("--vmin", type=float, default=-1.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--omin", type=float, default=-1.0)
    p.add_argument("--omax", type=float, default=1.0)
    p.add_argument("--goal-region-threshold", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Standard24Config(
        min_wall_dist=float(args.min_wall_dist),
        min_obstacle_gap=float(args.min_obstacle_gap),
        robot_radius=float(args.robot_radius),
        start_goal_min_dist=float(args.start_goal_min_dist),
    )
    robot_size = (float(args.robot_length), float(args.robot_width))
    control_limits = ((float(args.vmin), float(args.vmax)), (float(args.omin), float(args.omax)))

    rng = np.random.default_rng(int(args.seed))
    written = 0
    for i in range(int(args.variants)):
        idx = int(args.start_index) + i
        name = f"{str(args.name_prefix)}_{idx:04d}"
        task_path = out_dir / f"{name}.json"
        if task_path.exists() and not bool(args.overwrite):
            continue

        # per-map retry loop
        map_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        last_err: Exception | None = None
        for _attempt in range(int(cfg.max_maps_attempts)):
            try:
                obstacles = _generate_one(cfg, map_rng)
                if not _feasible_for_sampling(obstacles, cfg, map_rng):
                    continue
                task = _make_task(
                    obstacles=obstacles,
                    cfg=cfg,
                    robot_size=robot_size,
                    control_limits=control_limits,
                    goal_region_threshold=float(args.goal_region_threshold),
                )
                task = _round_floats(task, ndigits=int(cfg.float_ndigits))
                tmp_path = task_path.with_name(f".{task_path.name}.tmp.{os.getpid()}")
                try:
                    tmp_path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    os.replace(tmp_path, task_path)
                finally:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass

                if bool(args.write_preview):
                    _plot_task(task, out_dir / f"{name}.preview.png")
                if bool(args.print_metrics):
                    print(f"[map_gen_standard24] wrote {name}: n_obstacles={len(obstacles)}", flush=True)
                written += 1
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise RuntimeError(f"Failed to generate a valid map after {cfg.max_maps_attempts} attempts. Last error: {last_err}")

    print(f"[map_gen_standard24] done: wrote {written} maps under {out_dir}", flush=True)


if __name__ == "__main__":
    main()

