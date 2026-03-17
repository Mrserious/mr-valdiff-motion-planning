from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from unicycle_value_guided.crop_coverage import crop_forward_backward_m
from unicycle_value_guided.task_io import get_range, load_task
from unicycle_value_guided.se2 import angle_scalor_from_range


def _sample_uniform_disk(rng: np.random.Generator, n: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    # Uniform in area.
    r = float(radius) * np.sqrt(rng.uniform(0.0, 1.0, size=int(n)))
    t = rng.uniform(0.0, 2.0 * math.pi, size=int(n))
    return r * np.cos(t), r * np.sin(t)


def _plot(
    *,
    out_path: Path,
    crop_size: int,
    mpp: float,
    crop_bias_forward_m: float,
    dt: float,
    rho: float,
    control_limits: np.ndarray,
    angle_scalor: float,
    car_length_m: float,
    car_width_m: float,
    samples_per_control: int,
    seed: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    fwd, back = crop_forward_backward_m(crop_size=int(crop_size), meters_per_pixel=float(mpp), crop_bias_forward_m=float(crop_bias_forward_m))
    half_side = ((float(int(crop_size)) - 1.0) / 2.0) * float(mpp)

    vmin, vmax = float(min(control_limits[0])), float(max(control_limits[0]))
    wmin_s, wmax_s = float(min(control_limits[1])), float(max(control_limits[1]))  # scaled omega (theta_scaled/sec)
    v_grid = np.linspace(vmin, vmax, 11, dtype=np.float64)
    w_grid = np.linspace(wmin_s, wmax_s, 11, dtype=np.float64)
    max_speed = float(max(abs(vmin), abs(vmax)))

    # One-step yaw change (deg) for each omega_scaled bin (graph construction semantics).
    dyaw_deg = (w_grid / float(angle_scalor) * math.pi) * float(dt) * 180.0 / math.pi
    rho_deg = float(rho) / float(angle_scalor) * 180.0

    # Conservative XY radius bound for children: dt*|v|max + rho.
    child_radius_xy = float(dt) * float(max_speed) + float(rho)

    rng = np.random.default_rng(int(seed))
    # Sample points that represent possible children in XY (robot frame, yaw=0):
    # candidate centers are at (dx=dt*v, dy=0); neighbors are within 3D ball radius rho in (x,y,theta_scaled),
    # so XY radius is <= sqrt(rho^2 - dtheta^2). We sample dtheta uniformly and draw XY within that disk.
    pts_x: list[np.ndarray] = []
    pts_y: list[np.ndarray] = []
    pts_color: list[np.ndarray] = []

    for v in v_grid:
        dx_center = float(dt) * float(v)
        # Color forward/backward (sign of v)
        if v > 1e-9:
            base_color = np.array([0.12, 0.47, 0.71, 0.18], dtype=np.float32)  # blue-ish
        elif v < -1e-9:
            base_color = np.array([0.89, 0.10, 0.11, 0.18], dtype=np.float32)  # red-ish
        else:
            base_color = np.array([0.2, 0.2, 0.2, 0.18], dtype=np.float32)

        # 11 omega bins per v (121 total). We sample the same count per control for visual density.
        for _ in w_grid:
            n = int(samples_per_control)
            dtheta = rng.uniform(-float(rho), float(rho), size=n)
            r_xy = np.sqrt(np.maximum(0.0, float(rho) ** 2 - dtheta**2))
            # Sample each point with its own radius cap.
            dx = np.empty((n,), dtype=np.float64)
            dy = np.empty((n,), dtype=np.float64)
            for i in range(n):
                sx, sy = _sample_uniform_disk(rng, 1, float(r_xy[i]))
                dx[i] = float(dx_center + sx[0])
                dy[i] = float(sy[0])
            pts_x.append(dx)
            pts_y.append(dy)
            pts_color.append(np.repeat(base_color[None, :], repeats=n, axis=0))

    X = np.concatenate(pts_x, axis=0)
    Y = np.concatenate(pts_y, axis=0)
    C = np.concatenate(pts_color, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Crop coverage rectangle (robot frame).
    x_min = -float(back)
    x_max = float(fwd)
    y_min = -float(half_side)
    y_max = float(half_side)
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            width=(x_max - x_min),
            height=(y_max - y_min),
            fill=False,
            edgecolor="tab:blue",
            linewidth=2.0,
            linestyle="-",
            label="value crop coverage",
        )
    )

    # Conservative child radius circle (XY bound).
    t = np.linspace(0.0, 2.0 * math.pi, 400, dtype=np.float64)
    ax.plot(child_radius_xy * np.cos(t), child_radius_xy * np.sin(t), color="tab:orange", linewidth=2.0, label="children XY bound (dt*|v|max+rho)")

    # Draw the car footprint at origin (yaw=0, robot frame).
    ax.add_patch(
        Rectangle(
            (-float(car_length_m) / 2.0, -float(car_width_m) / 2.0),
            width=float(car_length_m),
            height=float(car_width_m),
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            label="car footprint",
        )
    )
    ax.annotate("", xy=(float(car_length_m) / 2.0 + 0.2, 0.0), xytext=(0.0, 0.0), arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "black"})
    ax.text(float(car_length_m) / 2.0 + 0.25, 0.05, "+x forward", fontsize=10)

    # Plot sampled children points.
    ax.scatter(X, Y, s=4.0, c=C, marker=".", linewidths=0.0, label="children samples (approx)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dx (m, robot frame, +x forward)")
    ax.set_ylabel("dy (m, robot frame, +y left)")
    ax.grid(True, alpha=0.25)

    ax.set_title(
        "Fine graph one-step children (XY) vs value crop coverage (robot frame)\n"
        f"dt={float(dt):.4f}s rho={float(rho):.4f} (≈{rho_deg:.2f}°) | "
        f"Δyaw(control)≈[{float(dyaw_deg.min()):.2f},{float(dyaw_deg.max()):.2f}]° | "
        f"crop x∈[{x_min:.2f},{x_max:.2f}] y∈[{y_min:.2f},{y_max:.2f}] | "
        f"children bound≈{float(child_radius_xy):.3f}m"
    )

    # Set view limits with a bit of padding.
    pad = 0.25
    lim = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), float(child_radius_xy)) + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.legend(loc="upper right", framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plot a 2D visualization in robot frame: car footprint at origin, an approximation of fine one-step children "
            "(from dt/rho/control grid), and the local value crop coverage rectangle."
        )
    )
    p.add_argument(
        "--goal-dir",
        type=str,
        required=True,
        help="Path to .../<map>/goal_k directory that contains meta_fine.json (for dt/rho) and task_path.",
    )
    p.add_argument("--out", type=str, required=True, help="Output PNG path.")
    p.add_argument("--crop-size", type=int, default=84)
    p.add_argument("--mpp", type=float, default=0.05)
    p.add_argument("--crop-bias-forward-m", type=float, default=0.9375)
    p.add_argument("--crop-mode", type=str, default="centered", choices=["biased", "centered"])
    p.add_argument("--car-length-m", type=float, default=0.625)
    p.add_argument("--car-width-m", type=float, default=0.4375)
    p.add_argument("--samples-per-control", type=int, default=12, help="Monte Carlo points per (v,omega) control.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    goal_dir = Path(args.goal_dir)
    meta_fine_path = goal_dir / "meta_fine.json"
    if not meta_fine_path.exists():
        raise FileNotFoundError(f"Missing meta_fine.json under --goal-dir: {meta_fine_path}")
    meta_fine = json.loads(meta_fine_path.read_text(encoding="utf-8"))
    summary = meta_fine.get("summary", {}) if isinstance(meta_fine.get("summary", {}), dict) else {}

    dt = summary.get("dt", None)
    rho = summary.get("rho", None)
    if dt is None or rho is None:
        raise ValueError(f"meta_fine.json missing summary.dt/rho: {meta_fine_path}")
    dt = float(dt)
    rho = float(rho)

    crop_bias = float(args.crop_bias_forward_m)
    if str(args.crop_mode).strip().lower() == "centered":
        crop_bias = 0.0

    task_path = meta_fine.get("task_path", meta_fine.get("tmp_task_path", summary.get("task", "")))
    if not task_path:
        raise ValueError(f"meta_fine.json missing task_path/tmp_task_path/summary.task: {meta_fine_path}")
    task = load_task(str(task_path))
    if not task.get("robots"):
        raise ValueError(f"Task has no robots[] entry: {task_path}")

    control_limits = np.asarray(task["robots"][0].get("control_limits", [[-1, 1], [-1, 1]]), dtype=np.float64).reshape(2, 2)
    xmin, xmax, _, _ = get_range(task)
    angle_scalor = float(angle_scalor_from_range(float(xmin), float(xmax)))

    _plot(
        out_path=Path(args.out),
        crop_size=int(args.crop_size),
        mpp=float(args.mpp),
        crop_bias_forward_m=float(crop_bias),
        dt=float(dt),
        rho=float(rho),
        control_limits=control_limits,
        angle_scalor=float(angle_scalor),
        car_length_m=float(args.car_length_m),
        car_width_m=float(args.car_width_m),
        samples_per_control=int(args.samples_per_control),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

