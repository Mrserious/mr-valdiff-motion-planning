from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from unicycle_value_guided.crop_coverage import crop_forward_backward_m, required_child_radius_m
from unicycle_value_guided.task_io import get_range, load_task
from unicycle_value_guided.vi_io import load_vi_robot
from unicycle_value_guided.se2 import angle_scalor_from_range


def _parse_offsets_deg(text: str) -> tuple[int, ...]:
    items = [s.strip() for s in str(text).split(",")]
    out: list[int] = []
    for s in items:
        if not s:
            continue
        d = int(s)
        # Match wrap semantics used by observe_valuewin12ch (wrap to [-pi,pi]).
        w = ((d + 180) % 360) - 180  # [-180,180)
        if w == -180 and d > 0:
            w = 180
        out.append(int(w))
    if not out:
        raise ValueError("--yaw-offsets-deg must contain at least one integer")
    if 0 not in out:
        raise ValueError("--yaw-offsets-deg must include 0")
    # Dedup + stable sort for plotting.
    out = sorted(set(out))
    return tuple(out)


def _plot(
    *,
    out_path: Path,
    yaw_offsets_deg: tuple[int, ...],
    crop_size: int,
    mpp: float,
    crop_bias_forward_m: float,
    child_radius_m: float,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    fwd, back = crop_forward_backward_m(crop_size=int(crop_size), meters_per_pixel=float(mpp), crop_bias_forward_m=float(crop_bias_forward_m))
    half_side = ((float(int(crop_size)) - 1.0) / 2.0) * float(mpp)

    x_min = -float(back)
    x_max = float(fwd)
    y_min = -float(half_side)
    y_max = float(half_side)

    z_list = [float(d) for d in yaw_offsets_deg]
    z_min = float(min(z_list))
    z_max = float(max(z_list))

    # Conservative inclusion check (XY only).
    ok = float(min(fwd, back, half_side)) + 1e-9 >= float(child_radius_m)
    status = "OK" if ok else "NOT_OK"

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw crop rectangles at each yaw slice.
    for z in z_list:
        xs = [x_min, x_max, x_max, x_min, x_min]
        ys = [y_min, y_min, y_max, y_max, y_min]
        zs = [z] * len(xs)
        ax.plot(xs, ys, zs, color="tab:blue", linewidth=1.0, alpha=0.55)

        # Draw child-radius circle on each slice for an easy visual "fits inside" check.
        t = np.linspace(0.0, 2.0 * math.pi, 200, dtype=np.float64)
        cx = float(child_radius_m) * np.cos(t)
        cy = float(child_radius_m) * np.sin(t)
        cz = np.full_like(cx, z)
        ax.plot(cx, cy, cz, color="tab:orange", linewidth=1.0, alpha=0.35)

    # Axis styling.
    ax.set_xlabel("dx (m, robot frame, +x forward)")
    ax.set_ylabel("dy (m, robot frame, +y left)")
    ax.set_zlabel("yaw offset (deg)")
    ax.set_title(
        f"{title}\n"
        f"crop: x∈[{x_min:.2f},{x_max:.2f}] (back={back:.2f}, fwd={fwd:.2f}) "
        f"y∈[{y_min:.2f},{y_max:.2f}] | child_radius≈{float(child_radius_m):.3f}m => {status}"
    )

    # Keep a consistent view.
    pad = 0.15
    ax.set_xlim(float(x_min) - pad, float(x_max) + pad)
    ax.set_ylim(float(y_min) - pad, float(y_max) + pad)
    ax.set_zlim(float(z_min) - 5.0, float(z_max) + 5.0)
    ax.view_init(elev=22, azim=-55)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plot a 3D visualization of valuewin coverage: (dx,dy) crop replicated across yaw slices, "
            "and a conservative one-step children radius overlay."
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
    p.add_argument(
        "--crop-mode",
        type=str,
        default="biased",
        choices=["biased", "centered"],
        help="centered forces bias=0 so forward/backward coverage matches.",
    )
    p.add_argument(
        "--yaw-offsets-deg",
        type=str,
        default="-45,-36,-27,-18,-9,0,9,18,27,36,45,135,144,153,162,171,180,189,198,207,216,225",
        help="Yaw offsets (deg) for value slices. Default: 11 forward (-45..45) + 11 backward (180±45).",
    )
    p.add_argument(
        "--fine-vi-robot",
        type=str,
        default="",
        help="Optional path to fine vi_robot.pkl. If provided, compute child-radius from the robot (dt/rho/limits). "
        "Otherwise uses meta_fine.json summary (fast).",
    )
    args = p.parse_args(argv)

    goal_dir = Path(args.goal_dir)
    meta_fine_path = goal_dir / "meta_fine.json"
    if not meta_fine_path.exists():
        raise FileNotFoundError(f"Missing meta_fine.json under --goal-dir: {meta_fine_path}")
    meta_fine = json.loads(meta_fine_path.read_text(encoding="utf-8"))

    crop_bias = float(args.crop_bias_forward_m)
    crop_mode = str(args.crop_mode).strip().lower()
    if crop_mode == "centered":
        crop_bias = 0.0

    yaw_offsets_deg = _parse_offsets_deg(str(args.yaw_offsets_deg))

    # Conservative one-step children radius (XY) from VI settings.
    # Prefer meta_fine.json summary (fast). Optionally compute from vi_robot.pkl (slow).
    child_radius_m: float | None = None
    if args.fine_vi_robot:
        fine_robot = load_vi_robot(Path(args.fine_vi_robot))
        child_radius_m = float(required_child_radius_m(fine_robot=fine_robot))
    else:
        summary = meta_fine.get("summary", {}) if isinstance(meta_fine.get("summary", {}), dict) else {}
        dt = summary.get("dt", None)
        rho = summary.get("rho", None)
        if dt is not None and rho is not None:
            task_path_for_limits = meta_fine.get("task_path", meta_fine.get("tmp_task_path", ""))
            task_for_limits = load_task(task_path_for_limits)
            lims = np.asarray((task_for_limits.get("robots", [{}])[0] or {}).get("control_limits", [[-1, 1], [-1, 1]]), dtype=np.float64).reshape(2, 2)
            max_speed = float(max(abs(float(lims[0, 0])), abs(float(lims[0, 1]))))
            child_radius_m = float(float(dt) * max_speed + float(rho))

    if child_radius_m is None:
        # Fallback: load vi_robot.pkl from meta_fine.json output (may be large).
        out_obj = meta_fine.get("output", {}) if isinstance(meta_fine.get("output", {}), dict) else {}
        vi_pkl = out_obj.get("vi_robot_pkl", "")
        if not vi_pkl:
            raise ValueError(f"meta_fine.json missing summary.dt/rho and output.vi_robot_pkl: {meta_fine_path}")
        fine_robot = load_vi_robot(Path(vi_pkl))
        child_radius_m = float(required_child_radius_m(fine_robot=fine_robot))

    # Title context: map/goal and angle_scalor (for completeness).
    task_path = meta_fine.get("task_path", meta_fine.get("tmp_task_path", ""))
    task = load_task(task_path)
    xmin, xmax, _, _ = get_range(task)
    angle_scalor = angle_scalor_from_range(xmin, xmax)
    map_name = str(meta_fine.get("map_name", goal_dir.parent.name))
    goal_index = int(meta_fine.get("goal_index", goal_dir.name.replace("goal_", "")))

    _plot(
        out_path=Path(args.out),
        yaw_offsets_deg=yaw_offsets_deg,
        crop_size=int(args.crop_size),
        mpp=float(args.mpp),
        crop_bias_forward_m=float(crop_bias),
        child_radius_m=float(child_radius_m),
        title=f"{map_name} goal_{goal_index} | angle_scalor={float(angle_scalor):.3f}",
    )


if __name__ == "__main__":
    main()
