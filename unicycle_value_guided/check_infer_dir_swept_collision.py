from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.se2 import angle_scalor_from_range, theta_scaled_from_yaw, wrap_theta_scaled
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.task_io import get_range, load_task


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run strict swept collision checking on existing infer_diffusion trajectories "
            "(episode_*_states.npy + episode_*_projected_actions.npy)."
        )
    )
    p.add_argument("--infer-dir", type=str, required=True, help="Path to an infer_diffusion output directory.")
    p.add_argument(
        "--actions",
        type=str,
        default="projected",
        choices=["projected", "raw"],
        help="Which action file to validate: projected (default) or raw.",
    )
    p.add_argument("--dt", type=float, default=None, help="Override dt (seconds). Default: read from goal_*/meta_fine.json.")
    p.add_argument("--collision-check-step", type=float, default=0.05, help="Swept collision linear step (meters).")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output report path (JSON). Default: <infer_dir>/swept_collision_report_<actions>.json",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    infer_dir = Path(args.infer_dir).expanduser()
    if not infer_dir.exists():
        raise FileNotFoundError(f"--infer-dir does not exist: {infer_dir}")
    if not infer_dir.is_dir():
        raise NotADirectoryError(f"--infer-dir is not a directory: {infer_dir}")

    summary_path = infer_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in infer dir: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    goal_dir = infer_dir.parent
    meta_coarse_path = goal_dir / "meta_coarse.json"
    if not meta_coarse_path.exists():
        raise FileNotFoundError(f"Missing meta_coarse.json next to infer dir: {meta_coarse_path}")
    meta_coarse = json.loads(meta_coarse_path.read_text(encoding="utf-8"))

    task = load_task(meta_coarse.get("task_path", meta_coarse.get("tmp_task_path")))
    xmin, xmax, _, _ = get_range(task)
    angle_scalor = float(angle_scalor_from_range(xmin, xmax))

    # Collision robot uses original (real) obstacles.
    robot0 = task.get("robots", [{}])[0]
    collision_robot = Unicycle(task["env"], robot0, angle_scalor=float(angle_scalor), robot_id=0)

    # Default dt: infer_diffusion uses fine graph temporal resolution.
    if args.dt is not None:
        dt = float(args.dt)
    else:
        meta_fine_path = goal_dir / "meta_fine.json"
        if not meta_fine_path.exists():
            raise FileNotFoundError(f"Missing meta_fine.json for default dt; pass --dt to override: {meta_fine_path}")
        meta_fine = json.loads(meta_fine_path.read_text(encoding="utf-8"))
        dt = float(meta_fine.get("summary", {}).get("dt", meta_fine.get("dt", 0.0)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt: {dt}")

    action_kind = str(args.actions)
    action_suffix = "projected_actions" if action_kind == "projected" else "actions"

    state_paths = sorted(infer_dir.glob("episode_*_states.npy"))
    if not state_paths:
        raise FileNotFoundError(f"No episode_*_states.npy files found in: {infer_dir}")

    results: list[dict[str, Any]] = []
    ok_eps: list[int] = []
    fail_eps: list[int] = []

    for sp in state_paths:
        ep_str = sp.name.split("_")[1]
        ep = int(ep_str)
        states_yaw = np.load(sp).astype(np.float32, copy=False)
        if states_yaw.ndim != 2 or states_yaw.shape[1] != 3 or states_yaw.shape[0] < 1:
            raise ValueError(f"Invalid states array in {sp}: shape={states_yaw.shape}")

        ap = infer_dir / f"episode_{ep:04d}_{action_suffix}.npy"
        if not ap.exists():
            raise FileNotFoundError(f"Missing action file for episode {ep}: {ap}")
        actions = np.load(ap).astype(np.float32, copy=False)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(f"Invalid actions array in {ap}: shape={actions.shape}")

        T = int(actions.shape[0])
        if int(states_yaw.shape[0]) != T + 1:
            # Be tolerant: allow mismatch but only validate min(T, len(states)-1) steps.
            T = min(T, int(states_yaw.shape[0] - 1))

        ok = True
        first_fail_step: int | None = None
        first_fail_state: list[float] | None = None
        first_fail_action: list[float] | None = None

        for t in range(T):
            x, y, yaw = float(states_yaw[t, 0]), float(states_yaw[t, 1]), float(states_yaw[t, 2])
            th = float(theta_scaled_from_yaw(float(yaw), float(angle_scalor)))
            th = float(wrap_theta_scaled(th, float(angle_scalor)))
            s_scaled = np.array([x, y, th], dtype=np.float32)
            a = actions[t].astype(np.float32, copy=False).reshape(2)
            if not trajectory_collision_free(
                robot=collision_robot,
                task=task,
                state_scaled=s_scaled,
                action_v_omega=a,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                step_size=float(args.collision_check_step),
            ):
                ok = False
                first_fail_step = int(t)
                first_fail_state = [float(x), float(y), float(yaw)]
                first_fail_action = [float(a[0]), float(a[1])]
                break

        if ok:
            ok_eps.append(ep)
        else:
            fail_eps.append(ep)

        results.append(
            {
                "episode": ep,
                "ok": bool(ok),
                "T_checked": int(T),
                "first_fail_step": first_fail_step,
                "first_fail_state_yaw": first_fail_state,
                "first_fail_action": first_fail_action,
            }
        )

    report = {
        "infer_dir": str(infer_dir),
        "map_name": summary.get("map_name", None),
        "goal_index": summary.get("goal_index", None),
        "goal_pose": summary.get("goal_pose", None),
        "episodes": int(len(results)),
        "ok_episodes": int(len(ok_eps)),
        "fail_episodes": int(len(fail_eps)),
        "ok_rate": float(len(ok_eps)) / float(len(results)) if results else 0.0,
        "collision_check": {"semantic": "swept", "check_step": float(args.collision_check_step), "dt": float(dt), "actions": action_kind},
        "ok_episode_indices": ok_eps,
        "fail_episode_indices": fail_eps,
        "per_episode": results,
    }

    out_path = Path(args.out).expanduser() if args.out is not None else (infer_dir / f"swept_collision_report_{action_kind}.json")
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[swept_check] actions={action_kind} ok={len(ok_eps)}/{len(results)} "
        f"({report['ok_rate']:.3f}) wrote={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

