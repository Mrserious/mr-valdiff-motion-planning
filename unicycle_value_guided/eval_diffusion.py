from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.crop_coverage import validate_crop_covers_children
from unicycle_value_guided.infer_diffusion import AntiRepeatConfig, _find_vi_robot, _load_policy_from_ckpt, _run_episode
from unicycle_value_guided.inflation import prepare_inflated_goal_assets
from unicycle_value_guided.se2 import angle_scalor_from_range, theta_scaled_from_yaw
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.task_io import get_range, load_json, load_task
from unicycle_value_guided.vi_io import load_vi_robot
from unicycle_value_guided.value_grid3d import load_regular_value_grid_3d
from unicycle_value_guided.rollout_fine import reconstruct_action_from_transition


def _clip_action_v_omega(
    action_v_omega: np.ndarray,
    *,
    control_limits_scaled: np.ndarray,
    angle_scalor: float,
) -> np.ndarray:
    """
    Clip (v, omega_rad/s) to the robot control limits.

    Note:
      - VI robots store omega limits in theta_scaled/sec; convert to rad/s via: omega_rad = omega_scaled/angle_scalor*pi.
    """
    a = np.asarray(action_v_omega, dtype=np.float32).reshape(2)
    lims = np.asarray(control_limits_scaled, dtype=np.float32).reshape(2, 2)
    vmin, vmax = float(min(lims[0])), float(max(lims[0]))
    omin_s, omax_s = float(min(lims[1])), float(max(lims[1]))  # scaled omega (theta_scaled/sec)
    omin = omin_s / float(angle_scalor) * float(np.pi)
    omax = omax_s / float(angle_scalor) * float(np.pi)
    v = float(np.clip(float(a[0]), vmin, vmax))
    w = float(np.clip(float(a[1]), omin, omax))
    return np.array([v, w], dtype=np.float32)


def _start_has_feasible_vi_path(
    *,
    fine_robot: Any,
    collision_robot: Any,
    collision_task: dict[str, Any],
    start_scaled: np.ndarray,
    angle_scalor: float,
    max_steps: int,
    collision_check_step: float,
    max_children_per_step: int,
    allow_self_candidate: bool,
) -> bool:
    """
    Check if there exists a *feasible* (swept-collision-free) path from the snapped start node to the goal,
    using the fine VI graph as a proposal structure.

    This is intentionally policy-independent so that multiple models evaluated with the same seed will see the
    same start set.

    Strategy: greedy descent by node.value with swept collision filtering on each candidate edge.
    """
    start_scaled = np.asarray(start_scaled, dtype=np.float32).reshape(3)
    idx_list = getattr(fine_robot, "query_kdtree")(start_scaled)
    if not idx_list:
        return False
    idx_cur = int(idx_list[0])

    dt = float(getattr(fine_robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from fine_robot.get_temporal_res(): {dt}")

    control_limits_scaled = np.asarray(getattr(fine_robot, "control_limits", [[-1, 1], [-1, 1]]), dtype=np.float32).reshape(2, 2)

    visited: set[int] = {idx_cur}
    for _ in range(int(max_steps)):
        cur_state = np.asarray(fine_robot.nodes[int(idx_cur)].state, dtype=np.float32).reshape(3)
        if bool(getattr(fine_robot, "within_goal")(cur_state)):
            return True

        children = list(getattr(fine_robot.nodes[int(idx_cur)].children, "indices", []))
        if bool(allow_self_candidate):
            children = [int(idx_cur)] + children
        if not children:
            return False

        # Rank by node.value ascending (best-first). This is cheap and matches the VI semantics.
        vals = np.array([float(getattr(fine_robot.nodes[int(c)], "value", 1.0)) for c in children], dtype=np.float64)
        order = np.argsort(vals, kind="stable")

        moved = False
        tried = 0
        for oi in order:
            cand = int(children[int(oi)])
            if cand in visited:
                continue
            tried += 1
            cand_state = np.asarray(fine_robot.nodes[int(cand)].state, dtype=np.float32).reshape(3)
            a = reconstruct_action_from_transition(
                cur_state_scaled=cur_state,
                nxt_state_scaled=cand_state,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
            )
            a = _clip_action_v_omega(a, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))
            if not trajectory_collision_free(
                robot=collision_robot,
                task=collision_task,
                state_scaled=cur_state,
                action_v_omega=a,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                step_size=float(collision_check_step),
            ):
                if int(max_children_per_step) > 0 and tried >= int(max_children_per_step):
                    break
                continue
            idx_cur = int(cand)
            visited.add(int(cand))
            moved = True
            break

        if not moved:
            return False

    return False


def _parse_index_spec(spec: str, n: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            out.append(int(part))
    seen = set()
    dedup: list[int] = []
    for i in out:
        if i in seen:
            continue
        if i < 0 or i >= n:
            raise ValueError(f"goal index out of range: {i} (n={n})")
        seen.add(i)
        dedup.append(i)
    return dedup


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch evaluate unicycle diffusion policy on multiple goals/starts.")
    p.add_argument("--goals", type=str, required=True, help="Path to data/unicycle_value_grids/<map>/goals.json")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path or checkpoints dir (uses latest.ckpt).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--goal-indices", type=str, default="", help='Subset of goals, e.g. "0-9,20". Empty=all.')
    p.add_argument("--starts-per-goal", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clearance", type=float, default=0.2)
    p.add_argument("--min-goal-dist", type=float, default=3.0)
    p.add_argument("--crop-size", type=int, default=84)
    p.add_argument("--mpp", type=float, default=0.05)
    p.add_argument("--rotate-with-yaw", action="store_true", default=True)
    p.add_argument("--no-rotate-with-yaw", action="store_false", dest="rotate_with_yaw")
    p.add_argument(
        "--crop-mode",
        type=str,
        default="biased",
        choices=["biased", "centered"],
        help="biased: use --crop-bias-forward-m; centered: force symmetric crop (bias=0).",
    )
    p.add_argument("--crop-bias-forward-m", type=float, default=0.9375)
    p.add_argument(
        "--require-crop-covers-children",
        action="store_true",
        help="Fail if crop does not conservatively cover one-step children radius in both forward/backward directions.",
    )
    p.add_argument(
        "--yaw-offsets-deg",
        type=str,
        default="-45,-36,-27,-18,-9,0,9,18,27,36,45,135,144,153,162,171,180,189,198,207,216,225",
        help=(
            "Comma-separated yaw offsets (deg) for value slices; must include 0. "
            "Default: 11 forward (-45..45) + 11 backward (180±45)."
        ),
    )
    p.add_argument("--footprint-length-m", type=float, default=0.625)
    p.add_argument("--footprint-width-m", type=float, default=0.4375)
    p.add_argument("--max-resample-attempts", type=int, default=30)
    p.add_argument("--collision-check-step", type=float, default=0.05)
    p.add_argument(
        "--collision-semantic",
        type=str,
        default="swept",
        choices=["swept", "discrete"],
        help=(
            "Collision checking semantic for projected actions: "
            "swept=continuous (sub-step) collision checking along the macro step; "
            "discrete=only check the final integrated state. Default: swept."
        ),
    )
    p.add_argument("--allow-self-candidate", action="store_true")
    p.add_argument("--anti-repeat", action="store_true", help="Enable best-effort anti-repeat (tabu) when selecting projected child/action.")
    p.add_argument("--anti-repeat-xy-q", type=float, default=0.005, help="XY quantization step (meters) for anti-repeat tabu (default: 0.005).")
    p.add_argument("--anti-repeat-xy-recent-n", type=int, default=80, help="How many recent XY keys to tabu (0=disable).")
    p.add_argument("--anti-repeat-child-recent-n", type=int, default=20, help="How many recent child indices to tabu (0=disable).")
    p.add_argument("--anti-repeat-edge-recent-n", type=int, default=20, help="How many recent edges (idx_cur,idx_next) to tabu (0=disable).")
    p.add_argument(
        "--anti-repeat-avoid-uturn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer not to immediately reverse the previous edge (A->B then B->A). Default: enabled.",
    )
    p.add_argument(
        "--require-feasible-starts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter sampled starts: keep only starts that admit a feasible (swept-collision-free) path on the fine graph "
            "used for inference (default: enabled). This is policy-independent (VI-greedy + swept filter)."
        ),
    )
    p.add_argument(
        "--max-start-attempts",
        type=int,
        default=200_000,
        help="Max random start samples per episode before failing (includes feasibility filtering).",
    )
    p.add_argument(
        "--feasible-start-max-steps",
        type=int,
        default=None,
        help="Max steps for the feasibility check (default: use --max-steps).",
    )
    p.add_argument(
        "--feasible-start-max-children-per-step",
        type=int,
        default=0,
        help="Optional cap on how many children (ranked by VI value) to try per step during feasibility check (0=all).",
    )

    # Opt-A: obstacle inflation planning semantics (cached, pluggable).
    p.add_argument(
        "--opt-a",
        action="store_true",
        help="Enable Opt-A (inflated obstacles for occupancy/value/gpath; defaults to inflated fine sampling).",
    )
    p.add_argument("--opt-a-delta", type=float, default=0.05, help="Obstacle inflation radius in meters (default: 0.05).")
    p.add_argument(
        "--opt-a-cache-dir",
        type=str,
        default="data/unicycle_value_grids_inflated",
        help="Cache directory for inflated VI assets (separate from original value_grids).",
    )
    p.add_argument("--opt-a-overwrite-cache", action="store_true", help="Recompute Opt-A cached assets even if they exist.")
    p.add_argument(
        "--opt-a-use-inflated-fine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inflated fine VI graph for snap-to-children semantics (default: enabled for --opt-a).",
    )
    p.add_argument(
        "--opt-a-collision-check",
        type=str,
        default="real",
        choices=["real", "inflated"],
        help="Which obstacle set to use for collision checking (default: real).",
    )
    p.add_argument(
        "--opt-a-keep-real-collision-check",
        action="store_const",
        const="real",
        dest="opt_a_collision_check",
        help="Alias of --opt-a-collision-check=real (recommended).",
    )
    p.add_argument(
        "--opt-a-use-inflated-collision-check",
        action="store_const",
        const="inflated",
        dest="opt_a_collision_check",
        help="Alias of --opt-a-collision-check=inflated (debug only).",
    )
    p.add_argument("--opt-a-vi-device", type=str, default=None, help="Device for building Opt-A VI cache (default: follow meta).")
    p.add_argument("--opt-a-vi-dtype", type=str, default=None, choices=["float16", "float32"], help="VI dtype for Opt-A cache (default: follow meta).")
    p.add_argument("--opt-a-vi-max-iters", type=int, default=None, help="VI max iters for Opt-A cache (default: follow meta).")
    p.add_argument("--opt-a-vi-tol", type=float, default=None, help="VI tol for Opt-A cache (default: follow meta).")

    p.add_argument("--opt-b-topk-children", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="", help="Override output dir (default: outputs/eval_unicycle_diffusion/<map>/<timestamp>).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(str(args.device))

    collision_semantic = str(getattr(args, "collision_semantic", "swept")).strip().lower()
    if collision_semantic not in ("swept", "discrete"):
        raise ValueError(f"Invalid --collision-semantic: {collision_semantic!r}")

    anti_repeat = AntiRepeatConfig(
        enabled=bool(getattr(args, "anti_repeat", False)),
        xy_q=float(getattr(args, "anti_repeat_xy_q", 0.005)),
        xy_recent_n=int(getattr(args, "anti_repeat_xy_recent_n", 0)),
        child_recent_n=int(getattr(args, "anti_repeat_child_recent_n", 0)),
        edge_recent_n=int(getattr(args, "anti_repeat_edge_recent_n", 0)),
        avoid_uturn=bool(getattr(args, "anti_repeat_avoid_uturn", True)),
    )
    if not bool(anti_repeat.enabled):
        anti_repeat = None
    else:
        if int(anti_repeat.xy_recent_n) < 0 or int(anti_repeat.child_recent_n) < 0 or int(anti_repeat.edge_recent_n) < 0:
            raise ValueError("Anti-repeat recent window sizes must be >= 0.")
        if not (np.isfinite(float(anti_repeat.xy_q)) and float(anti_repeat.xy_q) > 0):
            raise ValueError(f"Invalid --anti-repeat-xy-q: {anti_repeat.xy_q}")

    goals_path = Path(args.goals)
    goals_payload = load_json(goals_path)
    goals: list[list[float]] = goals_payload["goals"]

    task_path = Path(goals_payload["task_path"])
    if not task_path.is_absolute():
        task_path = (Path.cwd() / task_path).resolve()
    task = load_task(task_path)
    xmin, xmax, ymin, ymax = get_range(task)
    angle_scalor = angle_scalor_from_range(xmin, xmax)

    yaw_offsets_deg = tuple(int(s.strip()) for s in str(args.yaw_offsets_deg).split(",") if s.strip())
    if len(yaw_offsets_deg) == 0:
        raise ValueError("--yaw-offsets-deg must contain at least one offset (and must include 0).")
    if 0 not in yaw_offsets_deg:
        raise ValueError("--yaw-offsets-deg must include 0.")

    crop_mode = str(getattr(args, "crop_mode", "biased")).strip().lower()
    if crop_mode not in ("biased", "centered"):
        raise ValueError(f"Invalid --crop-mode: {crop_mode}")
    if crop_mode == "centered":
        args.crop_bias_forward_m = 0.0

    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.ckpt"
    policy = _load_policy_from_ckpt(ckpt_path, device=device, use_ema=bool(args.use_ema))

    map_name = str(goals_payload.get("map_name", goals_path.parent.name))
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("outputs") / "eval_unicycle_diffusion" / map_name / ts
    out_root.mkdir(parents=True, exist_ok=True)

    goal_indices = _parse_index_spec(str(args.goal_indices), n=len(goals))
    all_goal_summaries: list[dict] = []

    for goal_idx in goal_indices:
        goal_xyz = goals[goal_idx]
        if len(goal_xyz) != 3:
            raise ValueError(f"Expected goal pose [x,y,yaw_deg], got: {goal_xyz}")
        goal_xy = np.asarray(goal_xyz[:2], dtype=np.float64).reshape(2)
        goal_yaw = float(goal_xyz[2]) / 180.0 * float(np.pi)
        goal_dir = goals_path.parent / f"goal_{goal_idx}"

        coarse_meta = json.loads((goal_dir / "meta_coarse.json").read_text(encoding="utf-8"))
        fine_meta = json.loads((goal_dir / "meta_fine.json").read_text(encoding="utf-8"))

        task_obs = None
        collision_robot: Any | None = None
        collision_task: dict[str, Any] = task
        start_inflated_robot: Any | None = None

        if bool(args.opt_a):
            inflated = prepare_inflated_goal_assets(
                base_task=task,
                base_task_path=task_path,
                map_name=map_name,
                goal_idx=int(goal_idx),
                goal_xyz=goal_xyz,
                cache_root=Path(args.opt_a_cache_dir),
                delta=float(args.opt_a_delta),
                coarse_meta_src=coarse_meta,
                fine_meta_src=(fine_meta if bool(args.opt_a_use_inflated_fine) else None),
                vi_device=(None if args.opt_a_vi_device is None else str(args.opt_a_vi_device)),
                vi_dtype=(None if args.opt_a_vi_dtype is None else str(args.opt_a_vi_dtype)),
                max_iters=(None if args.opt_a_vi_max_iters is None else int(args.opt_a_vi_max_iters)),
                tol=(None if args.opt_a_vi_tol is None else float(args.opt_a_vi_tol)),
                overwrite=bool(args.opt_a_overwrite_cache),
                keep_pkl=True,
                use_inflated_fine=bool(args.opt_a_use_inflated_fine),
            )
            task_obs = inflated.task_obs
            coarse_grid3d = inflated.coarse_grid3d
            coarse_robot = inflated.coarse_robot

            if bool(args.opt_a_use_inflated_fine):
                if inflated.fine_robot is None:
                    raise RuntimeError("Opt-A requested inflated fine, but fine_robot cache is missing.")
                fine_robot = inflated.fine_robot
            else:
                fine_robot = load_vi_robot(_find_vi_robot(goal_dir, "fine"))

            collision_check_mode = str(getattr(args, "opt_a_collision_check", "real"))
            if collision_check_mode == "inflated":
                if task_obs is None:
                    raise RuntimeError("Opt-A internal error: task_obs is None.")
                collision_task = task_obs
            else:
                collision_task = task
            if not collision_task.get("robots"):
                raise RuntimeError("collision_task has no robots[] entry.")
            collision_robot = Unicycle(collision_task["env"], collision_task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)

            # Always ensure sampled starts are also collision-free under the inflated obstacle semantics.
            # This avoids "real-free but inflated-colliding" starts, which are out-of-distribution for Opt-A
            # (occupancy/value/gpath are computed on the inflated map), even if collision checking uses real obstacles.
            if task_obs is None:
                raise RuntimeError("Opt-A internal error: task_obs is None.")
            if not task_obs.get("robots"):
                raise RuntimeError("Opt-A task_obs has no robots[] entry.")
            if collision_check_mode == "inflated":
                start_inflated_robot = collision_robot
            else:
                start_inflated_robot = Unicycle(
                    task_obs["env"],
                    task_obs["robots"][0],
                    angle_scalor=float(angle_scalor),
                    robot_id=0,
                )
        else:
            coarse_grid3d = load_regular_value_grid_3d(goal_dir / "value_coarse_3d.npy", goal_dir / "meta_coarse_3d.json")
            coarse_robot = load_vi_robot(_find_vi_robot(goal_dir, "coarse"))
            fine_robot = load_vi_robot(_find_vi_robot(goal_dir, "fine"))
            collision_robot = fine_robot

        validate_crop_covers_children(
            fine_robot=fine_robot,
            crop_size=int(args.crop_size),
            meters_per_pixel=float(args.mpp),
            crop_bias_forward_m=float(args.crop_bias_forward_m),
            strict=bool(getattr(args, "require_crop_covers_children", False)),
            extra_margin_m=0.0,
            context="eval_diffusion",
        )

        fine_node_states = np.stack([np.asarray(n.state, dtype=np.float32).reshape(3) for n in fine_robot.nodes], axis=0)

        rng = np.random.default_rng(int(args.seed) * 100000 + int(goal_idx))
        successes = 0
        reasons: dict[str, int] = {}
        lengths: list[int] = []
        start_attempts_total = 0

        feasible_max_steps = int(args.max_steps) if args.feasible_start_max_steps is None else int(args.feasible_start_max_steps)
        max_children_per_step = int(getattr(args, "feasible_start_max_children_per_step", 0))

        for ep in range(int(args.starts_per_goal)):
            for attempt in range(int(args.max_start_attempts)):
                x = float(rng.uniform(xmin + float(args.clearance), xmax - float(args.clearance)))
                y = float(rng.uniform(ymin + float(args.clearance), ymax - float(args.clearance)))
                if float(np.hypot(x - float(goal_xy[0]), y - float(goal_xy[1]))) < float(args.min_goal_dist):
                    continue
                yaw = float(rng.uniform(-np.pi, np.pi))
                start_scaled = np.array([x, y, theta_scaled_from_yaw(yaw, float(angle_scalor))], dtype=np.float32)
                if bool(getattr(args, "opt_a", False)):
                    if start_inflated_robot is None:
                        raise RuntimeError("Opt-A internal error: start_inflated_robot is None.")
                    if not bool(getattr(start_inflated_robot, "obstacle_free")(start_scaled)):
                        continue
                    # If we are snapping on the inflated fine graph, also ensure the start is free under that graph's
                    # own obstacle_free() semantics (should match inflated obstacles, but this is a cheap extra guard).
                    if bool(getattr(args, "opt_a_use_inflated_fine", False)) and hasattr(fine_robot, "obstacle_free"):
                        if not bool(getattr(fine_robot, "obstacle_free")(start_scaled)):
                            continue
                else:
                    if not bool(getattr(collision_robot, "obstacle_free")(start_scaled)):
                        continue
                if bool(getattr(args, "require_feasible_starts", True)):
                    if not _start_has_feasible_vi_path(
                        fine_robot=fine_robot,
                        collision_robot=collision_robot,
                        collision_task=collision_task,
                        start_scaled=start_scaled,
                        angle_scalor=float(angle_scalor),
                        max_steps=int(feasible_max_steps),
                        collision_check_step=float(args.collision_check_step),
                        max_children_per_step=int(max_children_per_step),
                        allow_self_candidate=bool(args.allow_self_candidate),
                    ):
                        continue
                start_state = np.array([x, y, yaw], dtype=np.float32)
                start_attempts_total += int(attempt) + 1
                break
            else:
                raise RuntimeError("Failed to sample a valid start state.")

            res = _run_episode(
                task=task,
                task_obs=task_obs,
                coarse_grid3d=coarse_grid3d,
                coarse_robot=coarse_robot,
                fine_robot=fine_robot,
                fine_node_states=fine_node_states,
                collision_robot=collision_robot,
                goal_xy=goal_xy,
                goal_yaw=goal_yaw,
                angle_scalor=float(angle_scalor),
                policy=policy,
                start_state=start_state,
                max_steps=int(args.max_steps),
                crop_size=int(args.crop_size),
                meters_per_pixel=float(args.mpp),
                rotate_with_yaw=bool(args.rotate_with_yaw),
                crop_bias_forward_m=float(args.crop_bias_forward_m),
                yaw_offsets_deg=yaw_offsets_deg,
                footprint_length_m=float(args.footprint_length_m),
                footprint_width_m=float(args.footprint_width_m),
                max_resample_attempts=int(args.max_resample_attempts),
                collision_check_step=float(args.collision_check_step),
                collision_semantic=str(collision_semantic),
                allow_self_candidate=bool(args.allow_self_candidate),
                opt_b_topk_children=int(args.opt_b_topk_children),
                anti_repeat=anti_repeat,
            )
            successes += int(bool(res.success))
            reasons[res.reason] = int(reasons.get(res.reason, 0) + 1)
            lengths.append(int(res.actions.shape[0]))
            print(f"[eval] goal={goal_idx} ep={ep} success={res.success} reason={res.reason} T={res.actions.shape[0]}", flush=True)

        gsum = {
            "goal_index": int(goal_idx),
            "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
            "episodes": int(args.starts_per_goal),
            "successes": int(successes),
            "success_rate": float(successes) / float(max(int(args.starts_per_goal), 1)),
            "avg_start_attempts": float(start_attempts_total) / float(max(int(args.starts_per_goal), 1)),
            "reasons": reasons,
            "avg_T": float(np.mean(lengths)) if lengths else 0.0,
            "median_T": float(np.median(lengths)) if lengths else 0.0,
        }
        all_goal_summaries.append(gsum)

    overall = {
        "map_name": map_name,
        "goals": str(goals_path),
        "ckpt": str(ckpt_path),
        "device": str(device),
        "goal_indices": goal_indices,
        "starts_per_goal": int(args.starts_per_goal),
        "require_feasible_starts": bool(getattr(args, "require_feasible_starts", True)),
        "max_start_attempts": int(getattr(args, "max_start_attempts", 0)),
        "feasible_start_max_steps": (int(args.max_steps) if args.feasible_start_max_steps is None else int(args.feasible_start_max_steps)),
        "feasible_start_max_children_per_step": int(getattr(args, "feasible_start_max_children_per_step", 0)),
        "success_rate_mean": float(np.mean([g["success_rate"] for g in all_goal_summaries])) if all_goal_summaries else 0.0,
        "opt_a": {
            "enabled": bool(args.opt_a),
            "delta": (None if not bool(args.opt_a) else float(args.opt_a_delta)),
            "cache_dir": (None if not bool(args.opt_a) else str(args.opt_a_cache_dir)),
            "use_inflated_fine": (None if not bool(args.opt_a) else bool(args.opt_a_use_inflated_fine)),
            "collision_check": (None if not bool(args.opt_a) else str(getattr(args, "opt_a_collision_check", "real"))),
            "overwrite_cache": (None if not bool(args.opt_a) else bool(args.opt_a_overwrite_cache)),
        },
        "per_goal": all_goal_summaries,
    }
    (out_root / "summary.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[eval] wrote: {out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
