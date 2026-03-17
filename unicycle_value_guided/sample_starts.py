from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.inflation import inflate_task_obstacles, prepare_inflated_goal_assets
from unicycle_value_guided.rollout_fine import reconstruct_action_from_transition
from unicycle_value_guided.se2 import angle_scalor_from_range, theta_scaled_from_yaw
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.task_io import get_range, load_json, load_task
from unicycle_value_guided.vi_io import load_vi_robot


def _find_vi_robot(goal_dir: Path, level: str) -> Path:
    level = str(level).strip().lower()
    candidates = [
        goal_dir / f"vi_robot_{level}.pkl",
        goal_dir / f"logs_{level}" / "vi_robot.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {level} vi_robot.pkl under: {goal_dir}")


def _clip_action_v_omega(
    action_v_omega: np.ndarray,
    *,
    control_limits_scaled: Sequence[Sequence[float]],
    angle_scalor: float,
) -> np.ndarray:
    action_v_omega = np.asarray(action_v_omega, dtype=np.float32).reshape(2)
    lims = np.asarray(control_limits_scaled, dtype=np.float32).reshape(2, 2)
    vmin, vmax = float(min(lims[0])), float(max(lims[0]))
    omin_s, omax_s = float(min(lims[1])), float(max(lims[1]))
    omin = omin_s / float(angle_scalor) * float(np.pi)
    omax = omax_s / float(angle_scalor) * float(np.pi)
    v = float(np.clip(action_v_omega[0], vmin, vmax))
    w = float(np.clip(action_v_omega[1], omin, omax))
    return np.array([v, w], dtype=np.float32)


def _vi_has_feasible_path(
    *,
    robot: Any,
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
    Policy-independent feasibility check: does there exist a swept-collision-free path from the snapped start node
    to the goal on the given VI graph (coarse or fine)?

    Strategy: greedy descent by node.value with swept collision filtering on each candidate edge.
    """
    start_scaled = np.asarray(start_scaled, dtype=np.float32).reshape(3)
    idx_list = getattr(robot, "query_kdtree")(start_scaled)
    if not idx_list:
        return False
    idx_cur = int(idx_list[0])

    dt = float(getattr(robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from robot.get_temporal_res(): {dt}")

    control_limits_scaled = np.asarray(getattr(robot, "control_limits", [[-1, 1], [-1, 1]]), dtype=np.float32).reshape(2, 2)

    visited: set[int] = {idx_cur}
    for _ in range(int(max_steps)):
        cur_state = np.asarray(robot.nodes[int(idx_cur)].state, dtype=np.float32).reshape(3)
        if bool(getattr(robot, "within_goal")(cur_state)):
            return True

        children = list(getattr(robot.nodes[int(idx_cur)].children, "indices", []))
        if bool(allow_self_candidate):
            children = [int(idx_cur)] + children
        if not children:
            return False

        vals = np.array([float(getattr(robot.nodes[int(c)], "value", 1.0)) for c in children], dtype=np.float64)
        order = np.argsort(vals, kind="stable")

        moved = False
        tried = 0
        for oi in order:
            cand = int(children[int(oi)])
            if cand in visited:
                continue
            tried += 1
            cand_state = np.asarray(robot.nodes[int(cand)].state, dtype=np.float32).reshape(3)
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


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample valid starts for a single (map, goal) and save them for later inference runs.")
    p.add_argument("--goals", type=str, required=True, help="Path to data/unicycle_value_grids/<map>/goals.json")
    p.add_argument("--goal-index", type=int, required=True, help="Which goal_<k> to sample starts for.")
    p.add_argument("--out", type=str, default=None, help="Output JSON file path. Default: <goal_dir>/starts_seed{seed}_{mode}.json")
    p.add_argument("--overwrite", action="store_true", help="Overwrite --out if it already exists.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-starts", type=int, default=100, help="How many starts to output.")
    p.add_argument("--clearance", type=float, default=0.2, help="Boundary margin when sampling starts.")
    p.add_argument("--min-goal-dist", type=float, default=3.0, help="Min xy distance start->goal.")

    p.add_argument("--collision-check-step", type=float, default=0.05, help="Collision check linear step (meters).")
    p.add_argument("--allow-self-candidate", action="store_true", help="Allow staying at current node as candidate.")

    p.add_argument(
        "--start-feasible-mode",
        type=str,
        default="none",
        choices=["none", "fine", "coarse", "coarse+fine", "fine_only"],
        help=(
            "Optional policy-independent start filtering using VI graphs + swept collision: "
            "none: no feasibility filtering; "
            "fine: require feasible path on fine(level6) VI; "
            "coarse: require feasible path on coarse(level2) VI; "
            "coarse+fine: require both; "
            "fine_only: require fine feasible but coarse infeasible."
        ),
    )
    p.add_argument(
        "--start-max-attempts",
        type=int,
        default=10_000,
        help="Max random samples per output start before failing (includes feasibility filtering).",
    )
    p.add_argument(
        "--start-feasible-max-steps",
        type=int,
        default=None,
        help="Max steps for the feasibility check (default: 250).",
    )
    p.add_argument(
        "--start-feasible-max-children-per-step",
        type=int,
        default=0,
        help="Optional cap on how many children (ranked by VI value) to try per step during feasibility check (0=all).",
    )
    p.add_argument(
        "--start-feasible-inflation-delta",
        type=float,
        default=None,
        help=(
            "Optional override for feasibility planning semantics: use VI graphs built on obstacles inflated by this delta (meters). "
            "0 means non-inflated (original) graphs. If set, this overrides the feasibility inflation delta implied by --opt-a/--opt-a-delta."
        ),
    )

    # Opt-A: obstacle inflation planning semantics (cached, pluggable).
    p.add_argument("--opt-a", action="store_true", help="Enable Opt-A (inflated obstacles for occupancy/value/gpath; defaults to inflated fine sampling).")
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
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    goals_path = Path(args.goals)
    goals_payload = load_json(goals_path)
    goals: list[list[float]] = goals_payload["goals"]
    goal_idx = int(args.goal_index)
    goal_xyz = goals[goal_idx]
    if len(goal_xyz) != 3:
        raise ValueError(f"Expected goal pose [x,y,yaw_deg], got: {goal_xyz}")
    goal_xy = np.asarray(goal_xyz[:2], dtype=np.float64).reshape(2)

    goal_dir = goals_path.parent / f"goal_{goal_idx}"

    task_path = Path(goals_payload["task_path"])
    if not task_path.is_absolute():
        task_path = (Path.cwd() / task_path).resolve()
    task = load_task(task_path)

    xmin, xmax, ymin, ymax = get_range(task)
    angle_scalor = angle_scalor_from_range(xmin, xmax)

    mode = str(getattr(args, "start_feasible_mode", "none")).strip().lower()
    if mode not in ("none", "fine", "coarse", "coarse+fine", "fine_only"):
        raise ValueError(f"Invalid --start-feasible-mode: {mode}")

    need_coarse = mode in ("coarse", "coarse+fine", "fine_only")
    need_fine = mode in ("fine", "coarse+fine", "fine_only")

    # Optional inflation override for the feasibility planning graphs.
    feasible_delta_override = args.start_feasible_inflation_delta
    if feasible_delta_override is not None and bool(getattr(args, "opt_a", False)):
        try:
            if abs(float(feasible_delta_override) - float(args.opt_a_delta)) > 1e-12:
                print(
                    "[sample_starts] NOTE: --start-feasible-inflation-delta overrides --opt-a-delta for feasibility planning.",
                    flush=True,
                )
        except Exception:
            pass

    # Effective feasibility inflation delta:
    #   - explicit override wins
    #   - else: follow --opt-a/--opt-a-delta if enabled
    #   - else: 0 (original graphs)
    if feasible_delta_override is not None:
        feasible_delta = float(feasible_delta_override)
    elif bool(getattr(args, "opt_a", False)):
        feasible_delta = float(args.opt_a_delta)
    else:
        feasible_delta = 0.0
    if feasible_delta < 0:
        raise ValueError(f"--start-feasible-inflation-delta must be >=0, got {feasible_delta}")
    # Compatibility: treat delta==0 as "no inflation" (load original VI assets).
    use_feasible_inflation = bool(feasible_delta > 0.0)

    task_obs: dict[str, Any] | None = None
    coarse_robot: Any | None = None
    fine_robot: Any | None = None
    collision_robot: Any | None = None
    start_inflated_robot: Any | None = None

    map_name = str(goals_payload.get("map_name", goals_path.parent.name))

    collision_check_mode = str(getattr(args, "opt_a_collision_check", "real"))
    if not task.get("robots"):
        raise RuntimeError("Task has no robots[] entry.")

    # collision check uses real obstacles by default (matches inference sampling); can be switched to "inflated"
    # when feasible_delta>0.
    collision_robot = Unicycle(task["env"], task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)

    if use_feasible_inflation:
        if mode != "none":
            coarse_meta = json.loads((goal_dir / "meta_coarse.json").read_text(encoding="utf-8"))
            fine_meta = json.loads((goal_dir / "meta_fine.json").read_text(encoding="utf-8"))
            inflated = prepare_inflated_goal_assets(
                base_task=task,
                base_task_path=task_path,
                map_name=map_name,
                goal_idx=int(goal_idx),
                goal_xyz=goal_xyz,
                cache_root=Path(args.opt_a_cache_dir),
                delta=float(feasible_delta),
                coarse_meta_src=coarse_meta,
                fine_meta_src=(fine_meta if bool(need_fine) else None),
                vi_device=(None if args.opt_a_vi_device is None else str(args.opt_a_vi_device)),
                vi_dtype=(None if args.opt_a_vi_dtype is None else str(args.opt_a_vi_dtype)),
                max_iters=(None if args.opt_a_vi_max_iters is None else int(args.opt_a_vi_max_iters)),
                tol=(None if args.opt_a_vi_tol is None else float(args.opt_a_vi_tol)),
                overwrite=bool(args.opt_a_overwrite_cache),
                keep_pkl=True,
                use_inflated_fine=bool(need_fine),
            )
            task_obs = inflated.task_obs
            coarse_robot = inflated.coarse_robot if need_coarse else None
            fine_robot = inflated.fine_robot if need_fine else None
        else:
            # No feasibility planning requested; only inflate obstacles for start obstacle_free checks.
            task_obs = inflate_task_obstacles(task, delta=float(feasible_delta))

        if task_obs is None or not task_obs.get("robots"):
            raise RuntimeError("Inflated task is missing robots[] entry.")

        if collision_check_mode == "inflated":
            collision_robot = Unicycle(task_obs["env"], task_obs["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)
        start_inflated_robot = collision_robot if collision_check_mode == "inflated" else Unicycle(task_obs["env"], task_obs["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)
    else:
        # Original (non-inflated) VI graphs.
        if need_coarse:
            coarse_robot = load_vi_robot(_find_vi_robot(goal_dir, "coarse"))
        if need_fine:
            fine_robot = load_vi_robot(_find_vi_robot(goal_dir, "fine"))

    if collision_robot is None:
        raise RuntimeError("Internal error: collision_robot is None.")
    if need_coarse and coarse_robot is None:
        raise RuntimeError("Internal error: coarse_robot is None while coarse feasibility is requested.")
    if need_fine and fine_robot is None:
        raise RuntimeError("Internal error: fine_robot is None while fine feasibility is requested.")

    seed = int(args.seed)
    n_starts = int(args.n_starts)
    if n_starts <= 0:
        raise ValueError("--n-starts must be positive.")

    mode_tag = mode.replace("+", "plus")
    out_path = (Path(args.out) if args.out is not None else (goal_dir / f"starts_seed{seed}_{mode_tag}.json")).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out_path}")

    rng = np.random.default_rng(int(seed) * 100000 + int(goal_idx))
    feasible_max_steps = 250 if args.start_feasible_max_steps is None else int(args.start_feasible_max_steps)
    max_children_per_step = int(getattr(args, "start_feasible_max_children_per_step", 0))

    starts: list[list[float]] = []
    attempts_per_start: list[int] = []
    for si in range(n_starts):
        for attempt in range(int(getattr(args, "start_max_attempts", 10_000))):
            x = float(rng.uniform(xmin + float(args.clearance), xmax - float(args.clearance)))
            y = float(rng.uniform(ymin + float(args.clearance), ymax - float(args.clearance)))
            if float(np.hypot(x - float(goal_xy[0]), y - float(goal_xy[1]))) < float(args.min_goal_dist):
                continue
            yaw = float(rng.uniform(-np.pi, np.pi))
            start_scaled = np.array([x, y, theta_scaled_from_yaw(yaw, float(angle_scalor))], dtype=np.float32)

            if start_inflated_robot is not None:
                if not bool(getattr(start_inflated_robot, "obstacle_free")(start_scaled)):
                    continue
                if (fine_robot is not None) and hasattr(fine_robot, "obstacle_free"):
                    # If feasible planning uses an inflated fine graph, also ensure the start is free under that graph's semantics.
                    if not bool(getattr(fine_robot, "obstacle_free")(start_scaled)):
                        continue
            else:
                if not bool(getattr(collision_robot, "obstacle_free")(start_scaled)):
                    continue

            if mode and mode != "none":
                coarse_ok = None
                fine_ok = None

                def _coarse_ok() -> bool:
                    nonlocal coarse_ok
                    if coarse_ok is None:
                        if coarse_robot is None:
                            raise RuntimeError("Internal error: coarse_robot is None.")
                        coarse_ok = _vi_has_feasible_path(
                            robot=coarse_robot,
                            collision_robot=collision_robot,
                            collision_task=task,
                            start_scaled=start_scaled,
                            angle_scalor=float(angle_scalor),
                            max_steps=int(feasible_max_steps),
                            collision_check_step=float(args.collision_check_step),
                            max_children_per_step=int(max_children_per_step),
                            allow_self_candidate=bool(args.allow_self_candidate),
                        )
                    return bool(coarse_ok)

                def _fine_ok() -> bool:
                    nonlocal fine_ok
                    if fine_ok is None:
                        if fine_robot is None:
                            raise RuntimeError("Internal error: fine_robot is None.")
                        fine_ok = _vi_has_feasible_path(
                            robot=fine_robot,
                            collision_robot=collision_robot,
                            collision_task=task,
                            start_scaled=start_scaled,
                            angle_scalor=float(angle_scalor),
                            max_steps=int(feasible_max_steps),
                            collision_check_step=float(args.collision_check_step),
                            max_children_per_step=int(max_children_per_step),
                            allow_self_candidate=bool(args.allow_self_candidate),
                        )
                    return bool(fine_ok)

                ok = False
                if mode == "fine":
                    ok = _fine_ok()
                elif mode == "coarse":
                    ok = _coarse_ok()
                elif mode == "coarse+fine":
                    ok = _coarse_ok() and _fine_ok()
                elif mode == "fine_only":
                    ok = (not _coarse_ok()) and _fine_ok()
                else:
                    raise ValueError(f"Invalid --start-feasible-mode: {mode}")

                if not ok:
                    continue

            starts.append([float(x), float(y), float(yaw)])
            attempts_per_start.append(int(attempt) + 1)
            print(
                f"[sample_starts] map={str(goals_payload.get('map_name', goals_path.parent.name))} goal={goal_idx} "
                f"success={len(starts)}/{n_starts} attempt={int(attempt) + 1}",
                flush=True,
            )
            break
        else:
            raise RuntimeError(
                f"Failed to sample start {si + 1}/{n_starts} within {int(getattr(args, 'start_max_attempts', 0))} attempts "
                f"(try increasing --start-max-attempts, relaxing --min-goal-dist, or reducing --clearance)."
            )

    payload = {
        "version": 1,
        "map_name": str(goals_payload.get("map_name", goals_path.parent.name)),
        "goals_path": str(goals_path.resolve()),
        "goal_index": int(goal_idx),
        "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
        "task_path": str(task_path),
        "seed": int(seed),
        "n_starts": int(n_starts),
        "starts_format": "[x, y, yaw_rad]",
        "constraints": {"clearance": float(args.clearance), "min_goal_dist": float(args.min_goal_dist)},
        "collision": {"check_step": float(args.collision_check_step), "allow_self_candidate": bool(args.allow_self_candidate)},
        "feasible": {
            "mode": str(mode),
            "max_steps": int(feasible_max_steps),
            "max_children_per_step": int(max_children_per_step),
            "inflation_delta": float(feasible_delta),
            "inflation_cache_dir": (None if not bool(use_feasible_inflation) else str(Path(args.opt_a_cache_dir))),
        },
        "opt_a": {
            "enabled": bool(getattr(args, "opt_a", False)),
            "delta": (None if not bool(getattr(args, "opt_a", False)) else float(args.opt_a_delta)),
            "cache_dir": (None if not bool(getattr(args, "opt_a", False)) else str(args.opt_a_cache_dir)),
            "use_inflated_fine": (None if not bool(getattr(args, "opt_a", False)) else bool(args.opt_a_use_inflated_fine)),
            "collision_check": (None if not bool(getattr(args, "opt_a", False)) else str(getattr(args, "opt_a_collision_check", "real"))),
            "overwrite_cache": (None if not bool(getattr(args, "opt_a", False)) else bool(args.opt_a_overwrite_cache)),
        },
        "attempts_per_start": attempts_per_start,
        "starts": starts,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sample_starts] wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
