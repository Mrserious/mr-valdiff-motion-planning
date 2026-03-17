from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.crop_coverage import validate_crop_covers_children
from unicycle_value_guided.inflation import prepare_inflated_goal_assets
from unicycle_value_guided.global_ref import GlobalRefConfig, compute_gpath_from_vi
from unicycle_value_guided.observe_valuewin12ch import make_local_valuewin12ch
from unicycle_value_guided.rollout_fine import rollout_greedy
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.se2 import angle_scalor_from_range, yaw_from_theta_scaled
from unicycle_value_guided.se2 import theta_scaled_from_yaw
from unicycle_value_guided.task_io import get_range, load_json, load_task
from unicycle_value_guided.vi_io import load_vi_robot
from unicycle_value_guided.value_grid3d import load_regular_value_grid_3d
from unicycle_value_guided.zarr_writer_unicycle import ZarrDatasetWriterUnicycle, ZarrWriterConfig


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


def _ensure_float32_255(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.min() < -1e-3 or img.max() > 255.0 + 1e-3:
        raise ValueError(f"img out of [0,255] range: min={img.min()} max={img.max()}")
    return img


def _stable_hash32(text: str) -> int:
    b = hashlib.sha1(str(text).encode("utf-8")).digest()
    return int.from_bytes(b[:4], byteorder="little", signed=False)


def _mix_seed_per_map(seed_base_u32: int, map_name: str) -> int:
    return (int(seed_base_u32) + _stable_hash32(map_name)) & 0xFFFFFFFF


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt03: build unicycle value-guided dataset (fine rollout, coarse obs).")
    p.add_argument("--goals", type=str, required=True, help="Path to data/unicycle_value_grids/<map>/goals.json")
    p.add_argument("--out", type=str, required=True, help="Output zarr path, e.g. data/zarr/unicycle_value.zarr")
    p.add_argument("--goal-indices", type=str, default="", help='Subset of goals, e.g. "0-9,20". Empty=all.')
    p.add_argument("--starts-per-goal", type=int, default=50, help="Successful episodes per goal (MVP: 20-50).")
    p.add_argument("--max-attempts-per-goal", type=int, default=2000, help="Max start samples per goal.")
    p.add_argument("--seed", type=int, default=None, help="Override seed base (default: use goals.json seed)")
    p.add_argument(
        "--seed-mode",
        type=str,
        default="per_map",
        choices=["global", "per_map"],
        help=(
            "How to apply seed across maps when you run this command in a loop. "
            "per_map mixes the base seed with map_name to avoid starts repeating across maps."
        ),
    )
    p.add_argument("--clearance", type=float, default=0.2, help="Boundary margin when sampling starts (meters).")
    p.add_argument(
        "--min-goal-dist",
        type=float,
        default=None,
        help="Min distance start->goal (xy). Default: auto from (min_steps, dt, max_speed).",
    )
    p.add_argument("--max-steps", type=int, default=250, help="Max steps in fine greedy rollout.")
    p.add_argument("--min-steps", type=int, default=8, help="Drop episodes shorter than this.")
    p.add_argument("--crop-size", type=int, default=84)
    p.add_argument("--mpp", type=float, default=0.05, help="Meters per pixel for local crop.")
    p.add_argument(
        "--rotate-with-yaw",
        action="store_true",
        default=True,
        help="Rotate crop with robot yaw (required for footprint-aware valuewin12ch).",
    )
    p.add_argument("--no-rotate-with-yaw", action="store_false", dest="rotate_with_yaw", help="Disable rotate-with-yaw (not recommended).")
    p.add_argument(
        "--crop-mode",
        type=str,
        default="biased",
        choices=["biased", "centered"],
        help="biased: use --crop-bias-forward-m; centered: force symmetric crop (bias=0).",
    )
    p.add_argument(
        "--crop-bias-forward-m",
        type=float,
        default=0.9375,
        help="Front-biased crop shift in meters along +x (robot frame). Default=1.5*0.625=0.9375.",
    )
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
    p.add_argument("--footprint-length-m", type=float, default=0.625, help="Rectangle footprint length (meters).")
    p.add_argument("--footprint-width-m", type=float, default=0.4375, help="Rectangle footprint width (meters).")
    p.add_argument(
        "--strict-swept-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Super-strict expert mode: reject episodes if any step collides along the continuous trajectory "
            "(same swept collision semantics as infer/eval). Default: enabled."
        ),
    )
    p.add_argument("--collision-check-step", type=float, default=0.05, help="Swept collision check linear step (meters).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output zarr if exists.")
    p.add_argument(
        "--compressor",
        type=str,
        default="none",
        choices=["none", "default", "disk"],
        help="ReplayBuffer compressor preset.",
    )
    p.add_argument("--max-success-episodes", type=int, default=None, help="Optional cap on total episodes written.")

    # Opt-A: obstacle inflation planning semantics (optional; matches infer/eval).
    p.add_argument(
        "--opt-a",
        action="store_true",
        help="Enable Opt-A (inflated obstacles for occupancy/value slices/gpath; defaults to inflated fine rollout).",
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
        help="Use inflated fine VI graph for rollout_greedy (default: enabled for --opt-a).",
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
    task_path = Path(goals_payload["task_path"])
    if not task_path.is_absolute():
        task_path = (Path.cwd() / task_path).resolve()
    task = load_task(task_path)

    map_name = str(goals_payload.get("map_name", goals_path.parent.name))
    seed_mode = str(args.seed_mode).strip().lower()
    if seed_mode not in ("global", "per_map"):
        raise ValueError(f"Invalid --seed-mode: {seed_mode}")

    seed_base = int(args.seed) if args.seed is not None else None
    seed_base_from_file = goals_payload.get("seed_base", None)
    seed_from_file = goals_payload.get("seed", 42)
    seed_mode_from_file = str(goals_payload.get("seed_mode", "")).strip().lower()

    if seed_base is None:
        # Prefer the explicit base seed from goals.json if available.
        if seed_base_from_file is not None:
            seed_base = int(seed_base_from_file)
        else:
            seed_base = int(seed_from_file)

    seed_base_u32 = int(seed_base) & 0xFFFFFFFF
    if seed_mode == "global":
        seed_eff_u32 = seed_base_u32
    else:
        # If goals.json already recorded a per-map effective seed, reuse it to avoid double mixing.
        if args.seed is None and seed_base_from_file is not None and seed_mode_from_file in ("per_map", "auto"):
            seed_eff_u32 = int(seed_from_file) & 0xFFFFFFFF
        else:
            seed_eff_u32 = _mix_seed_per_map(seed_base_u32, map_name)

    goals: list[list[float]] = goals_payload["goals"]
    goal_indices = _parse_index_spec(str(args.goal_indices), n=len(goals))

    out_path = Path(args.out)
    writer = ZarrDatasetWriterUnicycle(
        out_path,
        overwrite=bool(args.overwrite),
        cfg=ZarrWriterConfig(compressor=str(args.compressor)),
    )

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

    written = 0
    for goal_idx in goal_indices:
        goal_xyz = goals[goal_idx]
        if len(goal_xyz) != 3:
            raise ValueError(f"Expected goal pose [x,y,yaw_deg], got: {goal_xyz}")
        goal_xy = np.asarray(goal_xyz[:2], dtype=np.float64).reshape(2)
        goal_yaw = float(goal_xyz[2]) / 180.0 * float(np.pi)

        goal_dir = goals_path.parent / f"goal_{goal_idx}"
        task_obs = task
        coarse_grid3d = load_regular_value_grid_3d(goal_dir / "value_coarse_3d.npy", goal_dir / "meta_coarse_3d.json")
        coarse_robot = load_vi_robot(_find_vi_robot(goal_dir, "coarse"))
        fine_robot = load_vi_robot(_find_vi_robot(goal_dir, "fine"))

        if bool(args.opt_a):
            coarse_meta = json.loads((goal_dir / "meta_coarse.json").read_text(encoding="utf-8"))
            fine_meta = json.loads((goal_dir / "meta_fine.json").read_text(encoding="utf-8"))

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

            print(
                f"[make_dataset] opt-a enabled: map={map_name} goal={int(goal_idx)} delta={float(args.opt_a_delta):.3f} "
                f"cache={inflated.goal_dir} use_inflated_fine={bool(args.opt_a_use_inflated_fine)}",
                flush=True,
            )

        validate_crop_covers_children(
            fine_robot=fine_robot,
            crop_size=int(args.crop_size),
            meters_per_pixel=float(args.mpp),
            crop_bias_forward_m=float(args.crop_bias_forward_m),
            strict=bool(getattr(args, "require_crop_covers_children", False)),
            extra_margin_m=0.0,
            context="make_dataset",
        )

        # Collision checker for strict expert mode: use task (real) obstacle set.
        collision_robot = None
        if bool(getattr(args, "strict_swept_collision", False)):
            if not task.get("robots"):
                raise RuntimeError("Task has no robots[] entry.")
            collision_robot = Unicycle(task["env"], task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)

        dt = float(getattr(fine_robot, "get_temporal_res")())
        lims = np.asarray(getattr(fine_robot, "control_limits", [[-1, 1], [-1, 1]]), dtype=np.float32).reshape(2, 2)
        max_speed = float(max(abs(float(lims[0, 0])), abs(float(lims[0, 1]))))
        auto_min_goal_dist = max(1.0, float(args.min_steps) * max_speed * dt * 0.8)
        min_goal_dist = float(args.min_goal_dist) if args.min_goal_dist is not None else auto_min_goal_dist

        rng = np.random.default_rng(np.asarray([seed_eff_u32, int(goal_idx)], dtype=np.uint32))
        successes = 0
        attempts = 0
        gcfg = GlobalRefConfig()

        while successes < int(args.starts_per_goal) and attempts < int(args.max_attempts_per_goal):
            attempts += 1
            x = float(rng.uniform(xmin + float(args.clearance), xmax - float(args.clearance)))
            y = float(rng.uniform(ymin + float(args.clearance), ymax - float(args.clearance)))
            if float(np.hypot(x - float(goal_xy[0]), y - float(goal_xy[1]))) < min_goal_dist:
                continue
            yaw = float(rng.uniform(-np.pi, np.pi))
            start_scaled = np.array([x, y, theta_scaled_from_yaw(yaw, float(angle_scalor))], dtype=np.float32)
            if not bool(getattr(fine_robot, "obstacle_free")(start_scaled)):
                continue

            res = rollout_greedy(
                robot=fine_robot,
                start_state_scaled=start_scaled,
                max_steps=int(args.max_steps),
                angle_scalor=float(angle_scalor),
            )
            if not res.success:
                continue
            if res.actions_v_omega.shape[0] < int(args.min_steps):
                continue

            if bool(getattr(args, "strict_swept_collision", False)):
                assert collision_robot is not None
                ok = True
                T_check = int(res.actions_v_omega.shape[0])
                for t in range(T_check):
                    if not trajectory_collision_free(
                        robot=collision_robot,
                        task=task,
                        state_scaled=res.states_scaled[t],
                        action_v_omega=res.actions_v_omega[t],
                        dt=float(dt),
                        angle_scalor=float(angle_scalor),
                        step_size=float(args.collision_check_step),
                    ):
                        ok = False
                        break
                if not ok:
                    continue

            T = int(res.actions_v_omega.shape[0])
            state = np.zeros((T, 3), dtype=np.float32)
            state[:, 0:2] = res.states_scaled[:, 0:2]
            for t in range(T):
                state[t, 2] = float(yaw_from_theta_scaled(float(res.states_scaled[t, 2]), float(angle_scalor)))

            action = res.actions_v_omega.astype(np.float32, copy=False)

            if not bool(args.rotate_with_yaw):
                raise ValueError("valuewin12ch requires --rotate-with-yaw (local axes must align with robot frame).")

            img_list: list[np.ndarray] = []
            for t in range(T):
                img_list.append(
                    make_local_valuewin12ch(
                        task=task_obs,
                        coarse_grid3d=coarse_grid3d,
                        state=state[t],
                        crop_size=int(args.crop_size),
                        meters_per_pixel=float(args.mpp),
                        rotate_with_yaw=True,
                        crop_bias_forward_m=float(args.crop_bias_forward_m),
                        yaw_offsets_deg=yaw_offsets_deg,
                        footprint_length_m=float(args.footprint_length_m),
                        footprint_width_m=float(args.footprint_width_m),
                        scale_255=True,
                    )
                )
            img = np.stack(img_list, axis=0)
            img = _ensure_float32_255(img)

            gpath = np.zeros((T, 4, 3), dtype=np.float32)
            for t in range(T):
                gpath[t] = compute_gpath_from_vi(
                    coarse_robot=coarse_robot,
                    state=state[t],
                    goal_xy=goal_xy,
                    goal_yaw=goal_yaw,
                    angle_scalor=float(angle_scalor),
                    cfg=gcfg,
                )

            writer.add_episode(img=img, state=state, action=action, gpath=gpath)
            written += 1
            successes += 1
            print(
                f"[make_dataset] map={map_name} goal={goal_idx} "
                f"success={successes}/{int(args.starts_per_goal)} attempt={attempts} T={T} -> wrote episode {written}",
                flush=True,
            )

            if args.max_success_episodes is not None and written >= int(args.max_success_episodes):
                print("[make_dataset] reached --max-success-episodes, stopping.", flush=True)
                return

        if successes < int(args.starts_per_goal):
            print(
                f"[make_dataset] map={map_name} goal={goal_idx}: only {successes}/{int(args.starts_per_goal)} successes "
                f"after {attempts} attempts (min_goal_dist={min_goal_dist:.3f}).",
                flush=True,
            )


if __name__ == "__main__":
    main()
