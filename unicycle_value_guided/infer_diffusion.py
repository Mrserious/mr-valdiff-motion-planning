from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import dill
import numpy as np
import torch

import hydra
from omegaconf import OmegaConf

from value_guided.observe_coarse import load_regular_value_grid

from unicycle_value_cuda.unicycle_value_cuda.unicycle import Unicycle

from unicycle_value_guided.crop_coverage import validate_crop_covers_children
from unicycle_value_guided.inflation import prepare_inflated_goal_assets
from unicycle_value_guided.global_ref import GlobalRefConfig, compute_gpath_from_vi
from unicycle_value_guided.observe_valuewin12ch import make_local_valuewin12ch
from unicycle_value_guided.rollout_fine import reconstruct_action_from_transition
from unicycle_value_guided.swept_collision import trajectory_collision_free
from unicycle_value_guided.se2 import (
    angle_scalor_from_range,
    theta_scaled_from_yaw,
    wrap_theta_scaled,
    yaw_from_theta_scaled,
)
from unicycle_value_guided.task_io import get_range, load_json, load_task
from unicycle_value_guided.vi_io import load_vi_robot
from unicycle_value_guided.value_grid3d import load_regular_value_grid_3d


def _policy_device(policy: torch.nn.Module) -> torch.device:
    d = getattr(policy, "device", None)
    if isinstance(d, torch.device):
        return d
    if isinstance(d, str):
        return torch.device(d)
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    try:
        n = float(n)
    except Exception:
        return "n/a"
    if not np.isfinite(n) or n < 0:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    k = 0
    while n >= 1024.0 and k < len(units) - 1:
        n /= 1024.0
        k += 1
    return f"{n:.2f}{units[k]}"


def _rss_bytes() -> int | None:
    # Linux-friendly: /proc is the most reliable without extra deps.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _log_mem(tag: str, *, device: torch.device) -> None:
    rss = _rss_bytes()
    msg = f"[mem] {tag} rss={_fmt_bytes(rss)}"
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            idx = int(device.index or 0)
            alloc = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            peak = torch.cuda.max_memory_reserved(idx)
            msg += f" cuda_alloc={_fmt_bytes(alloc)} cuda_reserved={_fmt_bytes(reserved)} cuda_peak_reserved={_fmt_bytes(peak)}"
        except Exception:
            pass
    print(msg, flush=True)


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


def _integrate_unicycle_scaled(
    *,
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
) -> np.ndarray:
    s = np.asarray(state_scaled, dtype=np.float64).reshape(3)
    v = float(action_v_omega[0])
    omega_rad = float(action_v_omega[1])
    yaw = float(yaw_from_theta_scaled(float(s[2]), float(angle_scalor)))
    x = float(s[0] + math.cos(yaw) * v * float(dt))
    y = float(s[1] + math.sin(yaw) * v * float(dt))
    omega_scaled = float(omega_rad / math.pi * float(angle_scalor))
    th = wrap_theta_scaled(float(s[2] + omega_scaled * float(dt)), float(angle_scalor))
    return np.array([x, y, th], dtype=np.float32)


def _integrate_unicycle_scaled_swept(
    *,
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
    step_size: float,
) -> np.ndarray:
    """
    Integrate one constant-control step using the same sub-stepping scheme as swept collision checking.

    This does *not* perform collision checks; it only computes the end state.
    """
    s = np.asarray(state_scaled, dtype=np.float64).reshape(3)
    a = np.asarray(action_v_omega, dtype=np.float64).reshape(2)
    v = float(a[0])
    omega = float(a[1])  # rad/s
    yaw = float(yaw_from_theta_scaled(float(s[2]), float(angle_scalor)))

    total_lin = abs(v) * float(dt)
    total_ang = abs(omega) * float(dt)
    n_lin = int(math.ceil(total_lin / max(float(step_size), 1e-6))) if total_lin > 0 else 1
    n_ang = int(math.ceil(total_ang / (math.pi / 36.0))) if total_ang > 0 else 1  # ~5deg per step
    n = max(2, n_lin, n_ang)
    dt_sub = float(dt) / float(n)

    x = float(s[0])
    y = float(s[1])
    th = float(s[2])
    for _ in range(n):
        x = float(x + math.cos(yaw) * v * dt_sub)
        y = float(y + math.sin(yaw) * v * dt_sub)
        yaw = float(yaw + omega * dt_sub)
        th = float(theta_scaled_from_yaw(yaw, float(angle_scalor)))
        th = wrap_theta_scaled(th, float(angle_scalor))
    return np.array([x, y, th], dtype=np.float32)


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


def _distance_children_to_hat(
    *,
    node_states: np.ndarray,
    child_indices: Sequence[int],
    hat_state_scaled: np.ndarray,
    angle_scalor: float,
) -> np.ndarray:
    """
    Returns distances (len(child_indices),) float64.
    """
    if len(child_indices) == 0:
        return np.zeros((0,), dtype=np.float64)
    hat = np.asarray(hat_state_scaled, dtype=np.float64).reshape(3)
    idx = np.asarray(list(int(i) for i in child_indices), dtype=np.int64)
    s = node_states[idx].astype(np.float64, copy=False)
    dx = s[:, 0] - hat[0]
    dy = s[:, 1] - hat[1]
    dth = np.abs(s[:, 2] - hat[2])
    p = 2.0 * float(angle_scalor)
    dth = np.minimum(dth, p - dth)
    return np.sqrt(dx * dx + dy * dy + dth * dth)


def _rank_children_by_hat(
    *,
    node_states: np.ndarray,
    children: Sequence[int],
    hat_state_scaled: np.ndarray,
    angle_scalor: float,
) -> list[int]:
    if not children:
        return []
    d = _distance_children_to_hat(node_states=node_states, child_indices=children, hat_state_scaled=hat_state_scaled, angle_scalor=float(angle_scalor))
    order = np.argsort(d, kind="stable")
    return [int(children[int(i)]) for i in order]


def _xy_key(x: float, y: float, *, q: float) -> tuple[int, int]:
    qq = float(q)
    if not (np.isfinite(qq) and qq > 0):
        raise ValueError(f"Invalid anti-repeat quantization q: {q}")
    return (int(round(float(x) / qq)), int(round(float(y) / qq)))


def _anti_push_xy(state: _AntiRepeatState, key: tuple[int, int], *, max_n: int) -> None:
    n = int(max_n)
    if n <= 0:
        return
    state.xy_recent.append(key)
    state.xy_counts[key] += 1
    while len(state.xy_recent) > n:
        k0 = state.xy_recent.popleft()
        state.xy_counts[k0] -= 1
        if state.xy_counts[k0] <= 0:
            del state.xy_counts[k0]


def _anti_push_child(state: _AntiRepeatState, idx_next: int, *, max_n: int) -> None:
    n = int(max_n)
    if n <= 0:
        return
    idx_next = int(idx_next)
    state.child_recent.append(idx_next)
    state.child_counts[idx_next] += 1
    while len(state.child_recent) > n:
        i0 = int(state.child_recent.popleft())
        state.child_counts[i0] -= 1
        if state.child_counts[i0] <= 0:
            del state.child_counts[i0]


def _anti_push_edge(state: _AntiRepeatState, edge: tuple[int, int], *, max_n: int) -> None:
    n = int(max_n)
    if n <= 0:
        return
    e = (int(edge[0]), int(edge[1]))
    state.edge_recent.append(e)
    state.edge_counts[e] += 1
    while len(state.edge_recent) > n:
        e0 = state.edge_recent.popleft()
        state.edge_counts[e0] -= 1
        if state.edge_counts[e0] <= 0:
            del state.edge_counts[e0]


def _pick_projected_action(
    *,
    ranked_children: Sequence[int],
    cur_state_scaled: np.ndarray,
    fine_node_states: np.ndarray,
    dt: float,
    angle_scalor: float,
    control_limits_scaled: Sequence[Sequence[float]],
    collision_robot: Any,
    collision_task: dict[str, Any],
    collision_check_step: float,
    collision_semantic: str,
    projected_collision_stage: str,
    idx_cur: int,
    anti_repeat: AntiRepeatConfig | None,
    anti_state: _AntiRepeatState | None,
) -> tuple[int, np.ndarray] | None:
    if not ranked_children:
        return None

    stage = str(projected_collision_stage).strip().lower()
    if stage not in ("pre", "post"):
        raise ValueError(f"Invalid projected collision stage: {projected_collision_stage!r} (expected: 'pre' or 'post')")
    precheck_collision = bool(stage == "pre")

    semantic = str(collision_semantic).strip().lower()
    if semantic not in ("swept", "discrete"):
        raise ValueError(f"Invalid collision semantic: {collision_semantic!r} (expected: 'swept' or 'discrete')")

    xmin = xmax = ymin = ymax = None
    if precheck_collision and semantic == "discrete":
        xmin, xmax, ymin, ymax = get_range(collision_task)

    if anti_repeat is None or (not bool(getattr(anti_repeat, "enabled", False))):
        for cand_idx in ranked_children:
            cand_state = fine_node_states[int(cand_idx)].astype(np.float32, copy=False)
            a = reconstruct_action_from_transition(
                cur_state_scaled=cur_state_scaled,
                nxt_state_scaled=cand_state,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
            )
            a = _clip_action_v_omega(a, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))
            if precheck_collision:
                if semantic == "swept":
                    if not _trajectory_collision_free(
                        robot=collision_robot,
                        task=collision_task,
                        state_scaled=cur_state_scaled,
                        action_v_omega=a,
                        dt=float(dt),
                        angle_scalor=float(angle_scalor),
                        step_size=float(collision_check_step),
                    ):
                        continue
                else:
                    assert xmin is not None and xmax is not None and ymin is not None and ymax is not None
                    s_end = _integrate_unicycle_scaled_swept(
                        state_scaled=cur_state_scaled,
                        action_v_omega=a,
                        dt=float(dt),
                        angle_scalor=float(angle_scalor),
                        step_size=float(collision_check_step),
                    )
                    x = float(s_end[0])
                    y = float(s_end[1])
                    if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
                        continue
                    if not bool(getattr(collision_robot, "obstacle_free")(np.asarray(s_end, dtype=np.float32).reshape(3))):
                        continue
            return (int(cand_idx), np.asarray(a, dtype=np.float32).reshape(2))
        return None

    if anti_state is None:
        raise RuntimeError("anti_repeat enabled but anti_state is None")

    best_key: tuple[bool, bool, bool, bool, int] | None = None
    best: tuple[int, np.ndarray] | None = None

    for rank_i, cand_idx in enumerate(ranked_children):
        cand_idx = int(cand_idx)
        cand_state = fine_node_states[cand_idx].astype(np.float32, copy=False)
        a = reconstruct_action_from_transition(
            cur_state_scaled=cur_state_scaled,
            nxt_state_scaled=cand_state,
            dt=float(dt),
            angle_scalor=float(angle_scalor),
        )
        a = _clip_action_v_omega(a, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))
        s_pred: np.ndarray | None = None
        if precheck_collision:
            if semantic == "swept":
                if not _trajectory_collision_free(
                    robot=collision_robot,
                    task=collision_task,
                    state_scaled=cur_state_scaled,
                    action_v_omega=a,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                ):
                    continue
            else:
                assert xmin is not None and xmax is not None and ymin is not None and ymax is not None
                s_pred = _integrate_unicycle_scaled_swept(
                    state_scaled=cur_state_scaled,
                    action_v_omega=a,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                )
                x = float(s_pred[0])
                y = float(s_pred[1])
                if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
                    continue
                if not bool(getattr(collision_robot, "obstacle_free")(np.asarray(s_pred, dtype=np.float32).reshape(3))):
                    continue

        is_recent_xy = False
        if int(anti_repeat.xy_recent_n) > 0:
            if s_pred is None:
                s_pred = _integrate_unicycle_scaled_swept(
                    state_scaled=cur_state_scaled,
                    action_v_omega=a,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                )
            k = _xy_key(float(s_pred[0]), float(s_pred[1]), q=float(anti_repeat.xy_q))
            is_recent_xy = bool(anti_state.xy_counts.get(k, 0) > 0)

        is_back_edge = False
        if bool(anti_repeat.avoid_uturn) and anti_state.prev_edge is not None:
            prev_cur, prev_next = anti_state.prev_edge
            if int(idx_cur) == int(prev_next) and int(cand_idx) == int(prev_cur):
                is_back_edge = True

        is_recent_edge = False
        if int(anti_repeat.edge_recent_n) > 0:
            is_recent_edge = bool(anti_state.edge_counts.get((int(idx_cur), int(cand_idx)), 0) > 0)

        is_recent_child = False
        if int(anti_repeat.child_recent_n) > 0:
            is_recent_child = bool(anti_state.child_counts.get(int(cand_idx), 0) > 0)

        key = (bool(is_recent_xy), bool(is_back_edge), bool(is_recent_edge), bool(is_recent_child), int(rank_i))
        if best_key is None or key < best_key:
            best_key = key
            best = (int(cand_idx), np.asarray(a, dtype=np.float32).reshape(2))
            if key[:4] == (False, False, False, False):
                break

    return best


def _postcheck_projected_action(
    *,
    collision_robot: Any,
    collision_task: dict[str, Any],
    cur_state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
    collision_check_step: float,
    collision_semantic: str,
) -> tuple[bool, str | None, np.ndarray]:
    """
    Execute-stage collision check for a selected projected action.

    Returns:
      (ok, fail_detail, s_end)

    where fail_detail is one of {"collision","out_of_bounds"} when ok=False, else None.
    """
    semantic = str(collision_semantic).strip().lower()
    if semantic not in ("swept", "discrete"):
        raise ValueError(f"Invalid collision semantic: {collision_semantic!r} (expected: 'swept' or 'discrete')")

    s_end = _integrate_unicycle_scaled_swept(
        state_scaled=cur_state_scaled,
        action_v_omega=action_v_omega,
        dt=float(dt),
        angle_scalor=float(angle_scalor),
        step_size=float(collision_check_step),
    )

    if semantic == "swept":
        ok = _trajectory_collision_free(
            robot=collision_robot,
            task=collision_task,
            state_scaled=cur_state_scaled,
            action_v_omega=action_v_omega,
            dt=float(dt),
            angle_scalor=float(angle_scalor),
            step_size=float(collision_check_step),
        )
        if ok:
            return True, None, s_end
        xmin, xmax, ymin, ymax = get_range(collision_task)
        x = float(s_end[0])
        y = float(s_end[1])
        fail_detail = "out_of_bounds" if (x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax)) else "collision"
        return False, str(fail_detail), s_end

    xmin, xmax, ymin, ymax = get_range(collision_task)
    x = float(s_end[0])
    y = float(s_end[1])
    if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
        return False, "out_of_bounds", s_end
    if not bool(getattr(collision_robot, "obstacle_free")(np.asarray(s_end, dtype=np.float32).reshape(3))):
        return False, "collision", s_end
    return True, None, s_end


def _trajectory_collision_free(
    *,
    robot: Any,
    task: dict[str, Any],
    state_scaled: np.ndarray,
    action_v_omega: np.ndarray,
    dt: float,
    angle_scalor: float,
    step_size: float,
) -> bool:
    return trajectory_collision_free(
        robot=robot,
        task=task,
        state_scaled=state_scaled,
        action_v_omega=action_v_omega,
        dt=float(dt),
        angle_scalor=float(angle_scalor),
        step_size=float(step_size),
    )


def _strip_boundary_wall_obstacles(env: dict[str, Any]) -> dict[str, Any]:
    """
    Return a shallow-copied env with boundary-wall obstacles removed.

    Boundary walls are Opt-A planning-margin artifacts. Since obstacle_free() checks the robot's footprint
    samples, treating boundary walls as regular obstacles can make otherwise-valid (center-in-range) starts
    fail the inflated start check. We identify boundary walls by obstacle["name"] starting with
    "boundary_wall_", which is injected by the Opt-A inflation helper.
    """
    obstacles = env.get("obstacles", [])
    if obstacles is None or not isinstance(obstacles, list):
        return dict(env)
    kept: list[Any] = []
    for o in obstacles:
        name = ""
        if isinstance(o, dict):
            name = str(o.get("name", "") or "")
        if name.startswith("boundary_wall_"):
            continue
        kept.append(o)
    out = dict(env)
    out["obstacles"] = kept
    return out


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


def _load_policy_from_ckpt(
    ckpt_path: Path,
    *,
    device: torch.device,
    use_ema: bool,
) -> torch.nn.Module:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)

    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and ("ema_model" in payload.get("state_dicts", {})) else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.to(device)
    policy.eval()
    return policy


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    reason: str
    states: np.ndarray  # (T+1,3) x,y,yaw_rad
    actions: np.ndarray  # (T,2) policy output (v,omega)
    projected_actions: np.ndarray  # (T,2) snapped edge action (v,omega)
    resample_counts: np.ndarray  # (T,) int64
    fail_detail: str | None = None


@dataclass(frozen=True)
class AntiRepeatConfig:
    enabled: bool
    xy_q: float
    xy_recent_n: int
    child_recent_n: int
    edge_recent_n: int
    avoid_uturn: bool


@dataclass
class _AntiRepeatState:
    xy_recent: deque[tuple[int, int]]
    xy_counts: Counter[tuple[int, int]]
    child_recent: deque[int]
    child_counts: Counter[int]
    edge_recent: deque[tuple[int, int]]
    edge_counts: Counter[tuple[int, int]]
    prev_edge: tuple[int, int] | None


@dataclass(frozen=True)
class ObsAblationConfig:
    """
    Test-time observation ablations ("feature corruption") for quick experiments.

    Important:
      - We keep tensor shapes identical to training (e.g., 23-channel valuewin image).
      - This is NOT equivalent to retraining without those features; it is useful to
        probe feature reliance and sensitivity.
    """

    ablate_value_slices: bool
    value_slices_fill: float
    ablate_occupancy: bool
    occupancy_fill: float
    ablate_gpath: bool
    gpath_fill: float


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _apply_obs_ablation(
    *,
    img_chw: np.ndarray,  # (C,H,W) float in [0,1]
    gpath: np.ndarray,  # (4,3) float
    cfg: ObsAblationConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg is None:
        return img_chw, gpath

    if bool(cfg.ablate_occupancy):
        if img_chw.shape[0] < 1:
            raise ValueError(f"Invalid image channels for occupancy ablation: shape={img_chw.shape}")
        img_chw[0, :, :] = _clamp01(float(cfg.occupancy_fill))

    if bool(cfg.ablate_value_slices):
        if img_chw.shape[0] < 2:
            raise ValueError(f"Invalid image channels for value-slice ablation: shape={img_chw.shape}")
        img_chw[1:, :, :] = _clamp01(float(cfg.value_slices_fill))

    if bool(cfg.ablate_gpath):
        gpath = np.full_like(np.asarray(gpath, dtype=np.float32), float(cfg.gpath_fill), dtype=np.float32)

    return img_chw, gpath


def _run_episode(
    *,
    task: dict[str, Any],
    task_obs: dict[str, Any] | None,
    coarse_grid3d,
    coarse_robot: Any,
    fine_robot: Any,
    fine_node_states: np.ndarray,
    collision_robot: Any | None,
    goal_xy: np.ndarray,
    goal_yaw: float | None,
    angle_scalor: float,
    policy: Any,
    start_state: np.ndarray,  # (3,) x,y,yaw_rad
    execute_mode: str,
    max_steps: int,
    crop_size: int,
    meters_per_pixel: float,
    rotate_with_yaw: bool,
    crop_bias_forward_m: float,
    yaw_offsets_deg: tuple[int, ...],
    footprint_length_m: float,
    footprint_width_m: float,
    max_resample_attempts: int,
    collision_check_step: float,
    collision_semantic: str,
    projected_collision_stage: str,
    allow_self_candidate: bool,
    opt_b_topk_children: int,
    anti_repeat: AntiRepeatConfig | None = None,
    obs_ablation: ObsAblationConfig | None = None,
) -> EpisodeResult:
    if task_obs is None:
        task_obs = task
    if collision_robot is None:
        collision_robot = fine_robot

    dt = float(getattr(fine_robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from fine_robot.get_temporal_res(): {dt}")

    To = int(getattr(policy, "n_obs_steps", 2))
    if To <= 0:
        raise ValueError(f"Invalid policy.n_obs_steps: {To}")

    execute_mode = str(execute_mode).strip().lower()
    if execute_mode not in ("projected", "raw"):
        raise ValueError(f"Invalid execute mode: {execute_mode!r} (expected: 'projected' or 'raw')")

    projected_collision_stage = str(projected_collision_stage).strip().lower()
    if projected_collision_stage not in ("pre", "post"):
        raise ValueError(
            f"Invalid projected collision stage: {projected_collision_stage!r} "
            "(expected: 'pre' or 'post' for --projected-collision-stage)"
        )
    postcheck_collision = bool(execute_mode == "projected" and projected_collision_stage == "post")

    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)

    # NOTE: We do NOT teleport the state onto a graph node.
    # We use the continuous (x,y,yaw) state for observations / collision checking,
    # and use the nearest VI node only as an anchor to fetch candidate children.
    start_state = np.asarray(start_state, dtype=np.float64).reshape(3)
    state = np.array([float(start_state[0]), float(start_state[1]), float(start_state[2])], dtype=np.float32)
    state_scaled = np.array(
        [
            float(state[0]),
            float(state[1]),
            float(wrap_theta_scaled(theta_scaled_from_yaw(float(state[2]), float(angle_scalor)), float(angle_scalor))),
        ],
        dtype=np.float32,
    )
    idx_cur = -1
    if execute_mode == "projected":
        idx_list = getattr(fine_robot, "query_kdtree")(state_scaled)
        if not idx_list:
            return EpisodeResult(
                False,
                "no_nearest_node",
                np.asarray([state], dtype=np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0,), np.int64),
            )
        idx_cur = int(idx_list[0])

    gcfg = GlobalRefConfig()

    img_hist: list[np.ndarray] = []
    state_hist: list[np.ndarray] = []
    gpath_hist: list[np.ndarray] = []

    states: list[np.ndarray] = [state.copy()]
    actions: list[np.ndarray] = []
    projected_actions: list[np.ndarray] = []
    resample_counts: list[int] = []

    control_limits_scaled = getattr(fine_robot, "control_limits", [[-1, 1], [-1, 1]])
    policy_dev = _policy_device(policy)

    anti_state: _AntiRepeatState | None = None
    if anti_repeat is not None and bool(anti_repeat.enabled):
        if not (np.isfinite(float(anti_repeat.xy_q)) and float(anti_repeat.xy_q) > 0):
            raise ValueError(f"Invalid --anti-repeat-xy-q: {anti_repeat.xy_q}")
        anti_state = _AntiRepeatState(
            xy_recent=deque(),
            xy_counts=Counter(),
            child_recent=deque(),
            child_counts=Counter(),
            edge_recent=deque(),
            edge_counts=Counter(),
            prev_edge=None,
        )
        k0 = _xy_key(float(state_scaled[0]), float(state_scaled[1]), q=float(anti_repeat.xy_q))
        _anti_push_xy(anti_state, k0, max_n=int(anti_repeat.xy_recent_n))

    for _step in range(int(max_steps)):
        if not bool(rotate_with_yaw):
            raise ValueError("valuewin12ch requires rotate_with_yaw=True.")

        img_255 = make_local_valuewin12ch(
            task=task_obs,
            coarse_grid3d=coarse_grid3d,
            state=state,
            crop_size=int(crop_size),
            meters_per_pixel=float(meters_per_pixel),
            rotate_with_yaw=True,
            crop_bias_forward_m=float(crop_bias_forward_m),
            yaw_offsets_deg=tuple(int(d) for d in yaw_offsets_deg),
            footprint_length_m=float(footprint_length_m),
            footprint_width_m=float(footprint_width_m),
            scale_255=True,
        )
        img = (img_255 / 255.0).astype(np.float32, copy=False)  # (H,W,C)
        img_chw = np.moveaxis(img, -1, 0)  # (C,H,W)

        gpath = compute_gpath_from_vi(
            coarse_robot=coarse_robot,
            state=state,
            goal_xy=goal_xy,
            goal_yaw=goal_yaw,
            angle_scalor=float(angle_scalor),
            cfg=gcfg,
        )

        img_chw, gpath = _apply_obs_ablation(img_chw=img_chw, gpath=gpath, cfg=obs_ablation)

        img_hist.append(img_chw)
        state_hist.append(state.copy())
        gpath_hist.append(gpath.copy())
        if len(img_hist) == 1:
            while len(img_hist) < To:
                img_hist.insert(0, img_hist[0])
                state_hist.insert(0, state_hist[0])
                gpath_hist.insert(0, gpath_hist[0])
        if len(img_hist) > To:
            img_hist = img_hist[-To:]
            state_hist = state_hist[-To:]
            gpath_hist = gpath_hist[-To:]

        obs_dict = {
            "image": torch.from_numpy(np.stack(img_hist, axis=0)[None, ...]).to(policy_dev),
            "agent_pos": torch.from_numpy(np.stack(state_hist, axis=0)[None, ...]).to(policy_dev),
            "global_path": torch.from_numpy(np.stack(gpath_hist, axis=0)[None, ...]).to(policy_dev),
        }

        if execute_mode == "raw":
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            action_seq = result["action"]  # (1,K,2)
            a0 = action_seq[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            a0 = _clip_action_v_omega(a0, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))

            # Collision check on the raw action (no projection / no resample).
            semantic = str(collision_semantic).strip().lower()
            if semantic not in ("swept", "discrete"):
                raise ValueError(f"Invalid collision semantic: {collision_semantic!r} (expected: 'swept' or 'discrete')")
            xmin, xmax, ymin, ymax = get_range(task)

            # Precompute end state for discrete checks and for state update.
            s_end = _integrate_unicycle_scaled_swept(
                state_scaled=state_scaled,
                action_v_omega=a0,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                step_size=float(collision_check_step),
            )

            ok = True
            if semantic == "swept":
                ok = _trajectory_collision_free(
                    robot=collision_robot,
                    task=task,
                    state_scaled=state_scaled,
                    action_v_omega=a0,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                )
            else:
                x = float(s_end[0])
                y = float(s_end[1])
                if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
                    ok = False
                elif not bool(getattr(collision_robot, "obstacle_free")(np.asarray(s_end, dtype=np.float32).reshape(3))):
                    ok = False

            if not ok:
                # Classify failure best-effort: out_of_bounds if end state is outside range, else collision.
                x = float(s_end[0])
                y = float(s_end[1])
                if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
                    fail_reason = "out_of_bounds"
                else:
                    fail_reason = "collision"
                return EpisodeResult(
                    success=False,
                    reason=str(fail_reason),
                    states=np.asarray(states, dtype=np.float32),
                    actions=np.asarray(actions, dtype=np.float32),
                    projected_actions=np.asarray(projected_actions, dtype=np.float32),
                    resample_counts=np.asarray(resample_counts, dtype=np.int64),
                )

            # accept
            actions.append(a0.astype(np.float32, copy=False))
            projected_actions.append(a0.astype(np.float32, copy=False))
            resample_counts.append(0)

            state_scaled = np.asarray(s_end, dtype=np.float32).reshape(3)
            nxt_yaw = float(yaw_from_theta_scaled(float(state_scaled[2]), float(angle_scalor)))
            state = np.array([float(state_scaled[0]), float(state_scaled[1]), float(nxt_yaw)], dtype=np.float32)
            states.append(state.copy())

            if bool(getattr(fine_robot, "within_goal")(np.asarray(state_scaled, dtype=np.float32).reshape(3))):
                return EpisodeResult(
                    success=True,
                    reason="reached_goal",
                    states=np.asarray(states, dtype=np.float32),
                    actions=np.asarray(actions, dtype=np.float32),
                    projected_actions=np.asarray(projected_actions, dtype=np.float32),
                    resample_counts=np.asarray(resample_counts, dtype=np.int64),
                )
            continue

        accepted = False
        used_resamples = 0
        max_attempts = int(max_resample_attempts)
        for attempt in range(max_attempts):
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            action_seq = result["action"]  # (1,K,2)
            a0 = action_seq[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            a0 = _clip_action_v_omega(a0, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))

            cur_scaled = state_scaled

            children = list(getattr(fine_robot.nodes[int(idx_cur)].children, "indices", []))
            if bool(allow_self_candidate):
                children = [int(idx_cur)] + children
            if not children:
                return EpisodeResult(
                    success=False,
                    reason="no_children",
                    states=np.asarray(states, dtype=np.float32),
                    actions=np.asarray(actions, dtype=np.float32),
                    projected_actions=np.asarray(projected_actions, dtype=np.float32),
                    resample_counts=np.asarray(resample_counts, dtype=np.int64),
                )

            idx_next = None
            proj = None
            hat = _integrate_unicycle_scaled(state_scaled=cur_scaled, action_v_omega=a0, dt=float(dt), angle_scalor=float(angle_scalor))
            ranked = _rank_children_by_hat(node_states=fine_node_states, children=children, hat_state_scaled=hat, angle_scalor=float(angle_scalor))
            if opt_b_topk_children > 0:
                ranked = ranked[: int(opt_b_topk_children)]

            pick = _pick_projected_action(
                ranked_children=ranked,
                cur_state_scaled=cur_scaled,
                fine_node_states=fine_node_states,
                dt=float(dt),
                angle_scalor=float(angle_scalor),
                control_limits_scaled=control_limits_scaled,
                collision_robot=collision_robot,
                collision_task=task,
                collision_check_step=float(collision_check_step),
                collision_semantic=str(collision_semantic),
                projected_collision_stage=str(projected_collision_stage),
                idx_cur=int(idx_cur),
                anti_repeat=anti_repeat,
                anti_state=anti_state,
            )
            if pick is not None:
                idx_next, proj = pick

            if idx_next is None or proj is None:
                used_resamples = attempt + 1
                continue

            s_end_post: np.ndarray | None = None
            if bool(postcheck_collision):
                ok, fail_detail, s_end_post = _postcheck_projected_action(
                    collision_robot=collision_robot,
                    collision_task=task,
                    cur_state_scaled=cur_scaled,
                    action_v_omega=np.asarray(proj, dtype=np.float32).reshape(2),
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    collision_check_step=float(collision_check_step),
                    collision_semantic=str(collision_semantic),
                )
                if not bool(ok):
                    return EpisodeResult(
                        success=False,
                        reason="resample_exhausted",
                        states=np.asarray(states, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                        projected_actions=np.asarray(projected_actions, dtype=np.float32),
                        resample_counts=np.asarray(resample_counts, dtype=np.int64),
                        fail_detail=(None if fail_detail is None else str(fail_detail)),
                    )

            # accept
            used_resamples = attempt
            accepted = True

            actions.append(a0.astype(np.float32, copy=False))
            projected_actions.append(proj.astype(np.float32, copy=False))
            resample_counts.append(int(used_resamples))

            idx_anchor = int(idx_cur)

            # Apply the accepted action and advance the *continuous* state.
            # Collision checking was performed on this same action, so we must not "teleport" to the child node.
            if bool(postcheck_collision) and s_end_post is not None:
                state_scaled = np.asarray(s_end_post, dtype=np.float32).reshape(3)
            else:
                state_scaled = _integrate_unicycle_scaled_swept(
                    state_scaled=state_scaled,
                    action_v_omega=proj,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                )

            nxt_yaw = float(yaw_from_theta_scaled(float(state_scaled[2]), float(angle_scalor)))
            state = np.array([float(state_scaled[0]), float(state_scaled[1]), float(nxt_yaw)], dtype=np.float32)

            idx_list = getattr(fine_robot, "query_kdtree")(state_scaled)
            idx_cur = int(idx_list[0]) if idx_list else int(idx_next)
            states.append(state.copy())

            if anti_repeat is not None and bool(anti_repeat.enabled) and anti_state is not None:
                k1 = _xy_key(float(state_scaled[0]), float(state_scaled[1]), q=float(anti_repeat.xy_q))
                _anti_push_xy(anti_state, k1, max_n=int(anti_repeat.xy_recent_n))
                _anti_push_child(anti_state, int(idx_next), max_n=int(anti_repeat.child_recent_n))
                _anti_push_edge(anti_state, (int(idx_anchor), int(idx_next)), max_n=int(anti_repeat.edge_recent_n))
                anti_state.prev_edge = (int(idx_anchor), int(idx_next))
            break

        if not accepted:
            return EpisodeResult(
                success=False,
                reason="resample_exhausted",
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                projected_actions=np.asarray(projected_actions, dtype=np.float32),
                resample_counts=np.asarray(resample_counts, dtype=np.int64),
            )

        # Success check at the new continuous state.
        if bool(getattr(fine_robot, "within_goal")(np.asarray(state_scaled, dtype=np.float32).reshape(3))):
            return EpisodeResult(
                success=True,
                reason="reached_goal",
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                projected_actions=np.asarray(projected_actions, dtype=np.float32),
                resample_counts=np.asarray(resample_counts, dtype=np.int64),
            )

    return EpisodeResult(
        success=False,
        reason="timeout",
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        projected_actions=np.asarray(projected_actions, dtype=np.float32),
        resample_counts=np.asarray(resample_counts, dtype=np.int64),
    )


def _run_episodes_batched(
    *,
    task: dict[str, Any],
    task_obs: dict[str, Any] | None,
    coarse_grid3d,
    coarse_robot: Any,
    fine_robot: Any,
    fine_node_states: np.ndarray,
    collision_robot: Any | None,
    goal_xy: np.ndarray,
    goal_yaw: float | None,
    angle_scalor: float,
    policy: Any,
    start_states: Sequence[np.ndarray],  # (B,3) x,y,yaw_rad
    execute_mode: str,
    max_steps: int,
    crop_size: int,
    meters_per_pixel: float,
    rotate_with_yaw: bool,
    crop_bias_forward_m: float,
    yaw_offsets_deg: tuple[int, ...],
    footprint_length_m: float,
    footprint_width_m: float,
    max_resample_attempts: int,
    collision_check_step: float,
    collision_semantic: str,
    projected_collision_stage: str,
    allow_self_candidate: bool,
    opt_b_topk_children: int,
    anti_repeat: AntiRepeatConfig | None = None,
    obs_ablation: ObsAblationConfig | None = None,
) -> list[EpisodeResult]:
    """
    Run multiple episodes concurrently on a single GPU by batching policy inference.

    Notes:
      - Environment stepping (crop/gpath/snap/collision) remains per-episode on CPU.
      - This improves GPU utilization by increasing the batch size for policy.predict_action().
    """
    if task_obs is None:
        task_obs = task
    if collision_robot is None:
        collision_robot = fine_robot

    dt = float(getattr(fine_robot, "get_temporal_res")())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt from fine_robot.get_temporal_res(): {dt}")

    To = int(getattr(policy, "n_obs_steps", 2))
    if To <= 0:
        raise ValueError(f"Invalid policy.n_obs_steps: {To}")

    if not bool(rotate_with_yaw):
        raise ValueError("valuewin12ch requires rotate_with_yaw=True.")

    execute_mode = str(execute_mode).strip().lower()
    if execute_mode not in ("projected", "raw"):
        raise ValueError(f"Invalid execute mode: {execute_mode!r} (expected: 'projected' or 'raw')")

    projected_collision_stage = str(projected_collision_stage).strip().lower()
    if projected_collision_stage not in ("pre", "post"):
        raise ValueError(
            f"Invalid projected collision stage: {projected_collision_stage!r} "
            "(expected: 'pre' or 'post' for --projected-collision-stage)"
        )
    postcheck_collision = bool(execute_mode == "projected" and projected_collision_stage == "post")

    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    start_states = [np.asarray(s, dtype=np.float32).reshape(3) for s in start_states]
    B = len(start_states)
    if B == 0:
        return []

    gcfg = GlobalRefConfig()
    control_limits_scaled = getattr(fine_robot, "control_limits", [[-1, 1], [-1, 1]])
    policy_dev = _policy_device(policy)

    # Per-episode buffers.
    done = [False for _ in range(B)]
    success = [False for _ in range(B)]
    reason = ["" for _ in range(B)]
    fail_detail: list[str | None] = [None for _ in range(B)]

    idx_cur: list[int] = [-1 for _ in range(B)]
    state_yaw: list[np.ndarray] = [np.zeros((3,), dtype=np.float32) for _ in range(B)]
    state_scaled: list[np.ndarray] = [np.zeros((3,), dtype=np.float32) for _ in range(B)]

    img_hist: list[list[np.ndarray]] = [[] for _ in range(B)]
    state_hist: list[list[np.ndarray]] = [[] for _ in range(B)]
    gpath_hist: list[list[np.ndarray]] = [[] for _ in range(B)]

    states_out: list[list[np.ndarray]] = [[] for _ in range(B)]
    actions_out: list[list[np.ndarray]] = [[] for _ in range(B)]
    proj_out: list[list[np.ndarray]] = [[] for _ in range(B)]
    resample_out: list[list[int]] = [[] for _ in range(B)]

    anti_states: list[_AntiRepeatState | None] = [None for _ in range(B)]
    if anti_repeat is not None and bool(anti_repeat.enabled):
        if not (np.isfinite(float(anti_repeat.xy_q)) and float(anti_repeat.xy_q) > 0):
            raise ValueError(f"Invalid --anti-repeat-xy-q: {anti_repeat.xy_q}")
        if execute_mode == "raw":
            # Anti-repeat is an execution-layer heuristic for projected mode only.
            anti_repeat = None
        else:
            for i in range(B):
                anti_states[i] = _AntiRepeatState(
                    xy_recent=deque(),
                    xy_counts=Counter(),
                    child_recent=deque(),
                    child_counts=Counter(),
                    edge_recent=deque(),
                    edge_counts=Counter(),
                    prev_edge=None,
                )

    # Initialize continuous starts; use nearest nodes only as anchors for children.
    for i, start_state in enumerate(start_states):
        s0 = np.asarray(start_state, dtype=np.float32).reshape(3)
        state_yaw[i] = s0
        states_out[i] = [s0.copy()]
        start_scaled = np.array(
            [
                float(s0[0]),
                float(s0[1]),
                float(wrap_theta_scaled(theta_scaled_from_yaw(float(s0[2]), float(angle_scalor)), float(angle_scalor))),
            ],
            dtype=np.float32,
        )
        state_scaled[i] = start_scaled
        if execute_mode == "projected":
            idx_list = getattr(fine_robot, "query_kdtree")(start_scaled)
            if not idx_list:
                done[i] = True
                success[i] = False
                reason[i] = "no_nearest_node"
                continue

            idx = int(idx_list[0])
            idx_cur[i] = idx
        if anti_repeat is not None and bool(anti_repeat.enabled) and anti_states[i] is not None:
            k0 = _xy_key(float(start_scaled[0]), float(start_scaled[1]), q=float(anti_repeat.xy_q))
            _anti_push_xy(anti_states[i], k0, max_n=int(anti_repeat.xy_recent_n))

    max_attempts = int(max_resample_attempts)

    for _step in range(int(max_steps)):
        active = [i for i in range(B) if not done[i]]
        if not active:
            break

        # Build per-episode obs for the active set.
        imgs_b: list[np.ndarray] = []
        poses_b: list[np.ndarray] = []
        gpaths_b: list[np.ndarray] = []
        for i in active:
            s = state_yaw[i]
            img_255 = make_local_valuewin12ch(
                task=task_obs,
                coarse_grid3d=coarse_grid3d,
                state=s,
                crop_size=int(crop_size),
                meters_per_pixel=float(meters_per_pixel),
                rotate_with_yaw=True,
                crop_bias_forward_m=float(crop_bias_forward_m),
                yaw_offsets_deg=tuple(int(d) for d in yaw_offsets_deg),
                footprint_length_m=float(footprint_length_m),
                footprint_width_m=float(footprint_width_m),
                scale_255=True,
            )
            img = (img_255 / 255.0).astype(np.float32, copy=False)
            img_chw = np.moveaxis(img, -1, 0)

            gpath = compute_gpath_from_vi(
                coarse_robot=coarse_robot,
                state=s,
                goal_xy=goal_xy,
                goal_yaw=goal_yaw,
                angle_scalor=float(angle_scalor),
                cfg=gcfg,
            )

            img_chw, gpath = _apply_obs_ablation(img_chw=img_chw, gpath=gpath, cfg=obs_ablation)

            img_hist[i].append(img_chw)
            state_hist[i].append(s.copy())
            gpath_hist[i].append(gpath.copy())
            if len(img_hist[i]) == 1:
                while len(img_hist[i]) < To:
                    img_hist[i].insert(0, img_hist[i][0])
                    state_hist[i].insert(0, state_hist[i][0])
                    gpath_hist[i].insert(0, gpath_hist[i][0])
            if len(img_hist[i]) > To:
                img_hist[i] = img_hist[i][-To:]
                state_hist[i] = state_hist[i][-To:]
                gpath_hist[i] = gpath_hist[i][-To:]

            imgs_b.append(np.stack(img_hist[i], axis=0))
            poses_b.append(np.stack(state_hist[i], axis=0))
            gpaths_b.append(np.stack(gpath_hist[i], axis=0))

        if execute_mode == "raw":
            obs_dict = {
                "image": torch.from_numpy(np.stack([imgs_b[j] for j in range(len(active))], axis=0)).to(policy_dev),
                "agent_pos": torch.from_numpy(np.stack([poses_b[j] for j in range(len(active))], axis=0)).to(policy_dev),
                "global_path": torch.from_numpy(np.stack([gpaths_b[j] for j in range(len(active))], axis=0)).to(policy_dev),
            }
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            action_seq = result["action"]  # (P,K,2)

            semantic = str(collision_semantic).strip().lower()
            if semantic not in ("swept", "discrete"):
                raise ValueError(f"Invalid collision semantic: {collision_semantic!r} (expected: 'swept' or 'discrete')")
            xmin, xmax, ymin, ymax = get_range(task)

            for pi, ep_idx in enumerate(active):
                if done[ep_idx]:
                    continue

                a0 = action_seq[pi, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                a0 = _clip_action_v_omega(a0, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))

                s_end = _integrate_unicycle_scaled_swept(
                    state_scaled=state_scaled[ep_idx],
                    action_v_omega=a0,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                )

                ok = True
                if semantic == "swept":
                    ok = _trajectory_collision_free(
                        robot=collision_robot,
                        task=task,
                        state_scaled=state_scaled[ep_idx],
                        action_v_omega=a0,
                        dt=float(dt),
                        angle_scalor=float(angle_scalor),
                        step_size=float(collision_check_step),
                    )
                else:
                    x = float(s_end[0])
                    y = float(s_end[1])
                    if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
                        ok = False
                    elif not bool(getattr(collision_robot, "obstacle_free")(np.asarray(s_end, dtype=np.float32).reshape(3))):
                        ok = False

                if not ok:
                    x = float(s_end[0])
                    y = float(s_end[1])
                    if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
                        reason[ep_idx] = "out_of_bounds"
                    else:
                        reason[ep_idx] = "collision"
                    done[ep_idx] = True
                    success[ep_idx] = False
                    continue

                actions_out[ep_idx].append(np.asarray(a0, dtype=np.float32).reshape(2))
                proj_out[ep_idx].append(np.asarray(a0, dtype=np.float32).reshape(2))
                resample_out[ep_idx].append(0)

                state_scaled[ep_idx] = np.asarray(s_end, dtype=np.float32).reshape(3)
                nxt_yaw = float(yaw_from_theta_scaled(float(state_scaled[ep_idx][2]), float(angle_scalor)))
                s_next = np.array([float(state_scaled[ep_idx][0]), float(state_scaled[ep_idx][1]), float(nxt_yaw)], dtype=np.float32)
                state_yaw[ep_idx] = s_next
                states_out[ep_idx].append(s_next.copy())

                if bool(getattr(fine_robot, "within_goal")(np.asarray(state_scaled[ep_idx], dtype=np.float32).reshape(3))):
                    done[ep_idx] = True
                    success[ep_idx] = True
                    reason[ep_idx] = "reached_goal"
            continue

        # Resample loop for episodes that haven't accepted an action at this step.
        pending = list(active)
        accepted: dict[int, tuple[np.ndarray, np.ndarray, int, int, int]] = {}
        # ep_idx -> (raw_action, projected_action, next_idx, resample_count, idx_anchor)

        for attempt in range(max_attempts):
            if not pending:
                break

            # Build batched obs for pending episodes from the prebuilt histories.
            pending_pos = {ep_idx: j for j, ep_idx in enumerate(active)}
            sel = [pending_pos[i] for i in pending]

            obs_dict = {
                "image": torch.from_numpy(np.stack([imgs_b[j] for j in sel], axis=0)[:, None, ...]).to(policy_dev),
                "agent_pos": torch.from_numpy(np.stack([poses_b[j] for j in sel], axis=0)[:, None, ...]).to(policy_dev),
                "global_path": torch.from_numpy(np.stack([gpaths_b[j] for j in sel], axis=0)[:, None, ...]).to(policy_dev),
            }
            # NOTE: the extra [:, None, ...] matches the (B,To,...) layout expected by the policy (To already stacked).
            obs_dict = {
                "image": obs_dict["image"].squeeze(1),
                "agent_pos": obs_dict["agent_pos"].squeeze(1),
                "global_path": obs_dict["global_path"].squeeze(1),
            }

            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            action_seq = result["action"]  # (P,K,2)

            still_pending: list[int] = []
            for pi, ep_idx in enumerate(pending):
                idx = int(idx_cur[ep_idx])
                cur_scaled = state_scaled[ep_idx]

                a0 = action_seq[pi, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                a0 = _clip_action_v_omega(a0, control_limits_scaled=control_limits_scaled, angle_scalor=float(angle_scalor))

                children = list(getattr(fine_robot.nodes[int(idx)].children, "indices", []))
                if bool(allow_self_candidate):
                    children = [int(idx)] + children
                if not children:
                    done[ep_idx] = True
                    success[ep_idx] = False
                    reason[ep_idx] = "no_children"
                    continue

                idx_next = None
                proj = None
                hat = _integrate_unicycle_scaled(state_scaled=cur_scaled, action_v_omega=a0, dt=float(dt), angle_scalor=float(angle_scalor))
                ranked = _rank_children_by_hat(node_states=fine_node_states, children=children, hat_state_scaled=hat, angle_scalor=float(angle_scalor))
                if opt_b_topk_children > 0:
                    ranked = ranked[: int(opt_b_topk_children)]

                pick = _pick_projected_action(
                    ranked_children=ranked,
                    cur_state_scaled=cur_scaled,
                    fine_node_states=fine_node_states,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    control_limits_scaled=control_limits_scaled,
                    collision_robot=collision_robot,
                    collision_task=task,
                    collision_check_step=float(collision_check_step),
                    collision_semantic=str(collision_semantic),
                    projected_collision_stage=str(projected_collision_stage),
                    idx_cur=int(idx),
                    anti_repeat=anti_repeat,
                    anti_state=anti_states[ep_idx] if anti_states else None,
                )
                if pick is not None:
                    idx_next, proj = pick

                if idx_next is None or proj is None:
                    still_pending.append(ep_idx)
                    continue

                accepted[ep_idx] = (
                    a0,
                    np.asarray(proj, dtype=np.float32).reshape(2),
                    int(idx_next),
                    int(attempt),
                    int(idx),
                )

            pending = still_pending

        # Apply accepted actions, terminate failures.
        for i in active:
            if done[i]:
                continue
            if i not in accepted:
                done[i] = True
                success[i] = False
                reason[i] = "resample_exhausted"
                continue

            a0, proj, idx_next, used_resamples, idx_anchor = accepted[i]
            s_end_post: np.ndarray | None = None
            if bool(postcheck_collision):
                ok, fd, s_end_post = _postcheck_projected_action(
                    collision_robot=collision_robot,
                    collision_task=task,
                    cur_state_scaled=state_scaled[i],
                    action_v_omega=np.asarray(proj, dtype=np.float32).reshape(2),
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    collision_check_step=float(collision_check_step),
                    collision_semantic=str(collision_semantic),
                )
                if not bool(ok):
                    done[i] = True
                    success[i] = False
                    reason[i] = "resample_exhausted"
                    fail_detail[i] = (None if fd is None else str(fd))
                    continue

            actions_out[i].append(np.asarray(a0, dtype=np.float32).reshape(2))
            proj_out[i].append(np.asarray(proj, dtype=np.float32).reshape(2))
            resample_out[i].append(int(used_resamples))

            # Apply the accepted action and advance the *continuous* state.
            if bool(postcheck_collision) and s_end_post is not None:
                state_scaled[i] = np.asarray(s_end_post, dtype=np.float32).reshape(3)
            else:
                state_scaled[i] = _integrate_unicycle_scaled_swept(
                    state_scaled=state_scaled[i],
                    action_v_omega=proj,
                    dt=float(dt),
                    angle_scalor=float(angle_scalor),
                    step_size=float(collision_check_step),
                )

            nxt_yaw = float(yaw_from_theta_scaled(float(state_scaled[i][2]), float(angle_scalor)))
            s_next = np.array([float(state_scaled[i][0]), float(state_scaled[i][1]), float(nxt_yaw)], dtype=np.float32)
            state_yaw[i] = s_next

            idx_list = getattr(fine_robot, "query_kdtree")(state_scaled[i])
            idx_cur[i] = int(idx_list[0]) if idx_list else int(idx_next)
            states_out[i].append(s_next.copy())

            if anti_repeat is not None and bool(anti_repeat.enabled) and anti_states[i] is not None:
                k1 = _xy_key(float(state_scaled[i][0]), float(state_scaled[i][1]), q=float(anti_repeat.xy_q))
                _anti_push_xy(anti_states[i], k1, max_n=int(anti_repeat.xy_recent_n))
                _anti_push_child(anti_states[i], int(idx_next), max_n=int(anti_repeat.child_recent_n))
                _anti_push_edge(anti_states[i], (int(idx_anchor), int(idx_next)), max_n=int(anti_repeat.edge_recent_n))
                anti_states[i].prev_edge = (int(idx_anchor), int(idx_next))

            # Goal check at the new continuous state.
            if bool(getattr(fine_robot, "within_goal")(np.asarray(state_scaled[i], dtype=np.float32).reshape(3))):
                done[i] = True
                success[i] = True
                reason[i] = "reached_goal"

    # Finalize timeouts.
    for i in range(B):
        if not done[i]:
            done[i] = True
            success[i] = False
            reason[i] = "timeout"

    out: list[EpisodeResult] = []
    for i in range(B):
        out.append(
            EpisodeResult(
                success=bool(success[i]),
                reason=str(reason[i]),
                states=np.asarray(states_out[i], dtype=np.float32),
                actions=np.asarray(actions_out[i], dtype=np.float32),
                projected_actions=np.asarray(proj_out[i], dtype=np.float32),
                resample_counts=np.asarray(resample_out[i], dtype=np.int64),
                fail_detail=(None if fail_detail[i] is None else str(fail_detail[i])),
            )
        )
    return out


def _plot_episode(
    *,
    task: dict[str, Any],
    coarse_grid,
    goal_xy: Sequence[float],
    states: np.ndarray,
    out_path: Path,
    title: str,
    footprint_length_m: float,
    footprint_width_m: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot") from e

    def footprint_corners_xy(*, x: float, y: float, yaw: float, length_m: float, width_m: float) -> np.ndarray:
        L = float(length_m)
        W = float(width_m)
        if not (np.isfinite(L) and np.isfinite(W) and L > 0 and W > 0):
            raise ValueError(f"Invalid footprint size: length_m={L} width_m={W}")
        c = float(math.cos(float(yaw)))
        s = float(math.sin(float(yaw)))
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        hx = 0.5 * L
        hy = 0.5 * W
        corners_local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
        corners_world = corners_local @ R.T + np.array([float(x), float(y)], dtype=np.float32)[None, :]
        return corners_world

    grid = coarse_grid.as_ascending()
    V = grid.V
    xs = grid.x_samples
    ys = grid.y_samples
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.imshow(V, origin="lower", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis", alpha=0.95)

    for ob in task.get("env", {}).get("obstacles", []):
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

    goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=80, c="red", marker="x", linewidths=2, label="goal")
    states = np.asarray(states, dtype=np.float32).reshape(-1, 3)
    if states.size > 0:
        states_xy = states[:, 0:2]
        ax.scatter([float(states_xy[0, 0])], [float(states_xy[0, 1])], s=40, c="white", edgecolors="k", label="start")
        ax.plot(states_xy[:, 0], states_xy[:, 1], "-o", color="white", linewidth=2, markersize=3, label="traj")

        T = max(int(states.shape[0] - 1), 0)
        stride = max(1, T // 30) if T > 0 else 1
        idxs = list(range(0, states.shape[0], stride))
        if idxs and idxs[-1] != states.shape[0] - 1:
            idxs.append(states.shape[0] - 1)
        for i in idxs:
            x, y, yaw = float(states[i, 0]), float(states[i, 1]), float(states[i, 2])
            corners = footprint_corners_xy(x=x, y=y, yaw=yaw, length_m=float(footprint_length_m), width_m=float(footprint_width_m))
            ax.add_patch(Polygon(corners, closed=True, edgecolor="cyan", facecolor="none", linewidth=1.0, alpha=0.35))

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt05: unicycle diffusion inference (receding-horizon) on CUDA VI graphs.")
    p.add_argument("--goals", type=str, required=True, help="Path to data/unicycle_value_grids/<map>/goals.json")
    p.add_argument("--goal-index", type=int, required=True, help="Which goal_<k> to run.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to training checkpoint (latest.ckpt).")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for episode artifacts. Default: <goals_dir>/goal_<k>/infer_diffusion",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use-ema", action="store_true", help="Use EMA weights from checkpoint if available.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--episode-batch-size",
        type=int,
        default=1,
        help="Run multiple episodes concurrently by batching policy inference (improves GPU utilization). Default: 1.",
    )
    p.add_argument(
        "--execute-mode",
        type=str,
        default="projected",
        choices=["projected", "raw"],
        help=(
            "Execution mode: "
            "projected=use VI children projection + collision filtering + optional resampling (default); "
            "raw=execute the raw policy action directly (pure diffusion baseline; no projection, no resampling)."
        ),
    )
    p.add_argument(
        "--starts-file",
        type=str,
        default=None,
        help=(
            "Optional pre-sampled starts JSON file (from `python -m unicycle_value_guided.sample_starts`). "
            "If provided, starts are taken from this file in order and random start sampling is skipped."
        ),
    )
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clearance", type=float, default=0.2, help="Boundary margin when sampling starts.")
    p.add_argument("--min-goal-dist", type=float, default=3.0, help="Min xy distance start->goal.")
    p.add_argument("--crop-size", type=int, default=84)
    p.add_argument("--mpp", type=float, default=0.05)
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

    # Quick, test-time observation ablations (feature corruption).
    # These keep input tensor shapes identical to training, but replace certain inputs with constants.
    p.add_argument(
        "--ablate-value-slices",
        action="store_true",
        help="Replace all value slice channels (image[1:]) with a constant (see --ablate-value-slices-fill).",
    )
    p.add_argument(
        "--ablate-value-slices-fill",
        type=float,
        default=0.5,
        help="Fill value in [0,1] for --ablate-value-slices (default: 0.5).",
    )
    p.add_argument(
        "--ablate-occupancy",
        action="store_true",
        help="Replace the occupancy channel (image[0]) with a constant (see --ablate-occupancy-fill).",
    )
    p.add_argument(
        "--ablate-occupancy-fill",
        type=float,
        default=0.0,
        help="Fill value in [0,1] for --ablate-occupancy (default: 0.0).",
    )
    p.add_argument(
        "--ablate-gpath",
        action="store_true",
        help="Replace gpath (global_path) with a constant (see --ablate-gpath-fill).",
    )
    p.add_argument(
        "--ablate-gpath-fill",
        type=float,
        default=0.0,
        help="Fill value for --ablate-gpath (default: 0.0).",
    )
    p.add_argument("--footprint-length-m", type=float, default=0.625, help="Rectangle footprint length (meters).")
    p.add_argument("--footprint-width-m", type=float, default=0.4375, help="Rectangle footprint width (meters).")
    p.add_argument("--max-resample-attempts", type=int, default=30, help="Max diffusion resamples per step.")
    p.add_argument("--collision-check-step", type=float, default=0.05, help="Collision check linear step (meters).")
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
    p.add_argument(
        "--projected-collision-stage",
        type=str,
        default="pre",
        choices=["pre", "post"],
        help=(
            "For execute_mode=projected, when to perform collision filtering: "
            "pre=filter candidate children by collision before selecting (default); "
            "post=do NOT filter candidates; select the child first, then do a single collision check on the selected projected action; "
            "if it fails, terminate the episode as resample_exhausted."
        ),
    )
    p.add_argument("--allow-self-candidate", action="store_true", help="Allow staying at current node as candidate.")

    p.add_argument(
        "--anti-repeat",
        action="store_true",
        help="Enable best-effort anti-repeat (tabu) when selecting the projected child/action.",
    )
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
        help="Max random start samples per episode before failing (includes feasibility filtering).",
    )
    p.add_argument(
        "--start-feasible-max-steps",
        type=int,
        default=None,
        help="Max steps for the feasibility check (default: use --max-steps).",
    )
    p.add_argument(
        "--start-feasible-max-children-per-step",
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
        "--opt-a-boundary-wall-thickness-m",
        type=float,
        default=0.0,
        help=(
            "Optional thin boundary walls added as obstacles *before* Opt-A inflation. "
            "This creates a boundary no-go band (inside env.range) so that fine VI children near the wall shrink inward. "
            "Default: 0.0 (disabled)."
        ),
    )
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

    p.add_argument("--opt-b-topk-children", type=int, default=5, help="Try up to K nearest children (0=all).")
    p.add_argument("--plot", action="store_true", help="Save a per-episode plot overlayed on coarse value grid.")
    p.add_argument("--plot-every", type=int, default=1, help="When --plot, only plot every N episodes (default: 1=all).")
    p.add_argument("--plot-failures-only", action="store_true", help="When --plot, only plot failed episodes.")
    p.add_argument("--log-mem", action="store_true", help="Print RSS/CUDA memory stats at key points.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(str(args.device))
    if bool(getattr(args, "log_mem", False)):
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(int(device.index or 0))
            except Exception:
                pass
        _log_mem("start", device=device)

    goals_path = Path(args.goals)
    goals_payload = load_json(goals_path)
    goals: list[list[float]] = goals_payload["goals"]
    goal_idx = int(args.goal_index)
    goal_xyz = goals[goal_idx]
    if len(goal_xyz) != 3:
        raise ValueError(f"Expected goal pose [x,y,yaw_deg], got: {goal_xyz}")
    goal_xy = np.asarray(goal_xyz[:2], dtype=np.float64).reshape(2)
    goal_yaw = float(goal_xyz[2]) / 180.0 * float(np.pi)

    goal_dir = goals_path.parent / f"goal_{goal_idx}"

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

    coarse_meta = json.loads((goal_dir / "meta_coarse.json").read_text(encoding="utf-8"))
    fine_meta = json.loads((goal_dir / "meta_fine.json").read_text(encoding="utf-8"))

    execute_mode = str(getattr(args, "execute_mode", "projected")).strip().lower()
    if execute_mode not in ("projected", "raw"):
        raise ValueError(f"Invalid --execute-mode: {execute_mode!r} (expected: 'projected' or 'raw')")

    start_feasible_mode = str(getattr(args, "start_feasible_mode", "none")).strip().lower()
    needs_fine_graph = bool(execute_mode == "projected") or (args.starts_file is None and start_feasible_mode not in ("", "none"))

    # Goal semantics: always use the goal-specific tmp_task (meta_coarse.json) if available.
    goal_task_path = Path(str(coarse_meta.get("tmp_task_path", "") or "")).expanduser()
    if not str(goal_task_path).strip():
        goal_task_path = task_path
    if not goal_task_path.is_absolute():
        goal_task_path = (Path.cwd() / goal_task_path).resolve()
    goal_task = load_task(goal_task_path)
    if not goal_task.get("robots"):
        raise RuntimeError(f"Goal task has no robots[] entry: {goal_task_path}")
    goal_robot = Unicycle(goal_task["env"], goal_task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)
    # Raw (non-VI) Unicycle instances have no nodes, which would make get_temporal_res() return 0.0.
    # Use the fine VI dt as the macro-step duration for inference-time integration.
    dt_hint = None
    try:
        fine_sum_path = goal_dir / "logs_fine" / "summary.json"
        if fine_sum_path.exists():
            dt_hint = float(json.loads(fine_sum_path.read_text(encoding="utf-8")).get("dt"))
    except Exception:
        dt_hint = None
    if dt_hint is not None and np.isfinite(float(dt_hint)) and float(dt_hint) > 0:
        setattr(goal_robot, "preset_temporal_res", float(dt_hint))

    task_obs = None
    collision_robot: Any | None = None
    start_inflated_robot: Any | None = None
    plot_goal_dir = goal_dir
    boundary_wall_thickness_m = float(getattr(args, "opt_a_boundary_wall_thickness_m", 0.0))

    if bool(args.opt_a):
        use_inflated_fine = bool(args.opt_a_use_inflated_fine) and bool(needs_fine_graph)
        inflated = prepare_inflated_goal_assets(
            base_task=task,
            base_task_path=task_path,
            map_name=str(goals_payload.get("map_name", goals_path.parent.name)),
            goal_idx=int(goal_idx),
            goal_xyz=goal_xyz,
            cache_root=Path(args.opt_a_cache_dir),
            delta=float(args.opt_a_delta),
            boundary_wall_thickness_m=float(getattr(args, "opt_a_boundary_wall_thickness_m", 0.0)),
            coarse_meta_src=coarse_meta,
            fine_meta_src=(fine_meta if bool(use_inflated_fine) else None),
            vi_device=(None if args.opt_a_vi_device is None else str(args.opt_a_vi_device)),
            vi_dtype=(None if args.opt_a_vi_dtype is None else str(args.opt_a_vi_dtype)),
            max_iters=(None if args.opt_a_vi_max_iters is None else int(args.opt_a_vi_max_iters)),
            tol=(None if args.opt_a_vi_tol is None else float(args.opt_a_vi_tol)),
            overwrite=bool(args.opt_a_overwrite_cache),
            keep_pkl=True,
            use_inflated_fine=bool(use_inflated_fine),
        )
        task_obs = inflated.task_obs
        coarse_grid3d = inflated.coarse_grid3d
        coarse_robot = inflated.coarse_robot
        plot_goal_dir = inflated.goal_dir

        if bool(needs_fine_graph):
            if bool(args.opt_a_use_inflated_fine):
                if inflated.fine_robot is None:
                    raise RuntimeError("Opt-A requested inflated fine, but fine_robot cache is missing.")
                fine_robot = inflated.fine_robot
            else:
                fine_robot = load_vi_robot(_find_vi_robot(goal_dir, "fine"))
        else:
            fine_robot = goal_robot

        collision_check_mode = str(getattr(args, "opt_a_collision_check", "real"))
        if collision_check_mode == "inflated":
            if task_obs is None:
                raise RuntimeError("Opt-A internal error: task_obs is None.")
            if not task_obs.get("robots"):
                raise RuntimeError("Opt-A task_obs has no robots[] entry.")
            collision_robot = Unicycle(task_obs["env"], task_obs["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)
        else:
            # For collision checking we only need obstacle_free(), not the full VI graph.
            # Loading the fine VI robot can easily take tens of GB RAM (python objects + edges).
            if not task.get("robots"):
                raise RuntimeError("Task has no robots[] entry.")
            collision_robot = Unicycle(task["env"], task["robots"][0], angle_scalor=float(angle_scalor), robot_id=0)

        # Always ensure sampled starts are also collision-free under the inflated obstacle semantics.
        # This avoids "real-free but inflated-colliding" starts, which are out-of-distribution for Opt-A
        # (occupancy/value/gpath are computed on the inflated map), even if collision checking uses real obstacles.
        #
        # Boundary walls (if enabled) are planning-margin artifacts meant to shrink VI children near env.range.
        # Since obstacle_free() accounts for robot footprint samples, those walls can make existing starts invalid.
        # We therefore ignore boundary-wall obstacles for the start validity check.
        if task_obs is None:
            raise RuntimeError("Opt-A internal error: task_obs is None.")
        if not task_obs.get("robots"):
            raise RuntimeError("Opt-A task_obs has no robots[] entry.")
        task_obs_start = task_obs
        if boundary_wall_thickness_m > 0:
            task_obs_start = dict(task_obs)
            task_obs_start["env"] = _strip_boundary_wall_obstacles(dict(task_obs["env"]))
        if collision_check_mode == "inflated" and boundary_wall_thickness_m <= 0:
            start_inflated_robot = collision_robot
        else:
            start_inflated_robot = Unicycle(
                task_obs_start["env"],
                task_obs_start["robots"][0],
                angle_scalor=float(angle_scalor),
                robot_id=0,
            )

        print(
            f"[infer_diffusion] opt-a enabled: delta={float(args.opt_a_delta):.3f} cache={inflated.goal_dir} "
            f"boundary_wall={float(getattr(args, 'opt_a_boundary_wall_thickness_m', 0.0)):.3f} "
            f"use_inflated_fine={bool(args.opt_a_use_inflated_fine)} collision_check={str(getattr(args, 'opt_a_collision_check', 'real'))}",
            flush=True,
        )
    else:
        coarse_grid3d = load_regular_value_grid_3d(goal_dir / "value_coarse_3d.npy", goal_dir / "meta_coarse_3d.json")
        coarse_robot = load_vi_robot(_find_vi_robot(goal_dir, "coarse"))
        if bool(needs_fine_graph):
            fine_robot = load_vi_robot(_find_vi_robot(goal_dir, "fine"))
            collision_robot = fine_robot
        else:
            fine_robot = goal_robot
            collision_robot = goal_robot

    coarse_grid2d = load_regular_value_grid(plot_goal_dir / "value_coarse.npy", plot_goal_dir / "meta_coarse.json") if bool(args.plot) else None
    if bool(needs_fine_graph):
        fine_node_states = np.stack([np.asarray(n.state, dtype=np.float32).reshape(3) for n in fine_robot.nodes], axis=0)
        validate_crop_covers_children(
            fine_robot=fine_robot,
            crop_size=int(args.crop_size),
            meters_per_pixel=float(args.mpp),
            crop_bias_forward_m=float(args.crop_bias_forward_m),
            strict=bool(getattr(args, "require_crop_covers_children", False)),
            extra_margin_m=0.0,
            context="infer_diffusion",
        )
    else:
        fine_node_states = np.zeros((0, 3), dtype=np.float32)
        print("[infer_diffusion] execute_mode=raw: skipping fine VI graph load/coverage checks.", flush=True)
    if bool(getattr(args, "log_mem", False)):
        _log_mem("after_assets", device=device)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.ckpt"
    policy = _load_policy_from_ckpt(ckpt_path, device=device, use_ema=bool(args.use_ema))
    if bool(getattr(args, "log_mem", False)):
        _log_mem("after_policy", device=device)

    out_dir = goal_dir / "infer_diffusion" if args.out_dir is None else Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    starts_from_file: np.ndarray | None = None
    starts_file_path: Path | None = None
    if args.starts_file is not None:
        starts_file_path = Path(str(args.starts_file)).expanduser()
        if not starts_file_path.is_absolute():
            starts_file_path = (Path.cwd() / starts_file_path).resolve()
        payload = json.loads(starts_file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "goal_index" in payload and int(payload["goal_index"]) != int(goal_idx):
                raise ValueError(f"--starts-file goal_index={payload['goal_index']} does not match --goal-index={goal_idx}")
            if "goals_path" in payload and str(payload["goals_path"]):
                # Best-effort: sanity check only; allow relocation of datasets across machines.
                pass
            starts_list = payload.get("starts", None)
        elif isinstance(payload, list):
            starts_list = payload
        else:
            raise ValueError(f"Invalid --starts-file JSON: expected object or list, got {type(payload)}")
        if starts_list is None:
            raise ValueError("--starts-file JSON missing key: 'starts'")
        starts_from_file = np.asarray(starts_list, dtype=np.float32)
        if starts_from_file.ndim != 2 or starts_from_file.shape[1] != 3:
            raise ValueError(f"--starts-file starts must be shape (N,3) [x,y,yaw_rad], got {starts_from_file.shape}")
        if int(args.episodes) > int(starts_from_file.shape[0]):
            raise ValueError(f"--episodes={int(args.episodes)} exceeds starts in --starts-file (N={int(starts_from_file.shape[0])})")
        if str(getattr(args, "start_feasible_mode", "none")).strip().lower() not in ("none", ""):
            print("[infer_diffusion] NOTE: --starts-file provided; ignoring --start-feasible-* flags (use sample_starts).", flush=True)

    successes = 0
    lengths: list[int] = []
    reason_counts: Counter[str] = Counter()
    total_resamples = 0
    total_steps = 0
    rng = np.random.default_rng(int(args.seed) * 100000 + int(goal_idx))
    feasible_max_steps = int(args.max_steps) if args.start_feasible_max_steps is None else int(args.start_feasible_max_steps)
    max_children_per_step = int(getattr(args, "start_feasible_max_children_per_step", 0))
    episode_batch_size = int(getattr(args, "episode_batch_size", 1))
    if episode_batch_size <= 0:
        raise ValueError("--episode-batch-size must be >= 1.")
    episode_batch_size = min(episode_batch_size, int(args.episodes))
    if episode_batch_size > 1:
        print(f"[infer_diffusion] episode_batch_size={episode_batch_size} (batched policy inference).", flush=True)

    plot_every = max(1, int(getattr(args, "plot_every", 1)))
    plot_failures_only = bool(getattr(args, "plot_failures_only", False))

    collision_semantic = str(getattr(args, "collision_semantic", "swept")).strip().lower()
    if collision_semantic not in ("swept", "discrete"):
        raise ValueError(f"Invalid --collision-semantic: {collision_semantic!r}")

    projected_collision_stage = str(getattr(args, "projected_collision_stage", "pre")).strip().lower()
    if projected_collision_stage not in ("pre", "post"):
        raise ValueError(f"Invalid --projected-collision-stage: {projected_collision_stage!r}")
    postcheck_fail_detail_counts: Counter[str] = Counter()

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
        print(
            "[infer_diffusion] anti_repeat enabled:"
            f" xy_q={float(anti_repeat.xy_q):.4f}"
            f" xy_recent_n={int(anti_repeat.xy_recent_n)}"
            f" child_recent_n={int(anti_repeat.child_recent_n)}"
            f" edge_recent_n={int(anti_repeat.edge_recent_n)}"
            f" avoid_uturn={bool(anti_repeat.avoid_uturn)}",
            flush=True,
        )

    obs_ablation = ObsAblationConfig(
        ablate_value_slices=bool(getattr(args, "ablate_value_slices", False)),
        value_slices_fill=float(getattr(args, "ablate_value_slices_fill", 0.5)),
        ablate_occupancy=bool(getattr(args, "ablate_occupancy", False)),
        occupancy_fill=float(getattr(args, "ablate_occupancy_fill", 0.0)),
        ablate_gpath=bool(getattr(args, "ablate_gpath", False)),
        gpath_fill=float(getattr(args, "ablate_gpath_fill", 0.0)),
    )
    if not (obs_ablation.ablate_value_slices or obs_ablation.ablate_occupancy or obs_ablation.ablate_gpath):
        obs_ablation = None
    else:
        # Best-effort validation.
        for k, v in {
            "ablate_value_slices_fill": obs_ablation.value_slices_fill,
            "ablate_occupancy_fill": obs_ablation.occupancy_fill,
            "ablate_gpath_fill": obs_ablation.gpath_fill,
        }.items():
            if not np.isfinite(float(v)):
                raise ValueError(f"Invalid --{k}: {v}")
        print(
            "[infer_diffusion] obs_ablation enabled:"
            f" value_slices={bool(obs_ablation.ablate_value_slices)}(fill={float(obs_ablation.value_slices_fill):.3f})"
            f" occupancy={bool(obs_ablation.ablate_occupancy)}(fill={float(obs_ablation.occupancy_fill):.3f})"
            f" gpath={bool(obs_ablation.ablate_gpath)}(fill={float(obs_ablation.gpath_fill):.3f})",
            flush=True,
        )

    def _sample_start_state(ep: int) -> np.ndarray:
        if bool(getattr(args, "log_mem", False)):
            _log_mem(f"episode_{ep:04d}_start", device=device)

        if starts_from_file is not None:
            start_state = np.asarray(starts_from_file[int(ep)], dtype=np.float32).reshape(3)
            x, y, yaw = float(start_state[0]), float(start_state[1]), float(start_state[2])
            if float(np.hypot(x - float(goal_xy[0]), y - float(goal_xy[1]))) < float(args.min_goal_dist):
                raise ValueError(f"--starts-file start[{ep}] violates --min-goal-dist: {start_state.tolist()}")
            start_scaled = np.array([x, y, theta_scaled_from_yaw(yaw, float(angle_scalor))], dtype=np.float32)
            if bool(getattr(args, "opt_a", False)):
                if start_inflated_robot is None:
                    raise RuntimeError("Opt-A internal error: start_inflated_robot is None.")
                if not bool(getattr(start_inflated_robot, "obstacle_free")(start_scaled)):
                    raise ValueError(f"--starts-file start[{ep}] is not obstacle_free under inflated obstacles: {start_state.tolist()}")
                if boundary_wall_thickness_m <= 0 and bool(getattr(args, "opt_a_use_inflated_fine", False)) and hasattr(fine_robot, "obstacle_free"):
                    if not bool(getattr(fine_robot, "obstacle_free")(start_scaled)):
                        raise ValueError(f"--starts-file start[{ep}] is not obstacle_free under inflated fine semantics: {start_state.tolist()}")
            else:
                if not bool(getattr(collision_robot, "obstacle_free")(start_scaled)):
                    raise ValueError(f"--starts-file start[{ep}] is not obstacle_free under current obstacles: {start_state.tolist()}")
            return start_state

        for _ in range(int(getattr(args, "start_max_attempts", 10_000))):
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
                if boundary_wall_thickness_m <= 0 and bool(getattr(args, "opt_a_use_inflated_fine", False)) and hasattr(fine_robot, "obstacle_free"):
                    if not bool(getattr(fine_robot, "obstacle_free")(start_scaled)):
                        continue
            else:
                if not bool(getattr(collision_robot, "obstacle_free")(start_scaled)):
                    continue

            mode = str(getattr(args, "start_feasible_mode", "none")).strip().lower()
            if mode and mode != "none":
                coarse_ok = None
                fine_ok = None

                def _coarse_ok() -> bool:
                    nonlocal coarse_ok
                    if coarse_ok is None:
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
                    # coarse infeasible, but fine feasible
                    ok = (not _coarse_ok()) and _fine_ok()
                else:
                    raise ValueError(f"Invalid --start-feasible-mode: {mode}")

                if not ok:
                    continue

            return np.array([x, y, yaw], dtype=np.float32)

        raise RuntimeError("Failed to sample a valid start state (increase clearance or reduce min_goal_dist).")

    total_episodes = int(args.episodes)
    for ep0 in range(0, total_episodes, episode_batch_size):
        batch_eps = list(range(ep0, min(ep0 + episode_batch_size, total_episodes)))
        batch_starts = [_sample_start_state(ep) for ep in batch_eps]

        if len(batch_eps) == 1:
            results = [
                _run_episode(
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
                    start_state=batch_starts[0],
                    execute_mode=str(args.execute_mode),
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
                    projected_collision_stage=str(projected_collision_stage),
                    allow_self_candidate=bool(args.allow_self_candidate),
                    opt_b_topk_children=int(args.opt_b_topk_children),
                    anti_repeat=anti_repeat,
                    obs_ablation=obs_ablation,
                )
            ]
        else:
            results = _run_episodes_batched(
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
                start_states=batch_starts,
                execute_mode=str(args.execute_mode),
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
                projected_collision_stage=str(projected_collision_stage),
                allow_self_candidate=bool(args.allow_self_candidate),
                opt_b_topk_children=int(args.opt_b_topk_children),
                anti_repeat=anti_repeat,
                obs_ablation=obs_ablation,
            )

        if len(results) != len(batch_eps):
            raise RuntimeError(f"Internal error: got {len(results)} results for batch of {len(batch_eps)} episodes.")

        for ep, res in zip(batch_eps, results):
            ep_name = f"episode_{ep:04d}"
            np.save(out_dir / f"{ep_name}_states.npy", res.states)
            np.save(out_dir / f"{ep_name}_actions.npy", res.actions)
            np.save(out_dir / f"{ep_name}_projected_actions.npy", res.projected_actions)
            np.save(out_dir / f"{ep_name}_resample_counts.npy", res.resample_counts)

            should_plot = bool(args.plot)
            if should_plot and plot_failures_only and bool(res.success):
                should_plot = False
            if should_plot and int(plot_every) > 1 and (int(ep) % int(plot_every) != 0):
                should_plot = False

            if should_plot:
                if coarse_grid2d is None:
                    raise RuntimeError("Internal error: coarse_grid2d is None while --plot is enabled.")
                _plot_episode(
                    task=task,
                    coarse_grid=coarse_grid2d,
                    goal_xy=goal_xy,
                    states=res.states,
                    out_path=out_dir / f"{ep_name}.png",
                    title=f"goal_{goal_idx} | {res.reason} | T={res.actions.shape[0]}",
                    footprint_length_m=float(args.footprint_length_m),
                    footprint_width_m=float(args.footprint_width_m),
                )

            successes += int(bool(res.success))
            reason_counts[str(res.reason)] += 1
            if str(projected_collision_stage) == "post" and res.fail_detail is not None:
                postcheck_fail_detail_counts[str(res.fail_detail)] += 1
            lengths.append(int(res.actions.shape[0]))
            if res.resample_counts.size > 0:
                total_resamples += int(np.sum(res.resample_counts))
                total_steps += int(res.resample_counts.shape[0])
            else:
                total_steps += 0
            print(f"[infer] ep={ep} success={res.success} reason={res.reason} T={res.actions.shape[0]}", flush=True)
            if bool(getattr(args, "log_mem", False)):
                _log_mem(f"episode_{ep:04d}_end", device=device)

    summary = {
        "map_name": str(goals_payload.get("map_name", goals_path.parent.name)),
        "goal_index": int(goal_idx),
        "goal_pose": [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])],
        "episodes": int(args.episodes),
        "successes": int(successes),
        "success_rate": float(successes) / float(max(int(args.episodes), 1)),
        "avg_T": float(np.mean(lengths)) if lengths else 0.0,
        "median_T": float(np.median(lengths)) if lengths else 0.0,
        "ckpt": str(ckpt_path),
        "device": str(device),
        "rotate_with_yaw": bool(args.rotate_with_yaw),
        "execute_mode": str(getattr(args, "execute_mode", "projected")),
        "obs_ablation": (
            None
            if obs_ablation is None
            else {
                "ablate_value_slices": bool(obs_ablation.ablate_value_slices),
                "value_slices_fill": float(obs_ablation.value_slices_fill),
                "ablate_occupancy": bool(obs_ablation.ablate_occupancy),
                "occupancy_fill": float(obs_ablation.occupancy_fill),
                "ablate_gpath": bool(obs_ablation.ablate_gpath),
                "gpath_fill": float(obs_ablation.gpath_fill),
            }
        ),
        "sampling": {
            "episode_batch_size": int(episode_batch_size),
            "collision_semantic": str(collision_semantic),
            "projected_collision_stage": str(projected_collision_stage),
            "execute_mode": str(getattr(args, "execute_mode", "projected")),
            "snap_ranking": "hat",
            "opt_b_topk_children": int(getattr(args, "opt_b_topk_children", 0)),
            "anti_repeat": (
                None
                if anti_repeat is None
                else {
                    "enabled": bool(anti_repeat.enabled),
                    "xy_q": float(anti_repeat.xy_q),
                    "xy_recent_n": int(anti_repeat.xy_recent_n),
                    "child_recent_n": int(anti_repeat.child_recent_n),
                    "edge_recent_n": int(anti_repeat.edge_recent_n),
                    "avoid_uturn": bool(anti_repeat.avoid_uturn),
                }
            ),
        },
        "reason_counts": dict(reason_counts),
        "postcheck_fail_detail_counts": (None if str(projected_collision_stage) != "post" else dict(postcheck_fail_detail_counts)),
        "avg_resample_per_step": (float(total_resamples) / float(max(total_steps, 1))),
        "start_filter": {
            "source": ("file" if starts_file_path is not None else "random"),
            "starts_file": (None if starts_file_path is None else str(starts_file_path)),
            "starts_file_n": (None if starts_from_file is None else int(starts_from_file.shape[0])),
            "mode": str(getattr(args, "start_feasible_mode", "none")),
            "max_attempts": int(getattr(args, "start_max_attempts", 0)),
            "feasible_max_steps": (int(args.max_steps) if args.start_feasible_max_steps is None else int(args.start_feasible_max_steps)),
            "feasible_max_children_per_step": int(getattr(args, "start_feasible_max_children_per_step", 0)),
        },
        "opt_a": {
            "enabled": bool(args.opt_a),
            "delta": (None if not bool(args.opt_a) else float(args.opt_a_delta)),
            "cache_dir": (None if not bool(args.opt_a) else str(args.opt_a_cache_dir)),
            "boundary_wall_thickness_m": (None if not bool(args.opt_a) else float(getattr(args, "opt_a_boundary_wall_thickness_m", 0.0))),
            "use_inflated_fine": (None if not bool(args.opt_a) else bool(args.opt_a_use_inflated_fine)),
            "collision_check": (None if not bool(args.opt_a) else str(getattr(args, "opt_a_collision_check", "real"))),
            "overwrite_cache": (None if not bool(args.opt_a) else bool(args.opt_a_overwrite_cache)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[infer] wrote summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
