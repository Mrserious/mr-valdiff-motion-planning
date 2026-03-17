from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .collision import obstacle_free_mask_unicycle
from .cuda_kernels.python.cuda_graph_builder import cuda_module
from .grid import available_levels, build_state_grid
from .task_io import get_range_limits, load_task
from .unicycle import Unicycle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unicycle CUDA value solver (node-level)")
    parser.add_argument("--task", type=str, required=True, help="Path to task json.")
    parser.add_argument("--grid-scheme", type=str, default="legacy", choices=["legacy", "multigrid"], help="Grid scheme.")
    parser.add_argument("--level", type=int, default=0, help="Grid level index (scheme-dependent).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda / cuda:0 / cpu.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"], help="Value dtype.")
    parser.add_argument("--log-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--cell-size", type=float, default=0.0, help="Bucket cell size (0 => use rho).")
    parser.add_argument("--cell-neighbor-radius", type=int, default=1, help="Neighbor cell Chebyshev radius.")
    parser.add_argument("--graph-chunk-nodes", type=int, default=2048, help="Nodes per CUDA graph build chunk.")
    parser.add_argument("--vi-chunk-nodes", type=int, default=8192, help="Nodes per CUDA VI sub-chunk.")
    parser.add_argument("--max-iters", type=int, default=500, help="Max sweeps.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance (used when strict-zero disabled).")
    parser.add_argument("--strict-zero", action="store_true", help="Stop only when max delta == 0 for patience steps.")
    parser.add_argument("--zero-patience", type=int, default=3, help="Strict-zero patience.")
    parser.add_argument("--summary", type=str, default="", help="Optional summary json path.")
    return parser.parse_args()


def _build_neighbor_offsets(radius: int) -> List[Tuple[int, int]]:
    r = int(max(0, radius))
    offsets: List[Tuple[int, int]] = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            offsets.append((dx, dy))
    return offsets


def _build_cell_index(states_xy: np.ndarray, origin_xy: np.ndarray, cell_size: float) -> Dict[Tuple[int, int], np.ndarray]:
    cell_size = float(max(cell_size, 1e-8))
    rel = (states_xy - origin_xy[None, :]) / cell_size
    cells = np.floor(rel).astype(np.int32)
    bucket: Dict[Tuple[int, int], List[int]] = {}
    for idx, (bx, by) in enumerate(cells.tolist()):
        key = (int(bx), int(by))
        bucket.setdefault(key, []).append(idx)
    return {k: np.asarray(v, dtype=np.int64) for k, v in bucket.items()}


def _bucket_tensors(
    bucket_map: Dict[Tuple[int, int], np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bucket_map:
        empty_keys = torch.empty((0, 2), dtype=torch.int32, device=device)
        empty_indices = torch.empty((0,), dtype=torch.long, device=device)
        empty_offsets = torch.zeros((1,), dtype=torch.long, device=device)
        return empty_keys, empty_indices, empty_offsets

    items = sorted(bucket_map.items())
    key_array = np.array([[k[0], k[1]] for k, _ in items], dtype=np.int32)
    bucket_keys = torch.tensor(key_array, dtype=torch.int32, device=device)
    offsets = [0]
    indices_list: List[torch.Tensor] = []
    for _, indices in items:
        if indices.size == 0:
            offsets.append(offsets[-1])
            continue
        tensor = torch.tensor(indices, dtype=torch.long, device=device)
        indices_list.append(tensor)
        offsets.append(offsets[-1] + tensor.numel())
    bucket_indices = torch.cat(indices_list, dim=0) if indices_list else torch.empty((0,), dtype=torch.long, device=device)
    bucket_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    return bucket_keys, bucket_indices, bucket_offsets


def _edges_to_csr_unique(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    chunk_start: int,
    chunk_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if src.size == 0:
        row_offsets = torch.zeros((chunk_nodes + 1,), dtype=torch.long)
        col_indices = torch.empty((0,), dtype=torch.int32)
        return row_offsets, col_indices

    order = np.lexsort((dst, src))
    src_s = src[order]
    dst_s = dst[order]
    keep = np.ones((src_s.size,), dtype=bool)
    keep[1:] = (src_s[1:] != src_s[:-1]) | (dst_s[1:] != dst_s[:-1])
    src_u = src_s[keep]
    dst_u = dst_s[keep]

    rel_src = (src_u - int(chunk_start)).astype(np.int64)
    counts = np.bincount(rel_src, minlength=int(chunk_nodes)).astype(np.int64)
    row_offsets = np.zeros((int(chunk_nodes) + 1,), dtype=np.int64)
    row_offsets[1:] = np.cumsum(counts, axis=0)
    return torch.tensor(row_offsets, dtype=torch.long), torch.tensor(dst_u.astype(np.int32), dtype=torch.int32)


def _edges_to_csr_unique_gpu(
    edges: torch.Tensor,
    *,
    chunk_start: int,
    chunk_nodes: int,
    total_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if edges.numel() == 0:
        row_offsets = torch.zeros((chunk_nodes + 1,), dtype=torch.long, device=edges.device)
        col_indices = torch.empty((0,), dtype=torch.int32, device=edges.device)
        return row_offsets, col_indices

    if edges.dim() != 2 or edges.size(0) != 2:
        raise ValueError(f"edges must be (2,E), got {tuple(edges.shape)}")
    if not edges.is_cuda:
        raise ValueError("edges must be CUDA tensor")
    if edges.dtype not in (torch.int64, torch.long):
        raise ValueError(f"edges must be int64, got {edges.dtype}")

    src = edges[0]
    dst = edges[1]
    key = src * int(total_nodes) + dst
    uniq_key = torch.unique(key, sorted=True)
    src_u = torch.div(uniq_key, int(total_nodes), rounding_mode="floor")
    dst_u = uniq_key - src_u * int(total_nodes)

    rel_src = (src_u - int(chunk_start)).to(torch.int64)
    counts = torch.bincount(rel_src, minlength=int(chunk_nodes))
    row_offsets = torch.zeros((int(chunk_nodes) + 1,), dtype=torch.long, device=edges.device)
    row_offsets[1:] = torch.cumsum(counts, dim=0)
    col_indices = dst_u.to(torch.int32)
    return row_offsets, col_indices


def _assign_children_from_chunks(robot: Unicycle, chunk_graphs: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> None:
    nodes = robot.nodes
    for chunk_start, row_offsets_cpu, col_indices in chunk_graphs:
        col_indices_cpu = col_indices.to("cpu", non_blocking=False) if col_indices.is_cuda else col_indices
        row_offsets = row_offsets_cpu.tolist()
        cols = col_indices_cpu.tolist()
        local_nodes = len(row_offsets) - 1
        for li in range(local_nodes):
            gidx = int(chunk_start + li)
            beg = int(row_offsets[li])
            end = int(row_offsets[li + 1])
            node = nodes[gidx]
            node.children.indices = cols[beg:end]
            node.children.update_iteration = -1


def run_pipeline() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("MVP requires CUDA device to build graph/value on GPU.")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    levels = available_levels(scheme=str(args.grid_scheme))
    if int(args.level) not in levels:
        raise ValueError(f"Unsupported level={args.level} for scheme={args.grid_scheme!r}. Available: {levels}")
    task = load_task(args.task)
    env = task.env
    robots = task.robots
    if not robots:
        raise ValueError("Task has no robots.")
    robot_meta = robots[0]

    (x0, x1), _ = get_range_limits(env)
    angle_scalor = (x1 - x0) / 2.0
    robot = Unicycle(env, robot_meta, angle_scalor=angle_scalor, robot_id=0)

    # 1) Build grid nodes (half-open boundary convention).
    raw_states, grid_spec = build_state_grid(
        env,
        angle_scalor=angle_scalor,
        level=int(args.level),
        scheme=str(args.grid_scheme),
    )

    # 2) Filter collision states (not swept), aligned with notebook semantics.
    mask = obstacle_free_mask_unicycle(
        env=env,
        body_samples_xy=robot.body.samples,
        states=raw_states,
        angle_scalor=angle_scalor,
    )
    states = raw_states[mask]

    # 3) Ensure goal exists by explicitly inserting goal_state (if collision-free).
    goal_state = robot.goal_state.astype(np.float32)
    if obstacle_free_mask_unicycle(
        env=env,
        body_samples_xy=robot.body.samples,
        states=goal_state.reshape(1, 3),
        angle_scalor=angle_scalor,
    )[0]:
        already_present = bool(np.any(np.all(states == goal_state.reshape(1, 3), axis=1)))
        if not already_present:
            states = np.vstack([goal_state.reshape(1, 3), states])

    for s in states:
        robot.add_node_state(s)
    robot.init_nodes()
    robot.update_kdtree()

    d_val = robot.get_spatial_res()
    dt = robot.get_temporal_res()
    rho = robot.get_perturbation_radius()

    cell_size = float(args.cell_size) if float(args.cell_size) > 0 else float(rho)
    origin_xy = np.array([min(env["range"]["limits"][0]), min(env["range"]["limits"][1])], dtype=np.float32)
    bucket_map = _build_cell_index(states[:, :2].astype(np.float32), origin_xy, cell_size)
    neighbor_offsets = _build_neighbor_offsets(int(args.cell_neighbor_radius))

    module = cuda_module()
    node_states_t = torch.tensor(states, dtype=torch.float32, device=device)
    controls_t = torch.tensor(np.asarray(robot.controls, dtype=np.float32), dtype=torch.float32, device=device)
    bucket_keys_t, bucket_indices_t, bucket_offsets_t = _bucket_tensors(bucket_map, device=device)
    neighbor_t = torch.tensor(np.asarray(neighbor_offsets, dtype=np.int32), dtype=torch.int32, device=device)
    origin_t = torch.tensor(origin_xy, dtype=torch.float32, device=device)

    # 4) Build graph in chunks.
    total_nodes = states.shape[0]
    chunk_limit = int(max(1, args.graph_chunk_nodes))
    chunk_graphs: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    total_edges_unique = 0
    build_start = time.time()
    chunk_start = 0
    while chunk_start < total_nodes:
        chunk_count = min(chunk_limit, total_nodes - chunk_start)
        edges = module.build_graph(
            node_states_t,
            controls_t,
            bucket_keys_t,
            bucket_indices_t,
            bucket_offsets_t,
            neighbor_t,
            origin_t,
            float(cell_size),
            float(dt),
            float(rho),
            float(angle_scalor),
            int(chunk_start),
            int(chunk_count),
        )
        row_offsets_gpu, col_indices_gpu = _edges_to_csr_unique_gpu(
            edges,
            chunk_start=chunk_start,
            chunk_nodes=chunk_count,
            total_nodes=total_nodes,
        )
        row_offsets_cpu = row_offsets_gpu.to("cpu", non_blocking=False)
        col_indices_gpu = col_indices_gpu.contiguous()
        total_edges_unique += int(col_indices_gpu.numel())
        chunk_graphs.append((int(chunk_start), row_offsets_cpu, col_indices_gpu))
        chunk_start += chunk_count
        del edges
        torch.cuda.empty_cache()

    build_sec = time.time() - build_start

    # 5) Value iteration (global sweep) until convergence.
    goal_mask = torch.tensor([bool(robot.within_goal(n.state)) for n in robot.nodes], dtype=torch.bool, device=device)
    values = torch.tensor([float(n.value) for n in robot.nodes], dtype=dtype, device=device)
    delta = 1.0 - math.exp(-(dt - d_val))
    beta = 1.0 - delta

    vi_start = time.time()
    current = values.clone()
    next_values = torch.empty_like(current)
    vi_chunk_limit = int(max(1, args.vi_chunk_nodes))
    delta_buffer = torch.empty((vi_chunk_limit,), dtype=torch.float32, device=device)
    iterations_done = 0
    zero_hits = 0
    last_delta = 0.0

    strict_zero = bool(args.strict_zero)
    patience = int(max(1, args.zero_patience))
    tol = float(args.tol)

    while iterations_done < int(max(0, args.max_iters)):
        max_delta_iter = 0.0
        for chunk_start, row_offsets_cpu, col_indices_gpu in chunk_graphs:
            chunk_nodes_total = int(row_offsets_cpu.size(0) - 1)
            if chunk_nodes_total <= 0:
                continue
            local_start = 0
            while local_start < chunk_nodes_total:
                local_nodes = min(vi_chunk_limit, chunk_nodes_total - local_start)
                row_slice_start = int(row_offsets_cpu[local_start].item())
                row_slice_end = int(row_offsets_cpu[local_start + local_nodes].item())
                local_rows_cpu = (row_offsets_cpu[local_start : local_start + local_nodes + 1] - row_offsets_cpu[local_start]).clone()
                local_cols_gpu = col_indices_gpu.narrow(0, row_slice_start, row_slice_end - row_slice_start)

                delta_view = delta_buffer.narrow(0, 0, local_nodes)
                chunk_row_gpu = local_rows_cpu.to(device=device, non_blocking=True)
                chunk_col_gpu = local_cols_gpu

                module.value_iteration_chunk(
                    int(chunk_start + local_start),
                    chunk_row_gpu,
                    chunk_col_gpu,
                    goal_mask,
                    current,
                    next_values,
                    float(delta),
                    float(beta),
                    delta_view,
                )
                chunk_delta = float(delta_view.max().item()) if local_nodes > 0 else 0.0
                if chunk_delta > max_delta_iter:
                    max_delta_iter = chunk_delta
                local_start += local_nodes

        iterations_done += 1
        current, next_values = next_values, current
        last_delta = max_delta_iter

        if strict_zero:
            if max_delta_iter == 0.0:
                zero_hits += 1
                if zero_hits >= patience:
                    break
            else:
                zero_hits = 0
        else:
            if max_delta_iter <= tol:
                break

    vi_sec = time.time() - vi_start

    values_cpu = current.to("cpu")

    # 6) Write back values/children and save robot.
    for idx, node in enumerate(robot.nodes):
        node.value = float(values_cpu[idx].item())
    out_pkl = log_dir / "vi_robot.pkl"
    tmp_pkl = out_pkl.with_name(f".{out_pkl.name}.tmp.{os.getpid()}")
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    try:
        _assign_children_from_chunks(robot, chunk_graphs)
        with tmp_pkl.open("wb") as fh:
            import pickle  # noqa: WPS433

            pickle.dump({"robot": robot}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
        os.replace(tmp_pkl, out_pkl)
    finally:
        if gc_enabled:
            gc.enable()
        try:
            if tmp_pkl.exists():
                tmp_pkl.unlink()
        except Exception:
            pass

    summary = {
        "task": str(Path(args.task).resolve()),
        "level": int(args.level),
        "grid_total_raw": int(grid_spec.total),
        "nodes_after_filter": int(states.shape[0]),
        "dt": float(dt),
        "rho": float(rho),
        "d": float(d_val),
        "delta": float(delta),
        "beta": float(beta),
        "cell_size": float(cell_size),
        "cell_neighbor_radius": int(args.cell_neighbor_radius),
        "controls": int(len(robot.controls)),
        "graph_chunk_nodes": int(args.graph_chunk_nodes),
        "vi_chunk_nodes": int(args.vi_chunk_nodes),
        "unique_edges_total": int(total_edges_unique),
        "vi_iterations": int(iterations_done),
        "vi_last_delta": float(last_delta),
        "timing_sec": {
            "total": float(time.time() - t0),
            "graph_build": float(build_sec),
            "value_iteration": float(vi_sec),
        },
        "output": {
            "vi_robot_pkl": str(out_pkl),
        },
    }

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_pipeline()
