from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from unicycle_value_guided.solve_value_grids import solve_values_for_goals


def _discover_goals_files(root: Path) -> list[Path]:
    goals_files = sorted(root.glob("*/goals.json"))
    return [p for p in goals_files if p.is_file()]


def _parse_csv(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _parse_index_spec(spec: str, n: int) -> list[int]:
    """
    spec examples:
      "0,1,2"
      "0-9"
      "0-9,15,20-25"
    """
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
    # keep order but drop duplicates
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


def _parse_gpus(spec: str) -> list[str]:
    """
    Physical GPU ids (as strings), e.g. "0,1" -> ["0","1"].
    """
    gpus = [s.strip() for s in (spec or "").split(",") if s.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def _load_goals_len(goals_json_path: Path) -> int:
    payload = json.loads(goals_json_path.read_text(encoding="utf-8"))
    goals = payload.get("goals", None)
    if not isinstance(goals, list):
        raise ValueError(f"Invalid goals.json (missing list goals): {goals_json_path}")
    return len(goals)


def _init_worker_env(gpu_id: str) -> None:
    """
    Initializer for ProcessPoolExecutor workers.

    Each worker process is pinned to a single *physical* GPU via CUDA_VISIBLE_DEVICES,
    so device strings like "cuda:0" will use that GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _solve_one_goal(
    *,
    goals_json_path: str,
    goal_index: int,
    opts: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.time()
    solve_values_for_goals(
        goals_json_path=Path(goals_json_path),
        levels=opts["levels"],
        grid_scheme=str(opts["grid_scheme"]),
        coarse_level=int(opts["coarse_level"]),
        fine_level=int(opts["fine_level"]),
        device=str(opts["device"]),
        dtype=str(opts["dtype"]),
        cell_size=float(opts["cell_size"]),
        cell_neighbor_radius=int(opts["cell_neighbor_radius"]),
        graph_chunk_nodes=int(opts["graph_chunk_nodes"]),
        vi_chunk_nodes=int(opts["vi_chunk_nodes"]),
        max_iters=int(opts["max_iters"]),
        tol=float(opts["tol"]),
        goal_index_spec=str(int(goal_index)),
        overwrite=bool(opts["overwrite"]),
        keep_pkl=bool(opts["keep_pkl"]),
    )
    return {
        "goals": str(goals_json_path),
        "goal_index": int(goal_index),
        "elapsed_sec": float(time.time() - t0),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multi-GPU concurrent runner for unicycle Prompt02 (CUDA VI).\n"
            "It schedules per-goal jobs across GPUs and can run multiple jobs per GPU.\n"
            "Note: fine level6 can be extremely heavy; tune --workers-per-gpu to avoid OOM."
        )
    )
    p.add_argument(
        "--root",
        type=str,
        default="data/unicycle_value_grids",
        help="Root directory containing per-map folders with goals.json (default: data/unicycle_value_grids).",
    )
    p.add_argument(
        "--maps",
        type=str,
        default="",
        help=(
            "Comma-separated map folder names to run (e.g. standard10x10_0000,standard10x10_0001). "
            "Empty means: auto-discover all maps under --root."
        ),
    )
    p.add_argument("--levels", type=str, default="coarse,fine", help="Comma-separated: coarse,fine")
    p.add_argument("--grid-scheme", type=str, default="multigrid", choices=["legacy", "multigrid"])
    p.add_argument("--coarse-level", type=int, default=2)
    p.add_argument("--fine-level", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda:0", help='Device string passed to unicycle_value_cuda (use "cuda:0").')
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    p.add_argument("--cell-size", type=float, default=0.0, help="Bucket cell size (0 => use solver rho).")
    p.add_argument("--cell-neighbor-radius", type=int, default=1)
    p.add_argument("--graph-chunk-nodes", type=int, default=2048)
    p.add_argument("--vi-chunk-nodes", type=int, default=8192)
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument(
        "--goal-indices",
        type=str,
        default="",
        help='Optional subset, e.g. "0-9" or "0,5,10-20". Empty means all goals.',
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute even if artifacts already exist.")
    p.add_argument("--keep-pkl", action="store_true", help="Also hardlink/copy vi_robot_{level}.pkl to goal dir root.")
    p.add_argument(
        "--gpus",
        "--cuda-visible-devices",
        type=str,
        default="0",
        help='Physical GPU ids (comma-separated), e.g. "0,1". Alias: --cuda-visible-devices.',
    )
    p.add_argument("--workers-per-gpu", type=int, default=1, help="Concurrent per-goal jobs per GPU (default: 1).")
    p.add_argument("--shuffle", action="store_true", default=True, help="Shuffle goal tasks to balance load (default: on).")
    p.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="Disable task shuffle.")
    p.add_argument("--dry-run", action="store_true", help="Only print planned task count and exit.")
    p.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="multiprocessing start method (default: spawn; recommended for CUDA).",
    )
    p.add_argument("--print-every-sec", type=float, default=10.0, help="Progress print interval (seconds).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    map_names = _parse_csv(args.maps)
    if map_names:
        goals_files = []
        for name in map_names:
            goals_path = root / name / "goals.json"
            if not goals_path.exists():
                raise SystemExit(f"Missing goals.json: {goals_path}")
            goals_files.append(goals_path)
    else:
        goals_files = _discover_goals_files(root)
        if not goals_files:
            raise SystemExit(f"No goals.json found under: {root}")

    levels = _parse_csv(args.levels)
    if not levels:
        raise SystemExit("--levels cannot be empty.")

    gpus = _parse_gpus(args.gpus)
    workers_per_gpu = int(max(1, args.workers_per_gpu))

    tasks: list[tuple[str, int]] = []
    for goals_path in goals_files:
        n_goals = _load_goals_len(goals_path)
        for goal_idx in _parse_index_spec(args.goal_indices, n_goals):
            tasks.append((str(goals_path), int(goal_idx)))

    if bool(args.shuffle):
        random.shuffle(tasks)

    print(
        f"[run_all_value_grids] maps={len(goals_files)} goals_per_map=varies total_jobs={len(tasks)} "
        f"gpus={gpus} workers_per_gpu={workers_per_gpu}",
        flush=True,
    )
    if bool(args.dry_run):
        return

    import multiprocessing as mp

    ctx = mp.get_context(str(args.start_method))

    opts = {
        "levels": levels,
        "grid_scheme": str(args.grid_scheme),
        "coarse_level": int(args.coarse_level),
        "fine_level": int(args.fine_level),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "cell_size": float(args.cell_size),
        "cell_neighbor_radius": int(args.cell_neighbor_radius),
        "graph_chunk_nodes": int(args.graph_chunk_nodes),
        "vi_chunk_nodes": int(args.vi_chunk_nodes),
        "max_iters": int(args.max_iters),
        "tol": float(args.tol),
        "overwrite": bool(args.overwrite),
        "keep_pkl": bool(args.keep_pkl),
    }

    start = time.time()
    ok = 0
    failures: list[dict[str, Any]] = []
    future_to_task: dict[Future, tuple[str, int, str]] = {}

    executors: list[tuple[str, ProcessPoolExecutor]] = []
    for gpu_id in gpus:
        exe = ProcessPoolExecutor(
            max_workers=workers_per_gpu,
            mp_context=ctx,
            initializer=_init_worker_env,
            initargs=(gpu_id,),
        )
        executors.append((gpu_id, exe))

    try:
        for idx, (goals_json_path, goal_idx) in enumerate(tasks):
            gpu_id, exe = executors[idx % len(executors)]
            fut = exe.submit(_solve_one_goal, goals_json_path=goals_json_path, goal_index=goal_idx, opts=opts)
            future_to_task[fut] = (goals_json_path, goal_idx, gpu_id)

        last_print = time.time()
        for fut in as_completed(future_to_task):
            goals_json_path, goal_idx, gpu_id = future_to_task[fut]
            try:
                _ = fut.result()
                ok += 1
            except Exception as e:
                failures.append(
                    {
                        "goals": goals_json_path,
                        "goal_index": int(goal_idx),
                        "gpu": gpu_id,
                        "error": repr(e),
                    }
                )

            now = time.time()
            if (now - last_print) >= float(max(args.print_every_sec, 0.5)) or (ok + len(failures) == len(tasks)):
                done = ok + len(failures)
                elapsed = now - start
                rate = done / max(elapsed, 1e-6)
                print(
                    f"[run_all_value_grids] done={done}/{len(tasks)} ok={ok} err={len(failures)} "
                    f"rate={rate:.2f} tasks/s elapsed={elapsed/60:.1f} min",
                    flush=True,
                )
                last_print = now
    finally:
        for _, exe in executors:
            exe.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - start
    print(f"[run_all_value_grids] finished ok={ok} err={len(failures)} elapsed={elapsed/3600:.2f} hours", flush=True)
    if failures:
        print("[run_all_value_grids] failures (first 20):", flush=True)
        for item in failures[:20]:
            print(
                f"  - gpu={item.get('gpu')} goals={item.get('goals')} goal={item.get('goal_index')} err={item.get('error')}",
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
