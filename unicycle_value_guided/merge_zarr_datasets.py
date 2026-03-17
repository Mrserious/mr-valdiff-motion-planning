from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from unicycle_value_guided.zarr_writer_unicycle import ZarrDatasetWriterUnicycle, ZarrWriterConfig


def _parse_csv(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _parse_index_spec(spec: str) -> list[int]:
    """
    spec examples:
      "0,1,2"
      "0-9"
      "0-9,15,20-25"
    """
    spec = (spec or "").strip()
    if not spec:
        return []
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
        seen.add(i)
        dedup.append(i)
    return dedup


def _iter_episode_slices(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    episode_ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    starts = np.concatenate([np.zeros((1,), dtype=np.int64), episode_ends[:-1]], axis=0)
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge (concatenate) multiple unicycle zarr datasets into one, preserving episode order.\n"
            "Use this with per-map zarr outputs to safely parallelize make_dataset."
        )
    )
    p.add_argument("--out", type=str, required=True, help="Output zarr path (directory store).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output zarr if exists.")
    p.add_argument(
        "--compressor",
        type=str,
        default="disk",
        choices=["none", "default", "disk"],
        help='Destination compressor preset (ReplayBuffer.resolve_compressor). Default: "disk".',
    )

    # Two ways to specify sources:
    p.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated input zarr paths (in the desired append order).",
    )
    p.add_argument(
        "--tmp-dir",
        type=str,
        default="",
        help="Directory containing per-map zarr files, e.g. data/zarr/tmp_xxx",
    )
    p.add_argument("--maps", type=str, default="", help='Comma-separated map names, e.g. "standard10x10_0000,standard10x10_0001".')
    p.add_argument("--map-range", type=str, default="", help='Index spec, e.g. "0-29". Used with --map-format.')
    p.add_argument(
        "--map-format",
        type=str,
        default="standard10x10_{i:04d}",
        help="Python format string with {i}, e.g. standard10x10_{i:04d}",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for debugging (stop after writing this many episodes).",
    )
    p.add_argument("--print-every", type=int, default=100, help="Progress print interval in episodes.")
    return p


def _resolve_sources(args: argparse.Namespace) -> list[Path]:
    sources = _parse_csv(str(args.sources))
    if sources:
        return [Path(s) for s in sources]

    tmp_dir = Path(str(args.tmp_dir)) if str(args.tmp_dir).strip() else None
    if tmp_dir is None:
        raise SystemExit("Must provide either --sources or --tmp-dir (with --maps or --map-range).")

    maps = _parse_csv(str(args.maps))
    if not maps:
        idxs = _parse_index_spec(str(args.map_range))
        if not idxs:
            raise SystemExit("When using --tmp-dir, must provide --maps or --map-range.")
        fmt = str(args.map_format)
        try:
            maps = [fmt.format(i=int(i)) for i in idxs]
        except Exception as e:
            raise SystemExit(f"Failed to apply --map-format={fmt!r}: {e}") from e

    return [tmp_dir / f"{m}.zarr" for m in maps]


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and bool(args.overwrite):
        shutil.rmtree(out_path)
    elif out_path.exists() and not bool(args.overwrite):
        raise SystemExit(f"--out already exists (use --overwrite): {out_path}")

    sources = _resolve_sources(args)
    if not sources:
        raise SystemExit("No sources resolved.")

    for p in sources:
        if not p.exists():
            raise SystemExit(f"Missing source zarr: {p}")

    writer = ZarrDatasetWriterUnicycle(
        out_path,
        overwrite=bool(args.overwrite),
        cfg=ZarrWriterConfig(compressor=str(args.compressor)),
    )

    t0 = time.time()
    written = 0
    import zarr  # type: ignore

    max_episodes = int(args.max_episodes) if args.max_episodes is not None else None
    print_every = int(max(1, args.print_every))

    for src_idx, src_path in enumerate(sources):
        src_root = zarr.open(str(src_path), mode="r")
        try:
            episode_ends = np.asarray(src_root["meta"]["episode_ends"][:], dtype=np.int64)
        except Exception as e:
            raise SystemExit(f"Invalid source zarr (missing meta/episode_ends): {src_path} ({e})") from e

        slices = _iter_episode_slices(episode_ends)
        print(f"[merge_zarr] src={src_idx+1}/{len(sources)} path={src_path} episodes={len(slices)}", flush=True)

        for ep_i, (a, b) in enumerate(slices):
            img = np.asarray(src_root["data"]["img"][a:b], dtype=np.float32)
            state = np.asarray(src_root["data"]["state"][a:b], dtype=np.float32)
            action = np.asarray(src_root["data"]["action"][a:b], dtype=np.float32)
            gpath = np.asarray(src_root["data"]["gpath"][a:b], dtype=np.float32)

            writer.add_episode(img=img, state=state, action=action, gpath=gpath)
            written += 1

            if (written % print_every) == 0:
                elapsed = time.time() - t0
                print(f"[merge_zarr] wrote={written} elapsed={elapsed/60:.1f} min", flush=True)

            if max_episodes is not None and written >= max_episodes:
                elapsed = time.time() - t0
                print(f"[merge_zarr] reached --max-episodes ({max_episodes}), stopping. elapsed={elapsed/60:.1f} min", flush=True)
                return

    elapsed = time.time() - t0
    print(f"[merge_zarr] done wrote={written} elapsed={elapsed/3600:.2f} hours out={out_path}", flush=True)


if __name__ == "__main__":
    main()

