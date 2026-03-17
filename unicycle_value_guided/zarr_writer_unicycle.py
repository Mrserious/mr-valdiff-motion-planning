from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZarrWriterConfig:
    compressor: str = "none"  # ReplayBuffer.resolve_compressor: "none" | "default" | "disk"


class ZarrDatasetWriterUnicycle:
    """
    Thin wrapper around diffusion_policy.common.replay_buffer.ReplayBuffer.

    This writer is intentionally separate from `value_guided/zarr_writer.py` to avoid
    breaking the single-integrator pipeline while unicycle upgrades `gpath` to (4,3).

    Writes keys:
      - data/img   : (T,H,W,C) float32 in [0,255]
      - data/state : (T,3) float32
      - data/action: (T,2) float32
      - data/gpath : (T,4,3) float32
      - meta/episode_ends : int64
    """

    def __init__(self, zarr_path: str | Path, *, overwrite: bool, cfg: ZarrWriterConfig = ZarrWriterConfig()):
        self.zarr_path = Path(zarr_path)
        self.cfg = cfg

        try:
            import zarr  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "zarr is required to write datasets. Please install it in your Python env "
                "(for example, install dependencies from quick_demo/requirements.txt)."
            ) from e

        from diffusion_policy.common.replay_buffer import ReplayBuffer

        if overwrite and self.zarr_path.exists():
            # zarr is stored as a directory.
            import shutil

            shutil.rmtree(self.zarr_path)

        root = zarr.open(str(self.zarr_path), mode="a")
        self.buffer = ReplayBuffer.create_from_group(root)

    def add_episode(
        self,
        *,
        img: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        gpath: np.ndarray,
    ) -> None:
        img = np.asarray(img, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        gpath = np.asarray(gpath, dtype=np.float32)

        T = int(action.shape[0])
        if img.shape[0] != T or state.shape[0] != T or gpath.shape[0] != T:
            raise ValueError(f"Episode length mismatch: img{img.shape} state{state.shape} action{action.shape} gpath{gpath.shape}")
        if img.ndim != 4:
            raise ValueError(f"img must be (T,H,W,C), got {img.shape}")
        if state.shape[1:] != (3,):
            raise ValueError(f"state must be (T,3), got {state.shape}")
        if action.shape[1:] != (2,):
            raise ValueError(f"action must be (T,2), got {action.shape}")
        if gpath.shape[1:] != (4, 3):
            raise ValueError(f"gpath must be (T,4,3), got {gpath.shape}")

        data: dict[str, Any] = {
            "img": img,
            "state": state,
            "action": action,
            "gpath": gpath,
        }
        self.buffer.add_episode(data=data, compressors=self.cfg.compressor)
