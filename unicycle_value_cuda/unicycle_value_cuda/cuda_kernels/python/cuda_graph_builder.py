from __future__ import annotations

import importlib.util
import hashlib
import os
import pathlib
import time
from types import ModuleType

from torch.utils.cpp_extension import load as load_extension

_MODULE: ModuleType | None = None


def _load_prebuilt(so_path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("unicycle_cuda", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_extension() -> ModuleType:
    source_dir = pathlib.Path(__file__).resolve().parents[1]
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"
        print(
            "[unicycle_cuda] TORCH_CUDA_ARCH_LIST not set; defaulting to 8.6+PTX to support CUDA 11.8 nvcc",
            flush=True,
        )
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    arch_tag = hashlib.sha1(arch_list.encode("utf-8")).hexdigest()[:10] if arch_list else "default"
    build_dir = source_dir / f"build_{arch_tag}"
    build_dir.mkdir(exist_ok=True)

    lock_path = build_dir / "lock"
    so_path = build_dir / "unicycle_cuda.so"
    if so_path.exists():
        if lock_path.exists():
            print(f"[unicycle_cuda] build lock exists at {lock_path}; loading prebuilt {so_path}", flush=True)
        try:
            return _load_prebuilt(so_path)
        except Exception as e:
            print(f"[unicycle_cuda] failed to load prebuilt {so_path}: {e!r}; falling back to build", flush=True)

    if lock_path.exists():
        try:
            age_sec = time.time() - lock_path.stat().st_mtime
            if age_sec > 3600:
                print(f"[unicycle_cuda] removing stale build lock {lock_path} (age={age_sec:.0f}s)", flush=True)
                lock_path.unlink()
        except Exception:
            pass

    module = load_extension(
        name="unicycle_cuda",
        sources=[
            str(source_dir / "src" / "bindings.cpp"),
            str(source_dir / "src" / "unicycle_candidate_kernels.cu"),
            str(source_dir / "src" / "value_iteration_kernels.cu"),
        ],
        extra_include_paths=[str(source_dir / "include")],
        build_directory=str(build_dir),
        verbose=True,
    )
    return module


def cuda_module() -> ModuleType:
    global _MODULE
    if _MODULE is None:
        _MODULE = _build_extension()
    return _MODULE
