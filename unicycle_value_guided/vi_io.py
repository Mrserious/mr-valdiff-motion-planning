from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _PickleCompatRule:
    src_prefix: str
    dst_prefix: str


_PICKLE_COMPAT_RULES: tuple[_PickleCompatRule, ...] = (
    # Numpy 2.0 moved internal modules from numpy.core -> numpy._core.
    # Some of our cached VI artifacts were pickled in an environment with numpy>=2,
    # but many runtime environments still ship numpy 1.x, where numpy._core doesn't exist.
    _PickleCompatRule(src_prefix="numpy._core", dst_prefix="numpy.core"),
    # scikit-learn moved dist metrics implementation modules across versions.
    # Newer pickles may reference sklearn.metrics._dist_metrics, but older runtimes
    # (e.g. 0.22) expose the implementation under sklearn.neighbors._dist_metrics.
    _PickleCompatRule(src_prefix="sklearn.metrics._dist_metrics", dst_prefix="sklearn.neighbors._dist_metrics"),
)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        mod = str(module)
        for rule in _PICKLE_COMPAT_RULES:
            if mod == rule.src_prefix or mod.startswith(rule.src_prefix + "."):
                mod = rule.dst_prefix + mod[len(rule.src_prefix) :]
                break
        # sklearn sometimes appends precision suffixes (e.g. EuclideanDistance64) that don't
        # exist in older versions. Fall back to the unsuffixed name when possible.
        if mod.endswith("._dist_metrics") and (str(name).endswith("32") or str(name).endswith("64")):
            base = str(name)[:-2]
            try:
                return super().find_class(mod, base)
            except Exception:
                pass
        return super().find_class(mod, name)


def load_vi_robot(vi_robot_pkl: str | Path) -> Any:
    vi_robot_pkl = Path(vi_robot_pkl)
    with vi_robot_pkl.open("rb") as fh:
        try:
            obj = pickle.load(fh)
        except ModuleNotFoundError as e:
            missing = str(getattr(e, "name", "") or "")
            # Retry with compatibility mappings (e.g. numpy._core -> numpy.core).
            if any(
                missing == rule.src_prefix or missing.startswith(rule.src_prefix + ".")
                for rule in _PICKLE_COMPAT_RULES
            ):
                fh.seek(0)
                obj = _CompatUnpickler(fh).load()
            else:
                raise
    if isinstance(obj, dict) and "robot" in obj:
        robot = obj["robot"]
    else:
        robot = obj
    try:
        robot.update_kdtree()
    except Exception:
        pass
    return robot
