"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import tempfile

# Work around Linux multiprocessing "AF_UNIX path too long" errors.
#
# Some workflows export a variable named `TMP` for dataset generation scripts
# (for example, a temporary zarr directory). Python's tempfile/multiprocessing also
# consults the `TMP` env var to decide where to create AF_UNIX sockets for FD sharing.
# If `TMP` is set to a long/relative path (and Hydra may chdir), the resulting socket
# path can exceed the AF_UNIX limit (~108 bytes) and crash DataLoader workers.
#
# We defensively force a short temp dir when the configured temp dir looks risky.
def _ensure_short_tempdir() -> None:
    env_tmp = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
    tmpdir = env_tmp or tempfile.gettempdir()
    # Leave some headroom for "/pymp-XXXX/listener-XXXX" etc.
    if len(tmpdir) <= 60 and os.path.isabs(tmpdir):
        return

    for cand in ("/tmp", "/dev/shm"):
        try:
            if os.path.isdir(cand) and os.access(cand, os.W_OK | os.X_OK):
                os.environ["TMPDIR"] = cand
                os.environ["TEMP"] = cand
                os.environ["TMP"] = cand
                tempfile.tempdir = cand
                return
        except Exception:
            continue


_ensure_short_tempdir()

try:
    import torch.multiprocessing as _tmp_mp

    _tmp_mp.set_sharing_strategy("file_system")
except Exception:
    # Best-effort; training can still run with the default strategy in many environments.
    pass

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    # Hydra config files are stored in the local `config/` folder.
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
