from typing import Dict
import torch
import numpy as np
import copy
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
	    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

_BLOSC_THREADS_CONFIGURED = False


def _ensure_blosc_nthreads(nthreads: int) -> None:
    """
    Zarr uses Blosc for compression/decompression. By default Blosc may use multiple threads.
    With PyTorch DataLoader(num_workers>0), this can cause severe CPU oversubscription.
    """
    global _BLOSC_THREADS_CONFIGURED
    if _BLOSC_THREADS_CONFIGURED:
        return
    try:
        import numcodecs.blosc as blosc

        blosc.set_nthreads(int(nthreads))
    except Exception:
        # Best-effort: if numcodecs is unavailable, just skip.
        pass
    _BLOSC_THREADS_CONFIGURED = True


class NavImageDataset(BaseImageDataset):
    def __init__(self,
	        zarr_path, 
            in_memory=True,
	        horizon=1,
	        pad_before=0,
	        pad_after=0,
	        seed=42,
	        val_ratio=0.0,
            max_train_episodes=None,
            n_obs_steps=None,
            blosc_nthreads: int = 1,
            ):
        
        super().__init__()
        zarr_path = os.path.expanduser(zarr_path)
        sample_keys = ['img', 'state', 'action', 'gpath']

        # Backward compatible `in_memory`:
        # - bool: True -> "numpy" (fully decompressed in RAM), False -> "disk"
        # - str: "disk" | "numpy" | "compressed"
        mode = in_memory
        if isinstance(in_memory, bool):
            mode = "numpy" if in_memory else "disk"
        mode = str(mode).lower()
        if mode in ("numpy", "ram", "decompressed", "true"):
            # Copy dataset into RAM as numpy arrays (fast per-step, but can take a long time and huge memory).
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=sample_keys)  # 'gmap' 'img',
        elif mode in ("compressed", "memstore", "zarr", "mem"):
            # Copy the compressed zarr store into RAM (moderate memory, avoids slow filesystem I/O).
            import zarr

            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, store=zarr.MemoryStore(), keys=sample_keys)
        elif mode in ("disk", "false"):
            # Read directly from on-disk zarr (starts quickly; per-step can be more I/O bound).
            self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        else:
            raise ValueError(f"Unsupported in_memory={in_memory!r}. Use: false|disk|compressed|numpy.")

        key_first_k = dict()
        if n_obs_steps is not None:
            n_obs_steps = int(n_obs_steps)
            # performance optimization: only load the first k observation steps
            # (the policy conditions only on the first n_obs_steps)
            for key in ['img', 'state', 'gpath']:
                key_first_k[key] = n_obs_steps
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=sample_keys,
            key_first_k=key_first_k,
            key_first_k_trim=True)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sample_keys = sample_keys
        self.key_first_k = key_first_k
        self.n_obs_steps = n_obs_steps
        self.blosc_nthreads = blosc_nthreads

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            keys=self.sample_keys,
            key_first_k=self.key_first_k,
            key_first_k_trim=True
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'global_path': self.replay_buffer['gpath'],
            # 'global_map': self.replay_buffer['gmap'],
            # 'reward': self.replay_buffer['reward'].reshape(-1, 1),
            # 'return_to_go': self.replay_buffer['rtg'].reshape(-1, 1),
            # 'lmf': self.replay_buffer['lmf'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        # normalizer['global_map'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        global_path = sample['gpath'].astype(np.float32)
        # global_map = np.moveaxis(sample['gmap'],-1,1)/255
        image_arr = sample['img']
        if self.n_obs_steps is not None:
            # only keep the first k obs steps to reduce CPU<->GPU transfer and pin_memory pressure
            k = int(self.n_obs_steps)
            agent_pos = agent_pos[:k]
            global_path = global_path[:k]
            image_arr = image_arr[:k]
        image = np.moveaxis(image_arr, -1, 1) / 255
        
        data = {
            'obs': {
                'image': image, # T, 1, 84, 84
                'agent_pos': agent_pos, # T, 3
                'global_path': global_path,
                # 'global_map': global_map,
            },
            'action': sample['action'].astype(np.float32) # T, 2
            # 'trajectory': {
            #     'lmf': sample['lmf'].astype(np.float32), #T, 64
            #     'action': sample['action'].astype(np.float32), # T, 2
            #     'reward': sample['reward'].astype(np.float32).reshape(-1, 1),
            #     'return_to_go': sample['rtg'].astype(np.float32).reshape(-1, 1)
            # }
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.blosc_nthreads is not None:
            _ensure_blosc_nthreads(self.blosc_nthreads)
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/ywh/diffusion_policy/data/static6_ped4_acc.zarr')
    dataset = NavImageDataset(zarr_path, horizon=16)

    # import zarr
    # store = zarr.DirectoryStore(zarr_path)
    # root = zarr.open(store, mode='r')
    # print("Shape:", root['data']['action'][1])

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
