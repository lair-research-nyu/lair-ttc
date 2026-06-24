from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TTCWindowDataset(Dataset):
    """
    Sliding-window dataset over cached DINO-v2 features.

    Each sample (sparse mode, default):
        x : (K, 768) float32  — K consecutive [CLS] features
        y : scalar float32    — log1p(ttc_s) for the LAST frame in the window

    Each sample (dense mode, predict_all=True):
        x : (K, 768) float32  — same
        y : (K,) float32      — log1p(ttc_s) for ALL K frames in the window
                                gives 32× more gradient signal per batch
    """

    def __init__(
        self,
        feat_cache: Path,
        episode_ids_to_use: set[int],
        window_size: int,
        window_stride: int = 1,
        dense: bool = False,
    ):
        data         = np.load(feat_cache)
        all_features = data["features"]     # (N, 768)
        all_ttc      = data["ttc_s"]        # (N,)
        all_ep_ids   = data["episode_ids"]  # (N,)

        self.K     = window_size
        self.dense = dense

        # Store per-episode arrays once; build a flat index of (ep_slot, start)
        self.ep_data: list[tuple[np.ndarray, np.ndarray]] = []
        self.index:   list[tuple[int, int]] = []

        for ep_idx in sorted(episode_ids_to_use):
            mask = all_ep_ids == ep_idx
            if not mask.any():
                continue
            ep_feats = all_features[mask].astype(np.float32)
            ep_ttc   = all_ttc[mask].astype(np.float32)
            n = len(ep_feats)
            if n < window_size:
                continue

            slot = len(self.ep_data)
            self.ep_data.append((ep_feats, ep_ttc))

            for start in range(0, n - window_size + 1, window_stride):
                self.index.append((slot, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        slot, start = self.index[i]
        feats, ttc  = self.ep_data[slot]
        x = torch.from_numpy(feats[start : start + self.K])             # (K, 768)
        if self.dense:
            y = torch.from_numpy(np.log1p(ttc[start : start + self.K]).astype(np.float32))  # (K,)
        else:
            y = torch.tensor(np.log1p(ttc[start + self.K - 1]), dtype=torch.float32)        # scalar
        return x, y


def make_splits(
    feat_cache: Path,
    hdf5_path: Path,
    cfg: dict,
) -> tuple[TTCWindowDataset, TTCWindowDataset, TTCWindowDataset]:
    """Return (train_ds, val_ds, test_ds) with episode-level splits."""

    data       = np.load(feat_cache)
    ep_ids_arr = data["episode_ids"]
    n_ep       = int(ep_ids_arr.max()) + 1

    # Resolve excluded episode names → indices
    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))

    excluded: set[int] = set()
    for name in cfg.get("excluded_episodes", []):
        if name in ep_keys:
            excluded.add(ep_keys.index(name))
        else:
            print(f"  WARNING: excluded episode '{name}' not found in HDF5")

    # Test set: last test_fraction of all episodes
    n_test   = max(1, round(n_ep * cfg["test_fraction"]))
    test_eps = set(range(n_ep - n_test, n_ep)) - excluded

    # Remaining → train + val (split by val_fraction)
    remaining = [i for i in range(n_ep - n_test) if i not in excluded]
    n_val     = max(1, round(len(remaining) * cfg["val_fraction"]))
    val_eps   = set(remaining[-n_val:])
    train_eps = set(remaining[:-n_val])

    K     = cfg["window_size"]
    S     = cfg["window_stride"]
    dense = cfg.get("predict_all", False)

    train_ds = TTCWindowDataset(feat_cache, train_eps, K, S, dense=dense)
    val_ds   = TTCWindowDataset(feat_cache, val_eps,   K, 1, dense=dense)
    test_ds  = TTCWindowDataset(feat_cache, test_eps,  K, 1, dense=dense)

    print(f"Episodes : {n_ep} total  |  excluded={sorted(excluded)}")
    print(f"           train={sorted(train_eps)}")
    print(f"           val  ={sorted(val_eps)}")
    print(f"           test ={sorted(test_eps)}")
    print(f"Windows  : {len(train_ds):,} train  |  {len(val_ds):,} val  |  {len(test_ds):,} test")

    return train_ds, val_ds, test_ds
