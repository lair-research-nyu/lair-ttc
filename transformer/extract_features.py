"""
extract_features.py — Extract frozen DINO-v2 features from an HDF5 dataset.

Run from repo root with hf-env active:
    python -m transformer.extract_features --hdf5 output/hdf5/cube-to-box/dataset.h5 \
        --output output/cube-to-box/dino_features.npz --camera head_left

Resumable: per-episode checkpoints are saved so extraction can be interrupted
and restarted. Only incomplete episodes are re-processed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


DINO_MODEL_ID = "facebook/dinov2-base"
DINO_DIM      = 768


# ──────────────────────────────────────────────────────────────────────────────

def load_dino(device: torch.device):
    from transformers import AutoModel, AutoImageProcessor
    print(f"Loading {DINO_MODEL_ID} …")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model     = AutoModel.from_pretrained(DINO_MODEL_ID).to(device).eval()
    print("  Done.")
    return processor, model


@torch.no_grad()
def extract_episode(
    frames: np.ndarray,
    processor,
    dino: torch.nn.Module,
    device: torch.device,
    batch: int = 32,
) -> np.ndarray:
    """Extract (N, 768) DINO [CLS] features for one episode's frames."""
    feats = []
    for i in range(0, len(frames), batch):
        batch_frames = [frames[j] for j in range(i, min(i + batch, len(frames)))]
        inputs = processor(images=batch_frames, return_tensors="pt").to(device)
        out    = dino(**inputs)
        feats.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())
    return np.concatenate(feats, axis=0)


# ──────────────────────────────────────────────────────────────────────────────

def extract_all(
    hdf5_path: Path,
    output_path: Path,
    camera: str = "head_left",
    batch: int = 32,
    device_str: str = "auto",
) -> None:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"Device  : {device}")
    print(f"HDF5    : {hdf5_path}")
    print(f"Camera  : {camera}")
    print(f"Output  : {output_path}")

    ckpt_dir = output_path.parent / "episodes"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor, dino = load_dino(device)

    with h5py.File(hdf5_path, "r") as f:
        ep_keys  = sorted(k for k in f.keys() if k.startswith("episode_"))
        n_total  = len(ep_keys)

        # Per-episode checkpoints
        def ckpt_path(ep_idx: int) -> Path:
            return ckpt_dir / f"ep_{ep_idx:04d}.npz"

        todo = [i for i in range(n_total) if not ckpt_path(i).exists()]
        done = n_total - len(todo)
        print(f"\nEpisodes: {n_total} total  |  {done} already done  |  {len(todo)} to process\n")

        for ep_idx in todo:
            ep_key = ep_keys[ep_idx]
            ep     = f[ep_key]

            if camera not in ep.get("cameras", {}):
                print(f"  {ep_key}: camera '{camera}' not found — skipping")
                np.savez(ckpt_path(ep_idx), features=np.zeros((0, DINO_DIM), np.float32),
                         ttc_s=np.zeros(0, np.float32))
                continue

            frames = ep["cameras"][camera][:]   # (N, H, W, 3) uint8

            # Compute TTC from timestamps
            ts  = ep["timestamps"][:]           # (N,) int64 nanoseconds
            end = int(ep.attrs.get("end_ns", ts[-1]))
            ttc = np.maximum(0.0, (end - ts) / 1e9).astype(np.float32)

            print(f"  {ep_key}: {len(frames)} frames, TTC {ttc[0]:.1f}->{ttc[-1]:.1f}s", flush=True)
            feats = extract_episode(frames, processor, dino, device, batch)

            np.savez(ckpt_path(ep_idx), features=feats, ttc_s=ttc)
            print(f"    saved checkpoint  ({feats.shape})")

    # Merge all checkpoints into final .npz
    print("\nMerging checkpoints …")
    all_features, all_ttc, all_ep_ids = [], [], []
    for ep_idx, ep_key in enumerate(ep_keys):
        cp = ckpt_path(ep_idx)
        if not cp.exists():
            print(f"  WARNING: missing checkpoint for {ep_key} — skipping")
            continue
        data = np.load(cp)
        if len(data["features"]) == 0:
            continue
        all_features.append(data["features"])
        all_ttc.append(data["ttc_s"])
        all_ep_ids.append(np.full(len(data["features"]), ep_idx, dtype=np.int64))

    features    = np.concatenate(all_features, axis=0)
    ttc_s       = np.concatenate(all_ttc,      axis=0)
    episode_ids = np.concatenate(all_ep_ids,   axis=0)

    np.savez(output_path, features=features, ttc_s=ttc_s, episode_ids=episode_ids)
    print(f"Saved  : {output_path}  ({len(features):,} frames, {len(ep_keys)} episodes)")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Extract claud-v2 features from HDF5 dataset")
    p.add_argument("--hdf5",    type=Path, required=True, help="Path to dataset.h5")
    p.add_argument("--output",  type=Path, required=True, help="Output .npz path")
    p.add_argument("--camera",  default="head_left",      help="Camera stream name")
    p.add_argument("--batch",   type=int, default=32,     help="Frames per DINO batch")
    p.add_argument("--device",  default="auto",           help="cuda / cpu / auto")
    args = p.parse_args()
    extract_all(args.hdf5, args.output, args.camera, args.batch, args.device)


if __name__ == "__main__":
    main()
