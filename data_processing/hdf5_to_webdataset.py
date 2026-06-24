"""
hdf5_to_webdataset.py  –  Convert HDF5 episodes to WebDataset shards for TTC training.

Each sample = one timestep.  Per-sample files inside the tar:

    {key}.{cam}.jpg     – JPEG-encoded camera frame (one file per selected camera)
    {key}.json          – TTC label + episode metadata

Time-to-completion (TTC) label
───────────────────────────────
    ttc_s         = (episode end_ns  –  frame timestamp_ns) / 1e9
    ttc_normalized = ttc_s / episode_duration_s   (1.0 at start → 0.0 at end)

    Both are written to the .json so you can choose which to regress against.
    ttc_normalized is useful when episodes have varying lengths; ttc_s in seconds
    is useful when you want absolute time predictions.

Episode filtering
─────────────────
    --success-only   keep only episodes where exit_type == "success"
                     (recommended: failed episodes have ambiguous TTC labels)

Shard policy
────────────
    Episodes are never split across shards.  A new shard is opened before each
    episode that would push the current shard past --shard-size samples.

Usage
─────
    python -m data_processing.hdf5_to_webdataset \\
        --hdf5       output/hdf5/dataset.h5 \\
        --output-dir output/webdataset \\
        --cameras    head_left left_wrist right_wrist \\
        --shard-size 500 \\
        --success-only

Training recipe (PyTorch)
─────────────────────────
    import webdataset as wds, json
    ds = (
        wds.WebDataset("output/webdataset/shard-{000000..000015}.tar")
        .decode("rgb8")
        .to_tuple("head_left.jpg", "json")
        .map(lambda img, meta: (img, json.loads(meta)["ttc_s"]))
    )

Architecture notes
──────────────────
    Recommended starting point for TTC regression:
      1. Frozen DINO-v2 or CLIP ViT → 768-d feature per frame
      2. Temporal window: sample K frames at ~0.5 s spacing (K=8 = 4 s context)
      3. Causal transformer (nhead=8, nlayers=2) over the K feature vectors
      4. Regression MLP head → scalar ttc_s (or ttc_normalized)
      5. Huber loss (robust to rare large-TTC outliers near episode start)

    Baseline before adding temporal: frozen DINO-v2 + linear regression.
    Normalise output as log1p(ttc_s) to handle the skewed distribution.

    Only train on successful episodes.  Failed episodes can be used to train a
    separate binary success-prediction head if desired.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Core converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_ttc_webdataset(
    hdf5_path: Path,
    output_dir: Path,
    cameras: Sequence[str] | None = None,
    shard_size: int = 500,
    jpeg_quality: int = 90,
    success_only: bool = False,
) -> None:
    """Convert an HDF5 episode file to WebDataset shards with TTC labels.

    Parameters
    ----------
    hdf5_path    : path to dataset.h5 produced by rrd_to_hdf5.py
    output_dir   : directory to write shard-*.tar + index.json
    cameras      : camera names to include (None = all cameras present)
    shard_size   : target number of frame-samples per shard; episodes are
                   never split across shards
    jpeg_quality : JPEG encoding quality for camera frames (1–95)
    success_only : if True, skip episodes where exit_type != "success"
    """
    hdf5_path  = Path(hdf5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Shard state ───────────────────────────────────────────────────────────
    global_sample_idx = 0
    shard_idx         = 0
    shard_samples     = 0
    manifest: list[dict] = []
    current_tar: tarfile.TarFile | None = None
    current_shard_path: Path | None = None

    def _open_shard() -> None:
        nonlocal current_tar, current_shard_path, shard_idx, shard_samples
        if current_tar is not None:
            current_tar.close()
            manifest.append({
                "shard":     current_shard_path.name,  # type: ignore[union-attr]
                "n_samples": shard_samples,
            })
        current_shard_path = output_dir / f"shard-{shard_idx:06d}.tar"
        current_tar        = tarfile.open(current_shard_path, "w")
        shard_idx         += 1
        shard_samples      = 0

    def _write(key: str, ext: str, data: bytes) -> None:
        buf  = io.BytesIO(data)
        info = tarfile.TarInfo(name=f"{key}.{ext}")
        info.size = len(data)
        current_tar.addfile(info, buf)  # type: ignore[union-attr]

    _open_shard()  # open shard-000000

    # ── Episode loop ──────────────────────────────────────────────────────────
    ttc_values: list[float] = []   # collect for summary stats

    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        print(f"HDF5: {hdf5_path.name}  |  {len(ep_keys)} episodes")

        for ep_global_idx, ep_key in enumerate(ep_keys):
            ep = f[ep_key]

            exit_type       = str(ep.attrs.get("exit_type",       ""))
            workflow_status = str(ep.attrs.get("workflow_status", ""))

            if success_only and exit_type.lower() not in (
                "success", "succeeded", "complete", "completed"
            ):
                print(f"  SKIP {ep_key}  exit_type={exit_type!r}")
                continue

            # ── Timing ────────────────────────────────────────────────────
            ts_ns    = ep["timestamps"][:].astype(np.int64)
            end_ns   = int(ep.attrs.get("end_ns",   ts_ns[-1]))
            start_ns = int(ep.attrs.get("start_ns", ts_ns[0]))
            episode_duration_s = (end_ns - start_ns) / 1e9
            n_frames = len(ts_ns)

            # ── Camera selection ──────────────────────────────────────────
            avail_cams  = list(ep["cameras"].keys())
            ep_cams     = [c for c in (cameras or avail_cams) if c in avail_cams]
            if not ep_cams:
                print(f"  SKIP {ep_key}  no matching cameras (have {avail_cams})")
                continue

            multi_cam = len(ep_cams) > 1
            print(f"  {ep_key}  {n_frames}f  {episode_duration_s:.1f}s  "
                  f"exit={exit_type}  cams={ep_cams}")

            # ── Shard rotation (never mid-episode) ────────────────────────
            if shard_samples > 0 and shard_samples + n_frames > shard_size:
                _open_shard()

            # ── Frame loop ────────────────────────────────────────────────
            for i in range(n_frames):
                key   = f"{global_sample_idx:09d}"
                t_ns  = int(ts_ns[i])
                ttc_s = (end_ns - t_ns) / 1e9
                ttc_s = max(0.0, ttc_s)   # clamp: last frame rounding
                ttc_norm = (
                    ttc_s / episode_duration_s
                    if episode_duration_s > 0 else 0.0
                )
                ttc_values.append(ttc_s)

                # Camera frames
                for cam in ep_cams:
                    frame = ep["cameras"][cam][i]   # (H, W, 3) uint8
                    img   = Image.fromarray(frame)
                    buf   = io.BytesIO()
                    img.save(buf, format="JPEG", quality=jpeg_quality)
                    ext = f"{cam}.jpg" if multi_cam else "jpg"
                    _write(key, ext, buf.getvalue())

                # TTC label + metadata
                meta: dict = {
                    "ttc_s":              round(float(ttc_s),    4),
                    "ttc_normalized":     round(float(np.clip(ttc_norm, 0.0, 1.0)), 4),
                    "episode_idx":        ep_global_idx,
                    "frame_idx":          i,
                    "n_frames":           n_frames,
                    "progress":           round(i / max(1, n_frames - 1), 4),
                    "timestamp_ns":       t_ns,
                    "episode_duration_s": round(episode_duration_s, 3),
                    "workflow_status":    workflow_status,
                    "exit_type":          exit_type,
                }
                _write(key, "json", json.dumps(meta).encode())

                global_sample_idx += 1
                shard_samples     += 1

    # ── Close final shard ─────────────────────────────────────────────────────
    if current_tar is not None:
        current_tar.close()
        manifest.append({
            "shard":     current_shard_path.name,  # type: ignore[union-attr]
            "n_samples": shard_samples,
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_samples = sum(s["n_samples"] for s in manifest)
    ttc_arr       = np.array(ttc_values)

    index = {
        "n_shards":        len(manifest),
        "n_samples":       total_samples,
        "cameras":         list(cameras) if cameras is not None else [],
        "success_only":    success_only,
        "jpeg_quality":    jpeg_quality,
        "ttc_stats": {
            "mean_s":   round(float(ttc_arr.mean()),  2) if len(ttc_arr) else None,
            "std_s":    round(float(ttc_arr.std()),   2) if len(ttc_arr) else None,
            "max_s":    round(float(ttc_arr.max()),   2) if len(ttc_arr) else None,
            "median_s": round(float(np.median(ttc_arr)), 2) if len(ttc_arr) else None,
        },
        "shards": manifest,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))

    print(f"\n{'='*50}")
    print(f"Wrote {total_samples} samples across {len(manifest)} shards")
    print(f"TTC stats (s): mean={index['ttc_stats']['mean_s']}  "
          f"std={index['ttc_stats']['std_s']}  "
          f"max={index['ttc_stats']['max_s']}")
    print(f"Manifest: {index_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Convert HDF5 robot episodes to WebDataset shards with "
            "per-frame Time-to-Completion (TTC) labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hdf5",         type=Path, required=True,
                   help="Path to dataset.h5 from rrd_to_hdf5.py")
    p.add_argument("--output-dir",   type=Path, default=Path("output/webdataset"))
    p.add_argument("--cameras",      nargs="*",
                   help="Camera names to include (default: all)")
    p.add_argument("--shard-size",   type=int, default=500,
                   help="Target samples per shard (episodes are never split)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--success-only", action="store_true",
                   help="Skip episodes where exit_type != success")
    args = p.parse_args()

    convert_to_ttc_webdataset(
        hdf5_path    = args.hdf5,
        output_dir   = args.output_dir,
        cameras      = args.cameras or None,
        shard_size   = args.shard_size,
        jpeg_quality = args.jpeg_quality,
        success_only = args.success_only,
    )


if __name__ == "__main__":
    main()
